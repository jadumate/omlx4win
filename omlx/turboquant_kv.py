# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache: codebook-quantized KV compression for torch backend.

PyTorch port of the MLX TurboQuant algorithm. Key design:
  - MSE codec: random rotation + Lloyd-Max codebook quantization
  - Compresses past_key_values between decode steps
  - ~4x memory reduction at 4-bit, ~2x at 8-bit vs fp16
  - Trade-off: decompress/compress overhead per decode step

Unlike the MLX version (which uses Metal kernels for fused decode attention),
this torch version stores compressed KV on GPU and dequantizes before each
forward pass. The memory savings are the same; the fused kernel speedup is not.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Codebook generation (Beta distribution Lloyd-Max quantizer) — numpy, cached
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _build_codebook(dim: int, bits: int) -> np.ndarray:
    """Optimal scalar codebook for Beta(dim/2, dim/2) via Lloyd's algorithm."""
    n_levels = 1 << bits
    alpha = dim / 2.0
    rng = np.random.default_rng(seed=0)
    samples = 2.0 * rng.beta(alpha, alpha, size=100_000) - 1.0
    centroids = np.linspace(samples.min(), samples.max(), n_levels)
    for _ in range(100):
        dists = np.abs(samples[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)
        for j in range(n_levels):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = samples[mask].mean()
    return centroids.astype(np.float32)


@lru_cache(maxsize=16)
def _build_rotation(dim: int, seed: int) -> np.ndarray:
    """Random orthogonal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed=seed)
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return (Q * signs[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Bit packing helpers (torch)
# ---------------------------------------------------------------------------

def _pack_4bit(indices: "torch.Tensor") -> "torch.Tensor":
    """Pack uint8 indices (values 0–15) into 4-bit pairs, 2 per byte."""
    import torch
    N, D = indices.shape
    # Pad D to even if needed
    if D % 2 != 0:
        indices = torch.cat(
            [indices, torch.zeros(N, 1, dtype=torch.uint8, device=indices.device)], dim=1
        )
    even = indices[:, 0::2]
    odd = indices[:, 1::2]
    return (even | (odd.to(torch.uint8) << 4))  # (N, ceil(D/2))


def _unpack_4bit(packed: "torch.Tensor", D: int) -> "torch.Tensor":
    """Unpack 4-bit pairs back to uint8 indices."""
    import torch
    even = packed & 0x0F
    odd = (packed >> 4) & 0x0F
    interleaved = torch.stack([even, odd], dim=-1).reshape(packed.shape[0], -1)
    return interleaved[:, :D]


def _pack_3bit(indices: "torch.Tensor") -> "torch.Tensor":
    """Pack uint8 indices (values 0–7) into 3-bit groups, 8 values per 3 bytes."""
    import torch
    N, D = indices.shape
    pad = (8 - D % 8) % 8
    if pad:
        indices = torch.cat(
            [indices, torch.zeros(N, pad, dtype=torch.uint8, device=indices.device)], dim=1
        )
    g = indices.reshape(N, -1, 8).long()
    b0 = g[:, :, 0] | (g[:, :, 1] << 3) | ((g[:, :, 2] & 0x3) << 6)
    b1 = (g[:, :, 2] >> 2) | (g[:, :, 3] << 1) | (g[:, :, 4] << 4) | ((g[:, :, 5] & 0x1) << 7)
    b2 = (g[:, :, 5] >> 1) | (g[:, :, 6] << 2) | (g[:, :, 7] << 5)
    return torch.stack([b0, b1, b2], dim=-1).reshape(N, -1).to(torch.uint8)


def _unpack_3bit(packed: "torch.Tensor", D: int) -> "torch.Tensor":
    """Unpack 3-bit groups back to uint8 indices."""
    import torch
    N = packed.shape[0]
    groups = packed.shape[1] // 3
    p = packed.reshape(N, groups, 3).long()
    b0, b1, b2 = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    v0 = b0 & 0x7
    v1 = (b0 >> 3) & 0x7
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x7
    v3 = (b1 >> 1) & 0x7
    v4 = (b1 >> 4) & 0x7
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x7
    v6 = (b2 >> 2) & 0x7
    v7 = (b2 >> 5) & 0x7
    indices = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).reshape(N, -1)
    return indices[:, :D].to(torch.uint8)


# ---------------------------------------------------------------------------
# Per-dimension codec (rotation + codebook) — one per (dim, bits, seed)
# ---------------------------------------------------------------------------

class TorchTurboQuantCodec:
    """Codebook codec for one head dimension. Thread-safe after init."""

    def __init__(self, dim: int, bits: int, seed: int, device: str):
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.device = device
        self._init()

    def _init(self):
        import torch
        cb_np = _build_codebook(self.dim, self.bits)
        rot_np = _build_rotation(self.dim, self.seed)
        self.codebook = torch.tensor(cb_np, dtype=torch.float32, device=self.device)
        self.rotation = torch.tensor(rot_np, dtype=torch.float32, device=self.device)
        logger.debug(
            f"TurboQuant codec: dim={self.dim} bits={self.bits} "
            f"levels={1 << self.bits} device={self.device}"
        )

    def quantize(self, x: "torch.Tensor"):
        """Quantize (B, H, T, D) → (norms (B,H,T), packed (B,H,T,W)).

        norms: float32, packed: uint8.  Both on same device as x.
        """
        import torch
        B, H, T, D = x.shape
        flat = x.reshape(-1, D).to(torch.float32)  # (N, D)

        norms = flat.norm(dim=-1)  # (N,)
        inv = torch.where(norms > 1e-10, 1.0 / norms, torch.ones_like(norms))
        normalized = flat * inv[:, None]

        rotated = normalized @ self.rotation  # (N, D)

        # Nearest codebook entry — chunked to keep memory manageable
        CHUNK = 4096
        indices_list = []
        for i in range(0, rotated.shape[0], CHUNK):
            chunk = rotated[i : i + CHUNK]
            dists = (chunk.unsqueeze(-1) - self.codebook).abs()  # (C, D, n_levels)
            indices_list.append(dists.argmin(dim=-1).to(torch.uint8))
        indices = torch.cat(indices_list, dim=0)  # (N, D)

        if self.bits == 4:
            packed = _pack_4bit(indices)
        elif self.bits == 3:
            packed = _pack_3bit(indices)
        elif self.bits == 8:
            packed = indices
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

        return norms.reshape(B, H, T), packed.reshape(B, H, T, -1)

    def dequantize(self, norms: "torch.Tensor", packed: "torch.Tensor") -> "torch.Tensor":
        """Dequantize (B,H,T) norms + (B,H,T,W) packed → (B,H,T,D) float16."""
        import torch
        B, H, T = norms.shape
        N = B * H * T
        p_flat = packed.reshape(N, -1)

        if self.bits == 4:
            indices = _unpack_4bit(p_flat, self.dim).long()
        elif self.bits == 3:
            indices = _unpack_3bit(p_flat, self.dim).long()
        elif self.bits == 8:
            indices = p_flat.long()
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

        values = self.codebook[indices]  # (N, D)
        reconstructed = values @ self.rotation.T  # (N, D)
        n_flat = norms.reshape(N, 1)
        reconstructed = reconstructed * n_flat

        return reconstructed.to(torch.float16).reshape(B, H, T, self.dim)


# ---------------------------------------------------------------------------
# KV cache compressor — wraps a transformers past_key_values tuple
# ---------------------------------------------------------------------------

class TorchTurboQuantKVCache:
    """Compresses transformers past_key_values with TurboQuant.

    Usage::

        tq_cache = TorchTurboQuantKVCache(past_key_values, bits=4)
        # later, before forward pass:
        past_key_values = tq_cache.decompress()
        out = model(input_ids=..., past_key_values=past_key_values, ...)
        tq_cache = TorchTurboQuantKVCache(out.past_key_values, bits=4)
    """

    def __init__(self, past_key_values, bits: int = 4, seed: int = 0):
        if bits not in (3, 4, 8):
            raise ValueError(f"TurboQuant bits must be 3, 4, or 8; got {bits}")
        self.bits = bits
        self.seed = seed
        self._codecs: dict[int, TorchTurboQuantCodec] = {}
        self._compressed: list[tuple] = []
        self._compress(past_key_values)

    def _get_codec(self, dim: int, device: str) -> TorchTurboQuantCodec:
        if dim not in self._codecs:
            self._codecs[dim] = TorchTurboQuantCodec(dim, self.bits, self.seed, device)
        return self._codecs[dim]

    def _compress(self, past_key_values) -> None:
        for k, v in past_key_values:
            dim = k.shape[-1]
            codec = self._get_codec(dim, str(k.device))
            k_norms, k_packed = codec.quantize(k)
            v_norms, v_packed = codec.quantize(v)
            self._compressed.append((codec, k_norms, k_packed, v_norms, v_packed))

    def decompress(self) -> tuple:
        """Return decompressed past_key_values (tuple of (K, V) per layer)."""
        result = []
        for (codec, k_norms, k_packed, v_norms, v_packed) in self._compressed:
            k = codec.dequantize(k_norms, k_packed)
            v = codec.dequantize(v_norms, v_packed)
            result.append((k, v))
        return tuple(result)

    @property
    def nbytes(self) -> int:
        total = 0
        for _codec, k_norms, k_packed, v_norms, v_packed in self._compressed:
            total += k_norms.nbytes + k_packed.nbytes + v_norms.nbytes + v_packed.nbytes
        return total


# ---------------------------------------------------------------------------
# Backward-compat stubs (MLX classes used in batched.py / attention patch)
# ---------------------------------------------------------------------------

class _MLXStub:
    """Stub replacing MLX-specific TurboQuant classes on Windows."""
    def __init__(self, *args, **kwargs):
        pass


TurboQuantKVCache = _MLXStub
BatchTurboQuantKVCache = _MLXStub


def turboquant_enabled(bits, scheme=None) -> bool:
    """Check if TurboQuant should be used for given bits/scheme."""
    if scheme == "turboquant":
        return True
    if bits is not None and not float(bits).is_integer():
        return True
    return False
