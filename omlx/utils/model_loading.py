# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with post-load transforms."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
    quantization: str | None = None,
):
    """Load an LLM model/tokenizer pair via transformers.

    Args:
        model_name: HuggingFace model name or local path.
        tokenizer_config: Extra kwargs for AutoTokenizer.from_pretrained.
        quantization: Weight quantization mode. Supported values:
            - "4bit": 4-bit NF4 quantization via bitsandbytes (CUDA only).
            - "8bit": 8-bit INT8 quantization via bitsandbytes (CUDA only).
            - None: Full precision (float16 on CUDA, float32 on CPU).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_config or {}))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    quant_mode = (quantization or "").lower().strip()
    if quant_mode in ("4bit", "int4") and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
            logger.info(f"Loading {model_name} with 4-bit NF4 quantization")
        except ImportError:
            logger.warning("bitsandbytes not installed — falling back to fp16")
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
    elif quant_mode in ("8bit", "int8") and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
            logger.info(f"Loading {model_name} with 8-bit INT8 quantization")
        except ImportError:
            logger.warning("bitsandbytes not installed — falling back to fp16")
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
    elif quant_mode and device == "cpu":
        logger.warning(
            f"Quantization '{quantization}' requires CUDA — loading in fp32 on CPU"
        )
        load_kwargs["torch_dtype"] = torch.float32
    elif device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device == "cpu" and "device_map" not in load_kwargs:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def apply_post_load_transforms(model: Any, model_settings: Any = None) -> Any:
    """Apply optional post-load model transforms based on settings.

    Currently supports:
    - IndexCache: skip redundant indexer computation in DSA layers
    - GatedDeltaNet advance: fix missing cache.advance() in qwen3_5

    Args:
        model: A loaded mlx-lm model instance.
        model_settings: A ModelSettings instance (or None).

    Returns:
        The (possibly patched) model.
    """
    # GatedDeltaNet advance patch: always applied for qwen3_5 models
    # (no settings needed — auto-detected by model type)
    from ..patches.gated_delta_advance import apply_gated_delta_advance_patch

    if apply_gated_delta_advance_patch(model):
        logger.info("GatedDeltaNet advance() patch applied")

    if model_settings is None:
        return model

    index_cache_freq = getattr(model_settings, "index_cache_freq", None)
    if index_cache_freq is not None and index_cache_freq >= 2:
        from ..patches.index_cache import apply_index_cache

        applied = apply_index_cache(model, index_cache_freq)
        if applied:
            logger.info(f"IndexCache applied: freq={index_cache_freq}")

    return model
