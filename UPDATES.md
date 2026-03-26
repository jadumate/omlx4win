# UPDATES

## 2026-03-26 (3) — Weight quantization, TurboQuant KV, Web UI

### New Features

#### Weight Quantization (BitsAndBytes)
Load models in 4-bit (NF4) or 8-bit (INT8) via BitsAndBytes to reduce VRAM usage.
Configured per-model in the admin dashboard or via API.

| Setting | VRAM reduction | Notes |
|---------|---------------|-------|
| `4bit` | ~75% vs fp16 | NF4 + double quant, compute in fp16 |
| `8bit` | ~50% vs fp16 | LLM.int8() |

| File | Change |
|------|--------|
| `omlx/utils/model_loading.py` | Added `quantization` parameter; builds `BitsAndBytesConfig` for 4-bit/8-bit on CUDA |
| `omlx/engine/batched.py` | Reads `quantization` from model settings and passes to loader |
| `omlx/engine/vlm.py` | Same BitsAndBytes support for VLM models |
| `omlx/model_settings.py` | Added `quantization: Optional[str]` field |
| `omlx/admin/routes.py` | Added `quantization` to `ModelSettingsRequest`; validates and persists; triggers model reload |

#### TurboQuant KV Cache (PyTorch port)
Codebook-based KV cache compression between decode steps. Reduces KV memory ~4x at 4-bit.
Original used Apple Metal kernels; this port dequantizes on GPU before each forward pass.

| File | Change |
|------|--------|
| `omlx/turboquant_kv.py` | Full rewrite: `TorchTurboQuantCodec` (Lloyd-Max codebook + random QR rotation), `TorchTurboQuantKVCache` (compress/decompress `past_key_values`). Supports 3-bit, 4-bit, 8-bit. Handles both legacy tuple-of-tuples and `DynamicCache` (transformers ≥ 4.36). |
| `omlx/scheduler.py` | `_RequestState` now compresses KV after prefill/decode and decompresses before next decode when TurboQuant is enabled |

#### Web UI — Models → Quantizer tab
New "Weight Quantization" card lists all LLM/VLM models with per-model controls:
- Quantization dropdown (fp16 / 4-bit NF4 / 8-bit INT8)
- TurboQuant KV toggle + bits selector (3 / 4 / 8)
- Apply button (saves settings + reloads model)

| File | Change |
|------|--------|
| `omlx/admin/templates/dashboard/_models.html` | Added Weight Quantization card to Quantizer tab |
| `omlx/admin/static/js/dashboard.js` | Added `wqSelected`, `wqTq`, `wqTqBits`, `wqSaving` state; `wqLlmModels()` filter; `applyWeightQuant()` API call |

### Fixes

| File | Fix |
|------|-----|
| `omlx/admin/static/js/dashboard.js` | Apply button was permanently grayed out — Alpine.js v3 proxy returns truthy wrapper for uninitialized keys; fixed by initializing `wqSaving[m.id] = false` in `loadModels()` and using `=== true` comparison in `:disabled` |
| `omlx/turboquant_kv.py` | `_iter_kv` now uses `_safe_pairs()` for legacy tuple-of-tuples, extracting only `item[0], item[1]` — guards against models that store extra tensors per layer (caused "too many values to unpack") |
| `omlx/scheduler.py` | Full traceback now logged on generation error (was only logging message string) |

---

## 2026-03-26 (2) — Bug fixes

### Fixes

| File | Fix |
|------|-----|
| `omlx/cache/paged_ssd_cache.py` | `os.sysconf` is POSIX-only; falls back to `psutil.virtual_memory().total` on Windows |
| `omlx/scheduler.py` | Added missing `initial_cache_blocks` field to `SchedulerConfig` |
| `omlx/admin/static/js/dashboard.js` | Default `hfMlxOnly` and `msMlxOnly` to `false` so model downloader shows all HuggingFace/ModelScope models instead of MLX-only |

---

## 2026-03-26 — Windows Port (torch + transformers)

Ported oMLX from Apple Silicon / MLX to Windows with PyTorch + HuggingFace Transformers.
Published to: https://github.com/jadumate/omlx4win

---

### Why

The original oMLX ran exclusively on Apple Silicon via Apple's MLX framework (Metal GPU).
This port makes the same LLM inference server — with its OpenAI-compatible API, admin dashboard,
multi-model serving, and continuous batching — available on Windows with NVIDIA CUDA or CPU.

---

### What Changed

#### Inference Backend (full replacement)

| File | Change |
|------|--------|
| `omlx/scheduler.py` | Complete rewrite. `mlx_lm.BatchGenerator` replaced with a torch-based scheduler using per-request `past_key_values` KV cache streaming. Maintains identical interface with `engine_core.py`. |
| `omlx/models/llm.py` | `mlx_lm.load` / `mlx_lm.generate` replaced with `transformers.AutoModelForCausalLM`. Added `MLXLanguageModel` alias for backward compatibility. |
| `omlx/models/embedding.py` | `mlx-embeddings` + `mlx.core` replaced with `transformers.AutoModel` + torch mean-pooling and L2 normalization. |
| `omlx/models/reranker.py` | Supports both `AutoModelForSequenceClassification` (encoder rerankers) and `AutoModelForCausalLM` (e.g. Qwen3-Reranker). |
| `omlx/models/vlm.py` | Removed `mlx.core` / `mlx.nn`. Replaced with a clean `VLMModelAdapter` stub compatible with the VLM engine. |
| `omlx/models/xlm_roberta.py` | MLX-native XLMRoberta implementation replaced with `ModelArgs`/`Model` stubs — transformers handles this architecture natively. |
| `omlx/models/base_model.py` | `mlx.core` array ops (`mx.sum`, `mx.clip`, `mx.linalg.norm`) replaced with `torch` tensor equivalents. |
| `omlx/engine/vlm.py` | `mlx_vlm.utils.load()` replaced with `transformers.AutoProcessor` + `AutoModelForVision2Seq`. |
| `omlx/utils/model_loading.py` | `mlx_lm.load()` replaced with `transformers.AutoModelForCausalLM` + `AutoTokenizer`. |

#### Engine & Server Layer

| File | Change |
|------|--------|
| `omlx/engine_core.py` | Removed `import mlx.core as mx`. Renamed `get_mlx_executor` → `get_torch_executor` with backward alias. |
| `omlx/engine_pool.py` | Removed `mlx.core`. Memory tracking via `psutil` and `torch.cuda` instead of `mx.get_active_memory()`. |
| `omlx/engine/embedding.py` | Removed `import mlx.core as mx`. |
| `omlx/engine/reranker.py` | Removed `import mlx.core as mx`. |
| `omlx/engine/batched.py` | Replaced `mlx_lm.load()` with torch model loader. Removed SpecPrefill (MLX-specific). |

#### Hardware & System

| File | Change |
|------|--------|
| `omlx/utils/hardware.py` | Complete rewrite. macOS syscalls (`sysctl`, `system_profiler`, `ioreg`) replaced with `psutil` + `platform` + `uuid`. `is_apple_silicon()` and `is_mlx_available()` always return `False`. |
| `omlx/cli.py` | Removed `import mlx.core as mx` / `mx.set_cache_limit()` block. |
| `omlx/memory_monitor.py` | `mx.get_active_memory()` replaced with `psutil.Process().memory_info().rss`. |
| `omlx/process_memory_enforcer.py` | MLX memory limit calls replaced with psutil equivalents or no-ops. |
| `omlx/api/thinking.py` | Logits processor made a no-op on torch backend (thinking budget not applicable). |

#### Stubs (MLX-specific features, not available on Windows)

| File | Change |
|------|--------|
| `omlx/turboquant_kv.py` | Replaced with no-op class stubs (`TurboQuantKVCache`, `BatchTurboQuantKVCache`). |
| `omlx/patches/specprefill.py` | MLX import wrapped in `try/except`. |
| `omlx/patches/turboquant_attention.py` | MLX import wrapped in `try/except`. |
| `omlx/optimizations.py` | MLX import wrapped in `try/except`. |

#### Dependencies

`pyproject.toml` — removed `mlx`, `mlx-lm`, `mlx-vlm`, `mlx-embeddings`. Added:
- `torch>=2.1.0`
- `accelerate>=0.26.0`
- `safetensors>=0.4.0`

#### Documentation

`README.md` — full rewrite for Windows/torch edition. Covers install (CUDA/CPU), quickstart,
feature list, supported model types, CLI reference, and a comparison table vs. the original macOS version.

---

### What Was NOT Changed

- `omlx/server.py` — FastAPI routes (identical)
- `omlx/admin/` — Web dashboard (identical)
- `omlx/api/` — API utilities and route handlers (identical)
- `omlx/engine/base.py` — Abstract engine interfaces (identical)
- `omlx/request.py` — Request/response data structures (identical)
- `omlx/cache/` — Cache files importable; MLX ops guarded by `try/except HAS_MLX`
- All MCP, eval, integrations, and settings modules

---

### Features Not Available in This Port

- **TurboQuant KV fused decode** — Metal kernel speedup not available; PyTorch port dequantizes on GPU instead (memory savings are identical)
- **SpecPrefill** — MLX sparse prefill for MoE models
- **oQ quantization** — MLX-native weight quantization tool
- **Tiered KV cache (SSD cold tier)** — requires MLX array serialization
- **macOS menu bar app** — PyObjC/macOS-only
- **Homebrew install** — macOS package manager
