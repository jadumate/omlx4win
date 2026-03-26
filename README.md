<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX — Windows Edition</h1>
<p align="center"><b>LLM inference server for Windows</b><br>Continuous batching · OpenAI-compatible API · Admin dashboard · torch + transformers backend</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Windows-0078D6?logo=windows" alt="Windows">
  <img src="https://img.shields.io/badge/backend-PyTorch-EE4C2C?logo=pytorch" alt="PyTorch">
  <a href="https://buymeacoffee.com/jundot"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#features">Features</a> ·
  <a href="#models">Models</a> ·
  <a href="#cli-configuration">CLI Configuration</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

---

<p align="center">
  <img src="docs/images/omlx_dashboard.png" alt="oMLX Admin Dashboard" width="800">
</p>

> *This is the Windows port of oMLX, rewritten to run on PyTorch + HuggingFace Transformers instead of Apple's MLX framework. The original project targeted Apple Silicon — this version brings the same server, API, and admin dashboard to Windows with NVIDIA CUDA or CPU.*

---

## What's Different from the Original

| Feature | Original (macOS/MLX) | This Version (Windows/torch) |
|---------|---------------------|------------------------------|
| Inference backend | Apple MLX (Metal GPU) | PyTorch (CUDA / CPU) |
| Model format | MLX safetensors | HuggingFace safetensors / PyTorch |
| Platform | macOS 15+, Apple Silicon | Windows 10/11, Python 3.10+ |
| Menu bar app | Native PyObjC | Not included (server only) |
| KV cache tiers | Hot RAM + Cold SSD | In-process KV cache (past_key_values) |
| Quantization (oQ) | MLX-native | Not available in this port |
| TurboQuant KV | MLX Metal | Not available in this port |

The server, admin dashboard, OpenAI-compatible API, multi-model serving, and all HTTP endpoints are identical.

---

## Install

### Requirements

- Windows 10 or Windows 11
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended) — CPU inference also works

### 1. Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your CUDA version, or use one of these:

```powershell
# CUDA 12.1 (recommended for RTX 30/40 series)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install oMLX

```powershell
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .

# With MCP (Model Context Protocol) support
pip install -e ".[mcp]"
```

### Verify

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
omlx --help
```

---

## Quickstart

```powershell
omlx serve --model-dir C:\models
```

The server discovers LLMs, VLMs, embedding models, and rerankers from subdirectories automatically and starts on `http://localhost:8000`.

- **Admin dashboard**: `http://localhost:8000/admin`
- **Chat UI**: `http://localhost:8000/admin/chat`
- **API**: `http://localhost:8000/v1` (OpenAI-compatible)

Any OpenAI-compatible client can connect by setting `base_url = "http://localhost:8000/v1"`.

---

## Features

### Admin Dashboard

Web UI at `/admin` for real-time monitoring, model management, chat, benchmark, and per-model settings. Supports English, Korean, Japanese, and Chinese. All CDN dependencies are vendored for fully offline operation.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.45.34.png" alt="oMLX Admin Dashboard" width="720">
</p>

### Multi-Model Serving

Load LLMs, VLMs, embedding models, and rerankers in the same server process. Memory is managed automatically:

- **LRU eviction** — least-recently-used models are unloaded when memory runs low
- **Manual load/unload** — control model loading from the admin panel
- **Model pinning** — keep specific models always resident
- **Per-model TTL** — auto-unload idle models after a configurable timeout
- **Process memory limit** — set a total memory ceiling to prevent OOM

### Continuous Batching

Handles concurrent requests using per-request KV cache (torch `past_key_values`). Each request streams tokens independently while sharing the same GPU.

### Vision-Language Models

Run VLMs via `transformers.AutoModelForVision2Seq`. Supports multi-image chat, base64/URL/file image inputs, and OCR models.

### Built-in Chat

Chat with any loaded model directly from the admin panel. Supports conversation history, model switching, dark mode, reasoning model output, and image upload for VLM/OCR models.

<p align="center">
  <img src="docs/images/ScreenShot_2026-03-14_104350_610.png" alt="oMLX Chat" width="720">
</p>

### Model Downloader

Search and download models from HuggingFace directly in the admin dashboard.

<p align="center">
  <img src="docs/images/downloader_omlx.png" alt="oMLX Model Downloader" width="720">
</p>

### Per-Model Settings

Configure sampling parameters, chat template kwargs, TTL, model alias, model type override, and more per model from the admin panel — no server restart needed.

### API Compatibility

Drop-in replacement for OpenAI and Anthropic APIs.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming) |
| `POST /v1/completions` | Text completions (streaming) |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/rerank` | Document reranking |
| `GET /v1/models` | List available models |

### Tool Calling

Supports function calling via the model's chat template `tools` parameter. Works with any HuggingFace model whose tokenizer supports `apply_chat_template(tools=...)`.

### Claude Code Integration

Works as a local API backend for Claude Code. SSE keep-alive prevents read timeouts during long prefill, and context scaling support helps trigger auto-compact at the right time.

### MCP Support

Use oMLX as an MCP server or connect to external MCP tools:

```powershell
pip install "omlx[mcp]"
omlx serve --model-dir C:\models --mcp-config mcp.json
```

---

## Models

Point `--model-dir` at a directory containing HuggingFace model subdirectories. Two-level organization (`org/model-name/`) is also supported.

```
C:\models\
├── Llama-3.2-3B-Instruct\
├── Qwen2.5-7B-Instruct\
├── mistral-7b-instruct-v0.3\
├── bge-m3\
└── bge-reranker-v2-m3\
```

Models are auto-detected by architecture. You can also download models from the admin dashboard.

| Type | Detection | Examples |
|------|-----------|---------|
| LLM | `AutoModelForCausalLM` | Llama, Qwen, Mistral, DeepSeek, Gemma |
| VLM | Vision architecture detected | LLaVA, Qwen2-VL, InternVL |
| Embedding | BERT/RoBERTa/E5 architecture | BGE-M3, all-MiniLM, ModernBERT |
| Reranker | SequenceClassification or CausalLM reranker | BGE-reranker, Qwen3-Reranker |

> **Model format**: This Windows port uses standard HuggingFace models (safetensors or PyTorch `.bin`). The original oMLX used MLX-converted models from `mlx-community`. For best CUDA performance, use `float16` weights. 4-bit quantization works via `bitsandbytes` (install separately with `pip install bitsandbytes`).

---

## CLI Configuration

```powershell
# Basic serve
omlx serve --model-dir C:\models

# Memory limit for loaded models
omlx serve --model-dir C:\models --max-model-memory 16GB

# Adjust concurrent request limit
omlx serve --model-dir C:\models --max-num-seqs 8

# Custom host and port
omlx serve --model-dir C:\models --host 0.0.0.0 --port 8080

# With MCP tools
omlx serve --model-dir C:\models --mcp-config mcp.json

# HuggingFace mirror endpoint (for restricted regions)
omlx serve --model-dir C:\models --hf-endpoint https://hf-mirror.com

# API key authentication
omlx serve --model-dir C:\models --api-key your-secret-key

# Set log level
omlx serve --model-dir C:\models --log-level debug
```

All settings can also be configured from the web admin panel at `/admin`. Settings are persisted to `~\.omlx\settings.json`, and CLI flags take precedence.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMLX_MODEL_DIR` | `~/.omlx/models` | Model directory path |
| `OMLX_PORT` | `8000` | Server port |
| `OMLX_HOST` | `127.0.0.1` | Listen host |
| `OMLX_API_KEY` | _(none)_ | API authentication key |
| `OMLX_LOG_LEVEL` | `info` | Logging level |
| `OMLX_HF_ENDPOINT` | _(HuggingFace)_ | HuggingFace mirror URL |
| `OMLX_MCP_CONFIG` | _(none)_ | MCP configuration file |

<details>
<summary>Architecture</summary>

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (multi-model, LRU eviction, TTL, manual load/unload)
    │   ├── BatchedEngine  (LLMs — torch AutoModelForCausalLM)
    │   ├── VLMEngine      (vision-language — torch AutoModelForVision2Seq)
    │   ├── EmbeddingEngine (torch AutoModel + mean pooling)
    │   └── RerankerEngine  (torch AutoModelForSequenceClassification)
    │
    ├── ProcessMemoryEnforcer (total memory limit, TTL checks)
    │
    └── Scheduler (FCFS, per-request past_key_values streaming)
```

</details>

---

## Development

```powershell
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

---

## Contributing

Contributions are welcome! See [Contributing Guide](docs/CONTRIBUTING.md) for details.

- Bug fixes and improvements
- Performance optimizations (batched decode, paged attention)
- Additional model support
- Documentation

---

## License

[Apache 2.0](LICENSE)

---

## Acknowledgments

- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — oMLX's continuous batching architecture originated from vllm-mlx
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — model loading and inference backbone for this Windows port
- [PyTorch](https://pytorch.org) — inference backend
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple — the original backend (macOS version)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — VLM support in the original
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — embedding support in the original
