# JC1 Tools — Edge AI Toolkit for Jetson Orin Nano 🔧

On-device inference toolkit. All tools run locally with no cloud dependencies.
Native inference at 19 t/s via `libedge-cuda.so`. OpenAI-compatible API. RAG. Fleet mesh.

## Stack Overview

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│  edge-chat   │     │  curl/OpenAI  │     │  flato (port     │
│  (:8080)     │     │  SDK clients  │     │  4003)           │
└──────┬───────┘     └───────┬───────┘     └────────┬─────────┘
       │                     │                       │
       └──────────┬──────────┴──────────┬────────────┘
                  ▼                     ▼
        ┌─────────────────────────────────────────┐
        │         edge-gateway.py (:11435)         │
        │  OpenAI-compatible API                   │
        │  Mode routing | Smart model routing      │
        │  Native fallback | RAG | SSE streaming   │
        └────────┬────────────────┬────────────────┘
                 │                │
        ┌────────▼─────┐  ┌──────▼──────────┐
        │  Ollama      │  │  libedge-cuda.so │
        │  (local HTTP)│  │  (native 19 t/s) │
        └──────────────┘  └──────────────────┘
```

## Services

| Tool | Port | Purpose |
|------|------|---------|
| `edge-gateway.py` | 11435 | OpenAI-compatible API (chat + embed + RAG + native) |
| `edge-chat.py` | 8080 | Local chat web UI |
| `edge-rag.py` | 8081 | RAG API server |
| `edge-monitor-web.py` | 8082 | Live edge dashboard |
| flato MUD | 4003 | C telnet server with `/think`, `/gpu`, `/cuda` |

## Shared Modules (`edge/`)

| Module | Purpose |
|--------|---------|
| `edge/config.py` | Shared config (OLLAMA_URL, paths, limits) |
| `edge/ollama_client.py` | Ollama API client + API key auth |
| `edge/monitoring.py` | Thermal, CMA, RAM reading |
| `edge/similarity.py` | Cosine similarity, vector ranking |
| `edge/storage.py` | SQLite-backed conversation + tile persistence |
| `edge-router.py` | Auto-fallback router (Ollama → native → cloud) |

## Gateway Features

| Feature | How |
|---------|-----|
| **Native inference** | `?native=true` → libedge-cuda.so at 18 t/s |
| **Auto-fallback** | 2s Ollama health check → native when down |
| **SSE streaming** | `&stream=true` → per-token callback to HTTP |
| **Mode routing** | `?mode=optimizer|debugger|analyzer|general` → specialist CUDA prompts |
| **Model routing** | `gpt-3.5-turbo` → `deepseek-r1:1.5b` automatically |
| **Conversations** | SQLite-backed, persist with `conversation_id` |
| **OOM protection** | Rejects models >3GB with helpful error |
| **DeepSeek fallback** | Cloud API for models that don't fit on-device |

## Quick Start

```bash
# 1. Start gateway (native + Ollama, auto-fallback)
python3 tools/edge-gateway.py --port 11435

# 2. Chat via curl
curl -X POST http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:1.5b","messages":[{"role":"user","content":"Hi"}]}'

# 3. Native inference (bypasses Ollama)
curl "http://localhost:11435/v1/chat/completions?native=true" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:1.5b","messages":[{"role":"user","content":"Hi"}]}'

# 4. Streaming mode
curl "http://localhost:11435/v1/chat/completions?native=true&stream=true" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:1.5b","messages":[{"role":"user","content":"Hi"}]}'

# 5. Specialist mode
curl "http://localhost:11435/v1/chat/completions?mode=optimizer" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:1.5b","messages":[{"role":"user","content":"Optimize my CUDA kernel"}]}'
```

### Use with OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://jetson:11435/v1", api_key="local")

# Chat
resp = client.chat.completions.create(
    model="deepseek-r1:1.5b",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Chat with specialist mode (workaround: use base_url + query params)
client = OpenAI(base_url="http://jetson:11435/v1/chat/completions?mode=optimizer", api_key="local")

# Embeddings
resp = client.embeddings.create(
    model="nomic-embed-text",
    input=["Search query"]
)
```

## Verified Models (Jetson Orin Nano 8GB)

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| deepseek-r1:1.5b | 1.1GB | 61 t/s (Ollama) / 19 t/s (native CPU) | ✅ Best all-round |
| moondream | 1.7GB | 79 t/s | ✅ Vision model |
| nomic-embed-text | 274MB | 15,922 t/s | ✅ Embeddings |
| phi3:mini | 2.3GB | TBD | ✅ Pulled |
| qwen3.5:2b | 2.7GB | — | ❌ OOM (CMA depleted) |
| nemotron-3-nano:4b | 2.8GB | — | ❌ OOM (CMA depleted) |

## Other Tools

| Tool | Purpose |
|------|---------|
| `jetson-monitor.py` | CLI monitoring + stress test |
| `gpu-bench.py` | GPU benchmarks (Ollama + CUDA + thermal) |
| `tensorrt-bench.py` | TensorRT ONNX→TRT benchmarks |
| `fleet-health.py` | Fleet system health check |
| `plato-cron.py` | Scheduled task runner |
| `tile-graph.py` | Graph knowledge index |
| `skill-tree.py` | Self-evolving agent skills |
| `mesh-bridge.py` | Fleet mesh hub (Evennia ↔ Oracle1 ↔ Forgemaster) |
| `cocapn-health.py` | cocapn.ai product health monitor |

## System Requirements

- NVIDIA Jetson Orin Nano 8GB (tested)
- Ollama 0.18+
- CUDA 12.6, TensorRT 10.3 (JetPack)
- Python 3.10+
- CMA ≥ 256MB (currently 1792KB free — needs reboot)
