# JC1 Tools — Edge AI Toolkit for Jetson Orin Nano

All tools run on-device with no cloud dependencies.

## Shared Modules (`edge/`)

| Module | Purpose |
|--------|---------|
| `edge/config.py` | Shared config (OLLAMA_URL, paths, limits) |
| `edge/ollama_client.py` | Ollama API client + API key auth |
| `edge/monitoring.py` | Thermal, CMA, RAM reading |
| `edge/similarity.py` | Cosine similarity, vector ranking |

## Quick Reference

| Tool | Port | Purpose |
|------|------|---------|
| `edge-gateway.py` | 11435 | OpenAI-compatible API (chat + embed + RAG) |
| `edge-chat.py` | 8080 | Local chat web UI |
| `edge-rag.py` | 8081 | RAG API server |
| `jetson-monitor.py` | — | CLI monitoring + stress test |
| `gpu-bench.py` | — | GPU benchmarks (Ollama + CUDA + thermal) |
| `tensorrt-bench.py` | — | TensorRT ONNX→TRT benchmarks |
| `fleet-health.py` | — | Fleet system health check |
| `plato-cron.py` | — | Scheduled task runner |
| `tile-graph.py` | — | Graph knowledge index |
| `skill-tree.py` | — | Self-evolving agent skills |
| `cocapn-health.py` | — | cocapn.ai product health monitor |

## Edge AI Stack

### Start everything
```bash
# 1. Start Ollama (if not running)
ollama serve &

# 2. Start the gateway (unified API)
python3 tools/edge-gateway.py --port 11435

# 3. (Optional) Start chat UI on a different port
python3 tools/edge-chat.py --port 8080
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
print(resp.choices[0].message.content)

# Embeddings
resp = client.embeddings.create(
    model="nomic-embed-text",
    input=["Search query"]
)
print(len(resp.data[0].embedding))  # 768
```

### Use with curl
```bash
# Health check
curl http://jetson:11435/v1/health

# Chat
curl -X POST http://jetson:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:1.5b","messages":[{"role":"user","content":"Hi"}]}'

# Embeddings
curl -X POST http://jetson:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-embed-text","input":["Hello world"]}'

# RAG query
curl -X POST http://jetson:11435/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Jetson GPU tips"}'
```

## Verified Models (Jetson Orin Nano 8GB)

| Model | Size | Speed | Works |
|-------|------|-------|-------|
| deepseek-r1:1.5b | 1.1GB | 61 t/s | ✅ Best LLM |
| moondream | 1.7GB | 79 t/s | ✅ Vision |
| nomic-embed-text | 274MB | 15,922 t/s | ✅ Embeddings |
| phi3:mini | 2.3GB | TBD | ✅ Pulled |
| qwen3.5:2b | 2.7GB | — | ❌ OOM (CMA) |
| nemotron-3-nano:4b | 2.8GB | — | ❌ OOM (CMA) |

## System Requirements

- NVIDIA Jetson Orin Nano 8GB (tested)
- Ollama 0.18+
- CUDA 12.6, TensorRT 10.3 (pre-installed on JetPack)
- Python 3.10+
- CMA ≥ 256MB (default), 2GB recommended for 4B models
