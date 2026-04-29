"""Shared configuration for edge AI tools."""

import os

# Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("EDGE_MODEL", "deepseek-r1:1.5b")
EMBED_MODEL = os.environ.get("EDGE_EMBED_MODEL", "nomic-embed-text")

# Paths
WORKSPACE = os.environ.get("EDGE_WORKSPACE", os.path.expanduser("~/.openclaw/workspace"))
DATA_DIR = os.path.join(WORKSPACE, "memory")
RAG_DIR = os.path.join(DATA_DIR, "rag-index")
BENCH_DIR = os.path.join(DATA_DIR, "gpu-benchmarks")

# Limits
MAX_REQUEST_BODY = 1 * 1024 * 1024  # 1MB
CHAT_TIMEOUT = 600  # 10 minutes
EMBED_TIMEOUT = 60
HEALTH_TIMEOUT = 5

# Chunking
CHUNK_SIZE = 512  # characters
CHUNK_OVERLAP = 64  # characters
TOP_K = 5  # search results
RELEVANCE_THRESHOLD = 0.2

# Jetson-specific
CMA_TOTAL_MB = 256  # default, may be higher with boot arg
THERMAL_THROTTLE_C = 80.0
THERMAL_WARN_C = 60.0
