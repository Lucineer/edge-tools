"""
edge/ — Shared modules for JC1 edge AI toolkit.
"""

from edge.config import OLLAMA_URL, DEFAULT_MODEL, WORKSPACE
from edge.ollama_client import ollama_request, ollama_chat, ollama_embed, ollama_generate
from edge.monitoring import get_snapshot, get_thermal, get_memory_info, get_cma
from edge.storage import EdgeStore
from edge.similarity import rank_results

__all__ = [
    "OLLAMA_URL", "DEFAULT_MODEL", "WORKSPACE",
    "ollama_request", "ollama_chat", "ollama_embed", "ollama_generate",
    "get_snapshot", "get_thermal", "get_memory_info", "get_cma",
    "EdgeStore",
    "rank_results",
]
