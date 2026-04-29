#!/usr/bin/env python3
"""
edge-gateway.py — Unified edge AI gateway for Jetson Orin Nano.

Single server that exposes:
  POST /v1/chat/completions     — OpenAI-compatible chat
  POST /v1/embeddings           — OpenAI-compatible embeddings
  POST /v1/rag/query            — RAG: search + generate
  GET  /v1/models               — List available models
  GET  /v1/stats                — System stats (GPU, RAM, CMA)
  GET  /v1/health               — Health check

Compatible with OpenAI SDK, LangChain, and any OpenAI-compatible client.
No cloud APIs — everything runs on-device via Ollama.

Usage:
  python3 edge-gateway.py                    # Start on port 11435
  python3 edge-gateway.py --port 8080        # Custom port
  python3 edge-gateway.py --api-key secret   # Require API key

Test with OpenAI SDK:
  from openai import OpenAI
  client = OpenAI(base_url="http://jetson:11435/v1", api_key="local")
  resp = client.chat.completions.create(model="deepseek-r1:1.5b", messages=[{"role":"user","content":"Hi"}])
"""

import json
import os
import sys
import time
import math
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import URLError
from datetime import datetime
from threading import Lock

# Import shared modules
sys.path.insert(0, os.path.dirname(__file__))
from edge.config import (
    OLLAMA_URL, DEFAULT_MODEL, WORKSPACE, RAG_DIR,
    MAX_REQUEST_BODY, RELEVANCE_THRESHOLD,
)
from edge.ollama_client import ollama_request, ollama_chat, ollama_embed, check_api_key
from edge.similarity import rank_results
from edge.monitoring import get_snapshot
from edge.storage import EdgeStore

# Server config
DEFAULT_PORT = 11435
DEFAULT_HOST = "127.0.0.1"  # Secure default
API_KEY = None  # Set via --api-key
REQUEST_LOG = []
_stats_lock = Lock()
store = EdgeStore()  # Persistent storage

# Track usage
usage_stats = {
    "start_time": datetime.now().isoformat(),
    "total_requests": 0,
    "chat_requests": 0,
    "embed_requests": 0,
    "rag_requests": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "errors": 0,
}


def _ollama_raw_request(endpoint, data=None, stream=False):
    """Low-level Ollama proxy (supports streaming response objects)."""
    url = f"{OLLAMA_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    req = Request(url, data=json.dumps(data).encode() if data else None,
                  headers=headers, method="POST" if data else "GET")
    try:
        resp = urlopen(req, timeout=600)
        if stream:
            return resp
        return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": str(e)}


def get_stats():
    """System stats using shared monitoring module."""
    snap = get_snapshot()
    snap["timestamp"] = time.time()
    snap["uptime_s"] = time.time() - (datetime.fromisoformat(usage_stats["start_time"]).timestamp())
    snap["usage"] = {
        k: v for k, v in usage_stats.items() if k != "start_time"
    }
    return snap


# RAG index cache (avoids reloading on every request)
_rag_cache = {}
_rag_cache_lock = Lock()


def rag_search(query, index_name="fleet-knowledge", top_k=5):
    """Search RAG index with caching."""
    path = os.path.join(RAG_DIR, f"{index_name}.json")
    if not os.path.exists(path):
        return []

    # Check cache
    with _rag_cache_lock:
        mtime = os.path.getmtime(path)
        if index_name in _rag_cache:
            cached_mtime, cached_index = _rag_cache[index_name]
            if cached_mtime == mtime:
                index = cached_index
            else:
                with open(path) as f:
                    index = json.load(f)
                _rag_cache[index_name] = (mtime, index)
        else:
            with open(path) as f:
                index = json.load(f)
            _rag_cache[index_name] = (mtime, index)

    q_vec = ollama_embed(query)
    if not q_vec:
        return []

    return rank_results(q_vec, index.get("chunks", []), top_k, RELEVANCE_THRESHOLD)


class GatewayHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible edge AI gateway."""

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _stream_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        usage_stats["total_requests"] += 1

        if self.path == "/v1/models":
            resp = ollama_request("/api/tags")
            models = resp.get("models", [])
            self._json({
                "object": "list",
                "data": [{"id": m["name"], "object": "model", "owned_by": "local",
                          "size": m.get("size", 0)} for m in models]
            })

        elif self.path == "/v1/stats":
            self._json(get_stats())

        elif self.path == "/v1/health":
            resp = ollama_request("/api/tags")
            healthy = "error" not in resp
            self._json({"status": "ok" if healthy else "degraded",
                        "ollama": "connected" if healthy else "disconnected",
                        "device": "Jetson Orin Nano 8GB"})

        elif self.path.startswith("/v1/conversations"):
            self._handle_conversations_get()

        elif self.path.startswith("/v1/usage"):
            self._handle_usage_get()

        else:
            self._json({"error": {"message": "Not found", "type": "not_found"}}, 404)

    def do_POST(self):
        with _stats_lock:
            usage_stats["total_requests"] += 1
        length = int(self.headers.get("Content-Length", 0))

        # Body size limit
        if length > MAX_REQUEST_BODY:
            self._json({"error": {"message": "Request body too large", "type": "invalid_request"}}, 413)
            return

        try:
            body = json.loads(self.rfile.read(length).decode()) if length else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._json({"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request"}}, 400)
            return

        # Auth check (constant-time comparison)
        if API_KEY:
            auth = self.headers.get("Authorization", "")
            if not check_api_key(auth, API_KEY):
                self._json({"error": {"message": "Invalid API key", "type": "auth_error"}}, 401)
                with _stats_lock:
                    usage_stats["errors"] += 1
                return

        if self.path == "/v1/chat/completions":
            self._handle_chat(body)
        elif self.path == "/v1/embeddings":
            self._handle_embeddings(body)
        elif self.path == "/v1/rag/query":
            self._handle_rag(body)
        elif self.path.startswith("/v1/conversations"):
            self._handle_conversations_post(body)
        else:
            self._json({"error": {"message": "Not found", "type": "not_found"}}, 404)

    def _handle_chat(self, body):
        """OpenAI-compatible chat completions."""
        model = body.get("model", "deepseek-r1:1.5b")
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        if not messages:
            self._json({"error": {"message": "No messages", "type": "invalid_request"}}, 400)
            return

        usage_stats["chat_requests"] += 1

        # Convert OpenAI format to Ollama
        ollama_messages = []
        system_msg = None
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                ollama_messages.append(m)

        ollama_data = {
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
        }
        if system_msg:
            ollama_data["system"] = system_msg

        if stream:
            self._stream_chat(ollama_data, model)
        else:
            resp = ollama_request("/api/chat", ollama_data)
            if "error" in resp:
                usage_stats["errors"] += 1
                self._json({"error": {"message": resp["error"]}}, 502)
                return

            prompt_tokens = resp.get("prompt_eval_count", 0)
            completion_tokens = resp.get("eval_count", 0)
            usage_stats["total_prompt_tokens"] += prompt_tokens
            usage_stats["total_completion_tokens"] += completion_tokens
            store.log_usage(model, "/v1/chat/completions", prompt_tokens, completion_tokens)

            # Auto-save to conversation if conv_id provided
            conv_id = body.get("conversation_id")
            if conv_id:
                store.add_message(conv_id, "assistant",
                    resp.get("message", {}).get("content", ""),
                    prompt_tokens, completion_tokens)

            self._json({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.get("message", {}).get("content", "")},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })

    def _stream_chat(self, ollama_data, model):
        """Stream chat in OpenAI SSE format."""
        self._stream_sse()
        chat_id = f"chatcmpl-{threading.get_ident()}-{int(time.time()*1000)}"
        chunk_idx = 0
        req = Request(f"{OLLAMA_URL}/api/chat",
                      data=json.dumps(ollama_data).encode(),
                      headers={"Content-Type": "application/json"})
        try:
            resp = urlopen(req, timeout=600)
            for line in resp:
                decoded = line.decode().strip()
                if not decoded.startswith("data: "):
                    continue
                chunk = json.loads(decoded[6:])
                content = chunk.get("message", {}).get("content", "")
                if content:
                    chunk_idx += 1
                    sse_data = {
                        "id": f"{chat_id}-{chunk_idx}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }],
                    }
                    self.wfile.write(f"data: {json.dumps(sse_data)}\n\n".encode())
                    self.wfile.flush()
                if chunk.get("done"):
                    usage_stats["total_prompt_tokens"] += chunk.get("prompt_eval_count", 0)
                    usage_stats["total_completion_tokens"] += chunk.get("eval_count", 0)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break
        except Exception as e:
            usage_stats["errors"] += 1
            self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
            self.wfile.flush()

    def _handle_embeddings(self, body):
        """OpenAI-compatible embeddings."""
        model = body.get("model", "nomic-embed-text")
        input_data = body.get("input", [])
        if isinstance(input_data, str):
            input_data = [input_data]

        usage_stats["embed_requests"] += 1

        resp = ollama_request("/api/embed", {"model": model, "input": input_data})
        if "error" in resp:
            usage_stats["errors"] += 1
            self._json({"error": {"message": resp["error"]}}, 502)
            return

        embeddings = resp.get("embeddings", [])
        total_tokens = sum(len(t.split()) for t in input_data)
        usage_stats["total_prompt_tokens"] += total_tokens

        self._json({
            "object": "list",
            "data": [{"object": "embedding", "embedding": emb, "index": i}
                     for i, emb in enumerate(embeddings)],
            "model": model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        })

    def _handle_rag(self, body):
        """RAG query: search + generate."""
        query = body.get("query", "")
        model = body.get("model", "deepseek-r1:1.5b")
        index = body.get("index", "fleet-knowledge")
        top_k = body.get("top_k", 5)

        if not query:
            self._json({"error": {"message": "No query provided"}}, 400)
            return

        usage_stats["rag_requests"] += 1

        start = time.time()
        results = rag_search(query, index, top_k)

        if not results:
            self._json({"answer": "No relevant documents found.", "sources": [], "elapsed_s": 0})
            return

        # Build context from relevant chunks
        context = "\n\n".join(
            f"[{c['source']}]: {c['text']}" for c, s in results if s > 0.2
        )
        sources = [{"source": c["source"], "score": round(s, 3), "text": c["text"][:200]}
                   for c, s in results]

        prompt = f"Based on the following context, answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        gen_resp = ollama_request("/api/generate", {
            "model": model, "prompt": prompt, "stream": False,
            "system": "Answer based only on provided context. Be concise."
        })

        answer = gen_resp.get("response", "Error generating response")
        elapsed = time.time() - start

        tokens = gen_resp.get("eval_count", 0)
        usage_stats["total_completion_tokens"] += tokens

        self._json({
            "answer": answer,
            "sources": sources,
            "elapsed_s": round(elapsed, 2),
            "model": model,
            "tokens": tokens,
        })

    def log_message(self, fmt, *args):
        pass  # Suppress access logs

    def _handle_conversations_get(self):
        """GET /v1/conversations — list conversations, or /v1/conversations/:id — get one."""
        parts = self.path.rstrip("/").split("/")
        if len(parts) >= 4:
            # Get specific conversation
            conv_id = parts[3]
            conv = store.get_conversation(conv_id)
            if conv:
                self._json(conv)
            else:
                self._json({"error": {"message": "Conversation not found", "type": "not_found"}}, 404)
        else:
            # List conversations
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            limit = int(qs.get("limit", [20])[0])
            offset = int(qs.get("offset", [0])[0])
            convs = store.list_conversations(limit, offset)
            self._json({"object": "list", "data": convs, "total": len(convs)})

    def _handle_conversations_post(self, body):
        """POST /v1/conversations — create, or /v1/conversations/:id/messages — add message."""
        parts = self.path.rstrip("/").split("/")
        if len(parts) >= 5 and parts[4] == "messages":
            # Add message to conversation
            conv_id = parts[3]
            role = body.get("role", "user")
            content = body.get("content", "")
            if not content:
                self._json({"error": {"message": "No content", "type": "invalid_request"}}, 400)
                return
            store.add_message(conv_id, role, content,
                              body.get("tokens_prompt", 0),
                              body.get("tokens_completion", 0))
            self._json({"status": "ok", "conversation_id": conv_id})
        elif len(parts) >= 4 and parts[3] == "search":
            # Search conversations
            query = body.get("query", "")
            results = store.search_conversations(query, body.get("limit", 10))
            self._json({"results": results})
        else:
            # Create new conversation
            model = body.get("model", "unknown")
            title = body.get("title")
            conv_id = store.create_conversation(model=model, title=title)
            self._json({"id": conv_id, "model": model}, 201)

    def _handle_usage_get(self):
        """GET /v1/usage — aggregated usage stats."""
        stats = store.get_usage_stats()
        self._json({"object": "list", "data": stats})

    def do_DELETE(self):
        with _stats_lock:
            usage_stats["total_requests"] += 1
        if API_KEY:
            auth = self.headers.get("Authorization", "")
            if not check_api_key(auth, API_KEY):
                self._json({"error": {"message": "Invalid API key", "type": "auth_error"}}, 401)
                return
        parts = self.path.rstrip("/").split("/")
        if len(parts) >= 4 and parts[3]:
            store.delete_conversation(parts[3])
            self._json({"status": "ok"})
        else:
            self._json({"error": {"message": "Not found", "type": "not_found"}}, 404)


def main():
    global API_KEY
    port = DEFAULT_PORT
    host = DEFAULT_HOST

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--api-key" and i + 1 < len(sys.argv):
            API_KEY = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Health check Ollama
    resp = ollama_request("/api/tags")
    if "error" in resp:
        print(f"⚠️  Ollama not available: {resp['error']}")
        print("   Start with: ollama serve")
    else:
        models = resp.get("models", [])
        print(f"✅ Ollama connected — {len(models)} models")

    stats = get_stats()
    print(f"⚡ Edge Gateway starting on http://0.0.0.0:{port}")
    print(f"   GPU: {stats.get('gpu_temp_c', '?')}°C  RAM: {stats.get('ram_available_mb', '?')}MB  CMA: {stats.get('cma_free_mb', '?')}/{stats.get('cma_total_mb', '?')}MB")
    if API_KEY:
        print(f"   API key required")
    print(f"   Endpoints:")
    print(f"     POST /v1/chat/completions  — OpenAI-compatible chat")
    print(f"     POST /v1/embeddings        — OpenAI-compatible embeddings")
    print(f"     POST /v1/rag/query         — RAG: search + generate")
    print(f"     GET  /v1/models             — List models")
    print(f"     GET  /v1/stats              — System stats")
    print(f"     GET  /v1/health             — Health check")

    server = ThreadingHTTPServer((host, port), GatewayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopping edge gateway")
        server.shutdown()


if __name__ == "__main__":
    main()
