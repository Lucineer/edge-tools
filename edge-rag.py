#!/usr/bin/env python3
"""
edge-rag.py — Local RAG (Retrieval-Augmented Generation) pipeline for Jetson.

Uses nomic-embed-text for embeddings (15,922 t/s on-device!)
+ Ollama for generation. No cloud APIs needed.

Usage:
  python3 edge-rag.py index /path/to/docs          # Index documents
  python3 edge-rag.py search "query"               # Search indexed docs
  python3 edge-rag.py ask "question"               # RAG: search + generate
  python3 edge-rag.py ask "question" --model phi3:mini  # Custom model
  python3 edge-rag.py stats                        # Show index stats
  python3 edge-rag.py serve --port 8081            # RAG API server

Features:
  - Local embeddings via nomic-embed-text (Ollama)
  - Cosine similarity search
  - Markdown and text file parsing
  - Automatic chunking with overlap
  - Persistent index (JSON)
  - RAG pipeline: embed query → find relevant chunks → generate answer
"""

import json
import os
import sys
import time
import math
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEN_MODEL = "deepseek-r1:1.5b"
WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
DATA_DIR = os.path.join(WORKSPACE, "memory", "rag-index")
CHUNK_SIZE = 512  # chars
CHUNK_OVERLAP = 64  # chars
TOP_K = 5  # results to retrieve


def ollama_embed(texts):
    """Get embeddings for texts via Ollama."""
    if isinstance(texts, str):
        texts = [texts]
    req = Request(
        f"{OLLAMA_URL}/api/embed",
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urlopen(req, timeout=60)
    result = json.loads(resp.read().decode())
    return result.get("embeddings", [[]])[0] if len(texts) == 1 else result.get("embeddings", [])


def ollama_generate(prompt, model=DEFAULT_GEN_MODEL, system=None):
    """Generate text via Ollama."""
    data = {"model": model, "prompt": prompt, "stream": False}
    if system:
        data["system"] = system
    req = Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urlopen(req, timeout=300)
    result = json.loads(resp.read().decode())
    return result.get("response", "")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def parse_file(filepath):
    """Parse a file into text chunks with metadata."""
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        # Try UTF-8
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

        # Strip markdown headers for cleaner chunks (keep content)
        chunks = chunk_text(text)

        return [{
            "text": chunk,
            "source": os.path.basename(filepath),
            "path": filepath,
            "chunk_index": i,
            "total_chunks": len(chunks),
        } for i, chunk in enumerate(chunks) if chunk.strip()]
    except Exception as e:
        print(f"  ⚠️ Error parsing {filepath}: {e}")
        return []


def index_documents(paths, index_name="default"):
    """Index documents into the RAG store."""
    os.makedirs(DATA_DIR, exist_ok=True)
    index_path = os.path.join(DATA_DIR, f"{index_name}.json")

    # Load existing index
    index = {"name": index_name, "created": None, "updated": None, "documents": {}, "chunks": []}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    total_chunks = 0
    skipped_files = 0
    for path in paths:
        # Resolve path
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        if os.path.isdir(path):
            files = []
            for root, dirs, fnames in os.walk(path):
                for fname in fnames:
                    if fname.endswith((".md", ".txt", ".py", ".js", ".go", ".rs", ".html", ".css", ".json", ".yaml", ".yml", ".toml")):
                        files.append(os.path.join(root, fname))
            print(f"  📁 {path}: {len(files)} files found")
            for f in files:
                file_hash = hashlib.md5(open(f, "rb").read()).hexdigest()[:8]
                # Skip if file hasn't changed
                if f in index["documents"] and index["documents"][f].get("hash") == file_hash:
                    skipped_files += 1
                    continue
                # Remove old chunks from this file if re-indexing
                index["chunks"] = [c for c in index["chunks"] if c.get("path") != f]
                chunks = parse_file(f)
                if chunks:
                    index["documents"][f] = {"hash": file_hash, "chunks": len(chunks), "indexed": datetime.now().isoformat()}
                    total_chunks += len(chunks)
                    for chunk in chunks:
                        chunk["file_hash"] = file_hash
                        index["chunks"].append(chunk)
                    print(f"    ✅ {os.path.basename(f)}: {len(chunks)} chunks")
        elif os.path.isfile(path):
            file_hash = hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
            if path in index["documents"] and index["documents"][path].get("hash") == file_hash:
                skipped_files += 1
                continue
            # Remove old chunks from this file if re-indexing
            index["chunks"] = [c for c in index["chunks"] if c.get("path") != path]
            chunks = parse_file(path)
            if chunks:
                index["documents"][path] = {"hash": file_hash, "chunks": len(chunks), "indexed": datetime.now().isoformat()}
                total_chunks += len(chunks)
                for chunk in chunks:
                    chunk["file_hash"] = file_hash
                    index["chunks"].append(chunk)
                print(f"  ✅ {os.path.basename(path)}: {len(chunks)} chunks")
        else:
            print(f"  ⚠️ Not found: {path}")

    if skipped_files:
        print(f"  ⏭️  Skipped {skipped_files} unchanged files")

    if total_chunks == 0:
        print("  No chunks to embed.")
        return index

    # Embed all chunks
    print(f"\n  🔢 Embedding {total_chunks} chunks via {EMBED_MODEL}...")
    start = time.time()
    texts = [c["text"] for c in index["chunks"]]

    # Batch embed (Ollama supports batches)
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeds = ollama_embed(batch)
        if isinstance(embeds[0], list):
            all_embeddings.extend(embeds)
        else:
            all_embeddings.append(embeds)
        done = min(i + batch_size, len(texts))
        print(f"    [{done}/{len(texts)}]", end="\r")

    elapsed = time.time() - start
    for i, chunk in enumerate(index["chunks"]):
        if i < len(all_embeddings):
            chunk["embedding"] = all_embeddings[i]

    index["updated"] = datetime.now().isoformat()
    if not index["created"]:
        index["created"] = index["updated"]
    index["chunk_count"] = len(index["chunks"])
    index["embed_model"] = EMBED_MODEL

    # Save index
    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"\n  ✅ Indexed {total_chunks} chunks in {elapsed:.1f}s ({total_chunks/elapsed:.0f} chunks/s)")
    print(f"  💾 Saved to {index_path}")
    return index


def search_index(query, index_name="default", top_k=TOP_K):
    """Search the index for relevant chunks."""
    index_path = os.path.join(DATA_DIR, f"{index_name}.json")
    if not os.path.exists(index_path):
        print(f"❌ Index '{index_name}' not found. Run: edge-rag.py index <path>")
        return []

    with open(index_path) as f:
        index = json.load(f)

    # Embed query
    query_embed = ollama_embed(query)

    # Score all chunks
    scored = []
    for chunk in index["chunks"]:
        if "embedding" not in chunk:
            continue
        sim = cosine_similarity(query_embed, chunk["embedding"])
        scored.append((chunk, sim))

    # Sort by similarity
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def rag_ask(question, index_name="default", model=DEFAULT_GEN_MODEL):
    """RAG pipeline: search → generate."""
    results = search_index(question, index_name)

    if not results:
        return "No relevant documents found in the index.", []

    # Build context
    context_parts = []
    sources = []
    for chunk, score in results:
        if score > 0.3:  # relevance threshold
            context_parts.append(f"[{chunk['source']}]: {chunk['text']}")
            sources.append({"source": chunk["source"], "score": round(score, 3)})

    if not context_parts:
        return "No sufficiently relevant documents found.", []

    context = "\n\n".join(context_parts)

    prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

    start = time.time()
    answer = ollama_generate(prompt, model, system="You are a helpful assistant. Answer based only on the provided context. Cite your sources.")
    elapsed = time.time() - start

    return answer, sources, elapsed


def show_stats(index_name="default"):
    """Show index statistics."""
    index_path = os.path.join(DATA_DIR, f"{index_name}.json")
    if not os.path.exists(index_path):
        print(f"❌ Index '{index_name}' not found.")
        return

    with open(index_path) as f:
        index = json.load(f)

    print(f"📚 RAG Index: {index_name}")
    print(f"   Created: {index.get('created', '?')}")
    print(f"   Updated: {index.get('updated', '?')}")
    print(f"   Chunks: {index.get('chunk_count', len(index.get('chunks', [])))}")
    print(f"   Documents: {len(index.get('documents', {}))}")
    print(f"   Embed model: {index.get('embed_model', '?')}")

    # Check embedding dimension
    chunks_with_embed = [c for c in index.get("chunks", []) if "embedding" in c]
    if chunks_with_embed:
        dim = len(chunks_with_embed[0]["embedding"])
        print(f"   Embedding dim: {dim}")

    # File sizes
    for path, info in index.get("documents", {}).items():
        print(f"   📄 {os.path.basename(path)}: {info.get('chunks', 0)} chunks")

    index_size = os.path.getsize(index_path)
    print(f"   💾 Index size: {index_size / 1024:.1f} KB")


class RAGHandler(BaseHTTPRequestHandler):
    """HTTP handler for RAG API."""

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/api/stats":
            index_name = "default"
            index_path = os.path.join(DATA_DIR, f"{index_name}.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                self._send_json({
                    "name": index.get("name"),
                    "chunks": len(index.get("chunks", [])),
                    "documents": len(index.get("documents", {})),
                    "embed_model": index.get("embed_model"),
                })
            else:
                self._send_json({"error": "No index found"}, 404)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length).decode()) if content_length else {}

        if self.path == "/api/search":
            results = search_index(body.get("query", ""), body.get("index", "default"), body.get("top_k", TOP_K))
            self._send_json([{"text": c["text"], "source": c["source"], "score": round(s, 3)} for c, s in results])
        elif self.path == "/api/ask":
            result = rag_ask(body.get("question", ""), body.get("index", "default"), body.get("model", DEFAULT_GEN_MODEL))
            if len(result) == 3:
                answer, sources, elapsed = result
                self._send_json({"answer": answer, "sources": sources, "elapsed_s": round(elapsed, 2)})
            else:
                answer, sources = result
                self._send_json({"answer": answer, "sources": sources})
        else:
            self._send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        pass


def main():
    if len(sys.argv) < 2:
        print("edge-rag.py — Local RAG for Jetson Orin Nano")
        print("Usage:")
        print("  edge-rag.py index <path> [--name myindex]")
        print("  edge-rag.py search <query> [--top-k 5]")
        print("  edge-rag.py ask <question> [--model phi3:mini]")
        print("  edge-rag.py stats")
        print("  edge-rag.py serve [--port 8081]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        paths = []
        index_name = "default"
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--name" and i + 1 < len(sys.argv):
                index_name = sys.argv[i + 1]
                i += 2
            else:
                paths.append(sys.argv[i])
                i += 1
        if not paths:
            print("❌ Provide at least one path to index")
            sys.exit(1)
        index_documents(paths, index_name)

    elif cmd == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        if not query:
            print("❌ Provide a query")
            sys.exit(1)
        index_name = "default"
        top_k = 5
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--top-k" and i + 1 < len(sys.argv):
                top_k = int(sys.argv[i + 1])
                i += 2
            elif not sys.argv[i].startswith("--"):
                index_name = sys.argv[i]
                i += 1
            else:
                i += 1
        results = search_index(query, index_name, top_k)
        if results:
            print(f"\n🔍 Top {len(results)} results for: {query}\n")
            for chunk, score in results:
                print(f"  [{score:.3f}] {chunk['source']}")
                print(f"  {chunk['text'][:200]}...")
                print()
        else:
            print("No results found.")

    elif cmd == "ask":
        question = sys.argv[2] if len(sys.argv) > 2 else ""
        if not question:
            print("❌ Provide a question")
            sys.exit(1)
        model = DEFAULT_GEN_MODEL
        for i, arg in enumerate(sys.argv):
            if arg == "--model" and i + 1 < len(sys.argv):
                model = sys.argv[i + 1]

        print(f"🧠 Asking: {question}")
        print(f"   Model: {model}\n")

        result = rag_ask(question, "default", model)
        if len(result) == 3:
            answer, sources, elapsed = result
            print(f"⏱️  {elapsed:.1f}s\n")
        else:
            answer, sources = result
        print(answer)
        if sources:
            print(f"\n📚 Sources:")
            for s in sources:
                print(f"   {s['source']} (relevance: {s['score']:.3f})")

    elif cmd == "stats":
        name = sys.argv[2] if len(sys.argv) > 2 else "default"
        show_stats(name)

    elif cmd == "serve":
        port = 8081
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])

        print(f"📡 RAG API server on http://0.0.0.0:{port}")
        server = HTTPServer(("0.0.0.0", port), RAGHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
