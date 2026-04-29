#!/usr/bin/env python3
"""
edge-chat.py — Local AI chat server for Jetson Orin Nano.

Runs entirely on-device using Ollama. No cloud API needed.
Web UI with streaming responses. Built for cocapn.ai edge product.

Usage:
  python3 edge-chat.py                 # Start server on port 8080
  python3 edge-chat.py --port 3000     # Custom port
  python3 edge-chat.py --model qwen3.5:2b  # Custom model
  python3 edge-chat.py --list          # List available models

Features:
  - Streaming chat via Ollama REST API (localhost:11434)
  - Multi-model support (switch models mid-conversation)
  - Conversation history
  - System prompts
  - Token counting
  - GPU/RAM monitoring in UI
  - Embeddings endpoint for RAG
"""

import json
import time
import os
import sys
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError
import threading

# Config
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-r1:1.5b"
WORKSPACE = os.path.expanduser("~/.openclaw/workspace")

# Store conversations in memory
conversations = {}
system_prompts = {}

# Available models (pre-verified to work on Jetson 8GB)
JETSON_MODELS = {
    "deepseek-r1:1.5b": {"size": "1.1GB", "speed": "61 t/s", "type": "reasoning"},
    "moondream": {"size": "1.7GB", "speed": "79 t/s", "type": "vision"},
    "nomic-embed-text": {"size": "274MB", "speed": "15,922 t/s", "type": "embedding"},
}


def ollama_request(endpoint, data=None, method="GET", stream=False):
    """Make request to Ollama API."""
    url = f"{OLLAMA_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    req = Request(url, data=json.dumps(data).encode() if data else None,
                  headers=headers, method=method)

    try:
        resp = urlopen(req, timeout=300)
        if stream:
            return resp
        return json.loads(resp.read().decode())
    except URLError as e:
        return {"error": f"Ollama not available: {e.reason}"}


def get_system_stats():
    """Get current system stats."""
    stats = {"timestamp": time.time()}

    # Thermal zones
    for tz in ["gpu-thermal", "cpu-thermal", "tj-thermal"]:
        try:
            with open(f"/sys/class/thermal/thermal_zone{''.join(str(i) for i in range(20)) if False else ''}", "rb"):
                pass
        except:
            pass

    # Read specific thermal zones
    thermal_map = {}
    try:
        import glob as g
        for tz_path in sorted(g.glob("/sys/class/thermal/thermal_zone*")):
            with open(os.path.join(tz_path, "type"), "rb") as f:
                raw = f.read()
            if raw:
                name = raw.decode().strip()
            with open(os.path.join(tz_path, "temp"), "rb") as f:
                raw = f.read()
            if raw:
                temp_raw = raw.decode().strip()
                if temp_raw and temp_raw != "0":
                    thermal_map[name] = round(int(temp_raw) / 1000.0, 1)
    except:
        pass

    stats["gpu_temp"] = thermal_map.get("gpu-thermal", "?")
    stats["cpu_temp"] = thermal_map.get("cpu-thermal", "?")

    # CMA
    try:
        with open("/proc/meminfo", "rb") as f:
            raw = f.read().decode()
        for line in raw.split("\n"):
            if "CmaTotal" in line:
                stats["cma_total_mb"] = int(line.split()[1]) // 1024
            elif "CmaFree" in line:
                stats["cma_free_mb"] = int(line.split()[1]) // 1024
            elif "MemAvailable" in line:
                stats["ram_available_mb"] = int(line.split()[1]) // 1024
            elif "MemTotal" in line:
                stats["ram_total_mb"] = int(line.split()[1]) // 1024
    except:
        pass

    return stats


def get_ollama_models():
    """Get list of available Ollama models."""
    result = ollama_request("/api/tags")
    if "error" in result:
        return []
    return [m["name"] for m in result.get("models", [])]


class EdgeChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for edge chat server."""

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html(EDGE_CHAT_HTML)
        elif self.path == "/api/models":
            models = get_ollama_models()
            self._send_json({
                "models": models,
                "recommended": list(JETSON_MODELS.keys()),
                "model_info": JETSON_MODELS,
            })
        elif self.path == "/api/stats":
            self._send_json(get_system_stats())
        elif self.path.startswith("/api/conversation/"):
            conv_id = self.path.split("/")[-1]
            self._send_json({
                "id": conv_id,
                "messages": conversations.get(conv_id, []),
                "system_prompt": system_prompts.get(conv_id, ""),
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/api/chat":
            self._handle_chat(data)
        elif self.path == "/api/embed":
            self._handle_embed(data)
        elif self.path == "/api/system":
            conv_id = data.get("conversation_id", "default")
            system_prompts[conv_id] = data.get("prompt", "")
            self._send_json({"ok": True})
        elif self.path == "/api/conversation/clear":
            conv_id = data.get("conversation_id", "default")
            conversations.pop(conv_id, None)
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_chat(self, data):
        """Handle chat request with streaming."""
        model = data.get("model", DEFAULT_MODEL)
        prompt = data.get("message", "")
        conv_id = data.get("conversation_id", "default")

        if not prompt:
            self._send_json({"error": "No message"}, 400)
            return

        # Build message history
        messages = [{"role": "user", "content": m["content"]}
                    for m in conversations.get(conv_id, [])]
        messages.append({"role": "user", "content": prompt})

        sys_prompt = system_prompts.get(conv_id, "")
        ollama_data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if sys_prompt:
            ollama_data["system"] = sys_prompt

        # Stream response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        full_response = ""
        try:
            url = f"{OLLAMA_URL}/api/chat"
            req = Request(url, data=json.dumps(ollama_data).encode(),
                         headers={"Content-Type": "application/json"})
            resp = urlopen(req, timeout=300)

            for line in resp:
                decoded = line.decode().strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                chunk = json.loads(decoded[6:])
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    self.wfile.write(f"data: {json.dumps({'content': content})}\n\n".encode())
                    self.wfile.flush()
                if chunk.get("done"):
                    # Final stats
                    stats = chunk.get("prompt_eval_count", 0), chunk.get("eval_count", 0)
                    self.wfile.write(f"data: {json.dumps({'done': True, 'prompt_tokens': stats[0], 'eval_tokens': stats[1]})}\n\n".encode())
                    self.wfile.flush()
                    break
        except Exception as e:
            self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
            self.wfile.flush()
            return

        # Save to conversation
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append({"role": "user", "content": prompt})
        conversations[conv_id].append({"role": "assistant", "content": full_response})

    def _handle_embed(self, data):
        """Handle embedding request."""
        model = data.get("model", "nomic-embed-text")
        texts = data.get("texts", [])
        if not texts:
            self._send_json({"error": "No texts provided"}, 400)
            return

        result = ollama_request("/api/embed", {
            "model": model,
            "input": texts,
        })
        self._send_json(result)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


EDGE_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Edge Chat — Jetson Orin Nano</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; height: 100vh; display: flex; }
.sidebar { width: 280px; background: #111118; border-right: 1px solid #1e1e2e; display: flex; flex-direction: column; }
.sidebar-header { padding: 16px; border-bottom: 1px solid #1e1e2e; }
.sidebar-header h1 { font-size: 16px; color: #8b5cf6; }
.sidebar-header .sub { font-size: 11px; color: #666; margin-top: 4px; }
.model-select { width: 100%; background: #1a1a2e; color: #e0e0e0; border: 1px solid #2a2a3e; padding: 8px; border-radius: 6px; font-size: 12px; margin-top: 8px; }
.stats { padding: 12px 16px; border-bottom: 1px solid #1e1e2e; font-size: 11px; }
.stats .label { color: #666; }
.stats .value { color: #a78bfa; font-weight: 600; }
.stats .row { display: flex; justify-content: space-between; margin-bottom: 4px; }
.conversations { flex: 1; overflow-y: auto; padding: 8px; }
.conv-item { padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; margin-bottom: 2px; }
.conv-item:hover { background: #1a1a2e; }
.conv-item.active { background: #2a1a4e; color: #a78bfa; }
.main { flex: 1; display: flex; flex-direction: column; }
.chat-header { padding: 12px 20px; border-bottom: 1px solid #1e1e2e; display: flex; justify-content: space-between; align-items: center; }
.chat-header .title { font-size: 14px; font-weight: 600; }
.chat-header .tokens { font-size: 11px; color: #666; }
.messages { flex: 1; overflow-y: auto; padding: 20px; }
.msg { margin-bottom: 16px; max-width: 720px; }
.msg.user { margin-left: auto; }
.msg.assistant { margin-right: auto; }
.msg .bubble { padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
.msg.user .bubble { background: #2a1a4e; color: #e0d4f7; border-bottom-right-radius: 4px; }
.msg.assistant .bubble { background: #1a1a2e; border-bottom-left-radius: 4px; }
.msg .meta { font-size: 10px; color: #444; margin-top: 4px; }
.input-area { padding: 16px 20px; border-top: 1px solid #1e1e2e; }
.input-row { display: flex; gap: 8px; }
.input-row textarea { flex: 1; background: #1a1a2e; color: #e0e0e0; border: 1px solid #2a2a3e; padding: 10px 14px; border-radius: 8px; font-size: 14px; resize: none; height: 44px; max-height: 200px; font-family: inherit; }
.input-row textarea:focus { outline: none; border-color: #8b5cf6; }
.input-row button { background: #8b5cf6; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 14px; }
.input-row button:hover { background: #7c3aed; }
.input-row button:disabled { background: #3a3a4e; cursor: not-allowed; }
.system-input { width: 100%; background: #0f0f1a; color: #888; border: 1px solid #1e1e2e; padding: 6px 10px; border-radius: 4px; font-size: 11px; margin-top: 8px; }
.typing { display: inline-block; }
.typing::after { content: '...'; animation: dots 1.5s infinite; }
@keyframes dots { 0% { content: ''; } 25% { content: '.'; } 50% { content: '..'; } 75% { content: '...'; } }
@media (max-width: 768px) { .sidebar { display: none; } }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    <h1>⚡ Edge Chat</h1>
    <div class="sub">Jetson Orin Nano — On-Device AI</div>
    <select class="model-select" id="modelSelect">
      <option value="deepseek-r1:1.5b">deepseek-r1:1.5b (61 t/s)</option>
      <option value="moondream">moondream (79 t/s)</option>
    </select>
  </div>
  <div class="stats" id="stats">
    <div class="row"><span class="label">GPU</span><span class="value" id="gpuTemp">—</span></div>
    <div class="row"><span class="label">CMA</span><span class="value" id="cmaFree">—</span></div>
    <div class="row"><span class="label">RAM</span><span class="value" id="ramFree">—</span></div>
  </div>
  <div class="conversations" id="convList"></div>
</div>
<div class="main">
  <div class="chat-header">
    <span class="title" id="chatTitle">New Conversation</span>
    <span class="tokens" id="tokenCount"></span>
  </div>
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <div class="input-row">
      <textarea id="input" placeholder="Message (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
      <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
    <input class="system-input" id="systemInput" placeholder="System prompt (optional) — e.g. 'You are a helpful assistant'">
  </div>
</div>
<script>
const API = '';
let currentConv = 'default';
let convs = {};
let streaming = false;

async function loadModels() {
  try {
    const r = await fetch(API + '/api/models');
    const d = await r.json();
    const sel = document.getElementById('modelSelect');
    sel.innerHTML = '';
    d.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      const info = d.model_info[m];
      opt.textContent = info ? `${m} (${info.speed})` : m;
      sel.appendChild(opt);
    });
  } catch(e) { console.error(e); }
}

async function updateStats() {
  try {
    const r = await fetch(API + '/api/stats');
    const s = await r.json();
    document.getElementById('gpuTemp').textContent = s.gpu_temp + '°C';
    document.getElementById('cmaFree').textContent = (s.cma_free_mb || 0) + '/' + (s.cma_total_mb || 0) + 'MB';
    const ram = s.ram_available_mb;
    document.getElementById('ramFree').textContent = ram ? ram + 'MB free' : '—';
  } catch(e) {}
  setTimeout(updateStats, 5000);
}

async function sendMessage() {
  if (streaming) return;
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  input.style.height = '44px';

  const model = document.getElementById('modelSelect').value;
  const sysPrompt = document.getElementById('systemInput').value.trim();

  // Save system prompt
  if (sysPrompt) {
    await fetch(API + '/api/system', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({conversation_id: currentConv, prompt: sysPrompt})
    });
  }

  // Add user message to UI
  addMessage('user', msg);

  // Add assistant placeholder
  const assistDiv = addMessage('assistant', '');
  streaming = true;
  document.getElementById('sendBtn').disabled = true;

  try {
    const resp = await fetch(API + '/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model, message: msg, conversation_id: currentConv})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let totalTokens = 0;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.content) {
            fullText += data.content;
            assistDiv.querySelector('.bubble').textContent = fullText;
          }
          if (data.done) {
            totalTokens = (data.prompt_tokens || 0) + (data.eval_tokens || 0);
          }
        } catch(e) {}
      }
    }

    if (totalTokens) {
      document.getElementById('tokenCount').textContent = totalTokens + ' tokens';
    }
  } catch(e) {
    assistDiv.querySelector('.bubble').textContent = 'Error: ' + e.message;
  }

  streaming = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('input').focus();
}

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="bubble">${escapeHtml(content)}</div><div class="meta">${role} · ${new Date().toLocaleTimeString()}</div>`;
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({behavior: 'smooth'});
  return div;
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

document.getElementById('input').addEventListener('input', function() {
  this.style.height = '44px';
  this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

loadModels();
updateStats();
</script>
</body>
</html>"""


def main():
    port = 8080
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
        elif arg == "--list":
            models = get_ollama_models()
            print("Available models:")
            for m in models:
                info = JETSON_MODELS.get(m, {})
                print(f"  {m:50s} {info.get('speed', 'N/A'):>10s}  {info.get('type', '')}")
            return
        elif arg == "--model" and i + 1 < len(sys.argv):
            global DEFAULT_MODEL
            DEFAULT_MODEL = sys.argv[i + 1]

    # Check Ollama is running
    try:
        models = get_ollama_models()
        if not models:
            print("⚠️  No models found. Run 'ollama pull deepseek-r1:1.5b' first.")
            print("   Starting anyway — you can pull models later.")
        else:
            print(f"✅ Ollama connected — {len(models)} models available")
    except:
        print("⚠️  Cannot connect to Ollama. Make sure it's running: ollama serve")

    stats = get_system_stats()
    print(f"⚡ Edge Chat starting on http://0.0.0.0:{port}")
    print(f"   GPU: {stats.get('gpu_temp', '?')}°C  CMA: {stats.get('cma_free_mb', '?')}/{stats.get('cma_total_mb', '?')}MB")
    print(f"   Default model: {DEFAULT_MODEL}")
    print(f"   Press Ctrl+C to stop")

    server = HTTPServer(("0.0.0.0", port), EdgeChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopping edge-chat server")
        server.shutdown()


if __name__ == "__main__":
    main()
