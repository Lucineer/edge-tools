"""Shared Ollama API client for edge tools."""

import json
import hmac
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from .config import OLLAMA_URL, CHAT_TIMEOUT, EMBED_TIMEOUT, HEALTH_TIMEOUT


def ollama_request(endpoint, data=None, timeout=60):
    """Make a request to the Ollama API.

    Args:
        endpoint: API path (e.g., "/api/tags", "/api/chat")
        data: Optional dict to POST as JSON
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response dict. Has {"error": "..."} key on failure.
    """
    url = f"{OLLAMA_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    try:
        if data is not None:
            req = Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")
        else:
            req = Request(url, headers=headers, method="GET")

        resp = urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode())
    except HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def ollama_health():
    """Check if Ollama is reachable."""
    resp = ollama_request("/api/tags", timeout=HEALTH_TIMEOUT)
    return "error" not in resp


def ollama_list_models():
    """List available models."""
    resp = ollama_request("/api/tags", timeout=HEALTH_TIMEOUT)
    if "error" in resp:
        return []
    return resp.get("models", [])


def ollama_chat(messages, model=None, system=None, stream=False):
    """Send a chat completion request.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        model: Model name (uses default if None)
        system: Optional system prompt
        stream: If True, returns the raw HTTP response for SSE streaming

    Returns:
        If stream=False: parsed JSON response dict
        If stream=True: HTTP response object for line-by-line reading
    """
    from .config import DEFAULT_MODEL
    if model is None:
        model = DEFAULT_MODEL

    data = {"model": model, "messages": messages, "stream": stream}
    if system:
        data["system"] = system

    if stream:
        url = f"{OLLAMA_URL}/api/chat"
        req = Request(url, data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            return urlopen(req, timeout=CHAT_TIMEOUT)
        except Exception as e:
            return None
    else:
        return ollama_request("/api/chat", data, timeout=CHAT_TIMEOUT)


def ollama_generate(prompt, model=None, system=None):
    """Generate text (completion API).

    Args:
        prompt: Text prompt
        model: Model name
        system: Optional system prompt

    Returns:
        Generated text string, or empty string on error.
    """
    from .config import DEFAULT_MODEL
    if model is None:
        model = DEFAULT_MODEL

    data = {"model": model, "prompt": prompt, "stream": False}
    if system:
        data["system"] = system

    resp = ollama_request("/api/generate", data, timeout=CHAT_TIMEOUT)
    if "error" in resp:
        return ""
    return resp.get("response", "")


def ollama_embed(texts, model=None):
    """Get embeddings for one or more texts.

    Args:
        texts: String or list of strings
        model: Embedding model name

    Returns:
        If input is a string: single embedding vector (list of floats)
        If input is a list of strings: list of embedding vectors
        On error: empty list or empty vector
    """
    from .config import EMBED_MODEL
    if model is None:
        model = EMBED_MODEL

    if isinstance(texts, str):
        texts = [texts]

    resp = ollama_request("/api/embed", {"model": model, "input": texts}, timeout=EMBED_TIMEOUT)
    if "error" in resp:
        return [] if len(texts) > 1 else []

    embeddings = resp.get("embeddings", [])
    if len(texts) == 1:
        return embeddings[0] if embeddings else []
    return embeddings


def check_api_key(auth_header, expected_key):
    """Constant-time API key comparison.

    Args:
        auth_header: Value of Authorization header
        expected_key: Expected API key (without "Bearer " prefix)

    Returns:
        True if key matches
    """
    if not expected_key:
        return True  # No key required
    if not auth_header:
        return False
    expected = f"Bearer {expected_key}".encode()
    actual = auth_header.encode()
    return hmac.compare_digest(expected, actual)
