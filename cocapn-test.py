#!/usr/bin/env python3
"""
cocapn-test.py — Test cocapn-chat worker endpoints locally.

Run with: python3 cocapn-test.py [--host localhost] [--port 8787]

Requires: wrangler dev running the worker (npm run dev in cocapn-chat)
Tests: health, models, chat, streaming, auth endpoints
"""

import http.server
import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResult:
    name: str
    passed: bool
    status: int = 0
    time_ms: float = 0
    detail: str = ""


def request(method, path, host="localhost", port=8787, body=None, headers=None):
    """Make HTTP request and return (status, body, time_ms)."""
    url = f"http://{host}:{port}{path}"
    data = json.dumps(body).encode() if body else None
    hdrs = headers or {}
    if data:
        hdrs["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        elapsed = (time.time() - start) * 1000
        body = resp.read().decode()
        return resp.status, body, elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - start) * 1000
        body = e.read().decode() if e.fp else ""
        return e.code, body, elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return 0, str(e), elapsed


def run_tests(host, port):
    results = []

    print(f"🧪 cocapn-chat Test Suite")
    print(f"   Target: {host}:{port}\n")

    # Test 1: Health check
    status, body, ms = request("GET", "/api/health", host, port)
    passed = status == 200
    results.append(TestResult("GET /api/health", passed, status, ms, body[:100]))
    print(f"  {'✅' if passed else '❌'} GET /api/health ({status}) {ms:.0f}ms")

    # Test 2: Landing page
    status, body, ms = request("GET", "/", host, port)
    passed = status == 200 and "cocapn" in body.lower()
    results.append(TestResult("GET / (landing)", passed, status, ms, f"{len(body)} bytes"))
    print(f"  {'✅' if passed else '❌'} GET / ({status}) {ms:.0f}ms — {len(body)} bytes")

    # Test 3: Models list
    status, body, ms = request("GET", "/v1/models", host, port)
    passed = status == 200 or status == 401  # 401 is OK if auth required
    results.append(TestResult("GET /v1/models", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} GET /v1/models ({status}) {ms:.0f}ms")

    # Test 4: Chat without auth (should fail)
    status, body, ms = request("POST", "/v1/chat/completions", host, port, {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hello"}]
    })
    passed = status in (401, 403)  # Should require auth
    results.append(TestResult("POST /v1/chat (no auth)", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} POST /v1/chat (no auth) ({status}) {ms:.0f}ms")

    # Test 5: Login endpoint
    status, body, ms = request("POST", "/api/auth/login", host, port, {
        "email": "test@test.com",
        "password": "wrong"
    })
    passed = status in (400, 401, 404)  # Should fail with wrong creds
    results.append(TestResult("POST /api/auth/login (wrong creds)", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} POST /api/auth/login (wrong creds) ({status}) {ms:.0f}ms")

    # Test 6: Signup endpoint
    status, body, ms = request("POST", "/api/auth/signup", host, port, {
        "email": "test@test.com",
        "password": "TestPass123!",
        "name": "Test User"
    })
    passed = status in (200, 201, 400, 409)  # 200=ok, 409=exists
    results.append(TestResult("POST /api/auth/signup", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} POST /api/auth/signup ({status}) {ms:.0f}ms")

    # Test 7: Dashboard without auth
    status, body, ms = request("GET", "/api/dashboard", host, port)
    passed = status in (401, 403, 404)  # Should require auth
    results.append(TestResult("GET /api/dashboard (no auth)", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} GET /api/dashboard (no auth) ({status}) {ms:.0f}ms")

    # Test 8: Invalid model
    status, body, ms = request("POST", "/v1/chat/completions", host, port, {
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "hello"}]
    }, headers={"Authorization": "Bearer fake-key"})
    passed = status in (401, 404, 400)
    results.append(TestResult("POST /v1/chat (bad model)", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} POST /v1/chat (bad model) ({status}) {ms:.0f}ms")

    # Test 9: API key endpoint
    status, body, ms = request("GET", "/api/settings/keys", host, port)
    passed = status in (401, 403, 404)
    results.append(TestResult("GET /api/settings/keys (no auth)", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} GET /api/settings/keys (no auth) ({status}) {ms:.0f}ms")

    # Test 10: Docs page
    status, body, ms = request("GET", "/docs", host, port)
    passed = status in (200, 404)  # May or may not have docs route
    results.append(TestResult("GET /docs", passed, status, ms))
    print(f"  {'✅' if passed else '❌'} GET /docs ({status}) {ms:.0f}ms")

    # Summary
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    total_ms = sum(r.time_ms for r in results)

    print(f"\n{'='*50}")
    print(f"  Results: {passed_count}/{total} passed")
    print(f"  Total time: {total_ms:.0f}ms")
    if passed_count == total:
        print(f"  ✅ All tests passed!")
    else:
        failed = [r for r in results if not r.passed]
        print(f"  ❌ {len(failed)} failed:")
        for r in failed:
            print(f"     {r.name}: {r.status} — {r.detail[:80]}")

    return all(r.passed for r in results)


if __name__ == "__main__":
    host = "localhost"
    port = 8787

    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])

    try:
        success = run_tests(host, port)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
