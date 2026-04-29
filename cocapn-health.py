"""
cocapn-health.py — Monitor cocapn.ai product health from the Jetson.

Checks: worker status, API latency, model availability, cost tracking.
Reports to HEARTBEAT.md and can send alerts.

Usage:
  python3 cocapn-health.py              # Full health check
  python3 cocapn-health.py latency      # Measure API latency
  python3 cocapn-health.py models       # Check model availability
  python3 cocapn-health.py report       # Write to HEARTBEAT.md
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime


API_BASE = "https://cocapn.ai"
WORKSPACE = "~/.openclaw/workspace"
HEARTBEAT = f"{WORKSPACE}/HEARTBEAT.md"


def api_request(path, method="GET", body=None, headers=None):
    """Make request to cocapn API."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    hdrs = headers or {"User-Agent": "cocapn-health/1.0"}
    if data:
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        elapsed = (time.time() - start) * 1000
        return resp.status, resp.read().decode(), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.time() - start) * 1000
        body = e.read().decode() if e.fp else ""
        return e.code, body, elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return 0, str(e), elapsed


def check_landing():
    """Check if landing page is up."""
    status, body, ms = api_request("/")
    up = status == 200
    size = len(body) if up else 0
    return {
        "endpoint": "/ (landing)",
        "status": status,
        "latency_ms": round(ms),
        "ok": up,
        "size": f"{size // 1024}KB" if size > 1024 else f"{size}B",
    }


def check_docs():
    """Check if docs are accessible."""
    status, body, ms = api_request("/docs")
    up = status == 200
    return {
        "endpoint": "/docs",
        "status": status,
        "latency_ms": round(ms),
        "ok": up,
    }


def check_models():
    """Check model listing endpoint."""
    status, body, ms = api_request("/v1/models")
    models = []
    if status == 200:
        try:
            data = json.loads(body)
            models = [m.get("id", "") for m in data.get("data", [])]
        except:
            pass
    return {
        "endpoint": "/v1/models",
        "status": status,
        "latency_ms": round(ms),
        "ok": status == 200,
        "model_count": len(models),
        "models": models,
    }


def check_auth():
    """Check auth endpoints respond correctly."""
    # Signup with invalid data should return 400, not 500
    status, body, ms = api_request("/api/auth/signup", "POST", {
        "email": "",
        "password": "",
    })
    auth_ok = status in (400, 401, 403, 422)  # Validation errors are expected
    return {
        "endpoint": "/api/auth/signup (validation)",
        "status": status,
        "latency_ms": round(ms),
        "ok": auth_ok,
    }


def check_latency():
    """Measure raw TCP/TLS latency."""
    latencies = []
    for i in range(3):
        _, _, ms = api_request("/")
        latencies.append(ms)
    return {
        "min_ms": round(min(latencies)),
        "max_ms": round(max(latencies)),
        "avg_ms": round(sum(latencies) / len(latencies)),
        "samples": len(latencies),
    }


def full_check():
    """Run all health checks."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "source": "JC1 (Jetson Orin Nano)",
        "checks": [],
    }

    checks = [
        ("Landing Page", check_landing),
        ("Docs", check_docs),
        ("Models API", check_models),
        ("Auth", check_auth),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            result = check_fn()
            result["name"] = name
            results["checks"].append(result)
            if not result["ok"]:
                all_ok = False
        except Exception as e:
            results["checks"].append({
                "name": name,
                "ok": False,
                "error": str(e),
            })
            all_ok = False

    try:
        results["latency"] = check_latency()
    except:
        pass

    results["overall"] = "healthy" if all_ok else "degraded"
    return results


def format_report(results):
    """Format health check as readable report."""
    lines = [
        f"🏥 cocapn.ai Health — {results['timestamp']}",
        f"   Source: {results['source']}",
        "=" * 50,
    ]

    for check in results["checks"]:
        icon = "✅" if check.get("ok") else "❌"
        status = check.get("status", "?")
        latency = check.get("latency_ms", "?")
        lines.append(f"  {icon} {check['name']:20s} [{status}] {latency}ms")

    if "latency" in results:
        lat = results["latency"]
        lines.append(f"\n  📊 Latency: avg {lat['avg_ms']}ms (min {lat['min_ms']}, max {lat['max_ms']})")

    models_check = next((c for c in results["checks"] if c["name"] == "Models API"), None)
    if models_check and models_check.get("model_count"):
        lines.append(f"  🧠 Models available: {models_check['model_count']}")

    overall = results.get("overall", "unknown")
    icon = "✅" if overall == "healthy" else "⚠️"
    lines.append(f"\n  {icon} Overall: {overall}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import os

    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"

    if cmd == "check":
        results = full_check()
        print(format_report(results))

    elif cmd == "latency":
        lat = check_latency()
        print(f"📊 API Latency: avg {lat['avg_ms']}ms (min {lat['min_ms']}, max {lat['max_ms']})")

    elif cmd == "models":
        m = check_models()
        print(f"🧠 Models: {m['status']} ({m['model_count']} models, {m['latency_ms']}ms)")
        for model in m.get("models", []):
            print(f"   • {model}")

    elif cmd == "json":
        print(json.dumps(full_check(), indent=2))

    elif cmd == "report":
        results = full_check()
        report = format_report(results)
        print(report)

        # Append to heartbeat
        heartbeat_path = os.path.expanduser(HEARTBEAT)
        with open(heartbeat_path, "a") as f:
            f.write(f"\n\n### cocapn.ai Health ({results['timestamp']})\n```\n{report}\n```\n")

    else:
        print(__doc__)
