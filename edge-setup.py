#!/usr/bin/env python3
"""
edge-setup.py — First-run setup wizard for Jetson edge AI toolkit.

Auto-detects Jetson hardware, recommends models that fit available memory,
pulls Ollama models, validates the full stack.

Usage:
  python3 edge-setup.py              # Interactive wizard
  python3 edge-setup.py --detect     # Just show hardware info
  python3 edge-setup.py --install    # Pull recommended models
  python3 edge-setup.py --validate   # Test the full stack
  python3 edge-setup.py --json       # JSON output

For cocapn.ai: this is what customers run after flashing the SD card.
"""

import json
import os
import sys
import subprocess
import time
import glob
import math
from datetime import datetime

# ── Hardware Detection ──────────────────────────────────────────────

def read_file(path):
    """Read a sysfs file safely."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if raw:
            return raw.decode().strip()
    except (OSError, ValueError):
        pass
    return None


def detect_jetson_model():
    """Detect Jetson model from device tree or nvram."""
    # Method 1: /proc/device-tree/model
    model_str = read_file("/proc/device-tree/model")
    if model_str:
        return model_str.rstrip("\x00")

    # Method 2: dpkg
    try:
        out = subprocess.run(
            ["dpkg-query", "-W", "nvidia-jetpack"],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return f"Jetson (Jetpack {out.stdout.split()[-1]})"
    except Exception:
        pass

    return "Unknown Jetson"


def detect_cuda():
    """Detect CUDA version and GPU info."""
    info = {}
    nvcc = "/usr/local/cuda/bin/nvcc"
    if os.path.exists(nvcc):
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
            for line in out.stderr.split("\n"):
                if "release" in line.lower():
                    info["cuda_version"] = line.split("release")[-1].strip().rstrip(",")
                    break
        except Exception:
            pass
    else:
        # Search for nvcc
        for path in glob.glob("/usr/local/cuda-*/bin/nvcc"):
            try:
                out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
                for line in out.stderr.split("\n"):
                    if "release" in line.lower():
                        info["cuda_version"] = line.split("release")[-1].strip().rstrip(",")
                        info["nvcc_path"] = path
                        break
            except Exception:
                pass
            break

    # Jetson-specific: /sys/module/nv_gpu/version
    ver = read_file("/sys/module/nv_gpu/version")
    if ver:
        info["gpu_driver"] = ver

    return info


def detect_memory():
    """Read RAM, swap, and CMA from /proc/meminfo."""
    mem = {}
    try:
        with open("/proc/meminfo", "rb") as f:
            raw = f.read().decode()
        for line in raw.split("\n"):
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            val_kb = int(parts[1].strip().split()[0])
            if key == "MemTotal":
                mem["ram_total_mb"] = val_kb // 1024
            elif key == "MemAvailable":
                mem["ram_available_mb"] = val_kb // 1024
            elif "CmaTotal" in key:
                mem["cma_total_mb"] = val_kb // 1024
            elif "CmaFree" in key:
                mem["cma_free_mb"] = val_kb // 1024
    except Exception:
        pass
    return mem


def detect_thermal():
    """Read current temperatures."""
    temps = {}
    for tz_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        name = read_file(os.path.join(tz_path, "type"))
        val = read_file(os.path.join(tz_path, "temp"))
        if name and val and val != "0":
            temps[name] = round(int(val) / 1000.0, 1)
    return temps


def detect_ollama():
    """Check if Ollama is installed and running."""
    result = {"installed": False, "running": False, "models": []}

    # Check binary
    try:
        out = subprocess.run(["which", "ollama"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            result["installed"] = True
            result["path"] = out.stdout.strip()
    except Exception:
        pass

    # Check running + list models
    try:
        out = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            result["running"] = True
            for line in out.stdout.strip().split("\n")[1:]:  # Skip header
                parts = line.split()
                if parts:
                    # Strip :latest tag for matching
                    name = parts[0].replace(":latest", "")
                    result["models"].append({
                        "name": name,
                        "full_name": parts[0],
                        "size": parts[1] if len(parts) > 1 else "?",
                    })
    except Exception:
        pass

    return result


def detect_tensorrt():
    """Check TensorRT availability."""
    result = {"installed": False, "version": None}

    # Python
    try:
        import tensorrt
        result["installed"] = True
        result["version"] = tensorrt.__version__
        result["python"] = True
    except ImportError:
        pass

    # CLI (trtexec)
    try:
        out = subprocess.run(["trtexec", "--version"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            result["installed"] = True
            result["cli"] = True
            for line in out.stderr.split("\n"):
                if "TensorRT" in line:
                    result["version"] = line.strip()
    except Exception:
        pass

    return result


def detect_software():
    """Detect installed software stack."""
    sw = {}
    sw["python"] = sys.version.split()[0]
    sw["ollama"] = detect_ollama()
    sw["tensorrt"] = detect_tensorrt()

    # Check for useful tools
    for tool in ["curl", "wget", "git", "pip3", "uv"]:
        try:
            out = subprocess.run(["which", tool], capture_output=True, text=True, timeout=5)
            sw[tool] = out.returncode == 0
        except Exception:
            sw[tool] = False

    return sw


# ── Model Recommendations ───────────────────────────────────────────

MODEL_DB = {
    "nomic-embed-text": {
        "size_mb": 274,
        "purpose": "Embeddings (768-dim)",
        "min_ram_mb": 512,
        "min_cma_mb": 16,
        "required": True,
        "category": "embedding",
    },
    "deepseek-r1:1.5b": {
        "size_mb": 1100,
        "purpose": "Fast reasoning (61 t/s)",
        "min_ram_mb": 2048,
        "min_cma_mb": 12,
        "recommended": True,
        "category": "chat",
    },
    "moondream": {
        "size_mb": 1700,
        "purpose": "Vision + chat (79 t/s)",
        "min_ram_mb": 2048,
        "min_cma_mb": 0,  # survives on unified RAM
        "recommended": True,
        "category": "chat",
    },
    "phi3:mini": {
        "size_mb": 2300,
        "purpose": "Balanced quality (3.8B params)",
        "min_ram_mb": 3072,
        "min_cma_mb": 64,
        "category": "chat",
    },
    "qwen3.5:2b": {
        "size_mb": 2700,
        "purpose": "Good quality, high RAM",
        "min_ram_mb": 4096,
        "min_cma_mb": 256,
        "category": "chat",
    },
    "qwen3.5:4b": {
        "size_mb": 2800,
        "purpose": "Best quality (needs 2GB+ CMA)",
        "min_ram_mb": 4096,
        "min_cma_mb": 2048,
        "category": "chat",
    },
    "nemotron-3-nano:4b": {
        "size_mb": 2800,
        "purpose": "NVIDIA optimized (needs 2GB+ CMA)",
        "min_ram_mb": 4096,
        "min_cma_mb": 2048,
        "category": "chat",
    },
}


def recommend_models(ram_mb, cma_mb):
    """Recommend models that fit the hardware."""
    installed = detect_ollama().get("models", [])
    installed_names = {m["name"] for m in installed}

    results = {"install": [], "installed": [], "blocked": []}

    for name, info in MODEL_DB.items():
        fits_ram = ram_mb >= info["min_ram_mb"]
        fits_cma = cma_mb >= info["min_cma_mb"]
        is_installed = name in installed_names

        if is_installed:
            results["installed"].append({name: info})
        elif fits_ram and fits_cma:
            results["install"].append({name: info})
        else:
            reason = []
            if not fits_ram:
                reason.append(f"needs {info['min_ram_mb']}MB RAM")
            if not fits_cma:
                reason.append(f"needs {info['min_cma_mb']}MB CMA (you have {cma_mb}MB)")
            results["blocked"].append({name: info, "reason": ", ".join(reason)})

    return results


def pull_model(name):
    """Pull an Ollama model."""
    print(f"  🔄 Pulling {name}...")
    start = time.time()
    try:
        result = subprocess.run(
            ["ollama", "pull", name],
            capture_output=True, text=True, timeout=600
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  ✅ {name} pulled in {elapsed:.0f}s")
            return True
        else:
            print(f"  ❌ {name} failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ {name} timed out")
        return False
    except Exception as e:
        print(f"  ❌ {name} error: {e}")
        return False


# ── Validation ──────────────────────────────────────────────────────

def validate_stack():
    """Validate the full edge AI stack."""
    results = {"checks": [], "passed": 0, "failed": 0}

    def check(name, passed, detail=""):
        status = "✅" if passed else "❌"
        results["checks"].append({"name": name, "passed": passed, "detail": detail})
        if passed:
            results["passed"] += 1
            print(f"  {status} {name}")
        else:
            results["failed"] += 1
            print(f"  {status} {name}: {detail}")
        return passed

    print("🔧 Validating edge AI stack...\n")

    # Hardware
    mem = detect_memory()
    check("Jetson Orin Nano detected", "Jetson" in detect_jetson_model())
    check(f"RAM: {mem.get('ram_total_mb', '?')}MB", mem.get("ram_total_mb", 0) >= 4096,
          f"Only {mem.get('ram_total_mb', '?')}MB")
    check(f"CMA: {mem.get('cma_total_mb', '?')}MB", mem.get("cma_total_mb", 0) >= 64,
          f"Only {mem.get('cma_total_mb', '?')}MB (CMA increase recommended)")

    # Software
    ollama = detect_ollama()
    check("Ollama installed", ollama["installed"],
          "Run: curl -fsSL https://ollama.com/install.sh | sh")
    check("Ollama running", ollama["running"],
          "Run: ollama serve")

    # Models
    if ollama["running"]:
        model_names = {m["name"] for m in ollama["models"]}
        check("nomic-embed-text (embeddings)", "nomic-embed-text" in model_names,
              "Run: ollama pull nomic-embed-text")
        has_chat = any(n in model_names for n in ["deepseek-r1:1.5b", "moondream", "phi3:mini"])
        check("Chat model available", has_chat,
              "Run: ollama pull deepseek-r1:1.5b")

    # Gateway
    check("edge-gateway.py exists",
          os.path.exists(os.path.join(os.path.dirname(__file__), "edge-gateway.py")))

    # Shared modules
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from edge.config import OLLAMA_URL
        check("Shared edge/ modules", True)
    except ImportError as e:
        check("Shared edge/ modules", False, str(e))

    # API connectivity
    if ollama["running"]:
        try:
            from urllib.request import urlopen
            resp = urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
            check("Ollama API reachable", resp.status == 200)
        except Exception as e:
            check("Ollama API reachable", False, str(e))

    print(f"\n📊 {results['passed']} passed, {results['failed']} failed")
    return results


# ── Main ────────────────────────────────────────────────────────────

def detect_all():
    """Full hardware + software detection."""
    mem = detect_memory()
    return {
        "device": detect_jetson_model(),
        "cuda": detect_cuda(),
        "memory": mem,
        "thermal": detect_thermal(),
        "software": detect_software(),
        "model_recommendations": recommend_models(
            mem.get("ram_available_mb", 0),
            mem.get("cma_total_mb", 0)
        ),
        "timestamp": datetime.now().isoformat(),
    }


def print_report(info, json_output=False):
    """Print a human-readable report."""
    if json_output:
        print(json.dumps(info, indent=2, default=str))
        return

    print("=" * 60)
    print("🔧 Jetson Edge AI — Hardware Report")
    print("=" * 60)

    print(f"\n📱 Device: {info['device']}")
    print(f"   CUDA: {info['cuda'].get('cuda_version', 'not found')}")
    if info['cuda'].get('gpu_driver'):
        print(f"   GPU Driver: {info['cuda']['gpu_driver']}")

    mem = info["memory"]
    print(f"\n💾 Memory:")
    print(f"   RAM: {mem.get('ram_available_mb', '?')}/{mem.get('ram_total_mb', '?')}MB available")
    print(f"   CMA: {mem.get('cma_free_mb', '?')}/{mem.get('cma_total_mb', '?')}MB (GPU memory)")
    if mem.get('cma_total_mb', 0) < 1024:
        print(f"   ⚠️  CMA is only {mem['cma_total_mb']}MB — increase to 2GB for 4B models")
        print(f"   Edit /etc/kernel/cmdline-extra: add video=tegrafb mem=2G (needs sudo)")

    temps = info["thermal"]
    if temps:
        print(f"\n🌡️  Temperatures:")
        for name, temp in temps.items():
            status = "⚠️" if temp > 60 else "✅"
            print(f"   {status} {name}: {temp}°C")

    ollama = info["software"].get("ollama", {})
    print(f"\n🤖 Ollama:")
    if ollama.get("installed"):
        print(f"   ✅ Installed at {ollama.get('path', '?')}")
    else:
        print(f"   ❌ Not installed — curl -fsSL https://ollama.com/install.sh | sh")
    if ollama.get("running"):
        print(f"   ✅ Running — {len(ollama.get('models', []))} models")
        for m in ollama.get("models", []):
            print(f"      {m['name']:30s} {m['size']}")
    else:
        print(f"   ❌ Not running — ollama serve")

    trt = info["software"].get("tensorrt", {})
    if trt.get("installed"):
        print(f"\n⚡ TensorRT: {trt.get('version', '?')}")

    recs = info["model_recommendations"]
    if recs["install"]:
        print(f"\n📥 Recommended models to install:")
        for item in recs["install"]:
            for name, info in item.items():
                print(f"   • {name:30s} — {info['purpose']} ({info['size_mb']}MB)")

    if recs["blocked"]:
        print(f"\n🚫 Blocked (hardware limits):")
        for item in recs["blocked"]:
            for name, info in item.items():
                if name == "reason":
                    continue
                reason_str = item.get("reason", "")
                print(f"   ✗ {name:30s} — {reason_str}")

    if recs["installed"]:
        print(f"\n✅ Already installed:")
        for item in recs["installed"]:
            for name, info in item.items():
                print(f"   ✓ {name:30s} — {info['purpose']}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--detect":
        info = detect_all()
        print_report(info, "--json" in sys.argv)

    elif sys.argv[1] == "--install":
        mem = detect_memory()
        recs = recommend_models(mem.get("ram_available_mb", 0), mem.get("cma_total_mb", 0))

        if not recs["install"]:
            print("No models to install (all already present or hardware limits)")
            return

        print("📥 Installing recommended models...\n")
        for item in recs["install"]:
            for name, _ in item.items():
                pull_model(name)

        print(f"\n✅ Done. Run 'edge-setup.py --validate' to verify.")

    elif sys.argv[1] == "--validate":
        validate_stack()

    elif sys.argv[1] == "--json":
        print(json.dumps(detect_all(), indent=2, default=str))

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
