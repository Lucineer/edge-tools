#!/usr/bin/env python3
"""
edge-router.py — Edge model routing for Jetson Orin Nano
Inspired by vllm-project/semantic-router (3.9K⭐ trending)
and GenericAgent skill tree (8K⭐ trending)

Routes inference requests to the best available local model
based on task type, available models, and resource constraints.

Usage:
  python3 edge-router.py route "Explain quantum computing"
  python3 edge-router.py models        # List available edge models
  python3 edge-router.py benchmark     # Run throughput benchmark
"""

import os
import sys
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

# Jetson hardware constraints
@dataclass
class Hardware:
    ram_gb: float = 8.0
    cuda_cores: int = 1024
    arch: str = "arm64"
    max_model_size_gb: float = 6.0  # Leave room for OS + other processes

HARDWARE = Hardware()

@dataclass
class EdgeModel:
    name: str
    runtime: str        # 'litert-lm', 'whisper', 'llama.cpp', etc
    size_gb: float
    task: str           # 'chat', 'embedding', 'transcription', 'classification'
    quality: str        # 'low', 'medium', 'high'
    tokens_per_sec: int  # estimated on Jetson

# Models available on our Jetson
AVAILABLE_MODELS = [
    EdgeModel("DeepSeek-R1-Distill-Qwen-1.5B", "litert-lm", 1.2, "chat", "low", 35),
    EdgeModel("Qwen2.5-0.5B", "litert-lm", 0.4, "chat", "low", 60),
    EdgeModel("SmolLM-135M", "litert-lm", 0.15, "chat", "low", 120),
    # Planned (need HF token):
    # EdgeModel("Gemma-2-2B", "litert-lm", 1.8, "chat", "medium", 20),
    # EdgeModel("Phi-3-mini", "litert-lm", 2.5, "chat", "high", 15),
]

# Task classification patterns (from GenericAgent skill tree concept)
TASK_PATTERNS = {
    "chat": {
        "weight": 3,
        "keywords": ["hi", "hello", "how are you", "thanks", "goodbye", "help", "what", "tell me", "explain"],
    },
    "code": {
        "weight": 4,
        "keywords": ["code", "function", "program", "script", "api", "bug", "error", "debug", "compile", "sql", "python", "rust"],
    },
    "reasoning": {
        "weight": 5,
        "keywords": ["why", "how does", "explain", "analyze", "compare", "calculate", "solve", "prove", "theorem", "logic"],
    },
    "writing": {
        "weight": 2,
        "keywords": ["write", "create", "compose", "draft", "essay", "poem", "story", "edit", "rewrite"],
    },
    "embedding": {
        "weight": 1,
        "keywords": ["vectorize", "embed", "semantic search", "similarity", "cluster"],
    },
}

# Routing strategy: rule-based → resource-aware
def classify_task(text: str) -> str:
    """Classify a task based on keywords and patterns."""
    scores = {}
    for task, config in TASK_PATTERNS.items():
        score = 0
        for kw in config["keywords"]:
            if kw.lower() in text.lower():
                score += config["weight"]
        if score > 0:
            scores[task] = score
    
    if not scores:
        return "chat"  # default
    
    return max(scores, key=scores.get)


def resource_check(model: EdgeModel) -> bool:
    """Check if model fits on current hardware."""
    return model.size_gb <= HARDWARE.max_model_size_gb


def route_request(text: str, prefer_quality: str = "medium") -> dict:
    """Route a request to the best available model."""
    task = classify_task(text)
    
    # Filter by task type (most models handle chat)
    candidates = [m for m in AVAILABLE_MODELS if m.task in ("chat", "embedding") and resource_check(m)]
    
    if not candidates:
        return {"error": "No suitable edge model available", "fallback": "api"}
    
    # Quality preference
    quality_order = {"low": 0, "medium": 1, "high": 2}
    if prefer_quality != "any":
        candidates = [m for m in candidates if quality_order.get(m.quality, 0) >= quality_order.get(prefer_quality, 0)]
    
    # For code/reasoning tasks, prefer larger models
    if task in ("code", "reasoning"):
        candidates.sort(key=lambda m: (m.size_gb, m.tokens_per_sec), reverse=True)
    else:
        candidates.sort(key=lambda m: m.tokens_per_sec, reverse=True)
    
    model = candidates[0]
    
    return {
        "task": task,
        "model": model.name,
        "runtime": model.runtime,
        "estimated_tps": model.tokens_per_sec,
        "size_gb": model.size_gb,
        "quality": model.quality,
        "available_ram": HARDWARE.ram_gb - model.size_gb,
    }


def benchmark_model(model_name: str, prompt: str = "Hello, how are you today?") -> dict:
    """Run a quick throughput benchmark."""
    model = next((m for m in AVAILABLE_MODELS if m.name == model_name), None)
    if not model:
        return {"error": f"Model {model_name} not found"}
    
    if model.runtime == "litert-lm":
        cmd = f"litert-lm run {model.name} \"{prompt}\""
    else:
        return {"error": f"Runtime {model.runtime} not supported yet"}
    
    start = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        output_len = len(result.stdout.strip())
        tokens = output_len // 4  # approximate
        throughput = tokens / elapsed if elapsed > 0 else 0
        return {
            "model": model_name,
            "elapsed_sec": round(elapsed, 2),
            "output_chars": output_len,
            "estimated_tokens": tokens,
            "tokens_per_sec": round(throughput, 1),
            "stdout": result.stdout[:500] if result.stdout else "",
            "stderr": result.stderr[:200] if result.stderr else "",
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if cmd == "route":
        text = sys.argv[2] if len(sys.argv) > 2 else input("Prompt: ")
        quality = sys.argv[3] if len(sys.argv) > 3 else "medium"
        result = route_request(text, quality)
        print(json.dumps(result, indent=2))
    
    elif cmd == "models":
        print(f"{'Model':35s} {'Runtime':12s} {'Size':8s} {'Task':14s} {'Quality':10s} {'TPS':6s}")
        print("-" * 85)
        for m in AVAILABLE_MODELS:
            ok = "✓" if resource_check(m) else "✗"
            print(f"{m.name:35s} {m.runtime:12s} {m.size_gb:.1f}GB {m.task:14s} {m.quality:10s} {m.tokens_per_sec:>4d}t/s {ok}")
    
    elif cmd == "benchmark":
        model = sys.argv[2] if len(sys.argv) > 2 else "DeepSeek-R1-Distill-Qwen-1.5B"
        prompt = sys.argv[3] if len(sys.argv) > 3 else "What is the capital of France?"
        result = benchmark_model(model, prompt)
        print(json.dumps(result, indent=2))
    
    elif cmd == "classify":
        text = sys.argv[2] if len(sys.argv) > 2 else input("Text: ")
        task = classify_task(text)
        print(f"Task: {task}")
    
    else:
        print("""Usage: edge-router.py <command> [args]

Commands:
  route <text> [quality]   Route to best edge model
  models                    List available edge models
  benchmark [model] [text]  Run throughput benchmark
  classify <text>           Classify task type
""")
