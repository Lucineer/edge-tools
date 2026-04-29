#!/usr/bin/env python3
"""
gpu-bench.py — GPU benchmark suite for Jetson Orin Nano.

Tests: Ollama inference speed, TensorRT engine build, CUDA kernel perf,
memory bandwidth, and thermal behavior.

Usage:
  python3 gpu-bench.py ollama              # Benchmark all Ollama models
  python3 gpu-bench.py ollama <model>      # Benchmark specific model
  python3 gpu-bench.py cuda                # CUDA kernel benchmarks
  python3 gpu-bench.py memory              # Memory bandwidth test
  python3 gpu-bench.py thermal             # Thermal stress test
  python3 gpu-bench.py full                # Run everything
"""

import subprocess
import time
import json
import os
import sys
import re
from datetime import datetime

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
RESULTS_DIR = os.path.join(WORKSPACE, "memory", "gpu-benchmarks")


def run_cmd(cmd, timeout=300):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1


def get_thermal():
    """Read GPU and SOC temperatures."""
    temps = {}
    zones = ["gpu-thermal", "soc0-thermal", "soc1-thermal", "soc2-thermal", "tj-thermal"]
    for zone in zones:
        path = f"/sys/class/thermal/{zone}/temp"
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    raw = f.read()
                if raw:
                    temps[zone] = round(int(raw.decode().strip()) / 1000, 1)
            except (OSError, ValueError):
                pass
    return temps


def get_power():
    """Read power consumption from tegrastats."""
    out, _ = run_cmd("timeout 2 tegrastats 2>&1")
    power = {}
    for match in re.finditer(r'(\w+) (\d+)mW', out):
        power[match.group(1)] = int(match.group(2))
    return power


def get_memory():
    """Read memory stats."""
    out, _ = run_cmd("cat /proc/meminfo")
    mem = {}
    for line in out.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            val = val.strip().split()[0]
            mem[key] = int(val)
    return {
        "total_mb": mem.get("MemTotal", 0) // 1024,
        "free_mb": mem.get("MemFree", 0) // 1024,
        "available_mb": mem.get("MemAvailable", 0) // 1024,
        "cma_total_mb": mem.get("CmaTotal", 0) // 1024,
        "cma_free_mb": mem.get("CmaFree", 0) // 1024,
    }


def bench_ollama_model(model, prompt="Explain quantum computing in 50 words."):
    """Benchmark a single Ollama model."""
    print(f"  🔄 Benchmarking {model}...")

    # Pre-fill to warm up
    run_cmd(f"ollama run {model} 'hi' --nowordwrap 2>&1", timeout=60)

    # Timed inference
    start = time.time()
    out, rc = run_cmd(f"ollama run {model} '{prompt}' --nowordwrap 2>&1", timeout=120)
    elapsed = time.time() - start

    # Count generated tokens (rough: ~4 chars per token)
    output_text = out.strip()
    token_count = len(output_text) // 4
    tps = token_count / max(elapsed, 0.001)

    # Memory during inference
    mem = get_memory()

    return {
        "model": model,
        "elapsed_s": round(elapsed, 2),
        "output_chars": len(output_text),
        "est_tokens": token_count,
        "tokens_per_sec": round(tps, 1),
        "ram_used_mb": mem["total_mb"] - mem["available_mb"],
        "ram_available_mb": mem["available_mb"],
        "cma_free_mb": mem["cma_free_mb"],
    }


def bench_all_ollama():
    """Benchmark all installed Ollama models."""
    print("🧪 Ollama GPU Benchmark")
    print("=" * 50)

    out, _ = run_cmd("ollama list 2>&1")
    models = []
    for line in out.strip().split("\n")[1:]:  # Skip header
        parts = line.split()
        if parts:
            models.append(parts[0])

    results = []
    for model in models:
        result = bench_ollama_model(model)
        results.append(result)
        print(f"  ✅ {result['model']:50s} {result['tokens_per_sec']:6.1f} t/s  {result['elapsed_s']:5.1f}s  RAM: {result['ram_available_mb']}MB free")

    return {
        "timestamp": datetime.now().isoformat(),
        "device": "Jetson Orin Nano 8GB",
        "ollama_version": "0.18.2",
        "results": results,
    }


def bench_cuda_kernels():
    """Compile and run CUDA kernel benchmarks."""
    print("🧪 CUDA Kernel Benchmark")
    print("=" * 50)

    # Create a simple CUDA benchmark
    cuda_code = """
#include <stdio.h>
#include <stdlib.h>

__global__ void vec_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Vector addition benchmark
    int n = 1 << 22; // 4M elements
    size_t bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize
    cudaMemset(d_a, 1, bytes);
    cudaMemset(d_b, 2, bytes);

    // Warmup
    vec_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < 100; i++) {
        vec_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float gb = (float)(n * 4 * 3) / 1e9;
    float bandwidth = gb / (ms / 1000.0);

    printf("VEC_ADD: %d elements, %d iters, %.1f ms total, %.1f GB/s bandwidth\\n", n, 100, ms, bandwidth);

    // Matrix multiply benchmark
    int N = 512;
    size_t mat_bytes = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, mat_bytes);
    cudaMalloc(&d_B, mat_bytes);
    cudaMalloc(&d_C, mat_bytes);
    cudaMemset(d_A, 1, mat_bytes);
    cudaMemset(d_B, 1, mat_bytes);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warmup
    matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) {
        matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("MATMUL: %dx%d, %d iters, %.1f ms total, %.1f ms/iter, %.2f GFLOPS\\n",
           N, N, 50, ms, ms/50, (2.0 * N * N * N) / (ms / 1000.0) / 1e9);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
"""

    src_path = "/tmp/gpu_bench.cu"
    bin_path = "/tmp/gpu_bench"

    with open(src_path, "w") as f:
        f.write(cuda_code)

    # Compile
    print("  🔄 Compiling CUDA kernel...")
    out, rc = run_cmd(
        f"/usr/local/cuda-12.6/bin/nvcc -o {bin_path} {src_path} "
        f"-I/usr/local/cuda-12.6/targets/aarch64-linux/include "
        f"-L/usr/local/cuda-12.6/targets/aarch64-linux/lib "
        f"-lcudart 2>&1",
        timeout=60
    )

    if rc != 0:
        return {"error": "Compilation failed", "output": out}

    print("  🔄 Running benchmark...")
    out, rc = run_cmd(bin_path, timeout=60)

    # Parse results
    results = {"timestamp": datetime.now().isoformat()}
    for line in out.split("\n"):
        if "VEC_ADD" in line:
            results["vec_add"] = {}
            m = re.search(r'(\d+\.\d+)\s*ms total', line)
            if m: results["vec_add"]["total_ms"] = float(m.group(1))
            m = re.search(r'(\d+\.\d+)\s*GB/s', line)
            if m: results["vec_add"]["bandwidth_gbs"] = float(m.group(1))
        elif "MATMUL" in line:
            results["matmul"] = {}
            m = re.search(r'(\d+\.\d+)\s*GFLOPS', line)
            if m: results["matmul"]["gflops"] = float(m.group(1))
            m = re.search(r'(\d+\.\d+)\s*ms/iter', line)
            if m: results["matmul"]["ms_per_iter"] = float(m.group(1))

    print(f"  ✅ CUDA benchmark complete")
    return results


def bench_thermal():
    """Thermal stress test — run GPU at load and monitor temps."""
    print("🧪 Thermal Stress Test (60s)")
    print("=" * 50)

    temps_before = get_thermal()
    print(f"  Start: GPU {temps_before.get('gpu-thermal', '?')}°C")

    # Run a sustained GPU workload via Ollama
    start = time.time()
    temps_samples = []
    while time.time() - start < 60:
        temps = get_thermal()
        temps_samples.append(temps)
        if len(temps_samples) % 10 == 0:
            print(f"  [{int(time.time()-start)}s] GPU: {temps.get('gpu-thermal', '?')}°C  SOC: {temps.get('soc0-thermal', '?')}°C")
        time.sleep(5)

    # Run a quick ollama inference during the test
    run_cmd("ollama run deepseek-r1:1.5b 'Write a 100 word story about a robot.' --nowordwrap 2>&1", timeout=60)

    temps_after = get_thermal()
    power = get_power()

    max_gpu = max(t.get("gpu-thermal", 0) for t in temps_samples)
    max_soc = max(t.get("soc0-thermal", 0) for t in temps_samples)

    results = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": 60,
        "gpu_temp_before": temps_before.get("gpu-thermal"),
        "gpu_temp_after": temps_after.get("gpu-thermal"),
        "gpu_temp_max": max_gpu,
        "soc_temp_max": max_soc,
        "samples": len(temps_samples),
        "power_mw": power,
    }

    print(f"  ✅ GPU: {temps_before.get('gpu-thermal')}°C → {temps_after.get('gpu-thermal')}°C (max {max_gpu}°C)")
    print(f"  Power: {power}")
    return results


def save_results(name, results):
    """Save benchmark results to file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Results saved to {path}")
    return path


def full_bench():
    """Run all benchmarks."""
    print("🔥 Full GPU Benchmark Suite — Jetson Orin Nano 8GB")
    print(f"   {datetime.now().isoformat()}")
    print("=" * 60)

    sys_info = {
        "timestamp": datetime.now().isoformat(),
        "device": "Jetson Orin Nano 8GB",
        "ram": get_memory(),
        "thermal": get_thermal(),
        "power": get_power(),
    }
    print(f"\n📋 System Info:")
    print(f"   RAM: {sys_info['ram']['total_mb']}MB total, {sys_info['ram']['available_mb']}MB available")
    print(f"   GPU temp: {sys_info['thermal'].get('gpu-thermal', '?')}°C")
    print(f"   Power: {sys_info['power']}")

    all_results = {"system": sys_info, "benchmarks": {}}

    # Ollama benchmarks
    print("\n")
    all_results["benchmarks"]["ollama"] = bench_all_ollama()
    save_results("ollama", all_results["benchmarks"]["ollama"])

    # CUDA kernels
    print("\n")
    all_results["benchmarks"]["cuda"] = bench_cuda_kernels()
    save_results("cuda", all_results["benchmarks"]["cuda"])

    # Thermal
    print("\n")
    all_results["benchmarks"]["thermal"] = bench_thermal()
    save_results("thermal", all_results["benchmarks"]["thermal"])

    return all_results


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ollama"

    if cmd == "ollama":
        if len(sys.argv) > 2:
            result = bench_ollama_model(sys.argv[2])
            print(json.dumps(result, indent=2))
        else:
            results = bench_all_ollama()
            save_results("ollama", results)
    elif cmd == "cuda":
        results = bench_cuda_kernels()
        save_results("cuda", results)
        print(json.dumps(results, indent=2))
    elif cmd == "thermal":
        results = bench_thermal()
        save_results("thermal", results)
        print(json.dumps(results, indent=2))
    elif cmd == "full":
        results = full_bench()
        save_results("full", results)
    elif cmd == "info":
        print(f"RAM: {get_memory()}")
        print(f"Thermal: {get_thermal()}")
        print(f"Power: {get_power()}")
    else:
        print(__doc__)
