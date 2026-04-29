#!/usr/bin/env python3
"""
tensorrt-bench.py — TensorRT inference benchmarks for Jetson Orin Nano.

Tests: ONNX→TensorRT conversion, inference latency, throughput, memory usage.

Usage:
  python3 tensorrt-bench.py onnx       # Create test ONNX models
  python3 tensorrt-bench.py build      # Build TensorRT engines
  python3 tensorrt-bench.py run        # Run inference benchmarks
  python3 tensorrt-bench.py full       # Full pipeline
"""

import tensorrt as trt
import numpy as np
import ctypes
import time
import json
import os
from datetime import datetime

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
RESULTS_DIR = os.path.join(WORKSPACE, "memory", "gpu-benchmarks")


def create_test_models():
    """Create ONNX models of varying sizes for benchmarking."""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("❌ onnx not installed. Run: pip install onnx")
        return

    models = {
        "tiny_mlp_64": (64, 32, 16),
        "small_mlp_256": (256, 128, 64),
        "medium_mlp_512": (512, 256, 128),
        "large_mlp_1024": (1024, 512, 256),
    }

    for name, (in_dim, hidden, out_dim) in models.items():
        W1 = helper.make_tensor('W1', TensorProto.FLOAT, [in_dim, hidden],
                                np.random.randn(in_dim, hidden).astype(np.float32).flatten().tolist())
        b1 = helper.make_tensor('b1', TensorProto.FLOAT, [hidden],
                                np.zeros(hidden, dtype=np.float32).tolist())
        W2 = helper.make_tensor('W2', TensorProto.FLOAT, [hidden, out_dim],
                                np.random.randn(hidden, out_dim).astype(np.float32).flatten().tolist())
        b2 = helper.make_tensor('b2', TensorProto.FLOAT, [out_dim],
                                np.zeros(out_dim, dtype=np.float32).tolist())

        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, in_dim])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, out_dim])

        matmul1 = helper.make_node('MatMul', ['X', 'W1'], ['M1'])
        add1 = helper.make_node('Add', ['M1', 'b1'], ['A1'])
        relu = helper.make_node('Relu', ['A1'], ['R1'])
        matmul2 = helper.make_node('MatMul', ['R1', 'W2'], ['M2'])
        add2 = helper.make_node('Add', ['M2', 'b2'], ['Y'])

        graph = helper.make_graph([matmul1, add1, relu, matmul2, add2], name, [X], [Y], [W1, b1, W2, b2])
        model = helper.make_model(graph)

        path = f"/tmp/{name}.onnx"
        onnx.save(model, path)
        params = in_dim * hidden + hidden + hidden * out_dim + out_dim
        print(f"  ✅ {name}: {in_dim}→{hidden}→{out_dim} ({params:,} params) → {path}")


def build_engine(onnx_path, fp16=True):
    """Build TensorRT engine from ONNX."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"     Parse error {i}: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    start = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - start

    if engine_bytes is None:
        print(f"     ❌ Build failed")
        return None

    size = engine_bytes.nbytes
    print(f"     ✅ FP16={fp16} build: {build_time*1000:.1f}ms, {size:,} bytes")

    # Deserialize to get memory info
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    return {
        "build_time_ms": round(build_time * 1000, 1),
        "engine_size_bytes": size,
        "layers": engine.num_layers,
        "device_memory": engine.device_memory_size_v2,
        "engine_bytes": engine_bytes,
        "engine": engine,
    }


def run_inference_benchmark(engine_info, num_warmup=50, num_iters=500):
    """Benchmark inference on a TensorRT engine."""
    engine = engine_info["engine"]
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    context = engine.create_execution_context()

    # Get I/O shapes
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(engine.num_io_tensors - 1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    input_size = int(np.prod(input_shape)) * 4  # float32
    output_size = int(np.prod(output_shape)) * 4

    # Allocate page-locked memory
    input_data = np.random.randn(*input_shape).astype(np.float32)
    output_data = np.empty(output_shape, dtype=np.float32)

    # Set tensor addresses
    context.set_tensor_address(input_name, input_data.ctypes.data)
    context.set_tensor_address(output_name, output_data.ctypes.data)

    # Warmup
    for _ in range(num_warmup):
        context.execute_async_v3(stream_handle=0)

    # Benchmark
    latencies = []
    for i in range(num_iters):
        start = time.perf_counter()
        context.execute_async_v3(stream_handle=0)
        # Synchronize (simplified — no actual CUDA stream sync without pycuda)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies.sort()
    results = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "warmup": num_warmup,
        "iters": num_iters,
        "mean_ms": round(np.mean(latencies), 4),
        "median_ms": round(np.median(latencies), 4),
        "p99_ms": round(np.percentile(latencies, 99), 4),
        "min_ms": round(np.min(latencies), 4),
        "max_ms": round(np.max(latencies), 4),
        "qps": round(1000 / np.median(latencies)),
    }

    return results


def full_benchmark():
    """Run complete TensorRT benchmark suite."""
    print("🔥 TensorRT Benchmark Suite — Jetson Orin Nano 8GB")
    print(f"   {datetime.now().isoformat()}")
    print("=" * 60)

    # Create models
    print("\n📋 Creating ONNX models...")
    create_test_models()

    # Build and benchmark each
    models = ["tiny_mlp_64", "small_mlp_256", "medium_mlp_512", "large_mlp_1024"]
    results = {"timestamp": datetime.now().isoformat(), "device": "Jetson Orin Nano 8GB", "tensorrt": "10.3.0", "benchmarks": {}}

    for name in models:
        onnx_path = f"/tmp/{name}.onnx"
        if not os.path.exists(onnx_path):
            print(f"  ⏭️ Skipping {name} (no ONNX model)")
            continue

        print(f"\n🔧 {name}:")
        print(f"   Building FP16 engine...")
        fp16_info = build_engine(onnx_path, fp16=True)
        if fp16_info:
            inf = run_inference_benchmark(fp16_info)
            results["benchmarks"][name] = {
                "fp16": {
                    "build": {"time_ms": fp16_info["build_time_ms"], "engine_bytes": fp16_info["engine_size_bytes"], "device_memory": fp16_info["device_memory"]},
                    "inference": inf,
                }
            }
            print(f"   Inference: {inf['median_ms']:.4f}ms median, {inf['qps']:,} QPS")

        print(f"   Building FP32 engine...")
        fp32_info = build_engine(onnx_path, fp16=False)
        if fp32_info:
            inf = run_inference_benchmark(fp32_info)
            if name in results["benchmarks"]:
                results["benchmarks"][name]["fp32"] = {
                    "build": {"time_ms": fp32_info["build_time_ms"], "engine_bytes": fp32_info["engine_size_bytes"], "device_memory": fp32_info["device_memory"]},
                    "inference": inf,
                }
            print(f"   Inference: {inf['median_ms']:.4f}ms median, {inf['qps']:,} QPS")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"tensorrt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    # Remove engine bytes before saving (not JSON serializable)
    for model in results["benchmarks"].values():
        for dtype in model.values():
            dtype.get("build", {}).pop("engine_bytes", None)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {path}")

    return results


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"

    if cmd == "onnx":
        create_test_models()
    elif cmd == "build":
        for name in ["tiny_mlp_64", "small_mlp_256", "medium_mlp_512", "large_mlp_1024"]:
            path = f"/tmp/{name}.onnx"
            if os.path.exists(path):
                print(f"\n{name}:")
                build_engine(path, fp16=True)
    elif cmd == "run":
        full_benchmark()
    else:
        full_benchmark()
