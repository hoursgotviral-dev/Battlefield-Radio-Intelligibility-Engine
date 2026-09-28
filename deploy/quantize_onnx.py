"""
Post-Training Dynamic INT8 Quantization & Latency Benchmark for Snapdragon/Edge CPU/NPU.

ML Concept - Weight-Only & Dynamic Activation Quantization:
Dynamic quantization converts FP32 weights to INT8 offline, reducing model storage by ~4x,
while activations are dynamically quantized to INT8 at runtime on supported matrix multiplication
and linear kernels, drastically reducing memory bandwidth pressure.
"""

import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_branch_a(
    input_model_path: str = "artifacts/branch_a_denoiser.onnx",
    output_model_path: str = "artifacts/branch_a_denoiser_int8.onnx",
):
    print(f"[*] Quantizing FP32 model {input_model_path} -> INT8 model {output_model_path}...")
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QInt8,
    )
    
    fp32_size = os.path.getsize(input_model_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"[+] FP32 model size: {fp32_size:.2f} MB")
    print(f"[+] INT8 model size: {int8_size:.2f} MB ({(1 - int8_size/fp32_size)*100:.1f}% reduction)")


def benchmark_latency(model_path: str, num_runs: int = 100):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    input_chunk = np.random.randn(1, 1, 257, 4).astype(np.float32)
    h_in = np.zeros((2, 1, 64), dtype=np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {"input_chunk": input_chunk, "h_in": h_in})
        
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {"input_chunk": input_chunk, "h_in": h_in})
        times.append((time.perf_counter() - start) * 1000.0)  # ms
        
    median_ms = np.median(times)
    p95_ms = np.percentile(times, 95)
    print(f"[*] Latency Benchmark for {os.path.basename(model_path)}:")
    print(f"    - Median Latency: {median_ms:.3f} ms ({median_ms*1000:.1f} µs)")
    print(f"    - P95 Latency   : {p95_ms:.3f} ms ({p95_ms*1000:.1f} µs)")
    return {"median_ms": median_ms, "p95_ms": p95_ms}


if __name__ == "__main__":
    fp32_path = "artifacts/branch_a_denoiser.onnx"
    int8_path = "artifacts/branch_a_denoiser_int8.onnx"
    if os.path.exists(fp32_path):
        quantize_branch_a(fp32_path, int8_path)
        print("\n--- FP32 Benchmark ---")
        benchmark_latency(fp32_path)
        print("\n--- INT8 Benchmark ---")
        benchmark_latency(int8_path)
    else:
        print(f"Error: {fp32_path} not found.")
