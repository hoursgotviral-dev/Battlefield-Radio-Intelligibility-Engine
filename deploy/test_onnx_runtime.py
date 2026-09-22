"""
Numerical Parity Verification: PyTorch vs. ONNX Runtime for Streaming State.

ML Concept - Sequential Streaming Parity:
In recurrent streaming models, numerical divergence between PyTorch eager mode and ONNX Runtime
can accumulate across sequential chunks if internal recurrent weights or operators differ in precision.
This utility runs a multi-chunk sequence through both PyTorch and ONNX Runtime in parallel,
feeding the previous chunk's output state into the next chunk's input state, and asserts
that max absolute difference is well within FP32 tolerance (< 1e-4).
"""

import os
import argparse
import yaml
import numpy as np
import torch
import onnxruntime as ort

from models.dummy_conv_gru import DummyStreamingConvGRU
from deploy.export_onnx import export_model_to_onnx


def verify_streaming_parity(
    model: torch.nn.Module,
    onnx_path: str,
    num_chunks: int = 10,
    freq_bins: int = 257,
    chunk_frames: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    tolerance: float = 1e-4,
) -> bool:
    """
    Verifies that PyTorch and ONNX Runtime yield identical results over consecutive streaming chunks.
    """
    model.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Initialize states
    torch_state = model.init_hidden_state(batch_size=1)
    ort_state = np.zeros((num_layers, 1, hidden_dim), dtype=np.float32)

    torch.manual_seed(42)
    np.random.seed(42)

    max_diff_out = 0.0
    max_diff_state = 0.0

    print(f"[*] Running {num_chunks}-chunk streaming parity verification...")

    for chunk_idx in range(num_chunks):
        # Generate random audio chunk
        input_np = np.random.randn(1, 1, freq_bins, chunk_frames).astype(np.float32)
        input_torch = torch.from_numpy(input_np)

        # PyTorch forward
        with torch.no_grad():
            torch_out, torch_state = model(input_torch, torch_state)

        # ONNX Runtime forward
        ort_inputs = {
            "input_chunk": input_np,
            "h_in": ort_state,
        }
        ort_outputs = session.run(None, ort_inputs)
        ort_out = ort_outputs[0]
        ort_state = ort_outputs[1]

        # Compare outputs
        diff_out = np.max(np.abs(torch_out.numpy() - ort_out))
        diff_state = np.max(np.abs(torch_state.numpy() - ort_state))

        max_diff_out = max(max_diff_out, diff_out)
        max_diff_state = max(max_diff_state, diff_state)

        if diff_out > tolerance or diff_state > tolerance:
            print(f"[!] Parity failed at chunk {chunk_idx}: diff_out={diff_out:.6f}, diff_state={diff_state:.6f}")
            return False

    print(f"[+] Parity PASSED across {num_chunks} chunks!")
    print(f"    - Max output diff: {max_diff_out:.8e}")
    print(f"    - Max hidden state diff: {max_diff_state:.8e}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test ONNX Runtime streaming parity.")
    parser.add_argument("--onnx", type=str, default="artifacts/dummy_conv_gru.onnx", help="Path to ONNX model")
    parser.add_argument("--config", type=str, default="configs/model_convgru.yaml", help="Path to config YAML")
    parser.add_argument("--chunks", type=int, default=10, help="Number of streaming chunks to simulate")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    freq_bins = model_cfg.get("freq_bins", 257)
    conv_channels = model_cfg.get("conv_channels", 32)
    gru_hidden_dim = model_cfg.get("gru_hidden_dim", 64)
    num_gru_layers = model_cfg.get("num_gru_layers", 2)
    chunk_frames = model_cfg.get("streaming_chunk_frames", 4)

    model = DummyStreamingConvGRU(
        freq_bins=freq_bins,
        conv_channels=conv_channels,
        gru_hidden_dim=gru_hidden_dim,
        num_gru_layers=num_gru_layers,
    )

    if not os.path.exists(args.onnx):
        print(f"[*] ONNX file {args.onnx} not found, exporting first...")
        export_model_to_onnx(
            model=model,
            output_path=args.onnx,
            freq_bins=freq_bins,
            chunk_frames=chunk_frames,
            hidden_dim=gru_hidden_dim,
            num_layers=num_gru_layers,
        )

    success = verify_streaming_parity(
        model=model,
        onnx_path=args.onnx,
        num_chunks=args.chunks,
        freq_bins=freq_bins,
        chunk_frames=chunk_frames,
        hidden_dim=gru_hidden_dim,
        num_layers=num_gru_layers,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
