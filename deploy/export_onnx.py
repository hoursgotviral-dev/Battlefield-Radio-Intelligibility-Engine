"""
ONNX Export Utility for Branch A (Continuous Causal ConvGRU Denoiser).

ML Concept - Static Graph Freezing for Qualcomm Hexagon NPU:
Edge NPUs such as the Qualcomm Hexagon Processor compile neural networks into ahead-of-time (AOT)
optimized hardware binaries using Qualcomm AI Hub / QNN SDK. Dynamic tensor dimensions
degrade NPU compiler scheduling efficiency and cause compilation failures.

This exporter freezes the trained Branch A model with static chunk shapes (1 chunk = 4 time frames = 32 ms),
exposes explicit recurrent hidden states (h_in / h_out) as named input/output ports,
and validates the resulting ONNX graph for QNN execution.
"""

import os
import sys

# Ensure UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import onnx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.branch_a_denoiser import BranchADenoiser


class StreamingBranchAEngine(nn.Module):
    """
    Streaming production wrapper for Branch A Causal ConvGRU Denoiser.
    Applies estimated spectral gain mask directly to input chunk.
    Exposes explicit static recurrent states for Snapdragon NPU execution.
    """
    def __init__(self, denoiser: BranchADenoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        input_chunk: torch.Tensor,
        h_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_chunk: [1, 1, 257, 4] static streaming chunk
            h_in: [2, 1, 64] recurrent hidden state
        Returns:
            enhanced_chunk: [1, 1, 257, 4] enhanced magnitude chunk
            h_out: [2, 1, 64] updated hidden state
        """
        mask, _, h_out = self.denoiser(input_chunk, h_in)
        enhanced_chunk = input_chunk * mask
        return enhanced_chunk, h_out


def export_branch_a_onnx(
    checkpoint_path: str = "checkpoints/branch_a_curriculum_best.pt",
    output_path: str = "artifacts/branch_a_denoiser.onnx",
    freq_bins: int = 257,
    chunk_frames: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 1,
    opset_version: int = 17,
) -> str:
    """Exports trained Branch A denoiser to static ONNX for Qualcomm AI Hub."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    denoiser = BranchADenoiser(
        freq_bins=freq_bins,
        channels=32,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        denoiser.load_state_dict(sd)
        print(f"[+] Loaded trained Branch A weights from: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Exporting with initial weights.")

    model = StreamingBranchAEngine(denoiser)
    model.eval()

    # Exact static inputs
    dummy_input_chunk = torch.randn(batch_size, 1, freq_bins, chunk_frames, dtype=torch.float32)
    dummy_h_in = torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32)

    input_names = ["input_chunk", "h_in"]
    output_names = ["enhanced_chunk", "h_out"]

    print(f"[*] Exporting Branch A PyTorch model to ONNX -> {output_path}")
    print(f"    - Input chunk shape: {tuple(dummy_input_chunk.shape)}")
    print(f"    - Hidden state in shape: {tuple(dummy_h_in.shape)}")
    print(f"    - Opset version: {opset_version}")

    torch.onnx.export(
        model,
        (dummy_input_chunk, dummy_h_in),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # Strictly static for Qualcomm AI Hub NPU compilation
        dynamo=False,
    )

    # Ensure all tensor data is self-contained inside the ONNX file
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    onnx.checker.check_model(output_path, full_check=True)
    print(f"[+] Branch A ONNX model successfully verified with full_check=True (Size: {Path(output_path).stat().st_size / 1024:.1f} KB)!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Branch A streaming speech enhancement model to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/branch_a_curriculum_best.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="artifacts/branch_a_denoiser.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_branch_a_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )


# Backward compatibility alias
export_model_to_onnx = export_branch_a_onnx


if __name__ == "__main__":
    main()
