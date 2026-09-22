"""
ONNX Export Utility for Streaming Speech Enhancement Models.

ML Concept - Static Graph Freezing for Snapdragon NPU:
Edge NPUs such as the Qualcomm Hexagon Processor compile neural networks into ahead-of-time (AOT)
optimized hardware binaries using Qualcomm AI Hub / QNN SDK. Dynamic tensor dimensions
(e.g., variable time length or batch sizes) degrade NPU compiler scheduling efficiency and can cause
compilation failures. 

This exporter freezes the model with static, predetermined chunk shapes (e.g. 1 chunk = 4 time frames),
exposes explicit recurrent hidden states as named input/output ports, and validates the resulting ONNX graph.
"""

import os
import argparse
from pathlib import Path
import yaml
import torch
import onnx

from models.dummy_conv_gru import DummyStreamingConvGRU


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    freq_bins: int = 257,
    chunk_frames: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    batch_size: int = 1,
    opset_version: int = 17,
) -> str:
    """
    Exports a streaming stateful model to static ONNX format.

    Args:
        model: PyTorch model instance
        output_path: Target .onnx filepath
        freq_bins: Number of frequency bins (e.g., 257 for 512-pt STFT)
        chunk_frames: Number of time frames in streaming chunk
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        batch_size: Batch size (fixed to 1 for edge streaming)
        opset_version: ONNX opset version (17 recommended for AI Hub)

    Returns:
        output_path: Path to exported ONNX model
    """
    model.eval()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Prepare dummy inputs with exact static shapes
    dummy_input_chunk = torch.randn(batch_size, 1, freq_bins, chunk_frames, dtype=torch.float32)
    dummy_h_in = torch.zeros(num_layers, batch_size, hidden_dim, dtype=torch.float32)

    input_names = ["input_chunk", "h_in"]
    output_names = ["enhanced_chunk", "h_out"]

    print(f"[*] Exporting PyTorch model to ONNX -> {output_path}")
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
    )

    # Check ONNX validity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[+] ONNX model successfully verified with onnx.checker!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export streaming speech enhancement model to ONNX.")
    parser.add_argument("--config", type=str, default="configs/model_convgru.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="artifacts/dummy_conv_gru.onnx", help="Output ONNX path")
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

    deploy_cfg = config.get("deploy", {}).get("onnx_export", {})
    opset = deploy_cfg.get("opset", 17)
    out_file = args.output or deploy_cfg.get("output_file", "artifacts/dummy_conv_gru.onnx")

    model = DummyStreamingConvGRU(
        freq_bins=freq_bins,
        conv_channels=conv_channels,
        gru_hidden_dim=gru_hidden_dim,
        num_gru_layers=num_gru_layers,
    )

    export_model_to_onnx(
        model=model,
        output_path=out_file,
        freq_bins=freq_bins,
        chunk_frames=chunk_frames,
        hidden_dim=gru_hidden_dim,
        num_layers=num_gru_layers,
        batch_size=1,
        opset_version=opset,
    )


if __name__ == "__main__":
    main()
