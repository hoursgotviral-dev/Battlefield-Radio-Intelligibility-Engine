"""
Unit test & smoke test for ONNX export and parity.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

from models.branch_a_denoiser import BranchADenoiser
from deploy.export_onnx import export_branch_a_onnx, StreamingBranchAEngine
from deploy.test_onnx_runtime import verify_streaming_parity


def test_onnx_export_and_checker():
    """Verify that Branch A model exports to valid ONNX without checker errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "test_model.onnx")
        
        export_branch_a_onnx(
            checkpoint_path="",
            output_path=onnx_file,
            freq_bins=257,
            chunk_frames=4,
            hidden_dim=32,
            num_layers=2,
            batch_size=1,
            opset_version=17,
        )
        
        assert os.path.exists(onnx_file)
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)


def test_onnx_runtime_parity():
    """Verify numeric parity between PyTorch and ONNX Runtime for Branch A."""
    ckpt_path = "checkpoints/branch_a_curriculum_best.pt"
    denoiser = BranchADenoiser(freq_bins=257, channels=32, hidden_dim=64, num_layers=2)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        denoiser.load_state_dict(sd)

    model = StreamingBranchAEngine(denoiser)
    model.init_hidden_state = denoiser.init_hidden_state
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "test_parity.onnx")
        
        export_branch_a_onnx(
            checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else "",
            output_path=onnx_file,
            freq_bins=257,
            chunk_frames=4,
            hidden_dim=64,
            num_layers=2,
            batch_size=1,
            opset_version=17,
        )
        
        success = verify_streaming_parity(
            model=model,
            onnx_path=onnx_file,
            num_chunks=5,
            freq_bins=257,
            chunk_frames=4,
            hidden_dim=64,
            num_layers=2,
            tolerance=1e-4,
        )
        assert success is True

