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

from models.dummy_conv_gru import DummyStreamingConvGRU
from deploy.export_onnx import export_model_to_onnx
from deploy.test_onnx_runtime import verify_streaming_parity


def test_onnx_export_and_checker():
    """Verify that model exports to valid ONNX without checker errors."""
    model = DummyStreamingConvGRU(freq_bins=257, conv_channels=16, gru_hidden_dim=32, num_gru_layers=2)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "test_model.onnx")
        
        export_model_to_onnx(
            model=model,
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
    """Verify numeric parity between PyTorch and ONNX Runtime."""
    model = DummyStreamingConvGRU(freq_bins=257, conv_channels=16, gru_hidden_dim=32, num_gru_layers=2)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "test_parity.onnx")
        
        export_model_to_onnx(
            model=model,
            output_path=onnx_file,
            freq_bins=257,
            chunk_frames=4,
            hidden_dim=32,
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
            hidden_dim=32,
            num_layers=2,
            tolerance=1e-4,
        )
        assert success is True
