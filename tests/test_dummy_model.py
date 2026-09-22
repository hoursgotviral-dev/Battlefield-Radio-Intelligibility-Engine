"""
Unit tests for dummy streaming Conv-GRU and architectural modules.
"""

import pytest
import torch

from models.dummy_conv_gru import DummyStreamingConvGRU, CausalConv2d
from models.branch_a_denoiser import BranchADenoiser
from models.branch_b_impulse import BranchBImpulse
from models.context_encoder import ContextEncoder
from models.fused_model import FusedBattlefieldModel


def test_causal_conv2d_shape():
    """Verify that CausalConv2d preserves time and frequency dimensions with stride=1."""
    conv = CausalConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
    x = torch.randn(2, 1, 257, 4)
    out = conv(x)
    assert out.shape == (2, 16, 257, 4), f"Expected shape (2, 16, 257, 4), got {out.shape}"


def test_dummy_streaming_conv_gru_forward():
    """Verify streaming forward pass and state shape preservation."""
    B, F, T = 1, 257, 4
    hidden_dim = 64
    num_layers = 2

    model = DummyStreamingConvGRU(
        freq_bins=F,
        conv_channels=32,
        gru_hidden_dim=hidden_dim,
        num_gru_layers=num_layers,
    )
    model.eval()

    input_chunk = torch.randn(B, 1, F, T)
    h_in = model.init_hidden_state(batch_size=B)

    with torch.no_grad():
        enhanced_chunk, h_out = model(input_chunk, h_in)

    assert enhanced_chunk.shape == (B, 1, F, T)
    assert h_out.shape == (num_layers, B, hidden_dim)


def test_streaming_state_continuity():
    """Verify state transitions over sequential chunks."""
    model = DummyStreamingConvGRU(freq_bins=257, conv_channels=16, gru_hidden_dim=32, num_gru_layers=1)
    model.eval()

    h = model.init_hidden_state(batch_size=1)
    chunk1 = torch.randn(1, 1, 257, 4)
    chunk2 = torch.randn(1, 1, 257, 4)

    with torch.no_grad():
        out1, h1 = model(chunk1, h)
        out2, h2 = model(chunk2, h1)

    # Assert state changed across chunks
    assert not torch.allclose(h, h1)
    assert not torch.allclose(h1, h2)


def test_branch_a_denoiser():
    """Verify Branch A continuous denoiser."""
    branch_a = BranchADenoiser(freq_bins=257, channels=16, hidden_dim=32, num_layers=1)
    x = torch.randn(1, 1, 257, 4)
    h = branch_a.init_hidden_state(batch_size=1)
    mask, feat, h_out = branch_a(x, h)
    assert mask.shape == (1, 1, 257, 4)
    assert h_out.shape == (1, 1, 32)
    assert (mask >= 0.0).all() and (mask <= 1.0).all()


def test_branch_b_impulse():
    """Verify Branch B impulse suppressor."""
    branch_b = BranchBImpulse(freq_bins=257, channels=16, hidden_dim=32)
    x = torch.randn(1, 1, 257, 4)
    mask, feat = branch_b(x)
    assert mask.shape == (1, 1, 257, 4)
    assert (mask >= 0.0).all() and (mask <= 1.0).all()


def test_context_encoder():
    """Verify Context Encoder embedding generation."""
    encoder = ContextEncoder(freq_bins=257, embedding_dim=32)
    x = torch.randn(2, 1, 257, 4)
    emb, snr = encoder(x)
    assert emb.shape == (2, 32)
    assert snr.shape == (2, 1)


def test_fused_battlefield_model():
    """Verify full multi-branch fused model."""
    fused = FusedBattlefieldModel(
        freq_bins=257,
        denoiser_channels=16,
        denoiser_hidden_dim=32,
        denoiser_layers=1,
        impulse_channels=16,
        impulse_hidden_dim=24,
        context_dim=16,
    )
    x = torch.randn(1, 1, 257, 4)
    h = fused.init_hidden_state(batch_size=1)
    enh, h_out, est_snr = fused(x, h)
    assert enh.shape == (1, 1, 257, 4)
    assert h_out.shape == (1, 1, 32)
    assert est_snr.shape == (1, 1)
