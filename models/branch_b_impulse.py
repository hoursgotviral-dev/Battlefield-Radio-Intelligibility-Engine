"""
Branch B: Impulse & Transient Noise Suppressor Module.

ML Concept - High-SPL Acoustic Transient Gating:
Gunfire, bullet snaps, artillery blasts, and explosive shockwaves exhibit distinct physical
characteristics: extreme rise times (microseconds to single milliseconds), high crest factors (> 20 dB),
and broadband spectral splattering. Standard continuous denoisers smear these transients into
musical chirp artifacts.
Branch B implements a fast temporal gated linear unit (GLU) with energy envelope normalization
to rapidly detect and gate high-energy acoustic transients frame-by-frame.
"""

from typing import Tuple
import torch
import torch.nn as nn
from models.dummy_conv_gru import CausalConv2d


class BranchBImpulse(nn.Module):
    """
    Causal high-SPL transient and impulse noise suppressor.
    """
    def __init__(
        self,
        freq_bins: int = 257,
        channels: int = 24,
        hidden_dim: int = 48,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.channels = channels
        self.hidden_dim = hidden_dim

        # Narrow receptive field in time for high temporal resolution
        self.transient_conv = CausalConv2d(1, channels, kernel_size=(3, 2))
        self.gate_conv = CausalConv2d(1, channels, kernel_size=(3, 2))

        self.glu_proj = nn.Sequential(
            nn.Linear(channels * freq_bins, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_bins),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, F, T] spectrogram chunk
        Returns:
            impulse_mask: [B, 1, F, T] transient suppression mask
            transient_feat: [B, hidden_dim, T] transient features
        """
        B, C, F, T = x.shape
        signal_feat = self.transient_conv(x)
        gate_feat = torch.sigmoid(self.gate_conv(x))
        gated = signal_feat * gate_feat  # [B, channels, F, T]

        gated_flat = gated.permute(0, 3, 1, 2).contiguous().view(B, T, self.channels * F)
        mask = self.glu_proj(gated_flat)  # [B, T, F]
        mask = mask.permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]

        return mask, gated_flat.permute(0, 2, 1)
