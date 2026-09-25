"""
Branch B: Causal Impulse & Gunshot Noise Suppressor Module.

ML Concept - High-SPL Acoustic Transient Gating:
Gunfire, bullet snaps, artillery blasts, and explosive shockwaves exhibit distinct physical
characteristics: extreme rise times (< 1 ms), high crest factors (> 20 dB), and broadband
spectral splattering. Standard continuous denoisers smear these transients into chirp artifacts.

Branch B implements a fast temporal gated linear network with temporal spectral flux
and transient envelope normalization to rapidly detect and suppress high-energy acoustic
transients frame-by-frame while preserving underlying speech formants.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from models.dummy_conv_gru import CausalConv2d


class BranchBImpulse(nn.Module):
    """
    Causal high-SPL transient and impulse noise suppressor (Gunshots, Artillery, Blast Shockwaves).
    
    Inputs:
        x: [B, 1, F, T] spectrogram magnitude chunk (F=257, T=frames)
        state: Optional recurrent/temporal buffer state [B, channels, F, 1] for streaming
    Outputs:
        impulse_mask: [B, 1, F, T] transient attenuation mask in [0, 1]
        transient_prob: [B, 1, T] frame-level transient detection probability
        next_state: [B, channels, F, 1] updated buffer state
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

        # High temporal resolution causal convolution (kernel_size: (3 freq, 2 time))
        self.transient_conv = CausalConv2d(1, channels, kernel_size=(3, 2))
        self.gate_conv = CausalConv2d(1, channels, kernel_size=(3, 2))

        # Temporal delta flux projection to detect sudden energy bursts
        self.flux_conv = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spectral mask generator
        self.mask_proj = nn.Sequential(
            nn.Linear(channels * freq_bins, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, freq_bins),
            nn.Sigmoid(),
        )

    def init_hidden_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Initializes streaming context state for causal inference."""
        return torch.zeros(batch_size, self.channels, self.freq_bins, 1, device=device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, F, T] spectrogram chunk
            state: [B, channels, F, 1] optional streaming context
        Returns:
            impulse_mask: [B, 1, F, T] suppression mask in [0, 1]
            transient_prob: [B, 1, T] frame-level transient detection probability
            next_state: [B, channels, F, 1]
        """
        B, C, F, T = x.shape

        # Signal & Gate convolutions
        feat = self.transient_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        gated = feat * gate  # [B, channels, F, T]

        # Save latest frame context for streaming state
        next_state = gated[:, :, :, -1:]

        # 1. Frame-level transient detection probability
        # Average across frequency to get temporal energy envelope: [B, channels, T]
        temp_env = torch.mean(gated, dim=2)
        transient_prob = self.flux_conv(temp_env)  # [B, 1, T]

        # 2. Spectral mask generation
        gated_flat = gated.permute(0, 3, 1, 2).contiguous().view(B, T, self.channels * F)
        mask = self.mask_proj(gated_flat)  # [B, T, F]
        impulse_mask = mask.permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]

        return impulse_mask, transient_prob, next_state
