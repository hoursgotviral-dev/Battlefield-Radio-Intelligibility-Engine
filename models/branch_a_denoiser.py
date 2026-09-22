"""
Branch A: Continuous Noise Denoiser Module.

ML Concept - Stationary & Quasi-Stationary Noise Suppression:
Continuous environmental noise (diesel engine rumble, rotor chop, wind buffeting) exhibits
strong temporal continuity and correlated spectral envelopes over multiple frames.
Branch A employs causal depthwise separable convolutions coupled with multi-frame recurrent
state tracking (GRU) to estimate a stationary spectral floor and track slow variations
in background noise power.
"""

from typing import Tuple
import torch
import torch.nn as nn
from models.dummy_conv_gru import CausalConv2d


class BranchADenoiser(nn.Module):
    """
    Causal continuous noise suppression branch.
    Outputs continuous noise spectral mask and updated recurrent state.
    """
    def __init__(
        self,
        freq_bins: int = 257,
        channels: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conv1 = CausalConv2d(1, channels, kernel_size=(5, 3))
        self.act1 = nn.PReLU()
        self.conv2 = CausalConv2d(channels, channels, kernel_size=(3, 3))
        self.act2 = nn.PReLU()

        self.gru = nn.GRU(
            input_size=channels * freq_bins,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, freq_bins),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        h_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, F, T] spectrogram chunk
            h_in: [num_layers, B, hidden_dim]
        Returns:
            mask_a: [B, 1, F, T] estimated gain mask
            features: [B, hidden_dim, T] intermediate temporal features
            h_out: [num_layers, B, hidden_dim]
        """
        B, C, F, T = x.shape
        feat = self.act1(self.conv1(x))
        feat = self.act2(self.conv2(feat))

        feat = feat.permute(0, 3, 1, 2).contiguous().view(B, T, -1)
        gru_out, h_out = self.gru(feat, h_in)  # [B, T, hidden_dim]

        mask = self.out_proj(gru_out).permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]
        features = gru_out.permute(0, 2, 1)  # [B, hidden_dim, T]

        return mask, features, h_out

    def init_hidden_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
