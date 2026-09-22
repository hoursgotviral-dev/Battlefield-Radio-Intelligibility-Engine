"""
Streaming Conv-GRU Dummy Model for Tactical Speech Enhancement.

ML Concept - Streaming Recurrent Spectral Masking:
In real-time low-latency speech enhancement, full-utterance models (e.g. standard Transformers
or bidirectional RNNs) are unusable due to future-context dependency (lookahead latency).
Instead, a causal streaming architecture processes audio in short, fixed-length frames or chunks
(e.g., 32 ms STFT windows). 

To retain temporal memory of long-term acoustic context (such as sustained diesel engine rumble
or background radio hiss) without future lookahead, the model maintains an internal recurrent hidden
state (GRU state vector). This hidden state is explicitly passed into the model at step `t` along with
the current audio chunk, updated by the recurrent transitions, and returned as an explicit output
tensor `h_out` to be fed into step `t+1`.

By constraining convolutions to be strictly causal (padding only on the left/past time axis)
and using unidirectional GRUs, the computation graph has zero future receptive field and satisfies
strict NPU real-time budget constraints on edge platforms like Qualcomm Hexagon NPU.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CausalConv2d(nn.Module):
    """
    2D Convolution with causal padding on the time axis.
    Ensures that output at time frame t depends only on frames <= t.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.freq_pad = (kernel_size[0] - 1) // 2
        self.time_pad = kernel_size[1] - 1  # Full causal padding on left
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, F, T]
        # Pad: (pad_left, pad_right, pad_top, pad_bottom)
        x_padded = nn.functional.pad(x, (self.time_pad, 0, self.freq_pad, self.freq_pad))
        return self.conv(x_padded)


class DummyStreamingConvGRU(nn.Module):
    """
    Fixed-shape streaming Conv-GRU model designed for Qualcomm AI Hub NPU export.

    Inputs:
        input_chunk: torch.Tensor of shape [B, 1, F, T] or [B, F, T] (magnitude or feature spectrogram).
        h_in: torch.Tensor of shape [num_layers, B, hidden_dim] (recurrent hidden state from previous chunk).

    Outputs:
        enhanced_chunk: torch.Tensor of same shape as input_chunk (enhanced spectral magnitude).
        h_out: torch.Tensor of shape [num_layers, B, hidden_dim] (updated recurrent hidden state).
    """
    def __init__(
        self,
        freq_bins: int = 257,
        conv_channels: int = 32,
        gru_hidden_dim: int = 64,
        num_gru_layers: int = 2,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.conv_channels = conv_channels
        self.gru_hidden_dim = gru_hidden_dim
        self.num_gru_layers = num_gru_layers

        # Causal encoder conv
        self.encoder_conv = CausalConv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=(3, 3),
        )
        self.encoder_act = nn.PReLU()

        # Recurrent bottleneck (GRU)
        # Input to GRU: flattened across channels and freq bins -> [B, T, conv_channels * freq_bins]
        self.gru_in_dim = conv_channels * freq_bins
        self.gru = nn.GRU(
            input_size=self.gru_in_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Decoder projection to reconstruct spectral mask [B, T, freq_bins]
        self.mask_projection = nn.Sequential(
            nn.Linear(gru_hidden_dim, freq_bins),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_chunk: torch.Tensor,
        h_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward step for a streaming chunk.

        Args:
            input_chunk: Tensor of shape [B, 1, F, T] or [B, F, T]
            h_in: Tensor of shape [num_layers, B, hidden_dim]

        Returns:
            enhanced_chunk: Enhanced spectrogram chunk [B, 1, F, T]
            h_out: Updated recurrent state [num_layers, B, hidden_dim]
        """
        if input_chunk.dim() == 3:
            # [B, F, T] -> [B, 1, F, T]
            x = input_chunk.unsqueeze(1)
        else:
            x = input_chunk

        B, C, F, T = x.shape

        # 1. Spatial-spectral feature extraction
        feat = self.encoder_act(self.encoder_conv(x))  # [B, conv_channels, F, T]

        # 2. Reshape for GRU: [B, T, conv_channels * F]
        # Transpose to [B, T, conv_channels, F] then flatten
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat_flat = feat.view(B, T, self.conv_channels * F)

        # 3. Recurrent temporal modeling
        gru_out, h_out = self.gru(feat_flat, h_in)  # gru_out: [B, T, hidden_dim], h_out: [L, B, hidden_dim]

        # 4. Mask estimation
        mask = self.mask_projection(gru_out)  # [B, T, F]
        mask = mask.permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]

        # 5. Spectral suppression (Multiplicative masking)
        enhanced_chunk = x * mask

        if input_chunk.dim() == 3:
            enhanced_chunk = enhanced_chunk.squeeze(1)

        return enhanced_chunk, h_out

    def init_hidden_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Helper to create zero-initialized state tensor."""
        return torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden_dim, device=device)
