"""
Context Encoder Module.

ML Concept - Acoustic Environment & SNR Context Conditioning:
Tactical speech enhancement requires different filtering aggressiveness depending on whether
the user is inside an armored vehicle (low-frequency resonance), a helicopter cockpit (broadband tonal),
in open terrain with gunfire (transient-heavy), or facing RF channel degradation (bandpass & static).
The Context Encoder aggregates sub-band energy statistics and frame-level features into a low-dimensional
context embedding that dynamically conditions the downstream multi-branch fusion network.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Computes acoustic context embedding and estimated SNR condition vector.
    """
    def __init__(
        self,
        freq_bins: int = 257,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.embedding_dim = embedding_dim

        # Global average pooling across time frames followed by MLP
        self.encoder = nn.Sequential(
            nn.Linear(freq_bins, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.Tanh(),
        )

        # Auxiliary head for estimated SNR level (in dB normalized)
        self.snr_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 1, F, T] spectrogram chunk
        Returns:
            context_emb: [B, embedding_dim] context embedding vector
            estimated_snr: [B, 1] predicted SNR score
        """
        B, C, F, T = x.shape
        # Temporal mean pooling over chunk: [B, F]
        spectral_mean = x.mean(dim=3).squeeze(1)
        context_emb = self.encoder(spectral_mean)  # [B, embedding_dim]
        est_snr = self.snr_head(context_emb)       # [B, 1]
        return context_emb, est_snr
