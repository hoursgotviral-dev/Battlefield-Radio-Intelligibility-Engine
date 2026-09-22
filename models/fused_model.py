"""
Fused Battlefield Speech Enhancement Model.

ML Concept - Dynamic Multi-Branch Mixture & Stateful Routing:
Rather than applying a single monolithic filter to vastly different noise types, the fused model
orchestrates the continuous denoiser (Branch A) and impulse noise suppressor (Branch B) through
a gating network conditioned on the acoustic context vector. 

All state transitions are explicit and static, making the combined model 100% compliant with
Qualcomm AI Hub compilation for edge Snapdragon NPUs.
"""

from typing import Tuple
import torch
import torch.nn as nn

from models.branch_a_denoiser import BranchADenoiser
from models.branch_b_impulse import BranchBImpulse
from models.context_encoder import ContextEncoder


class FusedBattlefieldModel(nn.Module):
    """
    Complete multi-branch streaming model combining continuous noise suppression,
    impulse gating, and contextual SNR steering.
    """
    def __init__(
        self,
        freq_bins: int = 257,
        denoiser_channels: int = 32,
        denoiser_hidden_dim: int = 64,
        denoiser_layers: int = 2,
        impulse_channels: int = 24,
        impulse_hidden_dim: int = 48,
        context_dim: int = 32,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.branch_a = BranchADenoiser(
            freq_bins=freq_bins,
            channels=denoiser_channels,
            hidden_dim=denoiser_hidden_dim,
            num_layers=denoiser_layers,
        )
        self.branch_b = BranchBImpulse(
            freq_bins=freq_bins,
            channels=impulse_channels,
            hidden_dim=impulse_hidden_dim,
        )
        self.context_encoder = ContextEncoder(
            freq_bins=freq_bins,
            embedding_dim=context_dim,
        )

        # Fusion combiner network
        self.fusion_net = nn.Sequential(
            nn.Linear(freq_bins * 2 + context_dim, freq_bins),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_chunk: torch.Tensor,
        h_a_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_chunk: [B, 1, F, T] spectrogram chunk
            h_a_in: [denoiser_layers, B, denoiser_hidden_dim]

        Returns:
            enhanced_chunk: [B, 1, F, T]
            h_a_out: [denoiser_layers, B, denoiser_hidden_dim]
            est_snr: [B, 1]
        """
        B, C, F, T = input_chunk.shape

        # 1. Branch A: Continuous Denoiser
        mask_a, feat_a, h_a_out = self.branch_a(input_chunk, h_a_in)

        # 2. Branch B: Impulse Suppressor
        mask_b, feat_b = self.branch_b(input_chunk)

        # 3. Context Encoder
        context_emb, est_snr = self.context_encoder(input_chunk)  # context: [B, context_dim]

        # 4. Fusion
        # Flatten masks over time: [B, T, F]
        mask_a_flat = mask_a.squeeze(1).permute(0, 2, 1)  # [B, T, F]
        mask_b_flat = mask_b.squeeze(1).permute(0, 2, 1)  # [B, T, F]
        context_expanded = context_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, context_dim]

        fusion_input = torch.cat([mask_a_flat, mask_b_flat, context_expanded], dim=-1)
        fused_mask = self.fusion_net(fusion_input).permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]

        # 5. Output
        enhanced_chunk = input_chunk * fused_mask

        return enhanced_chunk, h_a_out, est_snr

    def init_hidden_state(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return self.branch_a.init_hidden_state(batch_size=batch_size, device=device)
