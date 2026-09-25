"""
Fused Battlefield Speech Enhancement Model.

ML Concept - Dynamic Multi-Branch Mixture & Stateful Routing:
Orchestrates:
1. Branch A: Continuous Noise Denoiser (Causal ConvGRU) for mechanical engine/wind rumble.
2. Branch B: Impulse Noise Suppressor (Causal Transient CNN + GLU) for gunfire/artillery shockwaves.
3. Context Encoder: Acoustic environment and dynamic SNR condition vector.
4. Gated Spectral Fusion: Dynamically weights continuous vs. transient suppression based on context.

All state transitions are explicit, static, and 100% compliant with ONNX Opset 17 and Qualcomm AI Hub.
"""

from typing import Tuple, Optional
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
        self.denoiser_layers = denoiser_layers
        self.denoiser_hidden_dim = denoiser_hidden_dim
        self.impulse_channels = impulse_channels

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

        # Gated fusion combiner: combines continuous mask, impulse mask, and context
        self.fusion_net = nn.Sequential(
            nn.Linear(freq_bins * 2 + 1 + context_dim, freq_bins),
            nn.PReLU(),
            nn.Linear(freq_bins, freq_bins),
            nn.Sigmoid(),
        )

    def init_hidden_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes recurrent states for Branch A and temporal buffer for Branch B."""
        h_a = torch.zeros(self.denoiser_layers, batch_size, self.denoiser_hidden_dim, device=device)
        state_b = torch.zeros(batch_size, self.impulse_channels, self.freq_bins, 1, device=device)
        return h_a, state_b

    def load_pretrained_branches(
        self,
        branch_a_path: str = "checkpoints/branch_a_curriculum_best.pt",
        branch_b_path: str = "checkpoints/branch_b_best.pt",
        device: torch.device = torch.device("cpu"),
    ):
        """Loads pretrained weights into Branch A and Branch B."""
        if branch_a_path and torch.cuda.is_available() or True:
            try:
                ckpt_a = torch.load(branch_a_path, map_location=device)
                sd_a = ckpt_a.get("model_state_dict", ckpt_a)
                self.branch_a.load_state_dict(sd_a, strict=False)
                print(f"[+] Loaded pretrained Branch A weights from {branch_a_path}")
            except Exception as e:
                print(f"[-] Note: Could not load Branch A checkpoint: {e}")

        if branch_b_path:
            try:
                ckpt_b = torch.load(branch_b_path, map_location=device)
                sd_b = ckpt_b.get("model_state_dict", ckpt_b)
                self.branch_b.load_state_dict(sd_b, strict=False)
                print(f"[+] Loaded pretrained Branch B weights from {branch_b_path}")
            except Exception as e:
                print(f"[-] Note: Could not load Branch B checkpoint: {e}")

    def forward(
        self,
        input_chunk: torch.Tensor,
        h_a_in: torch.Tensor,
        state_b_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_chunk: [B, 1, F, T] spectrogram chunk
            h_a_in: [denoiser_layers, B, denoiser_hidden_dim]
            state_b_in: [B, impulse_channels, F, 1]

        Returns:
            enhanced_chunk: [B, 1, F, T]
            fused_mask: [B, 1, F, T]
            h_a_out: [denoiser_layers, B, denoiser_hidden_dim]
            state_b_out: [B, impulse_channels, F, 1]
            est_snr: [B, 1]
        """
        B, C, F, T = input_chunk.shape

        # 1. Branch A: Continuous Denoiser
        mask_a, feat_a, h_a_out = self.branch_a(input_chunk, h_a_in)

        # 2. Branch B: Impulse Suppressor
        mask_b, prob_b, state_b_out = self.branch_b(input_chunk, state_b_in)

        # 3. Context Encoder
        context_emb, est_snr = self.context_encoder(input_chunk)  # [B, context_dim]

        # 4. Gated Fusion
        # Re-arrange: [B, T, F]
        mask_a_flat = mask_a.squeeze(1).permute(0, 2, 1)  # [B, T, F]
        mask_b_flat = mask_b.squeeze(1).permute(0, 2, 1)  # [B, T, F]
        prob_b_flat = prob_b.permute(0, 2, 1)            # [B, T, 1]
        context_expanded = context_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, context_dim]

        fusion_in = torch.cat([mask_a_flat, mask_b_flat, prob_b_flat, context_expanded], dim=-1)
        fused_mask_flat = self.fusion_net(fusion_in)  # [B, T, F]
        fused_mask = fused_mask_flat.permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]

        # 5. Multiply input chunk by fused mask
        enhanced_chunk = input_chunk * fused_mask

        return enhanced_chunk, fused_mask, h_a_out, state_b_out, est_snr
