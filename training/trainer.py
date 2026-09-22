"""
Training Pipeline for Streaming Speech Enhancement Models.

ML Concept - Stateful Sequence Training:
To train streaming recurrent models without state resets causing discontinuity, utterances
are partitioned into consecutive chunks while carrying the hidden state forward across the sequence.
Gradients can be truncated (TBPTT) or computed over full utterances.
"""

from typing import Dict, Any, Optional
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dummy_conv_gru import DummyStreamingConvGRU
from data.dataset import TacticalSpeechDataset
from training.losses import MultiResolutionSTFTLoss, SISDRLoss


class TacticalTrainer:
    """
    Config-driven trainer for stateful streaming speech enhancement models.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)

        # Model instantiation
        model_cfg = config.get("model", {})
        self.model = DummyStreamingConvGRU(
            freq_bins=model_cfg.get("freq_bins", 257),
            conv_channels=model_cfg.get("conv_channels", 32),
            gru_hidden_dim=model_cfg.get("gru_hidden_dim", 64),
            num_gru_layers=model_cfg.get("num_gru_layers", 2),
        ).to(self.device)

        # Dataset & Loader
        self.dataset = TacticalSpeechDataset(
            sample_rate=config.get("audio", {}).get("sample_rate", 16000),
            seed=config.get("seed", 42),
            num_synthetic_samples=50,
        )
        self.dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)

        # Loss functions & Optimizer
        self.mrstft_loss = MultiResolutionSTFTLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.dataloader:
            clean = batch["clean"].to(self.device)      # [B, 1, T]
            degraded = batch["degraded"].to(self.device)# [B, 1, T]

            self.optimizer.zero_grad()

            # For dummy spectral demonstration: compute STFT outside model
            window = torch.hann_window(512, device=self.device)
            stft_deg = torch.stft(
                degraded.squeeze(1),
                n_fft=512,
                hop_length=128,
                win_length=512,
                window=window,
                return_complex=True,
            )
            mag_deg = torch.abs(stft_deg)  # [B, 257, T_frames]

            # Streaming chunk forward pass
            B, F, T = mag_deg.shape
            chunk_size = 4
            h = self.model.init_hidden_state(batch_size=B, device=self.device)
            enhanced_chunks = []

            for t_start in range(0, T - chunk_size + 1, chunk_size):
                chunk = mag_deg[:, :, t_start : t_start + chunk_size].unsqueeze(1)
                enh_chunk, h = self.model(chunk, h)
                enhanced_chunks.append(enh_chunk.squeeze(1))

            if enhanced_chunks:
                enhanced_mag = torch.cat(enhanced_chunks, dim=-1)
                # Compute loss on spectral magnitude
                target_stft = torch.stft(
                    clean.squeeze(1),
                    n_fft=512,
                    hop_length=128,
                    win_length=512,
                    window=window,
                    return_complex=True,
                )
                target_mag = torch.abs(target_stft)[:, :, : enhanced_mag.shape[-1]]

                loss = nn.functional.l1_loss(enhanced_mag, target_mag)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / max(1, len(self.dataloader))
