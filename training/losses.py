"""
Loss Functions for Battlefield Speech Enhancement.

ML Concept - Multi-Resolution STFT & Scale-Invariant SDR:
Single-domain losses (e.g. simple Time-domain L1/MSE) fail to capture perceptual speech structures
and often result in muffled or phase-distorted outputs.
This module provides:
1. Multi-Resolution STFT Loss: Evaluates spectral convergence and log magnitude distance across
   multiple FFT window sizes (e.g., 512, 1024, 2048), capturing both sharp harmonic pitch tracks
   and broadband impulse transients.
2. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio): Evaluates energy orthogonality independent
   of overall volume scale.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss combining spectral convergence and log magnitude loss
    across different analysis window resolutions.
    """
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x: torch.Tensor, n_fft: int, hop: int, win_len: int) -> torch.Tensor:
        window = torch.hann_window(win_len, device=x.device)
        stft_out = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            return_complex=True,
        )
        return torch.abs(stft_out)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, T] or [B, 1, T]
            target: [B, T] or [B, 1, T]
        """
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target

        total_loss = 0.0
        for n_fft, hop, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            mag_pred = self.stft(pred, n_fft, hop, win_len)
            mag_target = self.stft(target, n_fft, hop, win_len)

            # Spectral convergence loss
            sc_loss = torch.norm(mag_target - mag_pred, p="fro") / (torch.norm(mag_target, p="fro") + 1e-7)
            # Log magnitude loss
            log_loss = torch.mean(torch.abs(torch.log(mag_target + 1e-7) - torch.log(mag_pred + 1e-7)))

            total_loss += sc_loss + log_loss

        return total_loss / len(self.fft_sizes)


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss.
    Negative SI-SDR in dB for minimization.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target

        # Zero-mean normalization
        pred = pred - torch.mean(pred, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

        # Optimal scaling factor alpha = <pred, target> / ||target||^2
        dot = torch.sum(pred * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / target_energy) * target

        # Distortion residual
        e_noise = pred - s_target

        s_target_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps

        si_sdr = 10.0 * torch.log10(s_target_energy / e_noise_energy)
        return -torch.mean(si_sdr)
