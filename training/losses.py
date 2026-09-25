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


class WhisperPerceptualLoss(nn.Module):
    """
    Computes perceptual distance in Whisper encoder embedding space.
    Differentiably backpropagates phonetic and linguistic feature distance
    to steer the speech enhancement model toward high ASR intelligibility.
    """
    def __init__(self, model_name: str = "openai/whisper-tiny", device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        from transformers import WhisperModel
        import torchaudio.functional as F_audio

        whisper = WhisperModel.from_pretrained(model_name).to(device)
        self.encoder = whisper.encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80
        self.register_buffer("window", torch.hann_window(self.n_fft))
        mel_fb = F_audio.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=8000.0,
            n_mels=self.n_mels,
            sample_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("mel_fb", mel_fb)

    def wav_to_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Differentiable 80-channel log-Mel spectrogram matching Whisper specifications."""
        wav = wav.squeeze(1) if wav.dim() == 3 else wav
        window = torch.hann_window(self.n_fft, device=wav.device)
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        pwr = torch.abs(stft[:, :, :-1]) ** 2
        mel_spec = torch.matmul(self.mel_fb.to(wav.device).transpose(0, 1), pwr)
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        target_frames = 3000
        B, n_m, T_f = log_mel.shape
        if T_f < target_frames:
            log_mel = torch.nn.functional.pad(log_mel, (0, target_frames - T_f))
        else:
            log_mel = log_mel[:, :, :target_frames]
        return log_mel

    def forward(self, enh_wav: torch.Tensor, clean_wav: torch.Tensor) -> torch.Tensor:
        mel_enh = self.wav_to_log_mel(enh_wav)
        with torch.no_grad():
            mel_clean = self.wav_to_log_mel(clean_wav)
            feat_clean = self.encoder(mel_clean).last_hidden_state

        feat_enh = self.encoder(mel_enh).last_hidden_state
        return torch.mean(torch.abs(feat_enh - feat_clean))


class EnergyFloorLoss(nn.Module):
    """
    Prevents over-aggressive suppression / energy collapse under low SNR (< 2 dB).
    Penalizes enhanced audio if its total energy falls below a safe minimum ratio
    of the input degraded energy.
    """
    def __init__(self, min_ratio: float = 0.28):
        super().__init__()
        self.min_ratio = min_ratio

    def forward(self, enh_audio: torch.Tensor, deg_audio: torch.Tensor) -> torch.Tensor:
        enh_audio = enh_audio.squeeze(1) if enh_audio.dim() == 3 else enh_audio
        deg_audio = deg_audio.squeeze(1) if deg_audio.dim() == 3 else deg_audio
        e_enh = torch.sum(enh_audio ** 2, dim=-1) + 1e-8
        e_deg = torch.sum(deg_audio ** 2, dim=-1) + 1e-8
        ratio = e_enh / e_deg
        penalty = torch.relu(self.min_ratio - ratio) ** 2
        return torch.mean(penalty)


