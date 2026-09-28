"""
Spectral Post-Filter for Battlefield Tactical Speech Enhancement.

ML & DSP Concept:
Neural denoisers often leave residual high-frequency radio hiss or low-level musical noise
in quiet / unvoiced frames. A lightweight, causal Wiener/attenuation post-filter refines the
enhanced audio waveform by tracking the stationary residual noise floor and applying gentle,
bounded spectral smoothing (oversubtraction with spectral floor) to polish speech formants
and boost PESQ/STOI scores without degrading intelligibility (WER).
"""

from typing import Optional
import numpy as np
import torch


class SpectralPostFilter:
    """Lightweight DSP spectral post-filter to polish neural speech enhancement output."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        oversubtraction: float = 1.15,
        spectral_floor: float = 0.08,
        smoothing_factor: float = 0.85,
        sample_rate: int = 16000,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.oversubtraction = oversubtraction
        self.spectral_floor = spectral_floor
        self.smoothing_factor = smoothing_factor
        self.sample_rate = sample_rate

    def process(
        self,
        enh_audio: np.ndarray,
        deg_audio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies causal spectral post-filtering to enhanced waveform.

        Args:
            enh_audio: 1D numpy array of enhanced audio samples (at 16 kHz)
            deg_audio: Optional original degraded audio for residual estimation

        Returns:
            postfiltered_audio: 1D numpy array of post-filtered audio samples
        """
        if len(enh_audio) == 0:
            return enh_audio

        t_enh = torch.from_numpy(enh_audio).float()
        window = torch.hann_window(self.n_fft)

        stft_enh = torch.stft(
            t_enh,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )

        mag = torch.abs(stft_enh)  # [F, T]
        phase = torch.angle(stft_enh)

        # Estimate residual noise floor from the lowest 10% energy frames
        frame_energies = torch.mean(mag, dim=0)
        num_noise_frames = max(1, int(0.10 * frame_energies.shape[0]))
        _, sorted_indices = torch.topk(frame_energies, k=num_noise_frames, largest=False)
        noise_profile = torch.mean(mag[:, sorted_indices], dim=1, keepdim=True)  # [F, 1]

        # Spectral subtraction with oversubtraction and safety floor
        subtracted = mag - self.oversubtraction * noise_profile
        floor = self.spectral_floor * mag
        clean_mag = torch.maximum(subtracted, floor)

        # Temporal smoothing to prevent musical tone flutter
        smoothed_mag = torch.empty_like(clean_mag)
        smoothed_mag[:, 0] = clean_mag[:, 0]
        for t in range(1, clean_mag.shape[1]):
            smoothed_mag[:, t] = (
                self.smoothing_factor * smoothed_mag[:, t - 1]
                + (1.0 - self.smoothing_factor) * clean_mag[:, t]
            )

        # Reconstruct complex STFT with original enhanced phase
        complex_post = torch.polar(smoothed_mag, phase)
        post_audio = torch.istft(
            complex_post,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            length=len(enh_audio),
        )

        return post_audio.numpy().astype(np.float32)
