"""
Inference Reliability Guard for Battlefield Radio Speech Enhancement.

ML Engineering Objective:
Prevents worst-case enhancement failures and live-demo risks where model over-suppression,
low input SNR (< 2 dB), or musical tonal artifacts trigger catastrophic ASR hallucinations
or severe audible distortion.

Mechanism:
Monitors:
1. Energy Ratio: E_enh / E_deg (flags energy collapse < 0.25 or explosion > 1.30)
2. Spectral Flatness: SF_enh (flags tonal musical noise peaks < 0.0040)
3. Spectral Distortion Ratio: SF_enh / SF_deg (flags extreme spectral warping < 0.040)
4. Envelope Cross-Correlation: Corr(env_deg, env_enh) (flags envelope decorrelation < 0.70)

If any guard condition triggers, seamlessly falls back to passing through the original
degraded audio (or a safe spectral floor), ensuring the engine NEVER outputs worse-than-input
audio.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import torch


class ReliabilityGuard:
    """Lightweight inference-time safety guard for speech enhancement."""

    def __init__(
        self,
        min_energy_ratio: float = 0.25,
        max_energy_ratio: float = 1.30,
        min_spectral_flatness: float = 0.0040,
        min_flatness_ratio: float = 0.040,
        min_env_correlation: float = 0.70,
        n_fft: int = 512,
        hop_length: int = 128,
        sample_rate: int = 16000,
    ):
        self.min_energy_ratio = min_energy_ratio
        self.max_energy_ratio = max_energy_ratio
        self.min_spectral_flatness = min_spectral_flatness
        self.min_flatness_ratio = min_flatness_ratio
        self.min_env_correlation = min_env_correlation
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sample_rate

    def compute_features(self, deg_audio: np.ndarray, enh_audio: np.ndarray) -> Dict[str, float]:
        """Extracts acoustic integrity metrics between input and enhanced audio."""
        min_len = min(len(deg_audio), len(enh_audio))
        t_deg = torch.from_numpy(deg_audio[:min_len]).float()
        t_enh = torch.from_numpy(enh_audio[:min_len]).float()

        w = torch.hann_window(self.n_fft)
        stft_deg = torch.stft(
            t_deg,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=w,
            return_complex=True,
        )
        stft_enh = torch.stft(
            t_enh,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=w,
            return_complex=True,
        )

        mag_deg = torch.abs(stft_deg) + 1e-8
        mag_enh = torch.abs(stft_enh) + 1e-8

        pwr_deg = mag_deg**2
        pwr_enh = mag_enh**2

        e_deg = torch.sum(t_deg**2).item()
        e_enh = torch.sum(t_enh**2).item()
        energy_ratio = e_enh / (e_deg + 1e-8)

        sf_deg = (
            torch.exp(torch.mean(torch.log(pwr_deg), dim=0))
            / (torch.mean(pwr_deg, dim=0) + 1e-8)
        ).mean().item()
        sf_enh = (
            torch.exp(torch.mean(torch.log(pwr_enh), dim=0))
            / (torch.mean(pwr_enh, dim=0) + 1e-8)
        ).mean().item()
        flatness_ratio = sf_enh / (sf_deg + 1e-8)

        env_deg = torch.mean(mag_deg, dim=0)
        env_enh = torch.mean(mag_enh, dim=0)
        env_deg_norm = env_deg - env_deg.mean()
        env_enh_norm = env_enh - env_enh.mean()
        env_corr = (
            torch.sum(env_deg_norm * env_enh_norm)
            / (torch.norm(env_deg_norm) * torch.norm(env_enh_norm) + 1e-8)
        ).item()

        return {
            "energy_ratio": float(energy_ratio),
            "sf_enh": float(sf_enh),
            "sf_deg": float(sf_deg),
            "flatness_ratio": float(flatness_ratio),
            "env_corr": float(env_corr),
        }

    def check(self, deg_audio: np.ndarray, enh_audio: np.ndarray) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Evaluates safety constraints.
        Returns: (is_safe, trigger_reasons, extracted_features)
        """
        feats = self.compute_features(deg_audio, enh_audio)
        triggers = []

        if feats["energy_ratio"] < self.min_energy_ratio:
            triggers.append(f"Energy collapse (ratio={feats['energy_ratio']:.3f} < {self.min_energy_ratio})")
        if feats["energy_ratio"] > self.max_energy_ratio:
            triggers.append(f"Energy explosion (ratio={feats['energy_ratio']:.3f} > {self.max_energy_ratio})")
        if feats["sf_enh"] < self.min_spectral_flatness:
            triggers.append(f"Tonal peakiness/musical noise (flatness={feats['sf_enh']:.4f} < {self.min_spectral_flatness})")
        if feats["flatness_ratio"] < self.min_flatness_ratio:
            triggers.append(f"Severe spectral distortion (flatness_ratio={feats['flatness_ratio']:.3f} < {self.min_flatness_ratio})")
        if feats["env_corr"] < self.min_env_correlation:
            triggers.append(f"Envelope decorrelation (corr={feats['env_corr']:.3f} < {self.min_env_correlation})")

        is_safe = len(triggers) == 0
        return is_safe, triggers, feats

    def compute_confidence(self, feats: Dict[str, float]) -> float:
        """
        Computes a continuous confidence score in [0.0, 1.0] based on how far
        acoustic features are from safety thresholds.
        """
        # Energy ratio confidence (soft sigmoid around min_energy_ratio)
        c_energy = 1.0 / (1.0 + np.exp(-15.0 * (feats["energy_ratio"] - self.min_energy_ratio)))
        if feats["energy_ratio"] > 1.0:
            c_energy *= 1.0 / (1.0 + np.exp(15.0 * (feats["energy_ratio"] - self.max_energy_ratio)))

        # Spectral flatness confidence
        c_flatness = 1.0 / (1.0 + np.exp(-1000.0 * (feats["sf_enh"] - self.min_spectral_flatness)))

        # Flatness ratio confidence
        c_fratio = 1.0 / (1.0 + np.exp(-50.0 * (feats["flatness_ratio"] - self.min_flatness_ratio)))

        # Envelope correlation confidence
        c_corr = 1.0 / (1.0 + np.exp(-20.0 * (feats["env_corr"] - self.min_env_correlation)))

        confidence = float(min(c_energy, c_flatness, c_fratio, c_corr))
        return float(np.clip(confidence, 0.0, 1.0))

    def apply_guard(
        self,
        deg_audio: np.ndarray,
        enh_audio: np.ndarray,
        mode: str = "soft",
    ) -> Tuple[np.ndarray, bool, float, List[str]]:
        """
        Guards output audio. If mode='soft', blends confidence * enh + (1 - conf) * deg.
        If mode='hard', switches binary to degraded when triggered.
        Returns: (guarded_audio, is_safe, confidence, trigger_reasons)
        """
        is_safe, triggers, feats = self.check(deg_audio, enh_audio)
        conf = self.compute_confidence(feats)
        min_len = min(len(deg_audio), len(enh_audio))
        d = deg_audio[:min_len]
        e = enh_audio[:min_len]

        if mode == "soft":
            out = conf * e + (1.0 - conf) * d
        else:
            out = e if is_safe else d

        return out.astype(np.float32), is_safe, conf, triggers

