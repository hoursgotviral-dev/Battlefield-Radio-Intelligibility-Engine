"""
PyTorch Dataset & Streaming Data Loader for Battlefield Speech Enhancement.

ML Concept - Dynamic Online Simulation for Data Augmentation:
Rather than storing hundreds of gigabytes of pre-mixed degraded audio files, online simulation
dynamically degrades clean speech utterances on-the-fly during training. This provides infinite
variability in SNR, noise phase combinations, packet loss distributions, and non-linear distortion,
preventing the recurrent neural network from overfitting to specific noise profiles.
"""

from typing import Optional, List, Dict, Tuple
import os
import glob
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.simulation import BattlefieldAudioSimulator


class TacticalSpeechDataset(Dataset):
    """
    Dataset that loads clean speech waveforms and dynamically generates battlefield degraded pairs.
    """
    def __init__(
        self,
        clean_audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        segment_length_samples: int = 32000,  # 2.0 seconds
        seed: int = 42,
        num_synthetic_samples: int = 100,  # Used if clean_audio_dir is None or empty
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_length = segment_length_samples
        self.simulator = BattlefieldAudioSimulator(sample_rate=sample_rate, seed=seed)
        self.audio_files: List[str] = []

        if clean_audio_dir and os.path.isdir(clean_audio_dir):
            self.audio_files = sorted(
                glob.glob(os.path.join(clean_audio_dir, "**/*.wav"), recursive=True)
            )

        self.num_synthetic_samples = num_synthetic_samples
        self.use_synthetic = len(self.audio_files) == 0

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic_samples
        return len(self.audio_files)

    def _generate_synthetic_clean_speech(self, idx: int) -> np.ndarray:
        """Generates synthetic multi-tone speech-like formant signal for standalone testing."""
        rng = np.random.RandomState(idx + 1000)
        t = np.arange(self.segment_length) / self.sample_rate
        f0 = rng.uniform(100.0, 250.0)
        signal_out = np.zeros(self.segment_length, dtype=np.float32)
        # Formants
        for harmonic in [1, 2, 3, 4, 6]:
            signal_out += (1.0 / harmonic) * np.sin(2 * np.pi * (f0 * harmonic) * t)
        # Amplitude envelope modulation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t)
        return (signal_out * envelope * 0.5).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_synthetic:
            clean_audio = self._generate_synthetic_clean_speech(idx)
        else:
            file_path = self.audio_files[idx]
            clean_audio, sr = sf.read(file_path, dtype="float32")
            if clean_audio.ndim > 1:
                clean_audio = clean_audio.mean(axis=1)
            # Resample or pad/trim
            if len(clean_audio) < self.segment_length:
                clean_audio = np.pad(clean_audio, (0, self.segment_length - len(clean_audio)))
            else:
                clean_audio = clean_audio[: self.segment_length]

        # Apply battlefield degradation
        sim_res = self.simulator.simulate_battlefield_degradation(clean_audio)

        return {
            "clean": torch.from_numpy(sim_res["clean"]).unsqueeze(0),       # [1, T]
            "degraded": torch.from_numpy(sim_res["degraded"]).unsqueeze(0), # [1, T]
            "snr_db": torch.tensor(sim_res["snr_db"], dtype=torch.float32),
        }
