"""
Unit tests for Battlefield Audio Simulation pipeline.
"""

import pytest
import numpy as np
import torch

from data.simulation import BattlefieldAudioSimulator
from data.dataset import TacticalSpeechDataset


def test_simulation_reproducibility():
    """Verify that same seed yields identical simulated degradations."""
    sim1 = BattlefieldAudioSimulator(sample_rate=16000, seed=123)
    sim2 = BattlefieldAudioSimulator(sample_rate=16000, seed=123)

    clean = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
    res1 = sim1.simulate_battlefield_degradation(clean, snr_db=0.0)
    res2 = sim2.simulate_battlefield_degradation(clean, snr_db=0.0)

    np.testing.assert_allclose(res1["degraded"], res2["degraded"], rtol=1e-5, atol=1e-6)


def test_gunfire_impulse_generation():
    """Verify gunfire generator produces transient peaks."""
    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=42)
    gunfire = sim.generate_gunfire_impulse(length_samples=16000, burst_count=3)
    assert len(gunfire) == 16000
    assert np.max(np.abs(gunfire)) > 0.3
    # Verify non-zero energy
    assert np.sum(gunfire ** 2) > 0.0


def test_engine_noise_generation():
    """Verify armored vehicle engine noise generation."""
    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=42)
    engine = sim.generate_engine_noise(length_samples=16000, fundamental_hz=50.0)
    assert len(engine) == 16000
    assert np.std(engine) > 0.5


def test_bandpass_filtering():
    """Verify radio bandpass attenuates frequencies outside 300 - 3400 Hz."""
    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=42)
    t = np.linspace(0, 1, 16000)
    
    # 50 Hz sub-bass (should be attenuated)
    sub_bass = np.sin(2 * np.pi * 50 * t)
    # 1000 Hz mid-band (should pass)
    voice_mid = np.sin(2 * np.pi * 1000 * t)
    
    filt_bass = sim.apply_radio_bandpass(sub_bass)
    filt_mid = sim.apply_radio_bandpass(voice_mid)

    bass_energy = np.mean(filt_bass ** 2)
    mid_energy = np.mean(filt_mid ** 2)
    assert mid_energy > bass_energy * 5.0, "Passband energy should dominate stopped band."


def test_nonlinear_radio_clipping():
    """Verify push-to-talk microphone non-linear saturation."""
    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=42)
    high_input = np.ones(1000, dtype=np.float32) * 5.0
    clipped = sim.apply_nonlinear_radio_clipping(high_input, drive=3.0)
    assert np.max(clipped) <= 1.0


def test_tactical_speech_dataset():
    """Verify PyTorch dataset generation."""
    dataset = TacticalSpeechDataset(sample_rate=16000, num_synthetic_samples=5)
    assert len(dataset) == 5
    item = dataset[0]
    assert "clean" in item and "degraded" in item and "snr_db" in item
    assert item["clean"].shape == (1, 32000)
    assert item["degraded"].shape == (1, 32000)
