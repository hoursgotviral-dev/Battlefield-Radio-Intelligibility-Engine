"""
Unit tests for Speech Evaluation Suite (PESQ, STOI, WER).
"""

import pytest
import numpy as np

from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator
from eval.evaluate import evaluate_audio_pair


def test_stoi_identical_audio():
    """STOI between identical signals should be close to 1.0."""
    t = np.linspace(0, 1, 16000)
    audio = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    score = compute_stoi(audio, audio, sample_rate=16000)
    assert score > 0.95, f"Expected STOI > 0.95, got {score}"


def test_pesq_identical_audio():
    """PESQ for clean identical signal should be high."""
    t = np.linspace(0, 1, 16000)
    audio = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    score = compute_pesq(audio, audio, sample_rate=16000)
    assert score > 3.0, f"Expected PESQ > 3.0, got {score}"


def test_wer_calculation():
    """Verify Word Error Rate computation."""
    ref = "alpha leader this is bravo actual radio check over"
    hyp_exact = "alpha leader this is bravo actual radio check over"
    hyp_err = "alpha leader this is bravo radio check out"

    wer_zero = compute_wer(ref, hyp_exact)
    wer_positive = compute_wer(ref, hyp_err)

    assert wer_zero == 0.0
    assert wer_positive > 0.0


def test_evaluate_audio_pair():
    """Verify evaluation wrapper output dictionary."""
    t = np.linspace(0, 1, 16000)
    clean = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    noise = np.random.randn(16000).astype(np.float32) * 0.5
    degraded = clean + noise
    enhanced = clean + noise * 0.2

    res = evaluate_audio_pair(clean, degraded, enhanced)
    assert "pesq_degraded" in res
    assert "pesq_enhanced" in res
    assert "stoi_degraded" in res
    assert "stoi_enhanced" in res
    assert res["stoi_enhanced"] >= res["stoi_degraded"]
