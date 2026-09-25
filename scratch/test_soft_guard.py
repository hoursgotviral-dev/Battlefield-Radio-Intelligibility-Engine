import json
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.branch_a_denoiser import BranchADenoiser
from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator
from eval.reliability_guard import ReliabilityGuard
from training.train_branch_a import enhance_audio_waveform

def test_soft_blend_guard():
    print("=" * 80)
    print(" SOFT BLEND RELIABILITY GUARD TEST (50 UTTERANCES)")
    print("=" * 80)

    device = torch.device("cpu")
    model = BranchADenoiser(freq_bins=257, channels=32, hidden_dim=64, num_layers=2).to(device)
    ckpt = torch.load("checkpoints/branch_a_curriculum_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    guard = ReliabilityGuard()
    whisper_eval = WhisperEvaluator()

    manifest = json.load(open("data/simulated/test/manifest.json", "r", encoding="utf-8"))

    recs_hard = []
    recs_soft = []

    for idx, item in enumerate(manifest):
        tid = item["id"]
        ref_text = item["transcript"]
        c_audio, sr = sf.read(item["clean_path"], dtype="float32")
        d_audio, _ = sf.read(item["degraded_path"], dtype="float32")

        # Raw enhancement
        a_raw = enhance_audio_waveform(model, d_audio, sample_rate=16000)

        # Apply Hard Guard
        is_safe, trigs, feats = guard.check(d_audio, a_raw)
        min_len = min(len(c_audio), len(d_audio), len(a_raw))
        c_eval = c_audio[:min_len]
        d_eval = d_audio[:min_len]
        e_eval = a_raw[:min_len]

        a_hard = e_eval if is_safe else d_eval

        # Apply Soft Guard
        conf = guard.compute_confidence(feats)
        a_soft = conf * e_eval + (1.0 - conf) * d_eval

        # Metrics Hard
        hyp_hard = whisper_eval.transcribe(a_hard, sr)
        wer_hard = compute_wer(ref_text, hyp_hard)
        pesq_hard = compute_pesq(c_eval, a_hard, sr)
        stoi_hard = compute_stoi(c_eval, a_hard, sr)

        # Metrics Soft
        hyp_soft = whisper_eval.transcribe(a_soft, sr)
        wer_soft = compute_wer(ref_text, hyp_soft)
        pesq_soft = compute_pesq(c_eval, a_soft, sr)
        stoi_soft = compute_stoi(c_eval, a_soft, sr)

        recs_hard.append({"id": tid, "wer": wer_hard, "pesq": pesq_hard, "stoi": stoi_hard, "safe": is_safe, "conf": conf})
        recs_soft.append({"id": tid, "wer": wer_soft, "pesq": pesq_soft, "stoi": stoi_soft, "safe": is_safe, "conf": conf})

        if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
            print(f"  Evaluated [{idx+1:02d}/{len(manifest):02d}] utterances...")

    def summarize(recs, name):
        wers = [r["wer"] for r in recs]
        pesqs = [r["pesq"] for r in recs]
        stois = [r["stoi"] for r in recs]
        outliers = sum(1 for w in wers if w > 1.0)
        return {
            "name": name,
            "pesq_mean": round(float(np.mean(pesqs)), 3),
            "stoi_mean": round(float(np.mean(stois)), 3),
            "wer_median": round(float(np.median(wers)), 3),
            "wer_clipped_mean": round(float(np.mean(np.clip(wers, 0, 1))), 3),
            "wer_raw_mean": round(float(np.mean(wers)), 3),
            "outliers": outliers,
        }

    s_hard = summarize(recs_hard, "Hard Fallback Guard")
    s_soft = summarize(recs_soft, "Soft Blend Guard")

    print("\n" + "=" * 80)
    print(" COMPARISON: HARD FALLBACK VS. SOFT BLEND RELIABILITY GUARD")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Hard Fallback Guard':<22} | {'Soft Blend Guard':<20}")
    print("-" * 80)
    print(f"{'PESQ (-0.5 to 4.5)':<25} | {s_hard['pesq_mean']:<22.3f} | {s_soft['pesq_mean']:<20.3f}")
    print(f"{'STOI (0.0 to 1.0)':<25} | {s_hard['stoi_mean']:<22.3f} | {s_soft['stoi_mean']:<20.3f}")
    print(f"{'Whisper Median WER':<25} | {s_hard['wer_median']:<22.3f} | {s_soft['wer_median']:<20.3f}")
    print(f"{'Whisper Clipped Mean':<25} | {s_hard['wer_clipped_mean']:<22.3f} | {s_soft['wer_clipped_mean']:<20.3f}")
    print(f"{'Whisper Raw Mean':<25} | {s_hard['wer_raw_mean']:<22.3f} | {s_soft['wer_raw_mean']:<20.3f}")
    print(f"{'Outliers (WER > 1.0)':<25} | {s_hard['outliers']:<22} | {s_soft['outliers']:<20}")
    print("=" * 80)

if __name__ == "__main__":
    test_soft_blend_guard()
