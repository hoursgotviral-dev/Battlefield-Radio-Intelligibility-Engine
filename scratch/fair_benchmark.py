import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.branch_a_denoiser import BranchADenoiser
from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator
from eval.reliability_guard import ReliabilityGuard
from training.train_branch_a import enhance_audio_waveform

def run_fair_benchmark():
    device = torch.device("cpu")
    print("[*] Loading Branch A model from checkpoints/branch_a_asr_best.pt...")
    model = BranchADenoiser(freq_bins=257, channels=32, hidden_dim=64, num_layers=2).to(device)
    ckpt = torch.load("checkpoints/branch_a_asr_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    guard = ReliabilityGuard()
    whisper_eval = WhisperEvaluator()

    manifest_path = Path("data/simulated/test/manifest.json")
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))

    results = []
    
    print(f"[*] Running fair guarded benchmark across all {len(manifest)} test utterances...")
    for idx, item in enumerate(manifest):
        tid = item["id"]
        ref_text = item["transcript"]
        c_audio, sr = sf.read(item["clean_path"], dtype="float32")
        d_audio, _ = sf.read(item["degraded_path"], dtype="float32")
        
        # DeepFilterNet enhanced audio
        df_path = Path(f"data/simulated/test/enhanced_deepfilternet/{tid}_df_enh.wav")
        df_raw, _ = sf.read(str(df_path), dtype="float32")
        
        # Branch A enhanced audio
        a_raw = enhance_audio_waveform(model, d_audio, sample_rate=16000, device=device)
        
        # Apply the EXACT SAME ReliabilityGuard to both pipelines
        df_guarded, df_safe, df_trigs = guard.apply_guard(d_audio, df_raw)
        a_guarded, a_safe, a_trigs = guard.apply_guard(d_audio, a_raw)
        
        # Align lengths
        min_len = min(len(c_audio), len(d_audio), len(df_raw), len(df_guarded), len(a_raw), len(a_guarded))
        c_eval = c_audio[:min_len]
        d_eval = d_audio[:min_len]
        df_raw_eval = df_raw[:min_len]
        df_g_eval = df_guarded[:min_len]
        a_raw_eval = a_raw[:min_len]
        a_g_eval = a_guarded[:min_len]

        # Transcriptions
        hyp_deg = whisper_eval.transcribe(d_eval, sr)
        hyp_df_raw = whisper_eval.transcribe(df_raw_eval, sr)
        hyp_df_g = whisper_eval.transcribe(df_g_eval, sr)
        hyp_a_raw = whisper_eval.transcribe(a_raw_eval, sr)
        hyp_a_g = whisper_eval.transcribe(a_g_eval, sr)

        # WER
        wer_deg = compute_wer(ref_text, hyp_deg)
        wer_df_raw = compute_wer(ref_text, hyp_df_raw)
        wer_df_g = compute_wer(ref_text, hyp_df_g)
        wer_a_raw = compute_wer(ref_text, hyp_a_raw)
        wer_a_g = compute_wer(ref_text, hyp_a_g)

        # PESQ
        pesq_deg = compute_pesq(c_eval, d_eval, sr)
        pesq_df_raw = compute_pesq(c_eval, df_raw_eval, sr)
        pesq_df_g = compute_pesq(c_eval, df_g_eval, sr)
        pesq_a_raw = compute_pesq(c_eval, a_raw_eval, sr)
        pesq_a_g = compute_pesq(c_eval, a_g_eval, sr)

        # STOI
        stoi_deg = compute_stoi(c_eval, d_eval, sr)
        stoi_df_raw = compute_stoi(c_eval, df_raw_eval, sr)
        stoi_df_g = compute_stoi(c_eval, df_g_eval, sr)
        stoi_a_raw = compute_stoi(c_eval, a_raw_eval, sr)
        stoi_a_g = compute_stoi(c_eval, a_g_eval, sr)

        record = {
            "id": tid,
            "snr_db": item["snr_db"],
            "transcript": ref_text,
            "df_guard_triggered": not df_safe,
            "df_triggers": df_trigs,
            "branch_a_guard_triggered": not a_safe,
            "branch_a_triggers": a_trigs,
            "both_genuine_untriggered": (df_safe and a_safe),
            
            # Degraded
            "pesq_degraded": round(pesq_deg, 3),
            "stoi_degraded": round(stoi_deg, 3),
            "wer_degraded": round(wer_deg, 3),
            "hyp_degraded": hyp_deg,
            
            # DeepFilterNet Guarded
            "pesq_df_guarded": round(pesq_df_g, 3),
            "stoi_df_guarded": round(stoi_df_g, 3),
            "wer_df_guarded": round(wer_df_g, 3),
            "hyp_df_guarded": hyp_df_g,

            # DeepFilterNet Raw
            "pesq_df_raw": round(pesq_df_raw, 3),
            "stoi_df_raw": round(stoi_df_raw, 3),
            "wer_df_raw": round(wer_df_raw, 3),

            # Branch A Guarded
            "pesq_branch_a_guarded": round(pesq_a_g, 3),
            "stoi_branch_a_guarded": round(stoi_a_g, 3),
            "wer_branch_a_guarded": round(wer_a_g, 3),
            "hyp_branch_a_guarded": hyp_a_g,

            # Branch A Raw
            "pesq_branch_a_raw": round(pesq_a_raw, 3),
            "stoi_branch_a_raw": round(stoi_a_raw, 3),
            "wer_branch_a_raw": round(wer_a_raw, 3),
        }
        results.append(record)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
            print(f"  Processed [{idx+1:02d}/{len(manifest):02d}] (DF Guarded: {sum(1 for r in results if r['df_guard_triggered'])}, Branch A Guarded: {sum(1 for r in results if r['branch_a_guard_triggered'])})")

    # Metrics computation function
    def compute_stats(recs):
        return {
            "count": len(recs),
            "pesq": {
                "degraded": round(float(np.mean([r["pesq_degraded"] for r in recs])), 3),
                "df_guarded": round(float(np.mean([r["pesq_df_guarded"] for r in recs])), 3),
                "branch_a_guarded": round(float(np.mean([r["pesq_branch_a_guarded"] for r in recs])), 3),
                "df_raw": round(float(np.mean([r["pesq_df_raw"] for r in recs])), 3),
                "branch_a_raw": round(float(np.mean([r["pesq_branch_a_raw"] for r in recs])), 3),
            },
            "stoi": {
                "degraded": round(float(np.mean([r["stoi_degraded"] for r in recs])), 3),
                "df_guarded": round(float(np.mean([r["stoi_df_guarded"] for r in recs])), 3),
                "branch_a_guarded": round(float(np.mean([r["stoi_branch_a_guarded"] for r in recs])), 3),
                "df_raw": round(float(np.mean([r["stoi_df_raw"] for r in recs])), 3),
                "branch_a_raw": round(float(np.mean([r["stoi_branch_a_raw"] for r in recs])), 3),
            },
            "wer": {
                "degraded_mean": round(float(np.mean([r["wer_degraded"] for r in recs])), 3),
                "degraded_median": round(float(np.median([r["wer_degraded"] for r in recs])), 3),
                "degraded_clipped_mean": round(float(np.mean(np.clip([r["wer_degraded"] for r in recs], 0, 1))), 3),
                "degraded_outliers": sum(1 for r in recs if r["wer_degraded"] > 1.0),

                "df_guarded_mean": round(float(np.mean([r["wer_df_guarded"] for r in recs])), 3),
                "df_guarded_median": round(float(np.median([r["wer_df_guarded"] for r in recs])), 3),
                "df_guarded_clipped_mean": round(float(np.mean(np.clip([r["wer_df_guarded"] for r in recs], 0, 1))), 3),
                "df_guarded_outliers": sum(1 for r in recs if r["wer_df_guarded"] > 1.0),

                "branch_a_guarded_mean": round(float(np.mean([r["wer_branch_a_guarded"] for r in recs])), 3),
                "branch_a_guarded_median": round(float(np.median([r["wer_branch_a_guarded"] for r in recs])), 3),
                "branch_a_guarded_clipped_mean": round(float(np.mean(np.clip([r["wer_branch_a_guarded"] for r in recs], 0, 1))), 3),
                "branch_a_guarded_outliers": sum(1 for r in recs if r["wer_branch_a_guarded"] > 1.0),

                "df_raw_mean": round(float(np.mean([r["wer_df_raw"] for r in recs])), 3),
                "branch_a_raw_mean": round(float(np.mean([r["wer_branch_a_raw"] for r in recs])), 3),
            }
        }

    full_stats = compute_stats(results)
    genuine_subset = [r for r in results if r["both_genuine_untriggered"]]
    genuine_stats = compute_stats(genuine_subset) if len(genuine_subset) > 0 else {}

    payload = {
        "full_benchmark_50": full_stats,
        "genuine_untriggered_subset": genuine_stats,
        "per_utterance_results": results,
    }

    with open("results/fair_comparison_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 96)
    print(" TABLE 1: ALL 50 TEST UTTERANCES (EACH PIPELINE UNDER SAME RELIABILITY GUARD)")
    print("=" * 96)
    print(f"{'Metric':<26} | {'Degraded Input':<16} | {'Guarded DeepFilterNet3':<24} | {'Guarded Branch A':<20}")
    print("-" * 96)
    print(f"{'PESQ (-0.5 to 4.5)':<26} | {full_stats['pesq']['degraded']:<16.3f} | {full_stats['pesq']['df_guarded']:<24.3f} | {full_stats['pesq']['branch_a_guarded']:<20.3f}")
    print(f"{'STOI (0.0 to 1.0)':<26} | {full_stats['stoi']['degraded']:<16.3f} | {full_stats['stoi']['df_guarded']:<24.3f} | {full_stats['stoi']['branch_a_guarded']:<20.3f}")
    print(f"{'Whisper Median WER':<26} | {full_stats['wer']['degraded_median']:<16.3f} | {full_stats['wer']['df_guarded_median']:<24.3f} | {full_stats['wer']['branch_a_guarded_median']:<20.3f}")
    print(f"{'Whisper Clipped Mean WER':<26} | {full_stats['wer']['degraded_clipped_mean']:<16.3f} | {full_stats['wer']['df_guarded_clipped_mean']:<24.3f} | {full_stats['wer']['branch_a_guarded_clipped_mean']:<20.3f}")
    print(f"{'Whisper Raw Mean WER':<26} | {full_stats['wer']['degraded_mean']:<16.3f} | {full_stats['wer']['df_guarded_mean']:<24.3f} | {full_stats['wer']['branch_a_guarded_mean']:<20.3f}")
    print(f"{'Outlier Count (WER > 1.0)':<26} | {full_stats['wer']['degraded_outliers']:<16} | {full_stats['wer']['df_guarded_outliers']:<24} | {full_stats['wer']['branch_a_guarded_outliers']:<20}")
    print(f"{'Guard Trigger Count':<26} | {'0 / 50':<16} | {sum(1 for r in results if r['df_guard_triggered']):<24} | {sum(1 for r in results if r['branch_a_guard_triggered']):<20}")
    print("=" * 96)

    if genuine_stats:
        print("\n" + "=" * 96)
        print(f" TABLE 2: GENUINE SAMPLES ONLY ({genuine_stats['count']} / 50 SAMPLES WHERE NEITHER GUARD TRIGGERED)")
        print("=" * 96)
        print(f"{'Metric':<26} | {'Degraded Input':<16} | {'DeepFilterNet3 (Raw)':<24} | {'Branch A (Raw)':<20}")
        print("-" * 96)
        print(f"{'PESQ (-0.5 to 4.5)':<26} | {genuine_stats['pesq']['degraded']:<16.3f} | {genuine_stats['pesq']['df_raw']:<24.3f} | {genuine_stats['pesq']['branch_a_raw']:<20.3f}")
        print(f"{'STOI (0.0 to 1.0)':<26} | {genuine_stats['stoi']['degraded']:<16.3f} | {genuine_stats['stoi']['df_raw']:<24.3f} | {genuine_stats['stoi']['branch_a_raw']:<20.3f}")
        print(f"{'Whisper Median WER':<26} | {genuine_stats['wer']['degraded_median']:<16.3f} | {genuine_stats['wer']['df_guarded_median']:<24.3f} | {genuine_stats['wer']['branch_a_guarded_median']:<20.3f}")
        print(f"{'Whisper Clipped Mean WER':<26} | {genuine_stats['wer']['degraded_clipped_mean']:<16.3f} | {genuine_stats['wer']['df_guarded_clipped_mean']:<24.3f} | {genuine_stats['wer']['branch_a_guarded_clipped_mean']:<20.3f}")
        print(f"{'Whisper Raw Mean WER':<26} | {genuine_stats['wer']['degraded_mean']:<16.3f} | {genuine_stats['wer']['df_raw_mean']:<24.3f} | {genuine_stats['wer']['branch_a_raw_mean']:<20.3f}")
        print(f"{'Outlier Count (WER > 1.0)':<26} | {genuine_stats['wer']['degraded_outliers']:<16} | {genuine_stats['wer']['df_guarded_outliers']:<24} | {genuine_stats['wer']['branch_a_guarded_outliers']:<20}")
        print("=" * 96)

if __name__ == "__main__":
    run_fair_benchmark()
