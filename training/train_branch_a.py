"""
Day 3 & Day 5 Enhancement: Branch A Retraining with Low-SNR Curriculum + Energy Floor Loss + ASR Loss.

ML Interventions:
1. Low-SNR Curriculum Oversampling: Samples in the hard [-5dB, +2dB] range are weighted 2.5x,
   ensuring they represent ~45-50% of each training epoch to prevent low-SNR mask collapse.
2. Anti-Collapse Energy Floor Loss: Penalizes output energy dropping below 28% of degraded input,
   guiding the model toward safe conservative attenuation instead of aggressive zeroing.
3. Differentiable Whisper Perceptual Loss: Preserves phonetic intelligibility.
4. Identical Model Capacity: Kept at 2-layer ConvGRU (channels=32, hidden_dim=64).
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.branch_a_denoiser import BranchADenoiser
from training.losses import MultiResolutionSTFTLoss, SISDRLoss, WhisperPerceptualLoss, EnergyFloorLoss
from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator
from eval.reliability_guard import ReliabilityGuard


class TacticalWavDataset(Dataset):
    """Loads pre-generated clean and degraded audio pairs from a split directory."""
    def __init__(self, split_dir: str, segment_length_samples: int = 32000):
        super().__init__()
        self.split_dir = Path(split_dir)
        self.segment_len = segment_length_samples
        manifest_path = self.split_dir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    def get_snrs(self) -> List[float]:
        return [item["snr_db"] for item in self.manifest]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.manifest[idx]
        clean_audio, sr_c = sf.read(item["clean_path"], dtype="float32")
        deg_audio, sr_d = sf.read(item["degraded_path"], dtype="float32")

        if clean_audio.ndim > 1:
            clean_audio = clean_audio.mean(axis=1)
        if deg_audio.ndim > 1:
            deg_audio = deg_audio.mean(axis=1)

        # Pad or slice to segment_len for batch training
        if len(clean_audio) < self.segment_len:
            clean_audio = np.pad(clean_audio, (0, self.segment_len - len(clean_audio)))
            deg_audio = np.pad(deg_audio, (0, self.segment_len - len(deg_audio)))
        else:
            clean_audio = clean_audio[: self.segment_len]
            deg_audio = deg_audio[: self.segment_len]

        return {
            "id": item["id"],
            "clean": torch.from_numpy(clean_audio).unsqueeze(0),       # [1, T]
            "degraded": torch.from_numpy(deg_audio).unsqueeze(0),     # [1, T]
            "transcript": item["transcript"],
            "snr_db": item["snr_db"],
        }


def enhance_audio_waveform(
    model: BranchADenoiser,
    degraded_audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Enhances a single full audio waveform using BranchADenoiser with STFT/iSTFT."""
    model.eval()
    orig_len = len(degraded_audio)
    deg_tensor = torch.from_numpy(degraded_audio).unsqueeze(0).to(device)  # [1, T]

    window = torch.hann_window(n_fft, device=device)
    with torch.no_grad():
        stft_deg = torch.stft(
            deg_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mag_deg = torch.abs(stft_deg).unsqueeze(1)  # [1, 1, 257, T_frames]
        phase_deg = torch.angle(stft_deg)

        h0 = model.init_hidden_state(batch_size=1, device=device)
        mask, _, _ = model(mag_deg, h0)
        enh_mag = (mag_deg * mask).squeeze(1)

        enh_stft = torch.polar(enh_mag, phase_deg)
        enh_audio = torch.istft(
            enh_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            length=orig_len,
        )

    return enh_audio.squeeze(0).cpu().numpy().astype(np.float32)


def train_branch_a_curriculum(
    train_dir: str = "data/simulated/train",
    val_dir: str = "data/simulated/val",
    checkpoints_dir: str = "checkpoints",
    epochs: int = 12,
    batch_size: int = 8,
    lr: float = 1e-3,
    whisper_weight: float = 0.35,
    energy_floor_weight: float = 2.0,
    device_name: str = "cpu",
) -> BranchADenoiser:
    """Trains Branch A with low-SNR curriculum oversampling, Energy Floor loss, and Whisper loss."""
    device = torch.device(device_name)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing BranchADenoiser model on {device}...")
    model = BranchADenoiser(
        freq_bins=257,
        channels=32,
        hidden_dim=64,
        num_layers=2,
    ).to(device)

    train_dataset = TacticalWavDataset(train_dir, segment_length_samples=32000)
    val_dataset = TacticalWavDataset(val_dir, segment_length_samples=32000)

    # Calculate curriculum sampling weights (2.5x for SNR <= 2.0 dB)
    snrs = train_dataset.get_snrs()
    sample_weights = [2.5 if s <= 2.0 else 1.0 for s in snrs]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    mrstft_loss = MultiResolutionSTFTLoss()
    sisdr_loss = SISDRLoss()
    l1_loss = nn.L1Loss()
    whisper_loss = WhisperPerceptualLoss(device=device)
    floor_loss = EnergyFloorLoss(min_ratio=0.28)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_fft = 512
    hop_length = 128
    window = torch.hann_window(n_fft, device=device)

    best_val_loss = float("inf")
    best_checkpoint = Path(checkpoints_dir) / "branch_a_curriculum_best.pt"

    print(f"[*] Starting Curriculum Fine-Tuning ({epochs} epochs, low-SNR target ~45%)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_floor = 0.0
        train_w = 0.0

        for batch in train_loader:
            clean = batch["clean"].to(device).squeeze(1)       # [B, T]
            degraded = batch["degraded"].to(device).squeeze(1) # [B, T]
            B, T_len = clean.shape

            optimizer.zero_grad()

            stft_deg = torch.stft(
                degraded,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            mag_deg = torch.abs(stft_deg).unsqueeze(1)  # [B, 1, 257, T_frames]
            phase_deg = torch.angle(stft_deg)

            stft_clean = torch.stft(
                clean,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            mag_clean = torch.abs(stft_clean)

            h0 = model.init_hidden_state(batch_size=B, device=device)
            mask, _, _ = model(mag_deg, h0)
            enh_mag = (mag_deg * mask).squeeze(1)  # [B, 257, T_frames]

            enh_stft = torch.polar(enh_mag, phase_deg)
            enh_audio = torch.istft(
                enh_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                length=T_len,
            )

            l_mr = mrstft_loss(enh_audio, clean)
            l_spec = l1_loss(enh_mag, mag_clean)
            l_si = sisdr_loss(enh_audio, clean)
            l_w = whisper_loss(enh_audio, clean)
            l_fl = floor_loss(enh_audio, degraded)

            loss = l_mr + 0.5 * l_spec + 0.05 * l_si + whisper_weight * l_w + energy_floor_weight * l_fl

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_floor += l_fl.item()
            train_w += l_w.item()

        scheduler.step()
        num_b = max(1, len(train_loader))
        train_loss /= num_b
        train_floor /= num_b
        train_w /= num_b

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device).squeeze(1)
                degraded = batch["degraded"].to(device).squeeze(1)
                B, T_len = clean.shape

                stft_deg = torch.stft(
                    degraded,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window,
                    return_complex=True,
                )
                mag_deg = torch.abs(stft_deg).unsqueeze(1)
                phase_deg = torch.angle(stft_deg)

                stft_clean = torch.stft(
                    clean,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window,
                    return_complex=True,
                )
                mag_clean = torch.abs(stft_clean)

                h0 = model.init_hidden_state(batch_size=B, device=device)
                mask, _, _ = model(mag_deg, h0)
                enh_mag = (mag_deg * mask).squeeze(1)

                enh_stft = torch.polar(enh_mag, phase_deg)
                enh_audio = torch.istft(
                    enh_stft,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window,
                    length=T_len,
                )

                l_mr = mrstft_loss(enh_audio, clean)
                l_spec = l1_loss(enh_mag, mag_clean)
                l_si = sisdr_loss(enh_audio, clean)
                l_w = whisper_loss(enh_audio, clean)
                l_fl = floor_loss(enh_audio, degraded)
                v_loss = l_mr + 0.5 * l_spec + 0.05 * l_si + whisper_weight * l_w + energy_floor_weight * l_fl
                val_loss += v_loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Train: {train_loss:.4f} (Floor: {train_floor:.4f}, W: {train_w:.4f}) | Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, best_checkpoint)

    elapsed = time.time() - start_time
    print(f"[+] Curriculum fine-tuning complete in {elapsed:.1f}s. Checkpoint: {best_checkpoint}")

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_full_guarded_benchmark(
    model: BranchADenoiser,
    test_dir: str = "data/simulated/test",
    results_dir: str = "results",
    sample_rate: int = 16000,
):
    """Runs fair apples-to-apples guarded benchmark across all 50 test utterances."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(test_dir) / "manifest.json"
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))

    guard = ReliabilityGuard()
    whisper_eval = WhisperEvaluator()

    results = []
    print(f"[*] Running fair guarded benchmark across {len(manifest)} test utterances...")

    for idx, item in enumerate(manifest):
        tid = item["id"]
        ref_text = item["transcript"]
        c_audio, sr = sf.read(item["clean_path"], dtype="float32")
        d_audio, _ = sf.read(item["degraded_path"], dtype="float32")

        # DeepFilterNet enhanced audio
        df_path = Path(f"data/simulated/test/enhanced_deepfilternet/{tid}_df_enh.wav")
        df_raw, _ = sf.read(str(df_path), dtype="float32")

        # Branch A enhanced audio
        a_raw = enhance_audio_waveform(model, d_audio, sample_rate=16000)

        # Apply SAME Reliability Guard
        df_guarded, df_safe, df_trigs = guard.apply_guard(d_audio, df_raw)
        a_guarded, a_safe, a_trigs = guard.apply_guard(d_audio, a_raw)

        # Align
        min_len = min(len(c_audio), len(d_audio), len(df_raw), len(df_guarded), len(a_raw), len(a_guarded))
        c_eval = c_audio[:min_len]
        d_eval = d_audio[:min_len]
        df_raw_eval = df_raw[:min_len]
        df_g_eval = df_guarded[:min_len]
        a_raw_eval = a_raw[:min_len]
        a_g_eval = a_guarded[:min_len]

        # Hypotheses
        hyp_deg = whisper_eval.transcribe(d_eval, sr)
        hyp_df_g = whisper_eval.transcribe(df_g_eval, sr)
        hyp_df_raw = whisper_eval.transcribe(df_raw_eval, sr)
        hyp_a_g = whisper_eval.transcribe(a_g_eval, sr)
        hyp_a_raw = whisper_eval.transcribe(a_raw_eval, sr)

        # Metrics
        wer_deg = compute_wer(ref_text, hyp_deg)
        wer_df_g = compute_wer(ref_text, hyp_df_g)
        wer_df_raw = compute_wer(ref_text, hyp_df_raw)
        wer_a_g = compute_wer(ref_text, hyp_a_g)
        wer_a_raw = compute_wer(ref_text, hyp_a_raw)

        pesq_deg = compute_pesq(c_eval, d_eval, sr)
        pesq_df_g = compute_pesq(c_eval, df_g_eval, sr)
        pesq_df_raw = compute_pesq(c_eval, df_raw_eval, sr)
        pesq_a_g = compute_pesq(c_eval, a_g_eval, sr)
        pesq_a_raw = compute_pesq(c_eval, a_raw_eval, sr)

        stoi_deg = compute_stoi(c_eval, d_eval, sr)
        stoi_df_g = compute_stoi(c_eval, df_g_eval, sr)
        stoi_df_raw = compute_stoi(c_eval, df_raw_eval, sr)
        stoi_a_g = compute_stoi(c_eval, a_g_eval, sr)
        stoi_a_raw = compute_stoi(c_eval, a_raw_eval, sr)

        results.append({
            "id": tid,
            "snr_db": item["snr_db"],
            "transcript": ref_text,
            "df_guard_triggered": not df_safe,
            "df_triggers": df_trigs,
            "branch_a_guard_triggered": not a_safe,
            "branch_a_triggers": a_trigs,
            "both_genuine_untriggered": (df_safe and a_safe),
            "pesq_degraded": round(pesq_deg, 3),
            "stoi_degraded": round(stoi_deg, 3),
            "wer_degraded": round(wer_deg, 3),
            "pesq_df_guarded": round(pesq_df_g, 3),
            "stoi_df_guarded": round(stoi_df_g, 3),
            "wer_df_guarded": round(wer_df_g, 3),
            "pesq_df_raw": round(pesq_df_raw, 3),
            "stoi_df_raw": round(stoi_df_raw, 3),
            "wer_df_raw": round(wer_df_raw, 3),
            "pesq_branch_a_guarded": round(pesq_a_g, 3),
            "stoi_branch_a_guarded": round(stoi_a_g, 3),
            "wer_branch_a_guarded": round(wer_a_g, 3),
            "pesq_branch_a_raw": round(pesq_a_raw, 3),
            "stoi_branch_a_raw": round(stoi_a_raw, 3),
            "wer_branch_a_raw": round(wer_a_raw, 3),
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
            print(f"    Evaluated [{idx+1:02d}/{len(manifest):02d}] (Branch A triggers: {sum(1 for r in results if r['branch_a_guard_triggered'])}, DF triggers: {sum(1 for r in results if r['df_guard_triggered'])})")

    def calc_stats(recs):
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

    full_stats = calc_stats(results)
    genuine_subset = [r for r in results if r["both_genuine_untriggered"]]
    genuine_stats = calc_stats(genuine_subset) if len(genuine_subset) > 0 else {}

    payload = {
        "full_benchmark_50": full_stats,
        "genuine_untriggered_subset": genuine_stats,
        "per_utterance_results": results,
    }

    with open(f"{results_dir}/day3_curriculum_benchmark.json", "w", encoding="utf-8") as f:
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
        print("=" * 96)


def main():
    print("=" * 88)
    print(" BRANCH A RETRAINING: LOW-SNR CURRICULUM + ENERGY FLOOR + WHISPER LOSS")
    print("=" * 88)

    model = train_branch_a_curriculum(
        train_dir="data/simulated/train",
        val_dir="data/simulated/val",
        checkpoints_dir="checkpoints",
        epochs=12,
        batch_size=8,
        lr=1e-3,
        whisper_weight=0.35,
        energy_floor_weight=2.0,
    )

    run_full_guarded_benchmark(
        model=model,
        test_dir="data/simulated/test",
        results_dir="results",
    )


if __name__ == "__main__":
    main()
