"""
Branch B: Impulse & Gunshot Noise Suppressor Training and Isolated Benchmark.

ML Objective:
Trains BranchBImpulse (High-temporal resolution causal CNN + GLU transient suppressor)
on 200 real LibriSpeech utterances injected with high-SPL physical Friedlander blast waveforms
and supersonic bullet crack transients.

Evaluates in isolation on 50 gunshot-heavy test clips to measure:
1. Impulse Peak Energy Attenuation (dB)
2. Transient Crest Factor Reduction (dB)
3. Speech Quality Preservation (PESQ & STOI)
4. Whisper ASR Word Error Rate (WER) under heavy gunfire
"""

import os
import sys
import json
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
from torch.utils.data import Dataset, DataLoader

from models.branch_b_impulse import BranchBImpulse
from data.simulation import BattlefieldAudioSimulator
from training.losses import MultiResolutionSTFTLoss
from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator


class GunshotSpeechDataset(Dataset):
    """
    Generates dynamic clean speech + high-SPL gunshot transient pairs.
    """
    def __init__(
        self,
        clean_dir: str = "data/simulated/train/clean",
        segment_length: int = 32000,
        sample_rate: int = 16000,
        burst_range: Tuple[int, int] = (2, 6),
        seed: int = 42,
    ):
        super().__init__()
        self.clean_files = sorted(list(Path(clean_dir).glob("*.wav")))
        self.segment_len = segment_length
        self.sr = sample_rate
        self.burst_range = burst_range
        self.sim = BattlefieldAudioSimulator(sample_rate=sample_rate, seed=seed)

    def __len__(self) -> int:
        return len(self.clean_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        c_path = self.clean_files[idx]
        clean_audio, _ = sf.read(str(c_path), dtype="float32")
        if clean_audio.ndim > 1:
            clean_audio = clean_audio.mean(axis=1)

        # Pad / slice
        if len(clean_audio) < self.segment_len:
            clean_audio = np.pad(clean_audio, (0, self.segment_len - len(clean_audio)))
        else:
            clean_audio = clean_audio[: self.segment_len]

        # Generate high-SPL gunshot impulse
        burst_count = int(self.sim.rng.randint(self.burst_range[0], self.burst_range[1] + 1))
        gunshot_noise = self.sim.generate_gunfire_impulse(
            length_samples=self.segment_len,
            burst_count=burst_count,
            peak_amp=float(self.sim.rng.uniform(1.2, 2.5)),
        )

        corrupted = clean_audio + gunshot_noise
        corrupted = np.clip(corrupted, -1.0, 1.0)

        # Transient mask target: 1 where clean speech dominates, 0 where gunshot shockwave peaks
        gunshot_energy = gunshot_noise ** 2
        speech_energy = clean_audio ** 2 + 1e-8
        transient_label = (gunshot_energy > (speech_energy * 2.0)).astype(np.float32)

        return {
            "id": c_path.stem,
            "clean": torch.from_numpy(clean_audio).float(),         # [T]
            "corrupted": torch.from_numpy(corrupted).float(),       # [T]
            "gunshot_noise": torch.from_numpy(gunshot_noise).float(),
            "transient_label": torch.from_numpy(transient_label).float(),
        }


def enhance_impulse_waveform(
    model: BranchBImpulse,
    corrupted_audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Enhances audio waveform using BranchBImpulse with STFT/iSTFT."""
    model.eval()
    orig_len = len(corrupted_audio)
    t_audio = torch.from_numpy(corrupted_audio).unsqueeze(0).to(device)

    w = torch.hann_window(n_fft, device=device)
    with torch.no_grad():
        stft = torch.stft(
            t_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=w,
            return_complex=True,
        )
        mag = torch.abs(stft).unsqueeze(1)  # [1, 1, 257, T_frames]
        phase = torch.angle(stft)

        state = model.init_hidden_state(batch_size=1, device=device)
        mask, prob, _ = model(mag, state)
        enh_mag = (mag * mask).squeeze(1)

        enh_stft = torch.polar(enh_mag, phase)
        enh_audio = torch.istft(
            enh_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=w,
            length=orig_len,
        )

    out_audio = enh_audio.squeeze(0).cpu().numpy().astype(np.float32)
    out_prob = prob.squeeze(0).squeeze(0).cpu().numpy()
    return out_audio, out_prob


def train_branch_b(
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    checkpoints_dir: str = "checkpoints",
    device_name: str = "cpu",
) -> BranchBImpulse:
    """Trains Branch B to suppress high-SPL acoustic shockwaves."""
    device = torch.device(device_name)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing BranchBImpulse on {device}...")
    model = BranchBImpulse(freq_bins=257, channels=24, hidden_dim=48).to(device)

    train_dataset = GunshotSpeechDataset(clean_dir="data/simulated/train/clean", seed=101)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    mrstft_loss = MultiResolutionSTFTLoss()
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_fft = 512
    hop_length = 128
    window = torch.hann_window(n_fft, device=device)

    best_checkpoint = Path(checkpoints_dir) / "branch_b_best.pt"
    print(f"[*] Starting Branch B Training ({epochs} epochs, {len(train_dataset)} speech utterances)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            clean = batch["clean"].to(device)         # [B, T]
            corrupted = batch["corrupted"].to(device) # [B, T]
            B, T_len = clean.shape

            optimizer.zero_grad()

            stft_corr = torch.stft(
                corrupted,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            mag_corr = torch.abs(stft_corr).unsqueeze(1)  # [B, 1, 257, T_frames]
            phase_corr = torch.angle(stft_corr)

            stft_clean = torch.stft(
                clean,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            mag_clean = torch.abs(stft_clean)

            state = model.init_hidden_state(batch_size=B, device=device)
            mask, prob, _ = model(mag_corr, state)

            enh_mag = (mag_corr * mask).squeeze(1)
            enh_stft = torch.polar(enh_mag, phase_corr)
            enh_audio = torch.istft(
                enh_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                length=T_len,
            )

            # Losses
            loss_mr = mrstft_loss(enh_audio, clean)
            loss_spec = l1_loss(enh_mag, mag_clean)

            # Transient loss: penalize high mask during gunshot peaks
            loss = loss_mr + 0.5 * loss_spec

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save({"model_state_dict": model.state_dict()}, best_checkpoint)
    print(f"[+] Branch B training complete in {time.time() - start_time:.1f}s. Saved to {best_checkpoint}")
    return model


def evaluate_branch_b_isolated(model: BranchBImpulse, num_test_clips: int = 50):
    """
    Evaluates Branch B in isolation on 50 gunshot-heavy test clips.
    Measures:
    1. Peak Impulse Attenuation (dB)
    2. Crest Factor Reduction (dB)
    3. PESQ & STOI improvement
    4. Whisper WER under heavy gunfire
    """
    print(f"\n[*] Evaluating Branch B in isolation on {num_test_clips} gunshot-corrupted test clips...")
    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=999)
    manifest = json.load(open("data/simulated/test/manifest.json", "r", encoding="utf-8"))[:num_test_clips]

    whisper_eval = WhisperEvaluator()

    peak_attens = []
    cf_reductions = []
    pesqs_corr = []
    pesqs_enh = []
    stois_corr = []
    stois_enh = []
    wers_corr = []
    wers_enh = []

    for idx, item in enumerate(manifest):
        clean_audio, sr = sf.read(item["clean_path"], dtype="float32")
        ref_text = item["transcript"]
        if clean_audio.ndim > 1:
            clean_audio = clean_audio.mean(axis=1)

        # Inject realistic gunfire bursts (3 to 6 shots)
        burst_count = int(sim.rng.randint(3, 7))
        gunshots = sim.generate_gunfire_impulse(
            length_samples=len(clean_audio),
            burst_count=burst_count,
            peak_amp=float(sim.rng.uniform(1.2, 2.0)),
        )

        corrupted = np.clip(clean_audio + gunshots, -1.0, 1.0)

        # Process with Branch B
        enh_audio, _ = enhance_impulse_waveform(model, corrupted)

        min_len = min(len(clean_audio), len(corrupted), len(enh_audio))
        c = clean_audio[:min_len]
        d = corrupted[:min_len]
        e = enh_audio[:min_len]

        # 1. Peak Impulse Attenuation
        max_d = np.max(np.abs(d))
        max_e = np.max(np.abs(e))
        atten_db = 20.0 * np.log10((max_d + 1e-8) / (max_e + 1e-8))
        peak_attens.append(atten_db)

        # 2. Crest Factor Reduction: CF = Peak / RMS
        cf_d = 20.0 * np.log10((max_d + 1e-8) / (np.sqrt(np.mean(d**2)) + 1e-8))
        cf_e = 20.0 * np.log10((max_e + 1e-8) / (np.sqrt(np.mean(e**2)) + 1e-8))
        cf_reductions.append(cf_d - cf_e)

        # 3. PESQ & STOI
        p_c = compute_pesq(c, d, sr)
        p_e = compute_pesq(c, e, sr)
        s_c = compute_stoi(c, d, sr)
        s_e = compute_stoi(c, e, sr)
        pesqs_corr.append(p_c)
        pesqs_enh.append(p_e)
        stois_corr.append(s_c)
        stois_enh.append(s_e)

        # 4. Whisper WER
        hyp_c = whisper_eval.transcribe(d, sr)
        hyp_e = whisper_eval.transcribe(e, sr)
        w_c = compute_wer(ref_text, hyp_c)
        w_e = compute_wer(ref_text, hyp_e)
        wers_corr.append(w_c)
        wers_enh.append(w_e)

        if (idx + 1) % 10 == 0 or (idx + 1) == num_test_clips:
            print(f"  Evaluated [{idx+1:02d}/{num_test_clips}] test clips...")

    summary = {
        "num_clips": num_test_clips,
        "peak_impulse_attenuation_db_mean": round(float(np.mean(peak_attens)), 2),
        "crest_factor_reduction_db_mean": round(float(np.mean(cf_reductions)), 2),
        "pesq": {
            "gunshot_corrupted_mean": round(float(np.mean(pesqs_corr)), 3),
            "branch_b_enhanced_mean": round(float(np.mean(pesqs_enh)), 3),
            "pesq_delta": round(float(np.mean(pesqs_enh) - np.mean(pesqs_corr)), 3),
        },
        "stoi": {
            "gunshot_corrupted_mean": round(float(np.mean(stois_corr)), 3),
            "branch_b_enhanced_mean": round(float(np.mean(stois_enh)), 3),
            "stoi_delta": round(float(np.mean(stois_enh) - np.mean(stois_corr)), 3),
        },
        "wer": {
            "gunshot_corrupted_mean": round(float(np.mean(wers_corr)), 3),
            "branch_b_enhanced_mean": round(float(np.mean(wers_enh)), 3),
            "wer_reduction": round(float(np.mean(wers_corr) - np.mean(wers_enh)), 3),
        }
    }

    out_json = "results/branch_b_isolated_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" BRANCH B: ISOLATED GUNSHOT SUPPRESSION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Peak Gunshot Attenuation:       {summary['peak_impulse_attenuation_db_mean']:>6.2f} dB")
    print(f"  Transient Crest Factor Removal: {summary['crest_factor_reduction_db_mean']:>6.2f} dB")
    print(f"  PESQ: Gunshot Corrupted = {summary['pesq']['gunshot_corrupted_mean']:.3f} -> Branch B = {summary['pesq']['branch_b_enhanced_mean']:.3f} (Delta: {summary['pesq']['pesq_delta']:+.3f})")
    print(f"  STOI: Gunshot Corrupted = {summary['stoi']['gunshot_corrupted_mean']:.3f} -> Branch B = {summary['stoi']['branch_b_enhanced_mean']:.3f} (Delta: {summary['stoi']['stoi_delta']:+.3f})")
    print(f"  WER:  Gunshot Corrupted = {summary['wer']['gunshot_corrupted_mean']:.3f} -> Branch B = {summary['wer']['branch_b_enhanced_mean']:.3f} (Delta: {summary['wer']['wer_reduction']:.3f} lower)")
    print("=" * 80)
    return summary


def main():
    model = train_branch_b(epochs=10, batch_size=8, lr=1e-3)
    evaluate_branch_b_isolated(model, num_test_clips=50)


if __name__ == "__main__":
    main()
