"""
Generate 5 representative demo audio pairs for the interactive Gradio UI.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.branch_a_denoiser import BranchADenoiser
from eval.reliability_guard import ReliabilityGuard
from eval.spectral_postfilter import SpectralPostFilter
from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator


def main():
    samples_dir = Path("demo/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = BranchADenoiser(freq_bins=257, channels=32, hidden_dim=64, num_layers=2)
    ckpt = torch.load("checkpoints/branch_a_curriculum_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    opt_file = Path("results/optimal_guard_params.json")
    if opt_file.exists():
        with open(opt_file, "r") as f:
            opt = json.load(f)
        guard_params = opt.get("guard_params", {})
        pf_params = opt.get("postfilter_params", {})
    else:
        guard_params = {"min_energy_ratio": 0.20, "max_energy_ratio": 1.25, "min_env_correlation": 0.60, "min_spectral_flatness": 0.003}
        pf_params = {"oversubtraction": 1.2, "spectral_floor": 0.08, "smoothing_factor": 0.75}

    guard = ReliabilityGuard(**guard_params)
    postfilter = SpectralPostFilter(**pf_params)
    whisper_eval = WhisperEvaluator()

    # Load test manifest
    with open("data/simulated/test/manifest.json", "r", encoding="utf-8") as f:
        test_manifest = json.load(f)

    # Pick 5 diverse samples covering different SNR levels
    sorted_manifest = sorted(test_manifest, key=lambda x: x.get("snr_db", 0))
    indices = [0, len(sorted_manifest)//4, len(sorted_manifest)//2, 3*len(sorted_manifest)//4, len(sorted_manifest)-1]
    selected = [sorted_manifest[i] for i in indices]

    scenario_labels = [
        "Scenario 1: Extreme Low-SNR Combat Artillery (-5.0 dB)",
        "Scenario 2: Urban Breach & Gunfire Impulses (0.0 dB)",
        "Scenario 3: Tactical Vehicle Engine Rumble (5.0 dB)",
        "Scenario 4: Rotary-Wing Rotor Wash (10.0 dB)",
        "Scenario 5: Radio Channel Static & Codec2 Quantization (15.0 dB)",
    ]

    manifest_output = []

    print("[*] Generating 5 curated demo audio triplets...")
    for idx, (item, label) in enumerate(zip(selected, scenario_labels)):
        sid = item["id"]
        c, sr = sf.read(item["clean_path"], dtype="float32")
        d, _ = sf.read(item["degraded_path"], dtype="float32")
        if c.ndim > 1:
            c = c.mean(axis=1)
        if d.ndim > 1:
            d = d.mean(axis=1)

        # STFT
        orig_len = len(d)
        deg_tensor = torch.from_numpy(d).unsqueeze(0)
        window = torch.hann_window(512)
        with torch.no_grad():
            stft_deg = torch.stft(deg_tensor, n_fft=512, hop_length=128, win_length=512, window=window, return_complex=True)
            mag_deg = torch.abs(stft_deg).unsqueeze(1)
            phase_deg = torch.angle(stft_deg)
            h0 = model.init_hidden_state(batch_size=1)
            mask, _, _ = model(mag_deg, h0)
            enh_mag = (mag_deg * mask).squeeze(1)
            enh_stft = torch.polar(enh_mag, phase_deg)
            enh_audio = torch.istft(enh_stft, n_fft=512, hop_length=128, win_length=512, window=window, length=orig_len).squeeze(0).numpy()

        guarded, is_safe, conf, triggers = guard.apply_guard(d, enh_audio, mode="soft")
        final = postfilter.process(guarded)

        clean_out = samples_dir / f"demo_sample_{idx+1:02d}_clean.wav"
        deg_out = samples_dir / f"demo_sample_{idx+1:02d}_deg.wav"
        enh_out = samples_dir / f"demo_sample_{idx+1:02d}_enh.wav"

        sf.write(str(clean_out), c, 16000)
        sf.write(str(deg_out), d, 16000)
        sf.write(str(enh_out), final, 16000)

        min_len = min(len(c), len(d), len(final))
        pesq_deg = compute_pesq(c[:min_len], d[:min_len], 16000)
        pesq_enh = compute_pesq(c[:min_len], final[:min_len], 16000)
        stoi_deg = compute_stoi(c[:min_len], d[:min_len], 16000)
        stoi_enh = compute_stoi(c[:min_len], final[:min_len], 16000)
        hyp_deg = whisper_eval.transcribe(d[:min_len], 16000)
        hyp_enh = whisper_eval.transcribe(final[:min_len], 16000)
        wer_deg = compute_wer(item["transcript"], hyp_deg)
        wer_enh = compute_wer(item["transcript"], hyp_enh)

        entry = {
            "index": idx + 1,
            "label": label,
            "original_id": sid,
            "snr_db": item.get("snr_db", 0.0),
            "transcript": item["transcript"],
            "clean_file": str(clean_out),
            "degraded_file": str(deg_out),
            "enhanced_file": str(enh_out),
            "pesq_degraded": float(pesq_deg),
            "pesq_enhanced": float(pesq_enh),
            "stoi_degraded": float(stoi_deg),
            "stoi_enhanced": float(stoi_enh),
            "wer_degraded": float(wer_deg),
            "wer_enhanced": float(wer_enh),
            "hyp_degraded": hyp_deg,
            "hyp_enhanced": hyp_enh,
            "guard_confidence": float(conf),
        }
        manifest_output.append(entry)
        print(f"  [+] Sample {idx+1}: {label} | PESQ {pesq_deg:.2f}->{pesq_enh:.2f}, STOI {stoi_deg:.2f}->{stoi_enh:.2f}, WER {wer_deg:.2f}->{wer_enh:.2f}")

    with open(samples_dir / "samples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_output, f, indent=2)
    print(f"[+] Demo samples generated in {samples_dir}")


if __name__ == "__main__":
    main()
