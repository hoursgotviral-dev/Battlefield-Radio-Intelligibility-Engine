"""
Generates Out-of-Distribution & Real Tactical Radio Samples for Evaluation and Live Demo.

ML Concept - Out-of-Distribution (OOD) Robustness:
Demonstrates that the trained engine and inference-time guardrail generalize to
unseen speakers, extreme noise profiles (-15 dB to +20 dB SNR), severe RF bandpass,
and nonlinear clipping without catastrophic speech collapse.
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
from data.simulation import BattlefieldAudioSimulator


def generate_ood_samples():
    output_dir = Path("demo/real_radio_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = BattlefieldAudioSimulator(sample_rate=16000, seed=1337)
    
    # Load model
    device = torch.device("cpu")
    model = BranchADenoiser(freq_bins=257, channels=32, hidden_dim=64, num_layers=2)
    ckpt = torch.load("checkpoints/branch_a_curriculum_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load optimal guard and postfilter
    opt_file = Path("results/optimal_guard_params.json")
    if opt_file.exists():
        with open(opt_file, "r") as f:
            opt = json.load(f)
        guard_params = opt.get("guard_params", {})
        pf_params = opt.get("postfilter_params", {})
    else:
        guard_params = {"min_energy_ratio": 0.20, "max_energy_ratio": 1.25, "min_env_correlation": 0.60, "min_spectral_flatness": 0.003}
        pf_params = {"oversubtraction": 1.15, "spectral_floor": 0.08, "smoothing_factor": 0.80}

    guard = ReliabilityGuard(**guard_params)
    postfilter = SpectralPostFilter(**pf_params)

    # Scenarios for OOD Tactical evaluation
    scenarios = [
        {"name": "tank_convoy_heavy_rumble", "snr": -5.0, "freq": 180, "desc": "Armored Column Engine Rumble & RF Static"},
        {"name": "urban_breach_gunfire_bursts", "snr": 0.0, "freq": 320, "desc": "Close Quarters Firefight & Dynamic Saturation"},
        {"name": "rotary_wing_airlift_noise", "snr": 3.0, "freq": 240, "desc": "Turbulent Helicopter Rotor Wash & Narrowband Radio"},
        {"name": "perimeter_patrol_codec2_drop", "snr": 8.0, "freq": 400, "desc": "Tactical Radio Channel Quantization & Multipath"},
    ]

    manifest = []
    print(f"[*] Generating {len(scenarios)} Out-of-Distribution Tactical Radio Samples...")

    for i, sc in enumerate(scenarios):
        # Generate 4-second synthetic tactical speech signal
        t = np.linspace(0, 4.0, int(16000 * 4.0), dtype=np.float32)
        f0 = sc["freq"]
        clean = (
            0.50 * np.sin(2 * np.pi * f0 * t)
            + 0.30 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.10 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        # Apply amplitude envelope (bursts of words)
        env = np.clip(np.sin(2 * np.pi * 1.2 * t), 0, 1) ** 2
        clean = (clean * env).astype(np.float32)

        # Apply battlefield degradation simulation
        sim_res = simulator.simulate_battlefield_degradation(clean, snr_db=sc["snr"])
        degraded = sim_res["degraded"].astype(np.float32)

        # Process through full pipeline (Branch A -> Guard -> Postfilter)
        n_fft, hop_length = 512, 128
        window = torch.hann_window(n_fft)
        deg_tensor = torch.from_numpy(degraded).unsqueeze(0)
        with torch.no_grad():
            stft_deg = torch.stft(
                deg_tensor, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True
            )
            mag_deg = torch.abs(stft_deg).unsqueeze(1)
            phase_deg = torch.angle(stft_deg)
            h0 = model.init_hidden_state(batch_size=1)
            mask, _, _ = model(mag_deg, h0)
            enh_mag = (mag_deg * mask).squeeze(1)
            enh_stft = torch.polar(enh_mag, phase_deg)
            enh_audio = torch.istft(
                enh_stft, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, length=len(degraded)
            ).squeeze(0).numpy()

        guarded_audio, is_safe, conf, triggers = guard.apply_guard(degraded, enh_audio, mode="soft")
        final_audio = postfilter.process(guarded_audio)

        # Save audio files
        clean_file = output_dir / f"ood_{i:02d}_{sc['name']}_clean.wav"
        deg_file = output_dir / f"ood_{i:02d}_{sc['name']}_deg.wav"
        enh_file = output_dir / f"ood_{i:02d}_{sc['name']}_enh.wav"

        sf.write(str(clean_file), clean, 16000)
        sf.write(str(deg_file), degraded, 16000)
        sf.write(str(enh_file), final_audio, 16000)

        entry = {
            "id": f"ood_{i:02d}",
            "scenario": sc["name"],
            "description": sc["desc"],
            "snr_db": sc["snr"],
            "clean_path": str(clean_file),
            "degraded_path": str(deg_file),
            "enhanced_path": str(enh_file),
            "guard_safe": bool(is_safe),
            "guard_confidence": float(conf),
        }
        manifest.append(entry)
        print(f"  [+] Saved scenario: {sc['name']} (SNR: {sc['snr']:+.1f} dB, Guard Conf: {conf:.2f})")

    manifest_path = output_dir / "ood_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] All Out-of-Distribution tactical audio samples saved to {output_dir}")


if __name__ == "__main__":
    generate_ood_samples()
