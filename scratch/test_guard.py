import soundfile as sf
import numpy as np
import torch
import json
from pathlib import Path

def compute_audio_guard_features(deg_wav: np.ndarray, enh_wav: np.ndarray, sr: int = 16000):
    t_deg = torch.from_numpy(deg_wav).float()
    t_enh = torch.from_numpy(enh_wav).float()
    
    # Align
    min_len = min(len(t_deg), len(t_enh))
    t_deg = t_deg[:min_len]
    t_enh = t_enh[:min_len]

    w = torch.hann_window(512)
    stft_deg = torch.stft(t_deg, n_fft=512, hop_length=128, win_length=512, window=w, return_complex=True)
    stft_enh = torch.stft(t_enh, n_fft=512, hop_length=128, win_length=512, window=w, return_complex=True)

    mag_deg = torch.abs(stft_deg) + 1e-8
    mag_enh = torch.abs(stft_enh) + 1e-8

    pwr_deg = mag_deg**2
    pwr_enh = mag_enh**2

    # Energy ratio
    e_deg = torch.sum(t_deg**2).item()
    e_enh = torch.sum(t_enh**2).item()
    energy_ratio = e_enh / (e_deg + 1e-8)

    # Spectral flatness
    sf_deg = (torch.exp(torch.mean(torch.log(pwr_deg), dim=0)) / (torch.mean(pwr_deg, dim=0) + 1e-8)).mean().item()
    sf_enh = (torch.exp(torch.mean(torch.log(pwr_enh), dim=0)) / (torch.mean(pwr_enh, dim=0) + 1e-8)).mean().item()
    flatness_ratio = sf_enh / (sf_deg + 1e-8)

    # Envelope correlation
    env_deg = torch.mean(mag_deg, dim=0)
    env_enh = torch.mean(mag_enh, dim=0)
    env_deg_norm = env_deg - env_deg.mean()
    env_enh_norm = env_enh - env_enh.mean()
    env_corr = (torch.sum(env_deg_norm * env_enh_norm) / (torch.norm(env_deg_norm) * torch.norm(env_enh_norm) + 1e-8)).item()

    return {
        "energy_ratio": energy_ratio,
        "sf_enh": sf_enh,
        "sf_deg": sf_deg,
        "flatness_ratio": flatness_ratio,
        "env_corr": env_corr,
    }

def evaluate_guard(deg_wav, enh_wav):
    feats = compute_audio_guard_features(deg_wav, enh_wav)
    # Guard rules:
    # 1. Energy collapse (enh energy < 25% of input)
    # 2. Extreme spectral peakiness (flatness < 0.004 or flatness ratio < 0.040)
    # 3. Envelope decorrelation (< 0.70)
    triggers = []
    if feats["energy_ratio"] < 0.25:
        triggers.append(f"Energy collapse (ratio={feats['energy_ratio']:.3f} < 0.25)")
    if feats["energy_ratio"] > 1.30:
        triggers.append(f"Energy explosion (ratio={feats['energy_ratio']:.3f} > 1.30)")
    if feats["sf_enh"] < 0.0040:
        triggers.append(f"Musical noise / extreme tonal peakiness (flatness={feats['sf_enh']:.4f} < 0.004)")
    if feats["flatness_ratio"] < 0.040:
        triggers.append(f"Severe spectral distortion (flatness_ratio={feats['flatness_ratio']:.3f} < 0.040)")
    if feats["env_corr"] < 0.70:
        triggers.append(f"Envelope decorrelation (corr={feats['env_corr']:.3f} < 0.70)")
        
    is_safe = len(triggers) == 0
    return is_safe, triggers, feats

manifest = json.load(open('data/simulated/test/manifest.json'))
d2 = json.load(open('results/day3_finetuned.json'))['per_utterance_results']
d2_map = {r['id']: r for r in d2}

print(f"{'ID':<10} | {'SNR':<7} | {'Safe?':<6} | {'WER_enh':<8} | {'Triggers'}")
print("-" * 75)
for item in manifest:
    tid = item['id']
    d_wav, _ = sf.read(item['degraded_path'])
    enh_wav, _ = sf.read(f'data/simulated/test/enhanced_branch_a/{tid}_branch_a.wav')
    safe, triggers, feats = evaluate_guard(d_wav, enh_wav)
    wer = d2_map[tid]['wer_branch_a']
    if not safe or wer > 1.0 or tid in ['test_0000', 'test_0001', 'test_0002', 'test_0003', 'test_0004']:
        trig_str = "; ".join(triggers) if triggers else "NONE (HEALTHY)"
        print(f"{tid:<10} | {item['snr_db']:>5.2f}dB | {str(safe):<6} | {wer:<8.3f} | {trig_str}")
