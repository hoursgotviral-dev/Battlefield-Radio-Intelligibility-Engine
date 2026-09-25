import soundfile as sf
import numpy as np
import torch
import json
from pathlib import Path

def get_stats(wav, sr=16000):
    t = torch.from_numpy(wav).float()
    w = torch.hann_window(512)
    stft = torch.stft(t, n_fft=512, hop_length=128, win_length=512, window=w, return_complex=True)
    mag = torch.abs(stft) + 1e-8
    pwr = mag**2
    geom_mean = torch.exp(torch.mean(torch.log(pwr), dim=0))
    arith_mean = torch.mean(pwr, dim=0)
    flatness = (geom_mean / (arith_mean + 1e-8)).mean().item()
    crest = (torch.max(mag, dim=0)[0] / (torch.mean(mag, dim=0) + 1e-8)).mean().item()
    energy = torch.sum(t**2).item()
    return {
        'flatness': flatness,
        'crest': crest,
        'energy': energy,
    }

manifest = json.load(open('data/simulated/test/manifest.json'))
for item in manifest[:10]:
    tid = item['id']
    d, _ = sf.read(item['degraded_path'])
    enh_p = Path(f'data/simulated/test/enhanced_branch_a/{tid}_branch_a.wav')
    if not enh_p.exists():
        continue
    enh, _ = sf.read(str(enh_p))
    st_d = get_stats(d)
    st_enh = get_stats(enh)
    e_ratio = st_enh['energy'] / (st_d['energy'] + 1e-8)
    f_ratio = st_enh['flatness'] / (st_d['flatness'] + 1e-8)
    print(f"[{tid}] SNR: {item['snr_db']:>5.2f} dB | Energy Ratio: {e_ratio:.4f} | Flatness Ratio: {f_ratio:.4f} | Flatness: d={st_d['flatness']:.4f}, enh={st_enh['flatness']:.4f}")
