import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fused_model import FusedBattlefieldModel
from eval.metrics import compute_wer, WhisperEvaluator
from eval.reliability_guard import ReliabilityGuard

def diagnose():
    print("=" * 80)
    print("TASK 2.1: PER-UTTERANCE FUSED RESULTS (fused WER > degraded WER, worst first)")
    print("=" * 80)
    
    with open("results/fused_benchmark_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = data["records"]
    manifest = {item["id"]: item for item in json.load(open("data/simulated/test/manifest.json", "r"))}
    
    worse_records = []
    for r in records:
        diff = r["wer_fused"] - r["wer_deg"]
        if diff > 0.001:
            worse_records.append({**r, "wer_diff": round(diff, 3), "transcript": manifest[r["id"]]["transcript"]})
            
    worse_records.sort(key=lambda x: x["wer_diff"], reverse=True)
    
    print(f"Total test utterances: {len(records)}")
    print(f"Utterances where fused WER > degraded WER: {len(worse_records)} / {len(records)} ({len(worse_records)/len(records)*100:.1f}%)")
    print(f"Utterances where fused WER == degraded WER: {sum(1 for r in records if abs(r['wer_fused'] - r['wer_deg']) <= 0.001)} / {len(records)}")
    print(f"Utterances where fused WER < degraded WER: {sum(1 for r in records if r['wer_fused'] < r['wer_deg'] - 0.001)} / {len(records)}")
    print("\nAll regressed samples sorted by WER delta (worst first):")
    print(f"{'ID':<12} | {'SNR (dB)':<10} | {'Deg WER':<10} | {'Fused WER':<10} | {'Delta':<10} | {'Guard Conf':<12}")
    print("-" * 75)
    for r in worse_records:
        print(f"{r['id']:<12} | {r['snr_db']:<10.2f} | {r['wer_deg']:<10.3f} | {r['wer_fused']:<10.3f} | {r['wer_diff']:+<10.3f} | {r['guard_confidence']:<12.3f}")

    print("\n" + "=" * 80)
    print("TOP 5 WORST SAMPLES: REFERENCE vs DEGRADED vs FUSED HYPOTHESIS")
    print("=" * 80)
    
    # We will run Whisper on top 5 worst to get exact hypothesis strings
    device = torch.device("cpu")
    model = FusedBattlefieldModel(
        freq_bins=257, denoiser_channels=32, denoiser_hidden_dim=64, denoiser_layers=2,
        impulse_channels=24, impulse_hidden_dim=48, context_dim=32,
    ).to(device)
    model.load_pretrained_branches(
        branch_a_path="checkpoints/branch_a_curriculum_best.pt",
        branch_b_path="checkpoints/branch_b_best.pt",
        device=device,
    )
    model.eval()
    guard = ReliabilityGuard()
    whisper_eval = WhisperEvaluator()
    n_fft = 512
    hop_length = 128
    w = torch.hann_window(n_fft, device=device)

    top5 = worse_records[:5]
    for idx, r in enumerate(top5):
        m_item = manifest[r["id"]]
        c_audio, sr = sf.read(m_item["clean_path"], dtype="float32")
        d_audio, _ = sf.read(m_item["degraded_path"], dtype="float32")
        if c_audio.ndim > 1: c_audio = c_audio.mean(axis=1)
        if d_audio.ndim > 1: d_audio = d_audio.mean(axis=1)
        
        t_d = torch.from_numpy(d_audio).unsqueeze(0).to(device)
        with torch.no_grad():
            stft = torch.stft(t_d, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=w, return_complex=True)
            mag = torch.abs(stft).unsqueeze(1)
            phase = torch.angle(stft)
            h_a, state_b = model.init_hidden_states(batch_size=1, device=device)
            enh_mag, fused_mask, _, _, est_snr = model(mag, h_a, state_b)
            enh_mag = enh_mag.squeeze(1)
            enh_stft = torch.polar(enh_mag, phase)
            raw_fused_audio = torch.istft(enh_stft, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=w, length=len(d_audio)).squeeze(0).cpu().numpy()

        guarded_fused_audio, is_safe, conf, trigs = guard.apply_guard(d_audio, raw_fused_audio, mode="soft")
        
        hyp_deg = whisper_eval.transcribe(d_audio, sr)
        hyp_fused = whisper_eval.transcribe(guarded_fused_audio, sr)
        
        print(f"\n--- [#{idx+1}] Sample ID: {r['id']} (SNR: {r['snr_db']:.2f} dB, Guard Conf: {conf:.3f}) ---")
        print(f"  Reference:    \"{m_item['transcript']}\"")
        print(f"  Degraded Hyp: \"{hyp_deg}\"  (WER: {r['wer_deg']:.3f})")
        print(f"  Fused Hyp:    \"{hyp_fused}\"  (WER: {r['wer_fused']:.3f})")

    print("\n" + "=" * 80)
    print("TASK 2.2: GATED FUSION LAYER OUTPUT VALUES & MASK STATISTICS")
    print("=" * 80)
    
    # Inspect internal masks
    for idx, r in enumerate(top5[:3]):
        m_item = manifest[r["id"]]
        d_audio, _ = sf.read(m_item["degraded_path"], dtype="float32")
        if d_audio.ndim > 1: d_audio = d_audio.mean(axis=1)
        t_d = torch.from_numpy(d_audio).unsqueeze(0).to(device)
        with torch.no_grad():
            stft = torch.stft(t_d, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=w, return_complex=True)
            mag = torch.abs(stft).unsqueeze(1)
            h_a, state_b = model.init_hidden_states(batch_size=1, device=device)
            mask_a, _, _ = model.branch_a(mag, h_a)
            mask_b, prob_b, _ = model.branch_b(mag, state_b)
            enh_mag, fused_mask, _, _, est_snr = model(mag, h_a, state_b)
            
        print(f"\nSample {r['id']} Mask Statistics:")
        print(f"  Branch A Mask (Continuous): mean={mask_a.mean():.4f}, std={mask_a.std():.4f}, min={mask_a.min():.4f}, max={mask_a.max():.4f}")
        print(f"  Branch B Mask (Impulse):    mean={mask_b.mean():.4f}, std={mask_b.std():.4f}, min={mask_b.min():.4f}, max={mask_b.max():.4f}")
        print(f"  Branch B Prob (Transient):  mean={prob_b.mean():.4f}, std={prob_b.std():.4f}, min={prob_b.min():.4f}, max={prob_b.max():.4f}")
        print(f"  Fused Mask (Output):        mean={fused_mask.mean():.4f}, std={fused_mask.std():.4f}, min={fused_mask.min():.4f}, max={fused_mask.max():.4f}")

    print("\n" + "=" * 80)
    print("TASK 2.3: RELIABILITY GUARD METRICS ON FUSED VS BRANCH A VS DEGRADED")
    print("=" * 80)
    
    # Test guard feature extraction
    sample_item = manifest["test_0000"]
    d_audio, sr = sf.read(sample_item["degraded_path"], dtype="float32")
    if d_audio.ndim > 1: d_audio = d_audio.mean(axis=1)
    
    feats_deg = guard.compute_features(d_audio, d_audio)
    print("Guard features (Degraded vs Degraded - identity check):", feats_deg)
    
    t_d = torch.from_numpy(d_audio).unsqueeze(0).to(device)
    with torch.no_grad():
        stft = torch.stft(t_d, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=w, return_complex=True)
        mag = torch.abs(stft).unsqueeze(1)
        phase = torch.angle(stft)
        
        # Branch A standalone
        h_a = torch.zeros(2, 1, 64, device=device)
        mask_a, _, _ = model.branch_a(mag, h_a)
        a_mag = (mag * mask_a).squeeze(1)
        a_stft = torch.polar(a_mag, phase)
        a_audio = torch.istft(a_stft, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=w, length=len(d_audio)).squeeze(0).cpu().numpy()
        
        # Fused model
        h_a, state_b = model.init_hidden_states(batch_size=1, device=device)
        enh_mag, fused_mask, _, _, _ = model(mag, h_a, state_b)
        f_stft = torch.polar(enh_mag.squeeze(1), phase)
        raw_f_audio = torch.istft(f_stft, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=w, length=len(d_audio)).squeeze(0).cpu().numpy()
        
    feats_a = guard.compute_features(d_audio, a_audio)
    feats_f = guard.compute_features(d_audio, raw_f_audio)
    
    print("\nGuard features on Branch A Output:")
    for k, v in feats_a.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nGuard features on Raw Fused Output:")
    for k, v in feats_f.items():
        print(f"  {k}: {v:.4f}")

    is_safe_a, trig_a, _ = guard.check(d_audio, a_audio)
    conf_a = guard.compute_confidence(feats_a)
    is_safe_f, trig_f, _ = guard.check(d_audio, raw_f_audio)
    conf_f = guard.compute_confidence(feats_f)
    print(f"\nBranch A Guard Evaluation: is_safe={is_safe_a}, conf={conf_a:.4f}, triggers={trig_a}")
    print(f"Fused Model Guard Evaluation: is_safe={is_safe_f}, conf={conf_f:.4f}, triggers={trig_f}")

if __name__ == "__main__":
    diagnose()
