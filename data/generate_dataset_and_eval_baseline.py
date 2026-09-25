"""
Generate Battlefield Speech Dataset and Evaluate Pretrained DeepFilterNet Baseline.
"""

import os
import sys
import types
import json
import csv
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch

# Torchaudio backward-compatibility shim for DeepFilterNet
AudioMetaData = type("AudioMetaData", (), {})
_compat_mod = types.ModuleType("torchaudio.backend.common")
_compat_mod.AudioMetaData = AudioMetaData
sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
sys.modules["torchaudio.backend.common"] = _compat_mod

from df.enhance import enhance, init_df
from data.simulation import BattlefieldAudioSimulator
from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator

# NATO Tactical Phonetic vocabulary & military callsigns / standard sentences
MILITARY_PHRASES = [
    "alpha leader this is bravo actual radio check over",
    "bravo actual reading you lima charlie break coordinates zero four niner inbound",
    "mortar fire detected bearing two seven zero take cover immediately",
    "convoy moving along grid route tango november three holding perimeter",
    "close air support on station request immediate smoke marking on target",
    "charlie squad secure northern checkpoint awaiting further orders",
    "medevac requested grid square seven one eight four urgent litter",
    "visual contact hostile vehicle column moving south along riverline",
    "radio silence in effect maintain passive observation until dawn",
    "all stations switch to alternate frequency channel victor seven out",
    "artillery battery counter battery radar locked fire for effect",
    "perimeter defense report green across all sectors standing by",
    "heavy diesel armored column approaching checkpoint kilo four",
    "wind turbulence high helicopter extraction landing zone is hot",
    "recon team reports forward outpost abandoned clear to proceed",
    "tactical relay established signal strength five by five over",
    "suppressive fire on ridge line target neutralized advance cautiously",
    "ammunition resupply ready at forward operating base delta",
    "engine failure on lead vehicle requesting towing assistance over",
    "squad leader confirm transmission receipt acknowledge over",
]


import subprocess
import tempfile
import torchaudio.functional as F

_SPOKEN_CACHE = {}

def synthesize_spoken_utterance(
    transcript: str,
    sample_rate: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates realistic spoken speech audio for the given transcript.
    Uses native Windows SpeechSynthesizer if available for high-intelligibility ASR,
    falling back to multi-harmonic formant modeling.
    """
    cache_key = (transcript, sample_rate)
    if cache_key in _SPOKEN_CACHE:
        clean_wav = _SPOKEN_CACHE[cache_key].copy()
        # Add slight pitch / temporal variability per seed
        rng = np.random.RandomState(seed)
        noise = rng.randn(len(clean_wav)) * 1e-4
        return (clean_wav + noise).astype(np.float32)

    # Attempt TTS via Windows PowerShell SpeechSynthesizer
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav, \
             tempfile.NamedTemporaryFile(suffix=".ps1", mode="w", encoding="utf-8", delete=False) as tmp_ps1:
            tmp_wav_path = tmp_wav.name
            tmp_ps1_path = tmp_ps1.name

            ps_code = f"""
Add-Type -AssemblyName System.Speech
$speaker = New-Object -TypeName System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 0
$speaker.SetOutputToWaveFile("{tmp_wav_path}")
$speaker.Speak("{transcript}")
$speaker.Dispose()
"""
            tmp_ps1.write(ps_code)

        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", tmp_ps1_path],
            check=True,
            capture_output=True,
        )

        if os.path.exists(tmp_wav_path) and os.path.getsize(tmp_wav_path) > 1000:
            audio, native_sr = sf.read(tmp_wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Resample to target sample_rate if needed
            if native_sr != sample_rate:
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                audio = F.resample(audio_tensor, native_sr, sample_rate).squeeze(0).numpy()

            # Normalize RMS
            rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
            audio = (audio * (0.12 / rms)).astype(np.float32)

            _SPOKEN_CACHE[cache_key] = audio
            
            # Cleanup temp files
            for p in [tmp_wav_path, tmp_ps1_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            return audio

    except Exception:
        pass

    # Fallback to formant synthesis if TTS not available
    return synthesize_speech_utterance(transcript, sample_rate=sample_rate, seed=seed)


def synthesize_speech_utterance(transcript: str, sample_rate: int = 16000, duration_sec: float = 2.5, seed: int = 42) -> np.ndarray:
    """
    Generates rich acoustic speech-like formant carrier modulated by word syllabic envelopes
    with distinct fundamental frequency f0, formants F1-F4, and phoneme consonant friction.
    """
    rng = np.random.RandomState(seed)
    total_samples = int(duration_sec * sample_rate)
    t = np.arange(total_samples) / sample_rate
    
    words = transcript.split()
    num_words = len(words)
    
    # Syllabic envelope modulation
    envelope = np.zeros(total_samples, dtype=np.float32)
    for i in range(num_words):
        w_start = int((i / num_words) * total_samples)
        w_len = int((0.85 / num_words) * total_samples)
        w_end = min(total_samples, w_start + w_len)
        if w_end > w_start:
            env_slice = np.hanning(w_end - w_start) ** 1.5
            envelope[w_start:w_end] = env_slice
            
    # Formants F0 (pitch), F1, F2, F3, F4
    f0 = rng.uniform(110.0, 220.0)
    speech = np.zeros(total_samples, dtype=np.float32)
    
    harmonics = [
        (1.0, 1.0),
        (2.0, 0.7),
        (3.0, 0.5),
        (f0 * 4.0 / f0, 0.4),
        (f0 * 6.0 / f0, 0.25),
        (f0 * 9.0 / f0, 0.15),
    ]
    
    # Pitch drift / intonation
    pitch_mod = 1.0 + 0.05 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * f0 * np.cumsum(pitch_mod) / sample_rate
    
    for mult, gain in harmonics:
        speech += gain * np.sin(phase * mult)
        
    # Add unvoiced fricative turbulence during active speech
    fricative_noise = rng.randn(total_samples).astype(np.float32)
    speech += 0.15 * fricative_noise * envelope
    
    speech_signal = speech * envelope
    # Normalize clean speech RMS
    rms = np.sqrt(np.mean(speech_signal ** 2)) + 1e-8
    target_rms = 0.12
    return (speech_signal * (target_rms / rms)).astype(np.float32)



def load_librispeech_clean_corpus(corpus_dir: str = "data/librispeech_clean") -> List[Dict[str, str]]:
    """
    Loads real LibriSpeech human speech utterances and their ground-truth transcripts.
    """
    corpus_path = Path(corpus_dir)
    trans_files = list(corpus_path.glob("**/*.trans.txt"))
    trans_map = {}
    for tf in trans_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        trans_map[parts[0]] = parts[1].lower()
        except Exception:
            pass

    audio_files = sorted(list(corpus_path.glob("**/*.flac")) + list(corpus_path.glob("**/*.wav")))
    corpus = []
    for af in audio_files:
        stem = af.stem
        transcript = trans_map.get(stem, "")
        if transcript:
            corpus.append({
                "audio_path": str(af),
                "transcript": transcript,
                "id": stem,
            })
    return corpus


def generate_dataset(
    output_dir: str = "data/simulated",
    clean_source_dir: str = "data/librispeech_clean",
    counts: dict = None,
    sample_rate: int = 16000,
    seed: int = 42,
) -> dict:
    """
    Generates train, val, and test splits from real LibriSpeech human speech,
    applying physical battlefield radio degradation and saving matching transcripts.
    """
    if counts is None:
        counts = {"train": 200, "val": 50, "test": 50}

    corpus = load_librispeech_clean_corpus(clean_source_dir)
    total_needed = sum(counts.values())
    if len(corpus) < total_needed:
        raise ValueError(f"Insufficient LibriSpeech samples: found {len(corpus)}, need {total_needed}")

    # Shuffle deterministically
    rng_corpus = np.random.RandomState(seed)
    indices = rng_corpus.permutation(len(corpus))
    
    sim = BattlefieldAudioSimulator(sample_rate=sample_rate, seed=seed)
    rng = np.random.RandomState(seed)
    
    dataset_stats = {}
    cursor = 0
    
    for split, count in counts.items():
        split_dir = Path(output_dir) / split
        clean_dir = split_dir / "clean"
        deg_dir = split_dir / "degraded"
        clean_dir.mkdir(parents=True, exist_ok=True)
        deg_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = []
        snrs = []
        
        print(f"[*] Generating {count} real speech utterances for split '{split}'...")
        for i in range(count):
            item_idx = indices[cursor]
            cursor += 1
            sample_entry = corpus[item_idx]
            transcript = sample_entry["transcript"]
            
            # Read real clean speech
            clean_wav, native_sr = sf.read(sample_entry["audio_path"], dtype="float32")
            if clean_wav.ndim > 1:
                clean_wav = clean_wav.mean(axis=1)
            if native_sr != sample_rate:
                clean_tensor = torch.from_numpy(clean_wav).unsqueeze(0)
                clean_wav = F.resample(clean_tensor, native_sr, sample_rate).squeeze(0).numpy()

            # Normalize clean speech RMS
            rms = np.sqrt(np.mean(clean_wav ** 2)) + 1e-8
            clean_wav = (clean_wav * (0.10 / rms)).astype(np.float32)

            utt_seed = seed + (1000 if split == "train" else 2000 if split == "val" else 3000) + i
            
            # Target SNR uniformly between -5 dB and 20 dB
            target_snr = float(rng.uniform(-5.0, 20.0))
            snrs.append(target_snr)
            
            # Physical radio & acoustic degradation chain
            sim_res = sim.simulate_battlefield_degradation(
                clean_speech=clean_wav,
                snr_db=target_snr,
                include_gunfire=bool(rng.rand() > 0.3),
                include_engine=bool(rng.rand() > 0.2),
                include_wind=bool(rng.rand() > 0.3),
                bandpass_cutoff=(300.0, 3400.0),
                codec2_bitrate=2400 if rng.rand() > 0.4 else 1200,
                clipping_drive=float(rng.uniform(1.2, 2.8)),
            )
            
            deg_wav = sim_res["degraded"]
            duration = len(clean_wav) / sample_rate
            
            # Save audio
            clean_filename = f"{split}_{i:04d}_clean.wav"
            deg_filename = f"{split}_{i:04d}_deg.wav"
            
            sf.write(clean_dir / clean_filename, clean_wav, sample_rate)
            sf.write(deg_dir / deg_filename, deg_wav, sample_rate)
            
            manifest.append({
                "id": f"{split}_{i:04d}",
                "original_id": sample_entry["id"],
                "clean_path": str(clean_dir / clean_filename),
                "degraded_path": str(deg_dir / deg_filename),
                "transcript": transcript,
                "snr_db": round(target_snr, 2),
                "duration_sec": round(duration, 2),
            })
            
        with open(split_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        dataset_stats[split] = {
            "count": count,
            "source": "LibriSpeech dev-clean (Real Human Speech)",
            "snr_min": round(float(np.min(snrs)), 2),
            "snr_max": round(float(np.max(snrs)), 2),
            "snr_mean": round(float(np.mean(snrs)), 2),
            "snr_std": round(float(np.std(snrs)), 2),
            "snr_distribution": {
                "-5 to 0 dB": sum(1 for s in snrs if -5 <= s < 0),
                "0 to 5 dB": sum(1 for s in snrs if 0 <= s < 5),
                "5 to 10 dB": sum(1 for s in snrs if 5 <= s < 10),
                "10 to 15 dB": sum(1 for s in snrs if 10 <= s < 15),
                "15 to 20 dB": sum(1 for s in snrs if 15 <= s <= 20),
            },
        }
        
    return dataset_stats



def evaluate_baseline_on_test_set(
    test_dir: str = "data/simulated/test",
    results_dir: str = "results",
    sample_rate: int = 16000,
) -> dict:
    """
    Evaluates Pretrained DeepFilterNet3 baseline on test set utterances.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    manifest_file = Path(test_dir) / "manifest.json"
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
        
    print("[*] Initializing Pretrained DeepFilterNet3 model...")
    df_model, df_state, _ = init_df()
    df_sr = df_state.sr()
    print(f"[+] DeepFilterNet initialized (Internal native sample rate: {df_sr} Hz)")
    
    whisper_eval = WhisperEvaluator(model_name="openai/whisper-tiny")
    
    eval_results = []
    sample_pairs = []
    enhanced_out_dir = Path(test_dir) / "enhanced_deepfilternet"
    enhanced_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Evaluating {len(manifest)} test set utterances with DeepFilterNet3 baseline...")
    for idx, item in enumerate(manifest):
        clean_audio, sr_c = sf.read(item["clean_path"])
        deg_audio, sr_d = sf.read(item["degraded_path"])
        clean_audio = clean_audio.astype(np.float32)
        deg_audio = deg_audio.astype(np.float32)
        ref_text = item["transcript"]
        
        # DeepFilterNet expects torch float32 tensor (channels, time) at df_sr (48000 Hz)
        deg_tensor = torch.from_numpy(deg_audio).unsqueeze(0).to(torch.float32)
        if sr_d != df_sr:
            # Resample to 48kHz for DeepFilterNet
            import torchaudio.functional as F
            deg_48k = F.resample(deg_tensor, sr_d, df_sr).to(torch.float32)
        else:
            deg_48k = deg_tensor
            
        # Enhance with DeepFilterNet (reset state for utterance isolation)
        df_state.reset()
        with torch.no_grad():
            enhanced_48k = enhance(df_model, df_state, deg_48k)
            
        # Resample back to 16kHz for metrics
        if df_sr != sample_rate:
            enhanced_16k = F.resample(enhanced_48k, df_sr, sample_rate).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            enhanced_16k = enhanced_48k.squeeze(0).cpu().numpy().astype(np.float32)
            
        # Align lengths
        min_len = min(len(clean_audio), len(enhanced_16k), len(deg_audio))
        clean_eval = clean_audio[:min_len]
        deg_eval = deg_audio[:min_len]
        enh_eval = enhanced_16k[:min_len]
        
        # Save enhanced audio
        enh_path = enhanced_out_dir / f"{item['id']}_df_enh.wav"
        sf.write(enh_path, enh_eval, sample_rate)
        
        # Calculate PESQ
        pesq_deg = compute_pesq(clean_eval, deg_eval, sample_rate)
        pesq_enh = compute_pesq(clean_eval, enh_eval, sample_rate)
        
        # Calculate STOI
        stoi_deg = compute_stoi(clean_eval, deg_eval, sample_rate)
        stoi_enh = compute_stoi(clean_eval, enh_eval, sample_rate)
        
        # Calculate Whisper transcriptions & WER
        hyp_deg = whisper_eval.transcribe(deg_eval, sample_rate)
        hyp_enh = whisper_eval.transcribe(enh_eval, sample_rate)
        wer_deg = compute_wer(ref_text, hyp_deg)
        wer_enh = compute_wer(ref_text, hyp_enh)
        
        if len(sample_pairs) < 5:
            sample_pairs.append({
                "id": item["id"],
                "reference_text": ref_text,
                "hypothesis_degraded": hyp_deg,
                "hypothesis_enhanced": hyp_enh,
                "wer_degraded": round(wer_deg, 3),
                "wer_enhanced": round(wer_enh, 3),
            })
        
        res = {
            "id": item["id"],
            "snr_db": item["snr_db"],
            "transcript": ref_text,
            "hypothesis_degraded": hyp_deg,
            "hypothesis_enhanced": hyp_enh,
            "pesq_degraded": round(pesq_deg, 3),
            "pesq_enhanced": round(pesq_enh, 3),
            "delta_pesq": round(pesq_enh - pesq_deg, 3),
            "stoi_degraded": round(stoi_deg, 3),
            "stoi_enhanced": round(stoi_enh, 3),
            "delta_stoi": round(stoi_enh - stoi_deg, 3),
            "wer_degraded": round(wer_deg, 3),
            "wer_enhanced": round(wer_enh, 3),
            "wer_reduction": round(wer_deg - wer_enh, 3),
        }
        eval_results.append(res)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest):
            print(f"    Processed [{idx+1:02d}/{len(manifest):02d}] utterances...")
            
    # Aggregate Metrics
    avg_pesq_deg = float(np.mean([r["pesq_degraded"] for r in eval_results]))
    avg_pesq_enh = float(np.mean([r["pesq_enhanced"] for r in eval_results]))
    avg_stoi_deg = float(np.mean([r["stoi_degraded"] for r in eval_results]))
    avg_stoi_enh = float(np.mean([r["stoi_enhanced"] for r in eval_results]))
    avg_wer_deg = float(np.mean([r["wer_degraded"] for r in eval_results]))
    avg_wer_enh = float(np.mean([r["wer_enhanced"] for r in eval_results]))
    
    summary = {
        "baseline_model": "Pretrained DeepFilterNet3 (Checkpoint: model_120.ckpt.best)",
        "test_sample_count": len(eval_results),
        "pesq": {
            "degraded_mean": round(avg_pesq_deg, 3),
            "enhanced_mean": round(avg_pesq_enh, 3),
            "delta_pesq": round(avg_pesq_enh - avg_pesq_deg, 3),
        },
        "stoi": {
            "degraded_mean": round(avg_stoi_deg, 3),
            "enhanced_mean": round(avg_stoi_enh, 3),
            "delta_stoi": round(avg_stoi_enh - avg_stoi_deg, 3),
        },
        "wer": {
            "degraded_mean": round(avg_wer_deg, 3),
            "enhanced_mean": round(avg_wer_enh, 3),
            "wer_reduction": round(avg_wer_deg - avg_wer_enh, 3),
        },
    }
    
    # Save JSON & CSV
    json_path = Path(results_dir) / "day2_baseline.json"
    csv_path = Path(results_dir) / "day2_baseline.csv"
    
    output_payload = {
        "summary": summary,
        "sample_pairs": sample_pairs,
        "per_utterance_results": eval_results,
    }
    
    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=2)
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_results[0].keys())
        writer.writeheader()
        writer.writerows(eval_results)
        
    print(f"[+] Results saved to {json_path} and {csv_path}")
    return output_payload


def main():
    print("=" * 75)
    print(" DAY 2: Battlefield Dataset Generation & Pretrained Baseline Evaluation")
    print("=" * 75)
    
    dataset_stats = generate_dataset()
    print("\n" + "=" * 75)
    print(" DATASET STATISTICS")
    print("=" * 75)
    print(json.dumps(dataset_stats, indent=2))
    
    eval_results = evaluate_baseline_on_test_set()
    
    print("\n" + "=" * 75)
    print(" 5 EXAMPLE (REFERENCE_TEXT, WHISPER_HYPOTHESIS) TEST PAIRS")
    print("=" * 75)
    for idx, pair in enumerate(eval_results.get("sample_pairs", [])):
        print(f"[{idx+1}] ID: {pair['id']}")
        print(f"    REF:      {pair['reference_text']}")
        print(f"    DEG HYP:  {pair['hypothesis_degraded']} (WER: {pair['wer_degraded']:.3f})")
        print(f"    ENH HYP:  {pair['hypothesis_enhanced']} (WER: {pair['wer_enhanced']:.3f})")
        print()
    
    print("=" * 75)
    print(" BASELINE BENCHMARK SUMMARY TABLE")
    print("=" * 75)
    summary = eval_results["summary"]
    print(f"{'Metric':<18} | {'Degraded (Input)':<20} | {'Enhanced (DeepFilterNet3)':<26} | {'Delta / Improvement':<18}")
    print("-" * 88)
    print(f"{'PESQ (-0.5 to 4.5)':<18} | {summary['pesq']['degraded_mean']:<20.3f} | {summary['pesq']['enhanced_mean']:<26.3f} | {summary['pesq']['delta_pesq']:+<18.3f}")
    print(f"{'STOI (0.0 to 1.0)':<18} | {summary['stoi']['degraded_mean']:<20.3f} | {summary['stoi']['enhanced_mean']:<26.3f} | {summary['stoi']['delta_stoi']:+<18.3f}")
    print(f"{'Whisper WER':<18} | {summary['wer']['degraded_mean']:<20.3f} | {summary['wer']['enhanced_mean']:<26.3f} | {summary['wer']['wer_reduction']:+<18.3f} (Reduction)")
    print("=" * 75)


if __name__ == "__main__":
    main()

