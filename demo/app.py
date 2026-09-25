"""
Live Demonstration of Battlefield Radio Intelligibility Engine.

Production Pipeline:
1. Loads real tactical test-set speech (clean reference, military radio degraded).
2. Runs streaming ONNX inference chunk-by-chunk using Branch A (artifacts/branch_a_denoiser.onnx).
3. Evaluates acoustic safety with the Soft-Blend Reliability Guard.
4. Transcribes reference, degraded, and enhanced audio with Whisper ASR.
5. Displays full timing breakdown (STFT, ONNX inference, Guard, iSTFT) and intelligibility metrics.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import compute_pesq, compute_stoi, compute_wer, WhisperEvaluator
from eval.reliability_guard import ReliabilityGuard


class ProductionONNXStreamingEngine:
    """
    Simulates production on-device streaming audio pipeline using ONNX Runtime.
    Processes audio chunk-by-chunk with explicit static recurrent state propagation.
    """
    def __init__(
        self,
        onnx_model_path: str = "artifacts/branch_a_denoiser.onnx",
        freq_bins: int = 257,
        chunk_frames: int = 4,
        n_fft: int = 512,
        hop_length: int = 128,
        sample_rate: int = 16000,
    ):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Branch A ONNX model not found: {onnx_model_path}. Run deploy/export_onnx.py first.")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(onnx_model_path, opts, providers=["CPUExecutionProvider"])
        self.freq_bins = freq_bins
        self.chunk_frames = chunk_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sample_rate
        self.window = torch.hann_window(n_fft)

    def process_streaming(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Executes real streaming enhancement chunk-by-chunk.
        Returns: (raw_enhanced_audio, timing_stats)
        """
        t0_total = time.perf_counter()
        
        # 1. External STFT
        t0_stft = time.perf_counter()
        audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        stft = torch.stft(
            audio_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft).numpy()
        phase = torch.angle(stft)
        t_stft_ms = (time.perf_counter() - t0_stft) * 1000.0

        B, F, T = mag.shape
        enhanced_mag_chunks = []
        h_state = np.zeros((2, 1, 64), dtype=np.float32)
        
        # 2. Streaming ONNX Loop (4 frames = 32ms per chunk)
        chunk_latencies = []
        t0_onnx_loop = time.perf_counter()
        
        for t_idx in range(0, T - self.chunk_frames + 1, self.chunk_frames):
            chunk = mag[:, :, t_idx : t_idx + self.chunk_frames]
            chunk_in = chunk[:, np.newaxis, :, :].astype(np.float32) # [1, 1, 257, 4]
            
            t0_chunk = time.perf_counter()
            enh_chunk, h_state = self.session.run(
                None,
                {"input_chunk": chunk_in, "h_in": h_state}
            )
            chunk_latencies.append((time.perf_counter() - t0_chunk) * 1000.0)
            enhanced_mag_chunks.append(enh_chunk.squeeze(1))

        t_onnx_total_ms = (time.perf_counter() - t0_onnx_loop) * 1000.0

        if not enhanced_mag_chunks:
            return audio, {"stft_ms": t_stft_ms, "onnx_chunk_ms": 0.0, "istft_ms": 0.0, "total_ms": 0.0}

        # 3. External iSTFT
        t0_istft = time.perf_counter()
        enh_mag_tensor = torch.from_numpy(np.concatenate(enhanced_mag_chunks, axis=-1))
        valid_frames = enh_mag_tensor.shape[-1]
        
        enh_complex = torch.polar(enh_mag_tensor, phase[:, :, :valid_frames])
        enhanced_audio = torch.istft(
            enh_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            length=len(audio),
        ).squeeze(0).numpy()
        t_istft_ms = (time.perf_counter() - t0_istft) * 1000.0
        
        total_time_ms = (time.perf_counter() - t0_total) * 1000.0
        
        timing_stats = {
            "stft_ms": t_stft_ms,
            "onnx_chunk_mean_ms": float(np.mean(chunk_latencies)),
            "onnx_chunk_p95_ms": float(np.percentile(chunk_latencies, 95)),
            "onnx_total_ms": t_onnx_total_ms,
            "istft_ms": t_istft_ms,
            "total_ms": total_time_ms,
            "num_chunks": len(chunk_latencies),
            "rtf": total_time_ms / (len(audio) / self.sr * 1000.0),
        }
        
        return enhanced_audio, timing_stats


def run_live_demo(sample_id: str = "test_0000"):
    print("=" * 90)
    print(" BATTLEFIELD RADIO INTELLIGIBILITY ENGINE — LIVE ON-DEVICE DEMO")
    print(" Target: Qualcomm Snapdragon X Elite (Hexagon NPU streaming pipeline)")
    print("=" * 90)

    # 1. Load test manifest
    manifest_path = "data/simulated/test/manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Test manifest not found at {manifest_path}")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    sample = next((s for s in manifest if s["id"] == sample_id), manifest[0])
    print(f"\n[1/4] Selected Test Utterance: {sample['id']}")
    print(f"      Input SNR: {sample['snr_db']:.2f} dB | Duration: {sample['duration_sec']:.2f} s")
    print(f"      Reference Transcript: \"{sample['transcript']}\"")

    # 2. Load audio files
    clean_audio, sr = sf.read(sample["clean_path"], dtype="float32")
    deg_audio, _ = sf.read(sample["degraded_path"], dtype="float32")
    if clean_audio.ndim > 1: clean_audio = clean_audio.mean(axis=1)
    if deg_audio.ndim > 1: deg_audio = deg_audio.mean(axis=1)

    # 3. Process with Production ONNX Engine
    print("\n[2/4] Executing Branch A ONNX Streaming Engine (32ms chunk budget)...")
    engine = ProductionONNXStreamingEngine("artifacts/branch_a_denoiser.onnx")
    raw_enh_audio, timing = engine.process_streaming(deg_audio)

    # 4. Apply Soft-Blend Reliability Guard
    print("\n[3/4] Evaluating Soft-Blend Reliability Guard (energy/flatness/correlation checks)...")
    guard = ReliabilityGuard()
    t0_guard = time.perf_counter()
    guarded_audio, is_safe, conf, triggers = guard.apply_guard(deg_audio, raw_enh_audio, mode="soft")
    t_guard_ms = (time.perf_counter() - t0_guard) * 1000.0

    # 5. Whisper ASR & Metrics Evaluation
    print("\n[4/4] Transcribing with Whisper ASR & Computing Intelligibility Metrics...")
    whisper_eval = WhisperEvaluator()
    
    min_len = min(len(clean_audio), len(deg_audio), len(guarded_audio))
    c_eval = clean_audio[:min_len]
    d_eval = deg_audio[:min_len]
    e_eval = guarded_audio[:min_len]

    pesq_deg = compute_pesq(c_eval, d_eval, sr)
    pesq_enh = compute_pesq(c_eval, e_eval, sr)
    stoi_deg = compute_stoi(c_eval, d_eval, sr)
    stoi_enh = compute_stoi(c_eval, e_eval, sr)

    hyp_deg = whisper_eval.transcribe(d_eval, sr)
    hyp_enh = whisper_eval.transcribe(e_eval, sr)
    wer_deg = compute_wer(sample["transcript"], hyp_deg)
    wer_enh = compute_wer(sample["transcript"], hyp_enh)

    # Display Report
    print("\n" + "=" * 90)
    print(" INTELLIGIBILITY & ASR EVALUATION REPORT")
    print("=" * 90)
    print(f"Ground Truth Reference : \"{sample['transcript']}\"")
    print(f"Degraded Input Hyp     : \"{hyp_deg}\"  (WER: {wer_deg:.3f})")
    print(f"Guarded Enhanced Hyp   : \"{hyp_enh}\"  (WER: {wer_enh:.3f})")
    print("-" * 90)
    print(f"PESQ Quality Score     : {pesq_deg:.3f} -> {pesq_enh:.3f} (Delta: {pesq_enh - pesq_deg:+.3f})")
    print(f"STOI Intelligibility   : {stoi_deg:.3f} -> {stoi_enh:.3f} (Delta: {stoi_enh - stoi_deg:+.3f})")
    print(f"WER Reduction          : {wer_deg:.3f} -> {wer_enh:.3f} (Delta: {wer_deg - wer_enh:+.3f} lower)")
    print(f"Reliability Guard State: {'SAFE' if is_safe else 'TRIGGERED'} (Confidence: {conf:.3f}, Triggers: {triggers})")
    print("=" * 90)

    print("\n" + "=" * 90)
    print(" ON-DEVICE STREAMING LATENCY BREAKDOWN (Chunk Size: 4 frames = 32.0 ms)")
    print("=" * 90)
    print(f"External STFT Pre-processing : {timing['stft_ms']:.2f} ms")
    print(f"ONNX Model Inference / Chunk : {timing['onnx_chunk_mean_ms']:.2f} ms (p95: {timing['onnx_chunk_p95_ms']:.2f} ms | Budget: 20.0 ms)")
    print(f"Soft-Blend Reliability Guard : {t_guard_ms:.2f} ms")
    print(f"External iSTFT Synthesis     : {timing['istft_ms']:.2f} ms")
    print(f"Total Processing Time        : {timing['total_ms']:.2f} ms for {sample['duration_sec']:.2f}s audio")
    print(f"Real-Time Factor (RTF)       : {timing['rtf']:.4f}x ({(1.0/timing['rtf']):.1f}x faster than real-time)")
    print("=" * 90)

    # Save enhanced sample output to demo/
    os.makedirs("demo/output", exist_ok=True)
    out_deg_path = f"demo/output/{sample['id']}_degraded.wav"
    out_enh_path = f"demo/output/{sample['id']}_enhanced.wav"
    sf.write(out_deg_path, deg_audio, sr)
    sf.write(out_enh_path, guarded_audio, sr)
    print(f"\n[+] Audio artifacts saved:\n    - Input Degraded: {out_deg_path}\n    - Guarded Enhanced: {out_enh_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Live demo for Battlefield Radio Intelligibility Engine.")
    parser.add_argument("--sample-id", type=str, default="test_0000", help="Sample ID from test manifest (e.g. test_0000, test_0001)")
    args = parser.parse_args()
    run_live_demo(sample_id=args.sample_id)


if __name__ == "__main__":
    main()
