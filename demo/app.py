"""
Live Interactive Demo for Battlefield Radio Intelligibility Engine.

ML Concept - Real-Time Streaming Audio Simulation & Processing Loop:
Demonstrates real-time chunk-by-chunk streaming inference with external STFT/iSTFT and state propagation.
"""

import os
import argparse
import numpy as np
import soundfile as sf
import torch

from models.dummy_conv_gru import DummyStreamingConvGRU
from data.simulation import BattlefieldAudioSimulator
from eval.metrics import compute_pesq, compute_stoi


class StreamingAudioProcessor:
    """
    Simulates real-time chunked audio processing.
    """
    def __init__(self, model: torch.nn.Module, sample_rate: int = 16000, n_fft: int = 512, hop_len: int = 128, chunk_frames: int = 4):
        self.model = model.eval()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.chunk_frames = chunk_frames
        self.window = torch.hann_window(n_fft)
        self.hidden_state = model.init_hidden_state(batch_size=1)

    def process_utterance(self, audio: np.ndarray) -> np.ndarray:
        """Processes an entire audio signal chunk-by-chunk in streaming fashion."""
        audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        
        # External STFT
        stft_spec = torch.stft(
            audio_t,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft_spec)
        phase = torch.angle(stft_spec)

        B, F, T = mag.shape
        enhanced_mag_chunks = []
        h = self.hidden_state

        # Streaming loop over fixed chunks
        for t in range(0, T - self.chunk_frames + 1, self.chunk_frames):
            chunk = mag[:, :, t : t + self.chunk_frames].unsqueeze(1)
            with torch.no_grad():
                enh_chunk, h = self.model(chunk, h)
            enhanced_mag_chunks.append(enh_chunk.squeeze(1))

        if not enhanced_mag_chunks:
            return audio

        enhanced_mag = torch.cat(enhanced_mag_chunks, dim=-1)
        valid_frames = enhanced_mag.shape[-1]

        # Reconstruct complex STFT
        enh_complex = enhanced_mag * torch.exp(1j * phase[:, :, :valid_frames])

        # External iSTFT
        enhanced_audio = torch.istft(
            enh_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.n_fft,
            window=self.window,
            length=valid_frames * self.hop_len,
        )

        return enhanced_audio.squeeze().numpy()


def run_demo():
    print("=" * 65)
    print(" Battlefield Radio Speech Enhancement — Live App Demo")
    print("=" * 65)

    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=42)
    model = DummyStreamingConvGRU()
    processor = StreamingAudioProcessor(model)

    # Synthetic tactical clean audio
    t = np.linspace(0, 3, 48000, dtype=np.float32)
    clean_audio = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 600 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

    print("[*] Simulating tactical noise (Gunfire + Tank Diesel + Radio Bandpass + PTT Clipping)...")
    sim_res = sim.simulate_battlefield_degradation(clean_audio, snr_db=-5.0)
    degraded_audio = sim_res["degraded"]

    print("[*] Executing real-time streaming Conv-GRU processor...")
    enhanced_audio = processor.process_utterance(degraded_audio)

    pesq_deg = compute_pesq(clean_audio, degraded_audio)
    pesq_enh = compute_pesq(clean_audio, enhanced_audio)
    stoi_deg = compute_stoi(clean_audio, degraded_audio)
    stoi_enh = compute_stoi(clean_audio, enhanced_audio)

    print("\n--- Live Processing Summary ---")
    print(f"Degraded Audio : PESQ={pesq_deg:.2f}, STOI={stoi_deg:.2f}")
    print(f"Enhanced Audio : PESQ={pesq_enh:.2f}, STOI={stoi_enh:.2f}")
    print(f"Delta-STOI     : {stoi_enh - stoi_deg:+.3f}")
    print("[+] Demo run finished successfully!")


def main():
    run_demo()


if __name__ == "__main__":
    main()
