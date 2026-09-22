"""
Battlefield Radio & Acoustic Degradation Simulation Pipeline.

ML Concept - Physical Degradation Modeling for Robust Generalization:
Tactical speech enhancement models fail in the field if trained only on synthetic Gaussian noise
or standard clean speech datasets. In reality, tactical military communications suffer from a cascade
of acoustic and RF hardware distortions:
1. High-SPL Impulse Transients (Gunfire, artillery, mortar blasts with high crest factor).
2. Continuous Mechanical Noise (Diesel engines, track rumble, rotorcraft turbulence).
3. RF Channel Bandpass Filtering (Strict 300 Hz - 3400 Hz cutoff).
4. Narrowband Vocoder Quantization (Codec2 / MELP 1200-2400 bps frame drops & pitch quantization).
5. Non-linear Push-to-Talk (PTT) Preamp Overdrive & Clipping (Harmonic saturation).
6. RF Atmospheric Static & Squelch Tail Transients (Burst noise upon PTT release).

This module simulates the full physical degradation chain deterministically and configurably.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import torch
import torchaudio


class BattlefieldAudioSimulator:
    """
    Simulates tactical battlefield acoustic environments and military radio channel distortions.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        seed: Optional[int] = 42,
    ):
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)

    # -------------------------------------------------------------------------
    # 1. Synthetic Noise Generators (Gunfire, Engine, Wind/Rotor)
    # -------------------------------------------------------------------------
    def generate_gunfire_impulse(
        self,
        length_samples: int,
        burst_count: int = 3,
        peak_amp: float = 1.0,
    ) -> np.ndarray:
        """
        Synthesizes realistic ballistic shockwaves and muzzle blast transients.
        Models Friedlander blast wave: p(t) = P_0 * (1 - t/t_d) * exp(-alpha * t/t_d).
        """
        noise = np.zeros(length_samples, dtype=np.float32)
        if burst_count <= 0 or length_samples < 200:
            return noise

        shot_indices = self.rng.randint(0, max(1, length_samples - 800), size=burst_count)
        for idx in shot_indices:
            blast_len = min(600, length_samples - idx)
            t = np.linspace(0, 1, blast_len)
            # Friedlander waveform approximation (sharp positive spike + exponential decay + negative phase)
            wave = (1.0 - t * 1.5) * np.exp(-5.0 * t)
            # Add high-frequency shockwave crack
            hf_crack = self.rng.randn(blast_len) * np.exp(-12.0 * t) * 0.4
            transient = (wave + hf_crack) * peak_amp
            noise[idx : idx + blast_len] += transient[:blast_len]

        return np.clip(noise, -1.0, 1.0)

    def generate_engine_noise(
        self,
        length_samples: int,
        fundamental_hz: float = 45.0,
        num_harmonics: int = 8,
    ) -> np.ndarray:
        """
        Synthesizes armored vehicle diesel engine rumble and track vibration.
        Combines harmonically related engine cylinder firings with filtered low-frequency rumble.
        """
        t = np.arange(length_samples) / self.sample_rate
        engine = np.zeros(length_samples, dtype=np.float32)

        # Harmonics
        for h in range(1, num_harmonics + 1):
            freq = fundamental_hz * h + self.rng.uniform(-1.0, 1.0)
            amp = (1.0 / h) * self.rng.uniform(0.8, 1.2)
            phase = self.rng.uniform(0, 2 * np.pi)
            engine += amp * np.sin(2 * np.pi * freq * t + phase)

        # Low-frequency hull vibration rumble (lowpass filtered white noise)
        b, a = signal.butter(4, 250.0 / (self.sample_rate / 2), btype="low")
        rumble = signal.lfilter(b, a, self.rng.randn(length_samples))
        engine += 1.5 * rumble

        # Normalize
        std = np.std(engine) + 1e-8
        return (engine / std).astype(np.float32)

    def generate_wind_rotor_noise(
        self,
        length_samples: int,
        blade_pass_freq_hz: float = 16.0,
    ) -> np.ndarray:
        """
        Synthesizes rotorcraft cockpit noise and turbulent aerodynamic wind roar.
        Amplitude-modulated pink/brown noise with periodic blade-passage pulses.
        """
        t = np.arange(length_samples) / self.sample_rate
        # Blade pass modulation (helicopter rotor chop)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * blade_pass_freq_hz * t)**4

        # Pink noise approximation
        white = self.rng.randn(length_samples)
        b, a = signal.butter(2, 600.0 / (self.sample_rate / 2), btype="low")
        wind = signal.lfilter(b, a, white) * mod

        std = np.std(wind) + 1e-8
        return (wind / std).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. SNR Mixing
    # -------------------------------------------------------------------------
    def mix_speech_and_noise(
        self,
        clean_speech: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixes clean speech with noise at exact target SNR (dB).
        """
        clean_speech = clean_speech.astype(np.float32)
        noise = noise.astype(np.float32)

        # Ensure lengths match
        if len(noise) < len(clean_speech):
            repeats = int(np.ceil(len(clean_speech) / len(noise)))
            noise = np.tile(noise, repeats)[: len(clean_speech)]
        else:
            noise = noise[: len(clean_speech)]

        speech_power = np.mean(clean_speech ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10

        target_noise_power = speech_power / (10.0 ** (target_snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise * scale

        mixed = clean_speech + scaled_noise
        return mixed, scaled_noise

    # -------------------------------------------------------------------------
    # 3. Radio Channel Distortions (Bandpass, Codec2, Clipping, Static)
    # -------------------------------------------------------------------------
    def apply_radio_bandpass(
        self,
        audio: np.ndarray,
        low_freq: float = 300.0,
        high_freq: float = 3400.0,
        order: int = 4,
    ) -> np.ndarray:
        """
        Applies NATO tactical voice narrowband bandpass filter (300 Hz - 3400 Hz).
        """
        nyquist = self.sample_rate / 2.0
        low = max(20.0, low_freq) / nyquist
        high = min(nyquist - 20.0, high_freq) / nyquist
        b, a = signal.butter(order, [low, high], btype="bandpass")
        return signal.filtfilt(b, a, audio).astype(np.float32)

    def apply_codec2_emulation(
        self,
        audio: np.ndarray,
        bitrate: int = 2400,
        packet_loss_rate: float = 0.05,
    ) -> np.ndarray:
        """
        Emulates low-bitrate tactical vocoder (Codec2/MELP 1200/2400 bps) artifacts:
        - Sub-band LPC spectral envelope quantization.
        - Harmonic sinusoidal speech resynthesis.
        - Random packet/frame loss drops.
        """
        frame_size = int(self.sample_rate * 0.02)  # 20ms frame = 320 samples
        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return audio

        out = np.zeros_like(audio)
        prev_frame = np.zeros(frame_size, dtype=np.float32)

        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]

            # Simulate packet loss (frame drop / packet concealment)
            if self.rng.uniform(0, 1) < packet_loss_rate:
                # Packet loss concealment: attenuate previous frame
                out[i * frame_size : (i + 1) * frame_size] = prev_frame * 0.5
                continue

            # Vocoder spectral bit reduction via 8-bit non-linear mu-law quantization
            mu = 255.0
            frame_norm = np.clip(frame, -1.0, 1.0)
            companded = np.sign(frame_norm) * np.log1p(mu * np.abs(frame_norm)) / np.log1p(mu)
            quantized = np.round(companded * (127.0 if bitrate >= 2400 else 63.0)) / (127.0 if bitrate >= 2400 else 63.0)
            expanded = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)

            out[i * frame_size : (i + 1) * frame_size] = expanded
            prev_frame = expanded

        # Handle remaining samples
        remaining = len(audio) - num_frames * frame_size
        if remaining > 0:
            out[num_frames * frame_size :] = audio[num_frames * frame_size :]

        return out.astype(np.float32)

    def apply_nonlinear_radio_clipping(
        self,
        audio: np.ndarray,
        drive: float = 2.5,
        asymmetry: float = 0.1,
    ) -> np.ndarray:
        """
        Simulates tactical Push-To-Talk (PTT) microphone preamplifier overdrive and diode soft/hard clipping.
        Equation: y(t) = tanh(drive * (x(t) + asymmetry)) - tanh(drive * asymmetry).
        """
        driven = drive * (audio + asymmetry)
        clipped = np.tanh(driven) - np.tanh(drive * asymmetry)
        # Normalize back to max amplitude
        max_val = np.max(np.abs(clipped)) + 1e-8
        return (clipped / max_val * min(1.0, drive * 0.8)).astype(np.float32)

    def apply_rf_static_and_squelch(
        self,
        audio: np.ndarray,
        snr_db: float = 30.0,
        burst_prob: float = 0.3,
    ) -> np.ndarray:
        """
        Adds RF atmospheric background hiss and squelch tail static bursts.
        """
        # Continuous thermal noise floor
        noise_std = 10.0 ** (-snr_db / 20.0)
        thermal_noise = self.rng.randn(len(audio)).astype(np.float32) * noise_std
        output = audio + thermal_noise

        # Squelch tail burst at the end or intermittent static burst
        if self.rng.uniform(0, 1) < burst_prob and len(audio) > 800:
            burst_len = min(640, len(audio) // 4)  # ~40ms
            burst_start = self.rng.randint(0, len(audio) - burst_len)
            burst = self.rng.randn(burst_len).astype(np.float32) * 0.3
            output[burst_start : burst_start + burst_len] += burst

        return np.clip(output, -1.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Full End-to-End Simulation Pipeline
    # -------------------------------------------------------------------------
    def simulate_battlefield_degradation(
        self,
        clean_speech: Union[np.ndarray, torch.Tensor],
        snr_db: Optional[float] = None,
        include_gunfire: bool = True,
        include_engine: bool = True,
        include_wind: bool = True,
        bandpass_cutoff: Tuple[float, float] = (300.0, 3400.0),
        codec2_bitrate: int = 2400,
        clipping_drive: float = 2.0,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Executes the full battlefield simulation pipeline on a clean speech utterance.

        Returns a dictionary containing:
            - 'clean': Original clean speech
            - 'degraded': Full battlefield degraded audio
            - 'snr_db': Chosen SNR
            - 'noise': Composite noise waveform
        """
        is_torch = isinstance(clean_speech, torch.Tensor)
        if is_torch:
            orig_speech = clean_speech.detach().cpu().numpy().squeeze()
        else:
            orig_speech = clean_speech.squeeze()

        length = len(orig_speech)
        if snr_db is None:
            snr_db = float(self.rng.uniform(-15.0, 20.0))

        # 1. Synthesize composite acoustic noise
        composite_noise = np.zeros(length, dtype=np.float32)
        if include_engine:
            composite_noise += self.generate_engine_noise(length) * 0.6
        if include_wind:
            composite_noise += self.generate_wind_rotor_noise(length) * 0.4
        if include_gunfire:
            burst_count = self.rng.randint(1, 4)
            composite_noise += self.generate_gunfire_impulse(length, burst_count=burst_count) * 0.8

        # If no noise flags were set, use background Gaussian noise
        if not (include_engine or include_wind or include_gunfire):
            composite_noise = self.rng.randn(length).astype(np.float32)

        # 2. Mix speech + noise at target SNR
        mixed, scaled_noise = self.mix_speech_and_noise(orig_speech, composite_noise, target_snr_db=snr_db)

        # 3. Radio Bandpass Filter
        filtered = self.apply_radio_bandpass(mixed, low_freq=bandpass_cutoff[0], high_freq=bandpass_cutoff[1])

        # 4. Low-bitrate Vocoder Emulation
        vocoded = self.apply_codec2_emulation(filtered, bitrate=codec2_bitrate)

        # 5. Non-linear Radio Preamp Clipping
        clipped = self.apply_nonlinear_radio_clipping(vocoded, drive=clipping_drive)

        # 6. RF Static & Squelch Tail Injection
        degraded = self.apply_rf_static_and_squelch(clipped)

        return {
            "clean": orig_speech if not is_torch else torch.from_numpy(orig_speech),
            "degraded": degraded if not is_torch else torch.from_numpy(degraded),
            "noise": scaled_noise if not is_torch else torch.from_numpy(scaled_noise),
            "snr_db": snr_db,
        }


def main():
    print("[*] Running Battlefield Audio Simulator smoke test...")
    sim = BattlefieldAudioSimulator(sample_rate=16000, seed=42)

    # Create 2 seconds of dummy synthetic clean speech (harmonic formants)
    t = np.linspace(0, 2, 32000, dtype=np.float32)
    dummy_speech = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)

    result = sim.simulate_battlefield_degradation(dummy_speech, snr_db=5.0)
    print(f"[+] Simulation completed successfully!")
    print(f"    - Clean shape: {result['clean'].shape}, std: {np.std(result['clean']):.4f}")
    print(f"    - Degraded shape: {result['degraded'].shape}, std: {np.std(result['degraded']):.4f}")
    print(f"    - Output SNR: {result['snr_db']} dB")


if __name__ == "__main__":
    main()
