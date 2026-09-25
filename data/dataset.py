"""
PyTorch Dataset & Streaming Data Loader for Battlefield Speech Enhancement.

ML Concept - Dynamic Online Simulation for Data Augmentation:
Rather than storing hundreds of gigabytes of pre-mixed degraded audio files, online simulation
dynamically degrades clean speech utterances on-the-fly during training. This provides infinite
variability in SNR, noise phase combinations, packet loss distributions, and non-linear distortion,
preventing the recurrent neural network from overfitting to specific noise profiles.
"""

from typing import Optional, List, Dict, Tuple
import os
import glob
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.simulation import BattlefieldAudioSimulator

# Standard NATO Tactical phrases for synthetic fallback
DEFAULT_TACTICAL_PHRASES = [
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


class TacticalSpeechDataset(Dataset):
    """
    Dataset that loads clean speech waveforms and dynamically generates battlefield degraded pairs.
    Propagates ground-truth reference transcripts from LibriSpeech, VCTK, or manifest metadata.
    """
    def __init__(
        self,
        clean_audio_dir: Optional[str] = None,
        sample_rate: int = 16000,
        segment_length_samples: int = 32000,  # 2.0 seconds
        seed: int = 42,
        num_synthetic_samples: int = 100,  # Used if clean_audio_dir is None or empty
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_length = segment_length_samples
        self.simulator = BattlefieldAudioSimulator(sample_rate=sample_rate, seed=seed)
        self.samples: List[Dict[str, str]] = []

        if clean_audio_dir and os.path.isdir(clean_audio_dir):
            self.samples = self._load_dataset_entries(clean_audio_dir)

        self.num_synthetic_samples = num_synthetic_samples
        self.use_synthetic = len(self.samples) == 0

    def _load_dataset_entries(self, audio_dir: str) -> List[Dict[str, str]]:
        """
        Discovers audio files and parses matching transcripts from:
        1. manifest.json in audio_dir or parent
        2. LibriSpeech format (*.trans.txt)
        3. VCTK format (*.txt files alongside *.wav or in txt/ folder)
        """
        entries = []
        
        # 1. Check for manifest.json
        manifest_paths = [
            os.path.join(audio_dir, "manifest.json"),
            os.path.join(os.path.dirname(audio_dir), "manifest.json"),
        ]
        for mp in manifest_paths:
            if os.path.isfile(mp):
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data:
                        clean_path = item.get("clean_path") or item.get("audio_path")
                        if clean_path and os.path.isfile(clean_path):
                            entries.append({
                                "audio_path": clean_path,
                                "transcript": item.get("transcript", item.get("text", "")),
                                "id": item.get("id", Path(clean_path).stem),
                            })
                    if entries:
                        return entries
                except Exception:
                    pass

        # 2. Check for LibriSpeech transcripts (*.trans.txt)
        trans_files = glob.glob(os.path.join(audio_dir, "**/*.trans.txt"), recursive=True)
        librispeech_trans_map = {}
        for tf in trans_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            librispeech_trans_map[parts[0]] = parts[1].lower()
            except Exception:
                pass

        # Discover all wav files
        wav_files = sorted(glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True))
        for wav_path in wav_files:
            file_stem = Path(wav_path).stem
            transcript = ""
            
            # Check LibriSpeech map
            if file_stem in librispeech_trans_map:
                transcript = librispeech_trans_map[file_stem]
            else:
                # Check for matching .txt file alongside .wav
                txt_path = Path(wav_path).with_suffix(".txt")
                if txt_path.is_file():
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            transcript = f.read().strip()
                    except Exception:
                        pass
                else:
                    # Check VCTK mirrored txt/ directory
                    vctk_txt = str(wav_path).replace("wav48", "txt").replace(".wav", ".txt")
                    if os.path.isfile(vctk_txt):
                        try:
                            with open(vctk_txt, "r", encoding="utf-8") as f:
                                transcript = f.read().strip()
                        except Exception:
                            pass
            
            entries.append({
                "audio_path": wav_path,
                "transcript": transcript,
                "id": file_stem,
            })
            
        return entries

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic_samples
        return len(self.samples)

    def _generate_synthetic_clean_speech(self, idx: int) -> Tuple[np.ndarray, str]:
        """Generates synthetic multi-tone speech-like formant signal for standalone testing."""
        rng = np.random.RandomState(idx + 1000)
        t = np.arange(self.segment_length) / self.sample_rate
        f0 = rng.uniform(100.0, 250.0)
        signal_out = np.zeros(self.segment_length, dtype=np.float32)
        # Formants
        for harmonic in [1, 2, 3, 4, 6]:
            signal_out += (1.0 / harmonic) * np.sin(2 * np.pi * (f0 * harmonic) * t)
        # Amplitude envelope modulation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2.0, 5.0) * t)
        clean_audio = (signal_out * envelope * 0.5).astype(np.float32)
        phrase = DEFAULT_TACTICAL_PHRASES[idx % len(DEFAULT_TACTICAL_PHRASES)]
        return clean_audio, phrase

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_synthetic:
            clean_audio, transcript = self._generate_synthetic_clean_speech(idx)
            sample_id = f"synthetic_{idx:04d}"
        else:
            entry = self.samples[idx]
            file_path = entry["audio_path"]
            transcript = entry.get("transcript", "")
            sample_id = entry.get("id", f"sample_{idx:04d}")
            
            clean_audio, sr = sf.read(file_path, dtype="float32")
            if clean_audio.ndim > 1:
                clean_audio = clean_audio.mean(axis=1)
            # Resample or pad/trim
            if len(clean_audio) < self.segment_length:
                clean_audio = np.pad(clean_audio, (0, self.segment_length - len(clean_audio)))
            else:
                clean_audio = clean_audio[: self.segment_length]

        # Apply battlefield degradation
        sim_res = self.simulator.simulate_battlefield_degradation(clean_audio)

        return {
            "clean": torch.from_numpy(sim_res["clean"]).unsqueeze(0),       # [1, T]
            "degraded": torch.from_numpy(sim_res["degraded"]).unsqueeze(0), # [1, T]
            "snr_db": torch.tensor(sim_res["snr_db"], dtype=torch.float32),
            "transcript": transcript,
            "reference_text": transcript,
            "id": sample_id,
        }

