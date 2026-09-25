import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor

class WhisperPerceptualLoss(nn.Module):
    """
    Computes perceptual distance in Whisper encoder embedding space.
    Backpropagates gradients from frozen Whisper encoder representations
    back into the speech enhancement model to preserve acoustic-phonetic cues.
    """
    def __init__(self, model_name: str = "openai/whisper-tiny", device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        print(f"[*] Initializing WhisperPerceptualLoss with {model_name}...")
        whisper = WhisperModel.from_pretrained(model_name).to(device)
        self.encoder = whisper.encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 80-channel log-Mel spectrogram parameters for Whisper (16kHz, n_fft=400, hop=160)
        # Using PyTorch functional / modules for end-to-end differentiability
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80
        self.register_buffer("window", torch.hann_window(self.n_fft))
        # Build mel filterbank matrix
        import torchaudio.functional as F_audio
        mel_fb = F_audio.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=8000.0,
            n_mels=self.n_mels,
            sample_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("mel_fb", mel_fb)

    def wav_to_log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Differentiable log-Mel spectrogram computation matching Whisper specs:
        wav: [B, T] -> log_mel: [B, 80, 3000] (padded/truncated to 30s or matching chunk)
        """
        # Ensure window is on same device
        window = torch.hann_window(self.n_fft, device=wav.device)
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        # Power spectrogram
        pwr = torch.abs(stft[:, :, :-1]) ** 2  # [B, 201, T_frames]
        # Mel filterbank projection: [B, 201, T] -> [B, 80, T]
        mel_spec = torch.matmul(self.mel_fb.to(wav.device).transpose(0, 1), pwr)
        # Log-mel with Whisper dynamic range compression
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        # Whisper encoder expects 3000 frames (30s) or exact multiple, pad/truncate to 3000
        target_frames = 3000
        B, n_m, T_f = log_mel.shape
        if T_f < target_frames:
            log_mel = torch.nn.functional.pad(log_mel, (0, target_frames - T_f))
        else:
            log_mel = log_mel[:, :, :target_frames]
        return log_mel

    def forward(self, enh_wav: torch.Tensor, clean_wav: torch.Tensor) -> torch.Tensor:
        """
        Computes L1 distance between encoder representations of enhanced and clean audio.
        """
        mel_enh = self.wav_to_log_mel(enh_wav)
        with torch.no_grad():
            mel_clean = self.wav_to_log_mel(clean_wav)
            feat_clean = self.encoder(mel_clean).last_hidden_state

        feat_enh = self.encoder(mel_enh).last_hidden_state
        loss = torch.mean(torch.abs(feat_enh - feat_clean))
        return loss

# Test gradient flow
if __name__ == "__main__":
    device = torch.device("cpu")
    loss_fn = WhisperPerceptualLoss(device=device)
    x = torch.randn(2, 32000, requires_grad=True)
    target = torch.randn(2, 32000)
    l = loss_fn(x, target)
    print("Loss value:", l.item())
    l.backward()
    print("Gradient norm on x:", x.grad.norm().item())
    assert x.grad is not None and x.grad.norm().item() > 0
    print("[+] Gradient flow test PASSED!")
