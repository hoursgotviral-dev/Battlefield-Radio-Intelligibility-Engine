# Battlefield Radio Intelligibility Engine — Specification (SPEC.md)

## 1. System Overview & Problem Formulation

### 1.1 Goal
The **Battlefield Radio Intelligibility Engine** is an on-device, real-time speech enhancement system designed to restore voice intelligibility under extreme acoustic and RF degradations encountered in tactical environments. It targets edge deployment on **Qualcomm Snapdragon NPUs** (Hexagon Tensor Processor) via **Qualcomm AI Hub**.

### 1.2 Operational Environment & Degradation Taxonomy
In tactical battlefield communications, speech signals suffer from compound additive and channel degradations:
1. **High-SPL Impulse Noise**: Gunfire (small arms muzzle blast, supersonic bullet crack), artillery, mortar fire, and explosive transients (sharp crest factor > 25 dB, microsecond risetimes).
2. **Heavy Continuous Acoustic Noise**: Armored vehicle diesel engines, tracked vehicle hull resonance, turbofan/rotorcraft cabin noise, high-velocity wind turbulence.
3. **RF Channel & Tactical Vocoder Distortions**:
   - Bandpass limiting (Tactical narrowband NATO 300 Hz – 3400 Hz).
   - Low-bitrate vocoding artifacts (e.g., Codec2 at 1200 / 2400 bps, MELP, CVSD) with frame drop and packet loss.
   - Non-linear radio clipping and RF front-end preamplifier overdrive (push-to-talk microphone saturation).
   - Burst static, atmospheric thermal noise, and squelch tail transients.

---

## 2. Signal Processing & Streaming Constraints

### 2.1 Audio Signal Format
- **Sampling Rate ($f_s$)**: 16,000 Hz (16 kHz mono, PCM 16-bit / Float32).
- **Audio Framing & Chunking**:
  - Analysis Window Size: $N_{\text{FFT}} = 512$ samples (32 ms).
  - Hop Size / Stride ($R$): 128 samples (8 ms) or 256 samples (16 ms).
  - Streaming Chunk Size ($C$): Fixed at $T_c = 4$ frames ($4 \times 128 = 512$ samples, 32 ms) or $T_c = 2$ frames.
  - Spectrogram Frequency Bins: $F = N_{\text{FFT}}/2 + 1 = 257$ bins.

### 2.2 Algorithmic Latency Budget
- Maximum permissible algorithmic latency: **$\le 20\text{ ms}$**.
- Zero look-ahead (strictly causal convolutions and recurrent operations).
- STFT and iSTFT transforms are performed in native pre/post-processing **outside** the neural network execution graph on host CPU/DSP to minimize NPU operator overhead and maximize Hexagon NPU pipeline efficiency.

---

## 3. Neural Network Architecture

The architecture adopts a multi-branch stateful neural network:

```
                  [ Streaming STFT Magnitude/Complex Feature: (B, 257, T_c) ]
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
      ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
      │  Branch A:      │          │  Branch B:      │          │ Context Encoder │
      │  Continuous     │          │  Impulse Noise  │          │ (SNR / Noise    │
      │  Denoiser       │          │  Suppressor     │          │  Classification)│
      │ (Causal ConvGRU)│          │ (Transient Attn)│          │                 │
      └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
               │ (State A)                  │ (State B)                  │ (Context Emb)
               └────────────────────┬───────┴────────────────────────────┘
                                    ▼
                         ┌────────────────────┐
                         │  Fusion Engine     │
                         │ (Complex Masking/  │
                         │  Spectral Routing) │
                         └──────────┬─────────┘
                                    │ (State F)
                                    ▼
                 [ Enhanced Spectral Output: (B, 257, T_c) ]
```

### 3.1 Branch A: Continuous Noise Denoiser
- **Objective**: Suppress stationary and quasi-stationary engine rumble, rotor chop, and wind roar.
- **Topology**: Causal 2D/1D Depthwise-Separable Convolutions followed by a multi-layer streaming Gated Recurrent Unit (GRU).
- **State Management**: Accepts recurrent hidden state $h_A \in \mathbb{R}^{L_A \times B \times D_A}$; outputs updated state $h'_A$.

### 3.2 Branch B: Impulse Noise Suppressor
- **Objective**: Detect and attenuate high-energy acoustic shockwaves (gunshots, explosions) without causing voice distortion or musical noise.
- **Topology**: Causal temporal gated linear units with rapid energy envelope tracking and transient gating.

### 3.3 Context Encoder
- **Objective**: Estimate background acoustic regime (e.g., in-cockpit, urban gunfire, open terrain) and dynamic SNR level to adaptively steer fusion gains.

### 3.4 Fusion Engine
- **Objective**: Combine intermediate representations from Branch A and Branch B, conditioned on the context embedding, predicting a bounded complex spectral mask (cIRM) or real spectral gain mask $M \in [0, 1]^{B \times 257 \times T_c}$.

---

## 4. Qualcomm AI Hub & NPU Deployment Constraints

To ensure zero-error compilation on Qualcomm Hexagon NPU via Qualcomm AI Hub:

1. **Static Tensor Dimensions**:
   - All input/output tensors MUST have static, predetermined shapes (no dynamic `None` dimensions for batch or time during ONNX export).
   - Input shape: `(1, 257, T_c)` for spectrogram chunks.
   - Recurrent states: `(L, 1, D)` with fixed layer count $L$ and hidden size $D$.
2. **Explicit Recurrent State I/O**:
   - PyTorch `torch.nn.GRU` or custom cell states must be unrolled or cleanly passed as top-level graph inputs (`h_in`) and returned as graph outputs (`h_out`).
3. **Supported Operator Set**:
   - Restrict to ONNX Opset 13–17 standard operators: `Conv`, `ConvTranspose`, `Add`, `Mul`, `Sigmoid`, `Tanh`, `Relu`, `PRelu`, `MatMul`, `Reshape`, `Transpose`, `Split`, `Concat`.
   - Avoid non-causal operations (`Bidirectional GRU`), dynamic indexing, dynamic slicing, or tensor-dependent control flow.
4. **Quantization Target**:
   - FP16 baseline on Hexagon NPU; INT8 post-training quantization (PTQ) or quantization-aware training (QAT) with symmetric per-channel weights and asymmetric per-tensor activations.

---

## 5. Battlefield Degradation Simulation Pipeline

The data synthesis pipeline mathematically models physical tactical degradation stages:

$$\mathbf{y}(t) = \mathcal{G}_{\text{radio}}\left( \text{Codec2}\left( \text{BPF}\left( \mathbf{s}(t) + \sum_{k} \alpha_k \mathbf{n}_k(t) \right) \right) \right) + \mathbf{n}_{\text{static}}(t)$$

1. **Speech + Noise Mixing**:
   - Clean speech $\mathbf{s}(t)$ mixed with noise sources (gunfire, tank/APC diesel, helicopter rotor, wind).
   - SNR sampled uniformly from $\mathcal{U}(-15\text{ dB}, +20\text{ dB})$.
2. **Military Radio Bandpass Filter (BPF)**:
   - 4th-order Butterworth / Chebyshev bandpass filter ($f_{\text{low}} = 300\text{ Hz}$, $f_{\text{high}} = 3400\text{ Hz}$).
3. **Tactical Vocoder Simulation**:
   - Low-bitrate vocoding (Codec2 1200 / 2400 bps emulation) including randomized packet loss and pitch quantization.
4. **Nonlinear Overdrive & Radio Clipping**:
   - Push-To-Talk preamp saturation: $\tilde{x} = \tanh(\beta x)$ where $\beta \in [1.0, 5.0]$.
5. **Static & Squelch Injection**:
   - Burst static, atmospheric thermal noise, and squelch tail noise floor.

---

## 6. Evaluation Protocol & Target Metrics

| Metric | Target (Severe Degradation: SNR < 0 dB) | Target (Moderate Degradation: SNR 0 to 10 dB) |
| :--- | :--- | :--- |
| **PESQ-NB / WB** | $\ge 2.2$ (vs $< 1.3$ degraded) | $\ge 2.8$ (vs $< 1.8$ degraded) |
| **STOI / ESTOI** | $\ge 0.75$ (vs $< 0.45$ degraded) | $\ge 0.88$ (vs $< 0.65$ degraded) |
| **Whisper WER** | $\le 25\%$ (vs $> 65\%$ degraded) | $\le 12\%$ (vs $> 35\%$ degraded) |
| **Inference Latency** | $< 4\text{ ms}$ per 32 ms chunk on Snapdragon NPU ($>8\times$ real-time factor) |

---

## 7. Repository Directory Structure

```
hp/
├── SPEC.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── simulation.yaml
│   └── model_convgru.yaml
├── data/
│   ├── __init__.py
│   ├── simulation.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── dummy_conv_gru.py
│   ├── branch_a_denoiser.py
│   ├── branch_b_impulse.py
│   ├── context_encoder.py
│   └── fused_model.py
├── training/
│   ├── __init__.py
│   ├── losses.py
│   └── trainer.py
├── eval/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluate.py
├── deploy/
│   ├── __init__.py
│   ├── export_onnx.py
│   ├── test_onnx_runtime.py
│   └── aihub_compile_profile.py
├── demo/
│   ├── __init__.py
│   └── app.py
├── docs/
│   ├── architecture.md
│   └── aihub_guidelines.md
└── tests/
    ├── __init__.py
    ├── test_dummy_model.py
    ├── test_export.py
    ├── test_simulation.py
    └── test_eval.py
```
