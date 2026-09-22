# Battlefield Radio Intelligibility Engine

Real-time on-device speech enhancement engine designed for tactical radio-degraded speech under extreme acoustic conditions (gunfire, tracked/wheeled vehicle engines, rotorcraft/wind noise), targeted for **Qualcomm Snapdragon NPU** edge deployment via **Qualcomm AI Hub**.

---

## Key Capabilities

- **Stateful Streaming Architecture**: Sub-20ms latency with causal Conv-GRU, zero lookahead, and explicit recurrent state passing.
- **Physical Battlefield Simulation**: High-SPL gunfire transients, armored vehicle diesel harmonics, NATO 300–3400 Hz bandpass filtering, Codec2 vocoder emulation, PTT non-linear clipping, and RF static/squelch tails.
- **Snapdragon NPU Optimized**: Strict static graph design verified with ONNX Runtime and Qualcomm AI Hub QNN compiler with 0 CPU fallbacks.
- **Multi-Metric Evaluation Suite**: Perceptual quality (PESQ), Speech intelligibility (STOI), and Word Error Rate (Whisper WER).

---

## Directory Structure

```
hp/
├── SPEC.md                               # Complete engineering & architectural specification
├── configs/                              # YAML configuration files
├── data/                                 # Tactical degradation simulator & dataset loaders
├── models/                               # Streaming neural models (Conv-GRU, Denoiser, Impulse, Fused)
├── deploy/                               # ONNX export, Parity validation & Qualcomm AI Hub scripts
├── eval/                                 # PESQ, STOI, Whisper WER benchmark runners
├── training/                             # Loss functions & stateful trainer
├── demo/                                 # Live streaming application demo
├── docs/                                 # Architectural diagrams & AI Hub guidelines
└── tests/                                # Comprehensive unit tests
```

---

## Quickstart

### 1. Run Unit & Smoke Tests
```bash
pytest tests/ -v
```

### 2. Export Streaming Conv-GRU to ONNX & Verify Parity
```bash
python deploy/export_onnx.py --config configs/model_convgru.yaml --output artifacts/dummy_conv_gru.onnx
python deploy/test_onnx_runtime.py --onnx artifacts/dummy_conv_gru.onnx
```

### 3. Run Battlefield Simulation Demo
```bash
python data/simulation.py
```

### 4. Run Speech Enhancement Benchmark (PESQ / STOI / WER)
```bash
python eval/evaluate.py --demo
```

### 5. Live App Demo
```bash
python demo/app.py
```
