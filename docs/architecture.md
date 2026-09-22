# System Architecture & Signal Flow

## 1. Tactical Audio Signal Pipeline

The Battlefield Radio Intelligibility Engine processes tactical audio through a low-latency, streaming pipeline:

```mermaid
flowchart TD
    A[16 kHz Raw Audio Stream] --> B[External STFT Analysis Window]
    B --> C[Spectral Magnitude Chunk: 257 x 4]
    C --> D[Snapdragon NPU / Hexagon DSP]
    
    subgraph NPU Execution Graph
        D --> E[Branch A: Continuous Denoiser]
        D --> F[Branch B: Impulse Suppressor]
        D --> G[Context Encoder]
        
        H[(Recurrent State h_in)] --> E
        E --> I[(Recurrent State h_out)]
        
        E --> J[Fusion Combiner]
        F --> J
        G --> J
        J --> K[Enhanced Gain Mask: 257 x 4]
        C --> L[Spectral Masking Multiplication]
        K --> L
    end
    
    L --> M[Enhanced Spectral Magnitude]
    M --> N[External iSTFT Synthesis + Overlap-Add]
    N --> O[16 kHz Enhanced Audio Stream]
```

## 2. Stateful Streaming Chunk Contract

| Tensor Port | Dimension | Description |
| :--- | :--- | :--- |
| `input_chunk` | `[1, 1, 257, 4]` | Static 4-frame spectral magnitude chunk |
| `h_in` | `[2, 1, 64]` | Multi-layer GRU recurrent hidden state |
| `enhanced_chunk` | `[1, 1, 257, 4]` | Filtered spectral magnitude |
| `h_out` | `[2, 1, 64]` | Updated hidden state passed to next chunk |

## 3. Subsystem Breakdown

1. **Branch A (Continuous Denoiser)**: Causal 2D convolution and stateful 2-layer GRU for tracking low-frequency diesel engine harmonics, hull vibrations, and aerodynamic wind buffeting.
2. **Branch B (Impulse Suppressor)**: Temporal Gated Linear Unit (GLU) with microsecond transient detection to gate gunfire shockwaves and mortar explosions.
3. **Context Encoder**: Estimates current signal-to-noise ratio (SNR) and acoustic regime to adjust fusion weights.
4. **Qualcomm AI Hub Target**: Compiled to QNN context binary targeting Snapdragon Hexagon NPU with 0 CPU fallbacks.
