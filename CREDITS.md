# Credits & Third-Party Attributions

This project builds upon and benchmarks against exceptional open-source tools and datasets in the speech processing and edge AI communities:

## 1. Baseline Speech Enhancement Models & Tools
- **DeepFilterNet3**: R. Schröter et al., *DeepFilterNet: A Low-Complexity Speech Enhancement Framework for Full-Band Audio*. Used as an offline baseline benchmark comparison. [GitHub](https://github.com/Rikorose/DeepFilterNet)
- **OpenAI Whisper (`openai/whisper-tiny.en`)**: Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*. Used as an objective frozen downstream Word Error Rate (WER) intelligibility evaluator. [GitHub](https://github.com/openai/whisper)

## 2. Evaluation Metrics & DSP Libraries
- **PESQ (Perceptual Evaluation of Speech Quality)**: ITU-T P.862 / `pesq` python package by vBaiCai.
- **pystoi (Short-Time Objective Intelligibility)**: C. H. Taal et al. / `pystoi` python package by Manuel Pariente.
- **PyTorch, soundfile & SciPy**: Audio I/O, spectrogram STFT transformations, and digital signal processing.
- **librosa**: Audio feature extraction and signal analysis utilities.
- **jiwer**: Levenshtein distance calculation for standardized Word Error Rate (WER).

## 3. Deployment & Acceleration Stack
- **Qualcomm AI Hub**: Cloud-hosted compilation and physical hardware profiling on Snapdragon X Elite Hexagon NPU.
- **ONNX Runtime (Microsoft)**: Open Neural Network Exchange runtime for cross-platform model validation and parity testing.

## 4. Datasets
- **LibriSpeech ASR Corpus**: Vassil Panayotov et al., used for clean speech training and test partitions.
- **Freefield1010 & ESC-50**: Acoustic environment samples used to synthesize combat audio, engine noise, and blast impulses.
