# Qualcomm AI Hub Deployment Guidelines for Snapdragon NPU

## 1. Golden Rules for Qualcomm AI Hub Compilation

1. **Static Tensors Only**:
   - The Hexagon Tensor Processor (HTP) requires static memory allocation for optimal DMA transfers and hardware VTCM (Vector Tightly-Coupled Memory) buffer management.
   - Do NOT include dynamic axes (`None`, `-1`, or `?`) in ONNX exports.

2. **Explicit Recurrent State Passing**:
   - Do NOT hide recurrent state inside blackbox custom ops.
   - Expose `h_in` as an ONNX graph input and `h_out` as an ONNX graph output.
   - For consecutive streaming frames, the host application passes the tensor pointer of `h_out` from iteration $t$ directly to `h_in` of iteration $t+1$.

3. **External STFT / iSTFT**:
   - Keep complex Fourier transform kernels (`torch.stft`, `torch.istft`) in host DSP or CPU pre/post-processing libraries.
   - Feed real-valued magnitude/phase or real/imag components into the NPU graph.

4. **Supported Operators**:
   - Use standard ONNX Opset 13–17 operators (`Conv`, `Add`, `Mul`, `Sigmoid`, `Tanh`, `PRelu`, `MatMul`, `Transpose`, `Reshape`, `Concat`).
   - Avoid non-causal bidirectional recurrent cells or dynamic slicing.

## 2. Compilation & Profiling Commands

Export ONNX model:
```bash
python deploy/export_onnx.py --config configs/model_convgru.yaml --output artifacts/dummy_conv_gru.onnx
```

Verify numeric parity across streaming chunks:
```bash
python deploy/test_onnx_runtime.py --onnx artifacts/dummy_conv_gru.onnx
```

Submit compilation and profiling job to Qualcomm AI Hub:
```bash
python deploy/aihub_compile_profile.py --onnx artifacts/branch_a_denoiser.onnx --device "Snapdragon X Elite"
```
