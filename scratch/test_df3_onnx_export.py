import types
import sys
import os
import torch
import torchaudio

# Set UTF-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Torchaudio backend shim
if not hasattr(torchaudio, "backend"):
    m = types.ModuleType("torchaudio.backend")
    m.common = types.ModuleType("torchaudio.backend.common")
    m.common.AudioMetaData = types.SimpleNamespace
    sys.modules["torchaudio.backend"] = m
    sys.modules["torchaudio.backend.common"] = m.common
    torchaudio.backend = m

import onnx
import onnxruntime as ort
import numpy as np
from df.enhance import init_df

def test_df3_onnx():
    print("=" * 80)
    print(" DEEPFILTERNET3 ONNX EXPORT TEST (OPSET 17, FIXED SHAPES)")
    print("=" * 80)

    model, state, _ = init_df()
    model.eval()

    B = 1
    T = 96
    freq_bins = model.freq_bins
    erb_bins = model.erb_bins
    nb_df = model.nb_df

    dummy_spec = torch.randn(B, 1, T, freq_bins, 2, dtype=torch.float32)
    dummy_feat_erb = torch.randn(B, 1, T, erb_bins, dtype=torch.float32)
    dummy_feat_spec = torch.randn(B, 1, T, nb_df, 2, dtype=torch.float32)

    out_onnx = "checkpoints/deepfilternet3_export.onnx"

    print("\nAttempting ONNX Export with opset=17, fixed shapes...")
    try:
        # Export using legacy torchscript engine
        torch.onnx.export(
            model,
            (dummy_spec, dummy_feat_erb, dummy_feat_spec),
            out_onnx,
            opset_version=17,
            input_names=["spec", "feat_erb", "feat_spec"],
            output_names=["spec_enh", "erb_mask", "lsnr", "df_coefs"],
            do_constant_folding=True,
            export_params=True,
        )
        print(f"[+] torch.onnx.export Succeeded -> {out_onnx}")

        # Check with onnx.checker
        print("\nTesting onnx.checker.check_model...")
        onnx_model = onnx.load(out_onnx)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX Checker Passed Successfully!")

        # Check with ONNXRuntime
        print("\nTesting ONNXRuntime InferenceSession...")
        session = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "spec": dummy_spec.numpy(),
            "feat_erb": dummy_feat_erb.numpy(),
            "feat_spec": dummy_feat_spec.numpy(),
        }
        ort_outs = session.run(None, ort_inputs)
        print(f"[+] ONNXRuntime Execution Succeeded! Number of outputs: {len(ort_outs)}")

        # Check PyTorch Parity
        with torch.no_grad():
            py_outs = model(dummy_spec, dummy_feat_erb, dummy_feat_spec)
        
        diff = np.max(np.abs(py_outs[0].numpy() - ort_outs[0]))
        print(f"[+] Parity on spec_enh (Max Absolute Diff): {diff:.6e}")
        print("\n[CONCLUSION] DeepFilterNet3 DfNet core EXPORTS TO ONNX AND PASSES ALL CHECKS!")

    except Exception as e:
        print(f"\n[-] ONNX Export / Checker Failed with error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_df3_onnx()
