import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import torch
import onnx
import onnxruntime as ort
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fused_model import FusedBattlefieldModel

def test_fused_onnx():
    print("=" * 80)
    print(" FUSED BATTLEFIELD MODEL: ONNX EXPORT & NUMERICAL PARITY TEST")
    print("=" * 80)

    device = torch.device("cpu")
    model = FusedBattlefieldModel(
        freq_bins=257,
        denoiser_channels=32,
        denoiser_hidden_dim=64,
        denoiser_layers=2,
        impulse_channels=24,
        impulse_hidden_dim=48,
        context_dim=32,
    ).to(device)

    model.load_pretrained_branches(
        branch_a_path="checkpoints/branch_a_curriculum_best.pt",
        branch_b_path="checkpoints/branch_b_best.pt",
        device=device,
    )
    model.eval()

    # Static shapes for streaming inference: 1 batch, 1 channel, 257 bins, 4 frames (32ms chunk)
    B = 1
    F = 257
    T_c = 4
    dummy_input = torch.randn(B, 1, F, T_c, dtype=torch.float32)
    dummy_h_a, dummy_state_b = model.init_hidden_states(batch_size=B, device=device)

    print(f"[*] Input shape: {dummy_input.shape}")
    print(f"[*] State A shape: {dummy_h_a.shape}, State B shape: {dummy_state_b.shape}")

    # 1. PyTorch Forward Pass
    print("\n1. Testing PyTorch Native Forward Pass...")
    with torch.no_grad():
        enh_chunk, fused_mask, h_a_out, state_b_out, est_snr = model(dummy_input, dummy_h_a, dummy_state_b)
    print(f"[+] PyTorch Forward Succeeded!")
    print(f"    - enh_chunk:   {enh_chunk.shape}")
    print(f"    - fused_mask:  {fused_mask.shape}")
    print(f"    - h_a_out:     {h_a_out.shape}")
    print(f"    - state_b_out: {state_b_out.shape}")
    print(f"    - est_snr:     {est_snr.shape}")

    # 2. ONNX Export
    out_onnx = "checkpoints/fused_battlefield_engine.onnx"
    Path("checkpoints").mkdir(exist_ok=True)
    print(f"\n2. Exporting to ONNX (Opset 17, static shapes, no dynamic axes)...")

    try:
        torch.onnx.export(
            model,
            (dummy_input, dummy_h_a, dummy_state_b),
            out_onnx,
            opset_version=17,
            input_names=["input_chunk", "h_a_in", "state_b_in"],
            output_names=["enhanced_chunk", "fused_mask", "h_a_out", "state_b_out", "est_snr"],
            do_constant_folding=True,
        )
        print(f"[+] Exported successfully to {out_onnx}!")
    except Exception as e:
        print(f"[-] ONNX Export Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. onnx.checker validation
    print("\n3. Validating with onnx.checker...")
    try:
        onnx_model = onnx.load(out_onnx)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX Checker PASSED with 0 errors!")
    except Exception as e:
        print(f"[-] ONNX Checker Failed: {e}")
        return False

    # 4. ONNXRuntime Parity Test
    print("\n4. Testing ONNXRuntime InferenceSession & Parity...")
    try:
        session = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "input_chunk": dummy_input.numpy(),
            "h_a_in": dummy_h_a.numpy(),
            "state_b_in": dummy_state_b.numpy(),
        }
        ort_outs = session.run(None, ort_inputs)
        
        diff_enh = np.max(np.abs(enh_chunk.numpy() - ort_outs[0]))
        diff_mask = np.max(np.abs(fused_mask.numpy() - ort_outs[1]))
        diff_ha = np.max(np.abs(h_a_out.numpy() - ort_outs[2]))

        print(f"[+] ONNXRuntime Session Execution PASSED!")
        print(f"    - Parity Max Diff (enhanced_chunk): {diff_enh:.6e}")
        print(f"    - Parity Max Diff (fused_mask):     {diff_mask:.6e}")
        print(f"    - Parity Max Diff (h_a_out):        {diff_ha:.6e}")

        assert diff_enh < 1e-5, f"Parity diff {diff_enh} too high"
        print("\n[+] FULL FUSED MODEL ONNX EXPORT & PARITY PASSED WITH 0 ERRORS!")
        return True
    except Exception as e:
        print(f"[-] ONNXRuntime Session Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fused_onnx()
