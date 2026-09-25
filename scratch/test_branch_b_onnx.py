import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.branch_b_impulse import BranchBImpulse

def test_branch_b_onnx():
    print("=" * 80)
    print(" BRANCH B: ONNX EXPORT & EDGE DEPLOYABILITY TEST")
    print("=" * 80)

    device = torch.device("cpu")
    model = BranchBImpulse(freq_bins=257, channels=24, hidden_dim=48).to(device)
    model.eval()

    # Fixed shapes for streaming inference: 1 batch, 1 channel, 257 bins, 4 frames (32ms chunk)
    B = 1
    F = 257
    T_c = 4
    dummy_x = torch.randn(B, 1, F, T_c, dtype=torch.float32)
    dummy_state = model.init_hidden_state(batch_size=B, device=device)

    out_onnx = "checkpoints/branch_b_impulse.onnx"
    Path("checkpoints").mkdir(exist_ok=True)

    print("[*] Testing PyTorch forward pass...")
    with torch.no_grad():
        mask, prob, next_state = model(dummy_x, dummy_state)
    print(f"[+] Forward pass passed! mask: {mask.shape}, prob: {prob.shape}, next_state: {next_state.shape}")

    print("[*] Exporting Branch B to ONNX (Opset 17, static shapes)...")
    torch.onnx.export(
        model,
        (dummy_x, dummy_state),
        out_onnx,
        opset_version=17,
        input_names=["input_spec", "state_in"],
        output_names=["impulse_mask", "transient_prob", "state_out"],
        do_constant_folding=True,
    )
    print(f"[+] Exported successfully to {out_onnx}!")

    print("[*] Validating with onnx.checker...")
    onnx_model = onnx.load(out_onnx)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model checker PASSED!")

    print("[*] Validating numerical parity with ONNXRuntime...")
    session = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_spec": dummy_x.numpy(),
        "state_in": dummy_state.numpy(),
    }
    ort_outs = session.run(None, ort_inputs)
    diff = np.max(np.abs(mask.numpy() - ort_outs[0]))
    print(f"[+] ONNXRuntime Parity Max Diff: {diff:.6e}")
    assert diff < 1e-5, f"Parity diff {diff} exceeds threshold"
    print("[+] Branch B ONNX Deployability Test PASSED WITH 0 ERRORS!")

if __name__ == "__main__":
    test_branch_b_onnx()
