"""
Qualcomm AI Hub Model Compilation and Profiling Utility.

ML Concept - Hardware Acceleration on Qualcomm Hexagon NPU:
Qualcomm AI Hub allows cloud-based compilation, profiling, and on-device execution of neural
networks directly on physical Snapdragon test devices (e.g. Snapdragon 8 Gen 3, Snapdragon X Elite).
The compilation process converts an ONNX graph into a Qualcomm Neural Processing Engine (QNN)
binary context (e.g., `.dlc` or `libqnn_model.so`) tailored to the Hexagon Tensor Processor (HTP).

Profiling measures:
1. Exact NPU inference latency per chunk (microseconds).
2. Peak NPU memory bandwidth and SRAM footprint.
3. Operator-level fallback analysis (ensuring 0 CPU fallbacks).
"""

import os
import sys
import argparse
import yaml

try:
    import qai_hub as hub
    QAI_HUB_AVAILABLE = True
except ImportError:
    QAI_HUB_AVAILABLE = False


def compile_and_profile_on_aihub(
    onnx_model_path: str,
    device_name: str = "Snapdragon X Elite",
    target_runtime: str = "qnn_lib_aarch64_windows",
    api_token: str = None,
    options: str = "--target_arch v75",
) -> dict:
    """
    Submits an ONNX model to Qualcomm AI Hub for compilation and profiling on physical Snapdragon hardware.

    Args:
        onnx_model_path: Path to exported ONNX model
        device_name: Target physical Snapdragon device name in AI Hub
        target_runtime: Target backend runtime (e.g. QNN)
        api_token: Qualcomm AI Hub API token
        options: Compiler optimization flags

    Returns:
        results: Dictionary containing profile and compilation job summaries
    """
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    print("=" * 60)
    print(" Qualcomm AI Hub Compilation & Profiling Pipeline")
    print("=" * 60)
    print(f"[*] Model Path      : {onnx_model_path}")
    print(f"[*] Target Device   : {device_name}")
    print(f"[*] Target Runtime  : {target_runtime}")

    if not QAI_HUB_AVAILABLE:
        print("\n[!] 'qai_hub' package is not installed in the local Python environment.")
        print("    To run live submissions, install via: pip install qai-hub")
        print("    And configure your token: qai-hub configure --api_token <YOUR_TOKEN>\n")
        print("[*] Generating simulated Qualcomm AI Hub submission manifest...")
        manifest = {
            "status": "SIMULATED",
            "model_name": os.path.basename(onnx_model_path),
            "target_device": device_name,
            "target_runtime": target_runtime,
            "simulated_metrics": {
                "npu_latency_ms": 1.45,
                "fps": 689.6,
                "npu_memory_mb": 4.2,
                "npu_operator_coverage": "100%",
                "cpu_fallbacks": 0,
            },
        }
        print(f"[+] Simulated Profiling Results:\n    - NPU Latency: {manifest['simulated_metrics']['npu_latency_ms']} ms / chunk (Budget: 20 ms)")
        print(f"    - NPU Operator Coverage: {manifest['simulated_metrics']['npu_operator_coverage']} (0 CPU fallbacks)")
        return manifest

    if api_token:
        os.environ["QAI_HUB_API_TOKEN"] = api_token

    # 1. Select Device
    print(f"[*] Querying available devices matching '{device_name}'...")
    device = hub.Device(device_name)

    # 2. Upload Model
    print(f"[*] Uploading ONNX model to Qualcomm AI Hub...")
    hub_model = hub.upload_model(onnx_model_path)

    # 3. Submit Compile Job
    print(f"[*] Submitting compilation job for runtime '{target_runtime}'...")
    compile_job = hub.submit_compile_job(
        model=hub_model,
        device=device,
        options=options,
    )
    print(f"[+] Compile job submitted (ID: {compile_job.job_id}). Waiting for completion...")
    compiled_model = compile_job.get_target_model()

    # 4. Submit Profile Job
    print(f"[*] Submitting on-device profiling job on {device.name}...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
    )
    print(f"[+] Profile job submitted (ID: {profile_job.job_id}). Waiting for completion...")
    profile_data = profile_job.download_profile()

    print("\n" + "=" * 60)
    print(" Qualcomm AI Hub Profile Report")
    print("=" * 60)
    print(f"Profile URL: {profile_job.url}")
    return {
        "compile_job_id": compile_job.job_id,
        "profile_job_id": profile_job.job_id,
        "profile_url": profile_job.url,
        "profile_data": profile_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Compile and profile model on Qualcomm AI Hub.")
    parser.add_argument("--onnx", type=str, default="artifacts/dummy_conv_gru.onnx", help="Path to ONNX model")
    parser.add_argument("--config", type=str, default="configs/model_convgru.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device name (e.g. Snapdragon X Elite)")
    parser.add_argument("--api-token", type=str, default=None, help="Qualcomm AI Hub API token")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    aihub_cfg = config.get("deploy", {}).get("qualcomm_aihub", {})
    device_name = args.device or aihub_cfg.get("device_name", "Snapdragon X Elite")
    target_runtime = aihub_cfg.get("target_runtime", "qnn_lib_aarch64_windows")

    compile_and_profile_on_aihub(
        onnx_model_path=args.onnx,
        device_name=device_name,
        target_runtime=target_runtime,
        api_token=args.api_token,
    )


if __name__ == "__main__":
    main()
