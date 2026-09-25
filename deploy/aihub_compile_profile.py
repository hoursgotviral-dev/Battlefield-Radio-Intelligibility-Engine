"""
Qualcomm AI Hub Model Compilation and Profiling Pipeline.

Submits Branch A ONNX graph to Qualcomm AI Hub for live compilation and profiling
on physical Snapdragon X Elite hardware (Windows ARM64).
"""

import os
import sys
import argparse
import yaml
import json

# Ensure UTF-8 stdout/stderr on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import qai_hub as hub


def compile_and_profile_on_aihub(
    onnx_model_path: str,
    device_name: str = "Snapdragon X Elite CRD",
    target_runtime: str = "qnn_lib_aarch64_windows",
    api_token: str = None,
    options: str = "--target_arch v75",
) -> dict:
    """
    Submits an ONNX model to Qualcomm AI Hub for compilation and profiling on physical Snapdragon hardware.
    """
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    if api_token:
        os.environ["QAI_HUB_API_TOKEN"] = api_token

    print("=" * 80)
    print(" Qualcomm AI Hub Live Compilation & Profiling Pipeline")
    print("=" * 80)
    print(f"[*] Model Path      : {onnx_model_path}")
    print(f"[*] Target Device   : {device_name}")
    print(f"[*] Target Runtime  : {target_runtime}")
    print(f"[*] Compiler Options: {options}")

    # 1. Select Device
    print(f"\n[1/4] Querying available devices matching '{device_name}'...")
    devices = hub.get_devices(name=device_name)
    if not devices:
        print(f"[!] Warning: Exact device name '{device_name}' not found. Searching for 'Snapdragon X Elite'...")
        devices = hub.get_devices(name="Snapdragon X Elite")
    
    if not devices:
        raise ValueError(f"No Qualcomm AI Hub devices found matching '{device_name}'.")
    
    device = devices[0]
    print(f"[+] Selected Target Device: {device.name} (OS: {device.os})")

    # 2. Upload Model
    print(f"\n[2/4] Uploading ONNX model to Qualcomm AI Hub...")
    hub_model = hub.upload_model(onnx_model_path)
    print(f"[+] Model uploaded successfully (Model ID: {hub_model.model_id})")

    # 3. Submit Compile Job
    print(f"\n[3/4] Submitting compilation job for runtime '{target_runtime}'...")
    try:
        compile_job = hub.submit_compile_job(
            model=hub_model,
            device=device,
            options=options,
        )
        print(f"[+] Compile job submitted: ID={compile_job.job_id}")
        print(f"    Dashboard URL: {compile_job.url}")
        print("    Waiting for Qualcomm AI Hub compilation to complete...")
        compiled_model = compile_job.get_target_model()
        print(f"[+] Compilation succeeded!")
    except Exception as e:
        print(f"[!] Compilation with options '{options}' failed: {e}")
        print("[*] Retrying compile job with default optimization options...")
        compile_job = hub.submit_compile_job(
            model=hub_model,
            device=device,
        )
        print(f"[+] Compile job submitted: ID={compile_job.job_id}")
        print(f"    Dashboard URL: {compile_job.url}")
        compiled_model = compile_job.get_target_model()
        print(f"[+] Compilation succeeded!")

    # 4. Submit Profile Job on Physical Snapdragon Hardware
    print(f"\n[4/4] Submitting on-device profiling job on {device.name}...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
    )
    print(f"[+] Profile job submitted: ID={profile_job.job_id}")
    print(f"    Dashboard URL: {profile_job.url}")
    print("    Waiting for physical device profiling to complete...")
    
    profile_data = profile_job.download_profile()
    
    print("\n" + "=" * 80)
    print(" QUALCOMM AI HUB ON-DEVICE PROFILING REPORT")
    print("=" * 80)
    print(f"Device Name      : {device.name}")
    print(f"Profile URL      : {profile_job.url}")
    
    # Save results to results/
    os.makedirs("results", exist_ok=True)
    report_path = "results/qai_hub_branch_a_profile.json"
    
    results = {
        "device": device.name,
        "device_os": device.os,
        "compile_job_id": compile_job.job_id,
        "compile_url": compile_job.url,
        "profile_job_id": profile_job.job_id,
        "profile_url": profile_job.url,
        "raw_profile_data": str(profile_data),
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Full Profile Report saved to: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compile and profile model on Qualcomm AI Hub.")
    parser.add_argument("--onnx", type=str, default="artifacts/branch_a_denoiser.onnx", help="Path to ONNX model")
    parser.add_argument("--device", type=str, default="Snapdragon X Elite CRD", help="Device name")
    parser.add_argument("--api-token", type=str, default=None, help="Qualcomm AI Hub API token")
    args = parser.parse_args()

    compile_and_profile_on_aihub(
        onnx_model_path=args.onnx,
        device_name=args.device,
        api_token=args.api_token,
    )


if __name__ == "__main__":
    main()
