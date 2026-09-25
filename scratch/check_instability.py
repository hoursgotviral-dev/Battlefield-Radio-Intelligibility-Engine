import json
import numpy as np

d = json.load(open("results/fair_comparison_benchmark.json"))["per_utterance_results"]
trig_a = [r for r in d if r["branch_a_guard_triggered"]]
untrig_a = [r for r in d if not r["branch_a_guard_triggered"]]
trig_df = [r for r in d if r["df_guard_triggered"]]

print("=== BRANCH A GUARD TRIGGERS (13 samples) ===")
for r in trig_a:
    trigs = "; ".join(r["branch_a_triggers"])
    print(f"ID: {r['id']:<10} | SNR: {r['snr_db']:>6.2f} dB | Triggers: {trigs}")

snrs_trig = [r["snr_db"] for r in trig_a]
snrs_untrig = [r["snr_db"] for r in untrig_a]

print("\n=== SUMMARY STATISTICS ===")
print(f"Triggered SNR Range: min={min(snrs_trig):.2f} dB, max={max(snrs_trig):.2f} dB, mean={np.mean(snrs_trig):.2f} dB, median={np.median(snrs_trig):.2f} dB")
print(f"Untriggered SNR Range: min={min(snrs_untrig):.2f} dB, max={max(snrs_untrig):.2f} dB, mean={np.mean(snrs_untrig):.2f} dB, median={np.median(snrs_untrig):.2f} dB")

print(f"\nCount of Triggered with SNR < 3.0 dB: {sum(1 for s in snrs_trig if s < 3.0)} / {len(snrs_trig)} ({sum(1 for s in snrs_trig if s < 3.0)/len(snrs_trig)*100:.1f}%)")
print(f"Count of Triggered with SNR < 0.0 dB: {sum(1 for s in snrs_trig if s < 0.0)} / {len(snrs_trig)} ({sum(1 for s in snrs_trig if s < 0.0)/len(snrs_trig)*100:.1f}%)")
print(f"Total Test Set samples with SNR < 3.0 dB: {sum(1 for r in d if r['snr_db'] < 3.0)} / {len(d)}")

print("\n=== DEEPFILTERNET3 GUARD TRIGGERS (3 samples) ===")
for r in trig_df:
    trigs = "; ".join(r["df_triggers"])
    print(f"ID: {r['id']:<10} | SNR: {r['snr_db']:>6.2f} dB | Triggers: {trigs}")
