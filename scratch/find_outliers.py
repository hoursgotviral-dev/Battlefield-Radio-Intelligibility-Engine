import json

d = json.load(open('results/day3_finetuned.json'))['per_utterance_results']
outliers = []
for r in d:
    if r['wer_branch_a'] > 1.0:
        outliers.append(r)
        print(f"[{r['id']}] SNR: {r['snr_db']:>5.2f} dB | WER Deg: {r['wer_degraded']} | WER DF: {r['wer_deepfilternet']} | WER Branch A: {r['wer_branch_a']}")
        print(f"   Ref: {r['transcript']}")
        print(f"   Hyp: {r['hypothesis_branch_a']}")
print(f"Total outliers with WER > 1.0: {len(outliers)} / {len(d)}")
