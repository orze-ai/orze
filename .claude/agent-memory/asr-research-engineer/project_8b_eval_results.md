---
name: 8B Model Evaluation Results
description: Full baseline and LoRA eval results for HiggsAudio3 8B on the Open ASR Leaderboard benchmarks
type: project
---

## 8B Baseline (Full Dataset)
- LS-clean: 1.29%, LS-other: 2.42%, TED: 3.46%, VP: 7.08%, E22: 10.75%, AMI: 15.42%, GS: ~9.0%, SPG: ~3.05%
- Avg: ~6.56%

## LoRA v2 (AMI+VP 5K each, rank 16, lr 5e-5, 1 epoch)
Full dataset results:
- LS-clean: 1.29%, LS-other: 2.47%, TED: 2.75%, VP: 5.78%, E22: 11.21%, AMI: ~9.0% (still running)
- E22 regression: +0.46%, SPG regression: ~+0.95%
- Estimated avg: ~5.69%

## LoRA v4 (AMI-only 10K, rank 16, lr 2e-5, 1 epoch)
- AMI: 6.58% at 1000 samples (better than v2)
- E22: 9.74% at 500 samples (similar pattern to v2)
- SPG: 3.83% at 250 samples (less regression than v2)

## Key Insights
- ESB datasets sorted by audio length (longest first). 500-sample estimates are optimistic.
- LoRA helps AMI massively (-6 to -9%) but can regress E22 (+0.5%) and SPG (+1%)
- Thinking mode is essential for 8B (no-thinking = catastrophic, +20-30% WER)
- Target: 5.42% avg (Cohere #1)

**Why:** Understanding per-dataset dynamics is key to beating the leaderboard
**How to apply:** When planning training, account for the E22/SPG regression; AMI-focused training has the most avg WER leverage
