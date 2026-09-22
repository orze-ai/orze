# Real GPU research campaign — completed and verified

Sixteen episodes, four ASR corpora, two repetitions, eight shared A100 GPUs.
Both strategies accepted 5/8 final goals. Mean capped time to final acceptance:
ParallelRefine **60.38 min**, combined **73.63 min**. Combined minus control:
**+13.24 min**, descriptive problem-level 95% t interval **[-27.56, +54.04] min**.
The mean-based rule retains ParallelRefine as default; four related ASR corpora
do not establish a population optimum. See default-decision.json.

The goal is >=10% relative WER improvement with independent acceptance within
120 minutes; failures receive the entire 120-minute penalty. The shared model
is Whisper-small.en (244M), not the project's pending 1.7B acceptance test.
Model waiting, failures, training and audit are timed; shared setup is separate.

Run the public arithmetic check with Python, numpy, scipy and jiwer:

```sh
python3 rescore_public.py
```

This recomputes WER from per-utterance error counts, paired-group bootstrap,
final acceptance, problem-balanced time means, the t interval and sign flips.
It checks published arithmetic; raw-text edit distances, execution isolation,
checkpoint identity and closure were checked separately by the full verifier.

verification.json records 106 closed model calls, 120 closed GPU containers and
86 independently rescored predictions/comparisons. operational-summary.json
records failures and measured timing. Known usage estimate is $70.128305;
retained upper bound $89.686025; no new unknown-cost calls.

The private Pro repository has frozen-protocol.tar.gz and completed-records.tar.gz
with manifests. These omit credentials, audio, model weights, feature arrays
and learned checkpoints; large inputs and outputs remain at the local run path
with their recorded hashes. Full execution replay requires those local files.
No provider retries or resume into old paths were performed.

See ../../plans/2026-09-22-real-gpu-results.zh-CN.md for interpretation, limitations
and the next efficiency priorities. The original frozen protocol is plan.json.
