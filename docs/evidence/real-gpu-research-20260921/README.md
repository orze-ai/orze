# Real GPU research campaign — prospective protocol

Four real ASR corpora, two repetitions, two exploration policies; eight shared
A100 GPUs. The primary outcome is capped wall time to independently accepted
final delivery, including research-model latency, failures, training and audit.
This archive records the protocol at launch; it contains no result claim.

The model is Whisper-small.en (244M), not the project's 1.7B model. Each world
uses the better development result of the unadapted model and fixed LoRA100 as
the shared reference. The separate audit is never mounted with text labels in
candidate containers. Full rules, limitations, input hashes and budget are in
plan.json. Common setup is excluded from the strategy clocks and reported
separately. Existing inference services keep running.

The private Pro repository contains frozen-protocol.tar.gz with the exact
orchestrator, worker, policy, verifier, and Core/Pro source dependencies. It does
not contain credentials, audio, model weights or generated checkpoints. Local
data/model caches are required to reproduce execution; copying a protocol is
not a new spending authorization. No provider retries or resume into old paths.

See ../../plans/2026-09-22-real-gpu-research.zh-CN.md for the human-readable plan.
