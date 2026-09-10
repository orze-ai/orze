# Explicit evaluation-only retry

Use `orze retry-eval IDEA_ID -c /path/to/orze.yaml` after addressing a failed
evaluator. The command returns JSON and admits work; it does not run a GPU,
start training, install Pro, contact a provider, or imply that evaluation
succeeded. The normal controller consumes the durable admission, including
after restart with an empty ideas inbox and empty in-memory pending list.

Only agreed `FAILED / training COMPLETE / evaluation FAILED` state may become
`IN_PROGRESS / training COMPLETE / evaluation PENDING`. The existing training
retry, reset and checkpoint-resume commands retain their different meanings.
Do not call ordinary `FAILED -> QUEUED` to retry evaluation: it resets both
stages and may lead to another training run.

## Preservation and recovery

Before changing lifecycle, the coordinator holds the project SQLite write lock
and prepares `_evaluation_retries/<failed-transition-id>/manifest.json`. The
manifest is published completely before any move and records paths, actions,
content hashes, declared execution policy and completed-generation inputs.

Only the independent designated evaluation output, benchmark receipt and eval
log may move into this archive. Current benchmark provenance and shared failure
analysis are copied, not removed. Training metrics, checkpoint, claim, training
configuration/logs, lineage, compute receipts, audits, sealed/bundle files and
exposure ledgers are protected. Unknown metric-source files are not assumed to
be owned evaluator outputs. Redirected, hard-linked, escaping or conflicting
paths reject admission before file operations.

If file preparation or the database commit fails, the database stays failed;
already archived evidence stays recoverable. Reissue the same command after
resolving the storage error to resume that immutable manifest. It never
overwrites a conflicting archive or new source file. A changed policy, training
input or damaged manifest is an explicit rejection, not permission to clean
files or start again. Existing compatible authority is opened mode=rw; a typo
cannot create a new empty project database, and this command performs no schema
bootstrap or migration.

Duplicate pending requests create neither another archive nor another lifecycle
edge. The next launch rechecks the prepared archive and input/policy identity.
Python, declared environment/arguments and checkpoint changes cannot silently
reuse old admission. Full checkpoint hashing has a real cost; this is not a
claim of zero-overhead verification or a snapshot of every ambient dependency.

## Benchmark and legacy boundaries

Admission itself does not reserve or refund benchmark exposure. Old provenance
remains in place to keep its ledger reference live until the next prepare
validates history, reserves a new look and actually publishes new provenance.
Old receipt/nonce replay cannot authorize the new evaluation. Exhausted or
corrupt history is not repaired or reset by retry. Silent provenance publication
failure prevents launch, while the already reserved look remains counted.

Preparation still precedes GPU availability/Popen in the legacy launcher.
Launch failures may therefore consume looks; automatic preflight backoff and
reservation/launch ordering are subsequent work, not claimed solved here.

`eval_output=metrics.json` and `./metrics.json` remain in place and still reach
actual evaluation when pending. The framework never erases that completed
training document to force a rerun. An external legacy in-place evaluator can
itself modify it; this feature does not create an immutable generation store.

Other limits remain explicit: no distributed attempt fencing or atomic
filesystem-plus-database transaction; uncertain active process receipts reject
readmission; legacy failures without process receipts still depend on lifecycle
authority. The scheduler scan is bounded; broader queue fairness/backpressure
and cleanup policy belong to the subsequent scheduler work. Tests use real
temporary CLI/SQLite/files/receipts with GPU and child-process test doubles;
they do not demonstrate live GPU throughput, paid-provider behavior, scientific
independence or research-quality gains.
