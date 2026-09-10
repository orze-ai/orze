# Native posthoc execution

V1-05C2c covers new posthoc actions launched with the product's actual IdeaLake.
It does not turn CPU inference fixtures into model-training, GPU, scientific
validity or research-efficiency evidence. The generic CPU research policy loop
and verifiable crash adoption remain separate, unfinished work.

## Identity and launch

The execution AttemptRef has phase `posthoc`. It is never a borrowed training
attempt. The existing workload FSM stage is explicitly mapped with
`lifecycle_phase: training`; no third pipeline stage is invented. Claim,
prelaunch lifecycle, configuration digest, private work directory and output
contract are pinned before creating an executable worker. READY records the
actual allocated worker before GPU-lease exit, without sending GO. The actual
worker PID/start identity, claim and running lifecycle are published before
fresh configuration/runtime/control checks permit GO.

Native posthoc requires a finite positive numeric `posthoc_timeout` (default
3600 seconds). It is an execution bound, not an estimate of scientific value.
Booleans and numeric strings are not budgets. Invalid limits are rejected before
intent or worker creation; the normalized limit is part of the pinned execution
identity and is checked again before GO. Controller downtime is not a promised
wall-clock termination deadline.
It does not authorize training-only B3 replication requests; changing a replica
task's kind cannot acquire a different execution path. Native posthoc replication
and interrupted checkpoint resume are not implemented in this slice.

The effective adapter configuration is bounded strict JSON transported through
a fully sealed Linux memfd. Only the descriptor number, digest and generation
are command arguments, not the configuration itself. The fixed driver verifies
and closes its inherited descriptor before its explicit adapter-runner import
and call. Python startup hooks in the authorized environment are not sandboxed:
sitecustomize can execute earlier, as the test registration fixture demonstrates.
This ordering must not be advertised as isolation from malicious same-UID code.

The descriptor has write, grow, shrink and seal-set seals. Linux enforces those
restrictions on the underlying inode; merely sealing writes does not suffice
to prevent resizing. Unsupported environments fail without a pipe/file fallback.
See [Python memfd API](https://docs.python.org/3/library/os.html#os.memfd_create)
and [Linux file-seal semantics](https://man7.org/linux/man-pages/man2/F_ADD_SEALS.2const.html).
FDs are private launch inputs, not durable restart authority; each holder closes
its copy once. A close error after a possible fork cannot imply NOT_STARTED.

Configuration identity describes this legacy adapter invocation, not a content
hash of arbitrary paths inside its configuration. Final checks are not one
atomic filesystem snapshot, a general data-input sandbox, or protection against
an external writer after GO.
The opened YAML descriptor must match the captured file identity before and
after reading. The descriptor regression injects a real unrelated inode at the
open boundary; it is not evidence of a reproduced pathname ABA race.

## Candidates and parent publication

The unchanged dict-returning adapter API runs under the fresh create-only
`_posthoc_attempts/<attempt_id>/work` directory. Relative outputs and default
relative inputs therefore refer there; old shared prediction files are not
silently adopted. Existing data outside the attempt must be explicitly supplied
through the adapter's own input configuration.

The actual runner writes metrics only in that work directory. The native worker
does not call the optional legacy artifact registrar. The old
`orze.artifact_catalog` import is unavailable in this package, so old attempted
registration must not be described as evidence of successful catalog writes.
A private result envelope binds the full attempt identity and payload digest;
neither that envelope nor an integer root exit is permission to publish.

The parent requires the exact current TREE_CLOSED proof before reading candidate
metrics or copying any declared output, and rechecks it in the terminal writer.
The original detached CPU writer can continue after the root exits; publication
waits until the actual owned tree is closed. Only declared files from this exact
posthoc work directory may become B1 snapshots. Shared-directory decoys and
`*.npz` files are not discovered as new artifacts.

Metrics, compute receipt, existing FSM transition, B1 records and attempt terminal
are accepted through one short effect/SQLite boundary. File intents preserve
uncertain filesystem/SQL outcomes; this is not a claim of filesystem transactions.
Duplicate and stale callbacks do not republish another attempt's result.

A normal adapter return without a status is operational completion, preserving
zero and negative values. Exceptions, nonzero worker exit, explicit FAILED or
invalid candidates cannot become successful artifacts. No B2 observation,
comparison, significance or model-lineage approval is fabricated. Posthoc
CompletionEvents retain their own reference through fresh/pending dispatch;
they do not implicitly start the project's training evaluator. The existing
evaluation stage may be SKIPPED, which is not evaluation COMPLETE.

## Stop and compatibility

Explicit STOP cannot turn worker exit 0 into success. A confirmed scheduler
slot rollback can requeue its exact attempt; automatic VRAM/provider repair is
not inferred from posthoc failure. Shutdown records phase-qualified interrupted
compute and the mapped existing lifecycle, without inventing a resumable model
checkpoint. Lost supervision keeps active work and its reservations held.

Historical native rows cannot acquire authority from a missing token, mutable
is_posthoc flag or a raw PID. Standalone run_posthoc and genuinely unbound legacy
launchers retain their limited old API; they are not upgraded to native closure
proof by this work. Durable supervisor reconnection, restart adoption, consumer
ACK, pre/post scripts and director/startup OS control remain outside the slice.
