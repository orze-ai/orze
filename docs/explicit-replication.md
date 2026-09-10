# Explicit same-specification replication

Use the Core control API or CLI to request a new execution of a confirmed
native source. Do not change the seed, add a configuration salt, or ask a model
to bypass duplicate detection.

```sh
orze replicate idea-source --request-id repeat-2026-09-10-01 -c orze.yaml
```

The stable request key is required. A successful first call returns
`{"status":"created","request_id":"...","task_id":"idea-rep-..."}`.
An exact replay returns the same task with `status: already_requested`, without
requeueing, resetting a claim, or launching another process. Reuse the same key
after an uncertain response; a different key requests a different task.
Keys are scoped to the selected authority database. Reusing a key for a
different source, project scope, or reason is a conflict.

The command resolves the selected project's configuration and opens an existing
compatible database. It does not create a project, probe GPUs, load paid
extensions, or start workers. It returns one JSON document and exit code 0 on
success, or 2 on a control/storage rejection. An uncertain close/response is not
proof that an earlier transaction rolled back.

```python
from orze.engine.replication import request_replication

result = request_replication(
    source_task_id, results_dir, cfg, lake,
    request_id="repeat-2026-09-10-01", reason="explicit_replication",
)
```

## What is authorized

The first native adapter requires a currently confirmed B1 training completion:
its exact AttemptRef, terminal/effect evidence, source configuration, launch
identity, artifact contract and registered artifact set must agree. A legacy
completed flag or a mutable metrics file cannot manufacture that authority.
This adapter currently copies train-kind tasks; generic request storage does
not prescribe a source phase, hardware, metrics, or scientific verdict.

One short SQLite transaction creates the immutable request, a new queued task,
and its lifecycle edge. The configuration is copied unchanged, including its
seed. The request pins the original specification and execution fingerprints.
Source/configuration/script drift is rejected at admission or launch; it is not
silently reinterpreted as a new experiment. Large launch-input hashing occurs
outside the writer; only bounded metadata and receipts are rechecked inside.

Normal queue synchronization reads this task directly from the same Lake. No
model output or Markdown append transports the authorization. At dispatch,
the controller checks the request-to-task mapping, project, actual execution
identity, configuration and current claim. Native begin binds the request to
the new AttemptRef in its transaction.

## Independent execution occurrence, unchanged recipe

The ordinary execution owner remains at `<identity>.json`. An authorized repeat
uses `<identity>.replicas/<request_id>.json` in the existing identity registry.
It neither overwrites nor releases the original owner. Prelaunch cleanup uses
the exact captured repeat slot, not a path recomputed from later configuration.
Uncertain writes or owners retain a HOLD; elapsed time alone cannot steal them.

The same request authorizes only its fixed task. Existing, explicitly confirmed
failed/not-started and operational requeue rules still apply to that task; a completed, launching,
running, or uncertain attempt is not reset by repeating the control request.
Each completed execution has its own attempt and artifact occurrence IDs even
when all artifact bytes and specification fingerprints are identical.

## Ordinary proposals and limits

Ordinary configuration deduplication remains enabled. The exact config keys
`replication_request_id` and `replication_authorization`, like `force_launch`,
are reserved control fields and rejected at any nesting depth in proposals and
launch inputs, even when their values are false or null. Descriptive
`replication_role`/`replication_index` and unrelated domain fields are not a
blanket forbidden namespace; they do not grant an execution slot.

A model proposal appended to an inbox is not a task admission, execution, or
completed replication. In particular, descriptive labels may pass proposal
formatting but still be rejected by the ordinary semantic configuration gate.
Pro's existing accepted-proposal count must not be read as completed experiments.

Additional user-installed or third-party admission policies can still veto a
task. In particular, an external semantic-dedup hook needs to understand the
explicit request before allowing an identical recipe; Core authorization does
not silently disable such a hook. The built-in path is covered here, not every
optional external extension.

Explicit replication is not evaluation-only retry. It deliberately schedules
a new execution; `retry-eval` instead preserves the original training outputs.
Interrupted checkpoint resume is not yet supported for a new replica task;
it requires the separate verified-resume admission protocol. This limitation
does not change the original non-replica path.

This slice does not aggregate repeated measurements, infer statistical
independence, qualify a scientific conclusion, implement a CPU executor, or
provide automatic recovery of unknown external effects. Filesystem slots and
SQLite are not one atomic store. Their cooperative ownership checks are not
an OS sandbox against arbitrary same-user file or SQL writes.
