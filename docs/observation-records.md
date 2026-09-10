# Opt-in isolated JSON observations

`orze.json_observations.v1` connects confirmed native training artifacts to an
explicit evaluator, independent evaluator output snapshots, and zero to 32
observation records. It does not rank them, declare statistical independence,
authorize a repeat task, or convert evaluator-authored validity into a scientific
verdict. Legacy evaluation configurations remain on their original path.

```yaml
artifact_contract:
  version: 1
  outputs:
    checkpoint: {path: best_model.pt, max_bytes: 1073741824}
eval_script: evaluate.py
observation_contract:
  version: 1
  adapter: orze.json_observations.v1
  protocol_id: project-evaluation-v1
  inputs: [checkpoint]
  output: {path: observations.json, max_bytes: 65536}
```

Inputs name declared B1 outputs from the current confirmed training terminal,
not filenames to rediscover in the mutable training directory. Even an empty
input list requires a native, launch-bound training artifact contract (which
itself may declare zero outputs). Output is a canonical relative file under the
attempt's work directory, bounded by at most 1 MiB. Native SQLite authority is
required; this contract cannot manufacture provenance for pre-native imports.

## Two directory roles

Canonical `results/<task_id>` still owns task/stage state, compute receipts,
termination HOLDs, effect receipts, and training-source identity. An evaluation
gets `results/<task_id>/_evaluation_attempts/<attempt_id>/` containing:

- `entrypoint.py`: an independent, hash-verified copy of the configured main script;
- `input_manifest.json`: create-only, read-back-verified input records and scope;
- `work/`: this worker's output directory;
- `eval_output.log` and, on failed completion, `failure.json` diagnostics.

The interpreter and copied entrypoint are passed as absolute command paths. The
child cwd is `work/`; the controller does not chdir or reinterpret the canonical
task directory. These reserved variables are assigned after inherited and
configured environment values:

| Variable | Meaning |
|---|---|
| `ORZE_EVALUATION_INPUT_MANIFEST` | Absolute input manifest path |
| `ORZE_EVALUATION_OUTPUT_DIR` | Absolute `work/` directory |
| `ORZE_EVALUATION_OUTPUT_PATH` | Exact declared output file inside `work/` |

The manifest has `schema: 1`, `task_id`, `attempt_id`, the complete training
`source_ref`, `protocol_fingerprint`, `inputs` (complete artifact records), and
`output_path`. Input paths refer to B1 independent read-only snapshots, never
mutable training checkpoints. Evaluators must consume these explicit paths and
use explicit absolute external resources. Copying the main script does **not**
freeze imports, environment packages, or external datasets. Directory separation
is not an OS sandbox against a hostile worker with the same filesystem identity.

## Output protocol

The exact envelope is:

```json
{"schema":1,"observations":[{"name":"validation","values":{"quality":0,"delta":-0.1},"validation":{"status":"unknown","reason_code":"adapter_reported"},"comparison_scope":null}]}
```

Names are unique bounded identifiers. `values` is a bounded JSON object;
non-finite numbers and duplicate JSON keys are rejected. Validation status is
`valid`, `invalid`, or `unknown`, with a bounded reason code. Comparison scope is
an explicit nonempty string up to 256 UTF-8 bytes, or null; it is not inferred
from identical metric names. Each stored observation is bounded to 32 KiB and
depth 16 by the shared record validator.

A valid empty array produces a completed evaluator attempt, one raw result
artifact, and zero observations. Missing, malformed, or oversized output, or a
nonzero/confirmed interrupted process, produces no accepted result artifacts or
observations. It does not produce a zero score. Failed diagnostics are strictly
attempt-local; the adapter never uses the old shared `eval_output` presence
shortcut or shared failure-marker writer.

The raw JSON result is copied to a new immutable artifact inode. Every observation
binds its evaluator AttemptRef, the subject training specification fingerprint,
protocol fingerprint, adapter, input artifact IDs, result artifact IDs, name,
values, declared validation, and comparison scope. The result artifact's own
specification can be an evaluator composite; it is not substituted for the
subject specification. Observation IDs are occurrence identities, not scores or
content hashes.

## Acceptance and explicit retry

Input hashing, main-script copying, sealed verification, output parsing, and
result copying/readback happen outside the short effect lock and SQL writer.
Identities are captured before expensive reads and rechecked afterward. The
terminal writer rechecks cheap identities, registers result artifacts and the
complete observation batch, and closes lifecycle/attempt state in one SQLite
transaction. Final record readbacks include input dependencies. SQL/file receipt
uncertainty retains the existing HOLD; no filesystem/SQLite atomicity is claimed.

`request_evaluation_retry` keeps the failed attempt directory, accepted history,
training bytes, and original training source reference. Its schema-2 metadata
manifest pins original input records/IDs, the protocol, and source. Relaunch uses
a new evaluation AttemptRef and directory, but cannot silently choose newer
inputs or a changed protocol. Retry does not launch training or move/delete old
attempt output. Legacy retry manifests and their existing allowlisted behavior
are preserved for legacy evaluations. There is no automatic recovery of partial
snapshot/publication state or garbage collection in this slice.

Prelaunch preparation may leave a directory before any evaluation attempt row
was created. A subsequent admission allocates a fresh attempt identifier; this
slice does not claim B1-style cross-prelaunch-failure staging deduplication or
automatic cleanup. Input manifests and the generic attempt binding each have a
64 KiB metadata limit, so a large declaration can hit those limits before the
32-input count limit. Such rejection never authorizes Popen.

## Explicit unsupported combinations and legacy qualification

For this first adapter, nested `report.benchmark_contract`, enabled
`evaluation_bundle`, enabled `model_lineage`, or managed-run qualification
requirements are rejected explicitly. Their legacy paths are untouched. Policies
are never removed from cfg to make the adapter appear compatible.

`sealed_files` remains supported against the canonical results-root manifest,
including its captured identity across publication. The sealed manifest identity
contributes to protocol identity; changing it is not a silent same-protocol retry.

Old report-file qualification does not grant these observations scientific
authority: the explicit observation adapter is required, and canonical training
metrics are not a substitute. A later consumer must explicitly define selection,
comparison, qualification, and repeated-measurement semantics. Tests with fake
process/GPU boundaries are not actual CPU/GPU workloads or research-benefit
measurements.
