# Declared training artifact records

This opt-in contract snapshots declared regular files from an actually launched
native training attempt. It does not create an evaluation, an observation,
scientific validity, an independent replication, or permission to run a task.
Configurations without the contract retain their existing behavior.

```yaml
artifact_contract:
  version: 1
  outputs:
    weights:
      path: checkpoints/model.bin
      max_bytes: 1073741824
```

Paths are canonical, task-relative files. The declaration permits zero to 32
named outputs, with positive per-file byte limits and at most 1 TiB in aggregate.
Directories, redirects, hardlinks, changing files, and exceeded limits are not
accepted. The versioned contract is captured before `Popen`, together with the
project results scope, the artifact storage root, and the adapter's specification
fingerprint. Changing/removing that declaration during an execution cannot
redirect its outputs or silently disable publication. A pre-native execution
cannot acquire this provenance afterward through the legacy completion import.

## Identity and physical storage

An artifact occurrence ID is derived from the complete producer `AttemptRef`,
logical output name, and results scope. It is not a content hash or a scientific
observation ID. Different attempts producing identical bytes have different
occurrence IDs. The existing attempt `generation` remains a per-task/per-phase
attempt ordinal; this feature does not redefine it as a model generation.

Storage is `<configured .orze directory>/artifacts/<artifact_id>/content`.
Without a configured control directory, the project `.orze` directory is used.
Copying creates a new inode, never a hardlink or rename of the worker's file.
A worker retaining an old output file descriptor therefore cannot change the
published copy by writing through that descriptor. Snapshot readback hashes
must match the copied bytes. Mode `0400` is a cooperative-writer convention,
not protection from malicious direct filesystem access by the same user.

The record contains `schema`, `artifact_id`, `producer`, `spec_fingerprint`,
`scope`, `logical_name`, `path`, `content_sha256`, and `size_bytes`. The producer
includes task, phase, attempt, and generation. The fingerprint combines the
existing native training execution identity with this output contract; it does
not contain task/attempt IDs or storage locations.

## Publication and failure boundaries

1. After normal training output validation, copy, hash, fsync, and read back the
   declared files **outside** the per-task effect lock and SQLite transaction.
2. Recheck the current native attempt, the launch binding, and cheap source/copy
   identities inside the short existing completion transaction.
3. Persist the effect plan, register the complete declared artifact set, and
   close the attempt with its `artifact_ids` in the **same** SQLite transaction.
   Existing lifecycle/compute receipts and effect confirmation remain in use.

The SQLite table is `research_artifacts`. Its generic API is
`register_artifacts(conn, ref, records)`, `artifacts_for_attempt(conn, ref)`, and
`get_artifact(conn, artifact_id)`. Writers require a caller-owned transaction and
the current RUNNING producer. Historical readers are read-only metadata views;
they do not grant current ownership, inspect arbitrary files, or qualify results.

A staged file without accepted database records is not a published artifact.
An existing occurrence directory is never overwritten or blindly reused. Thus a
failed/partial copy does not allocate a new large orphan on every monitor tick:
it requires explicit resolution. This slice supplies no automatic recovery or
garbage collection for such staging. A failure after the terminal effect intent
retains the existing HOLD protocol; filesystem and SQLite commit are not atomic.
Direct external changes to the database/files remain outside the insert-only API
contract. Historical lookup alone does not prove a confirmed completion event.

Failed attempts and explicit empty declarations have no accepted artifacts;
opt-in terminal records include an empty `artifact_ids` list. Legacy-off records
do not gain that field or create an artifact table/root merely for monitoring.

## Deliberately not included

Evaluation outputs still use their existing paths. Immutable evaluation
observations, physical evaluator attempt isolation, explicit same-seed repeat
admission, and a real CPU execution loop are later slices. Synthetic subprocess
and GPU-ownership boundaries in tests do not demonstrate actual GPU/CPU training
or improved research throughput. No existing benchmark exposure ledger,
training output, checkpoint, or evaluation-retry provenance is moved/deleted.
