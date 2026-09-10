# Promotion evidence and the optional anomaly policy

Promotion always requires the current project's agreed completed lifecycle and
qualified result artifacts. The declared primary metric and source are used
exactly; a claim must be a finite, non-boolean number equal to that source value.
Disabling the anomaly policy never disables qualification.

The optional top-level artifact field `honest` is a declaration, not a proof.
If present, it must be boolean. Explicit false vetoes qualification; true still
has to pass all source, validity, coverage, lifecycle and provenance checks.
Missing remains supported. Reports invalidate older qualification caches.

## Optional operational policy

The anomaly policy is now off by default. Existing projects that explicitly
set `champion_guard.enabled: true` retain blocking behavior unless they choose
`action: warn`.

```yaml
champion_guard:
  enabled: true
  action: hold       # hold or warn
  z_threshold: 4.0
  min_history: 10
  history_size: 50
```

Enabled policies use the absolute primary-metric z-score in either objective
direction. A short or constant history supplies no anomaly verdict. This is
an operational outlier heuristic over selected prior ideas, not a significance
test, independent replication, proof of improvement, or a scientific veto.

History lives in the existing project SQLite database, isolated by objective,
source definitions and qualification/benchmark policy. Each retained idea ID
has at most one accepted value and a qualification snapshot identity. Prior
samples are requalified before use; a changed primary value is excluded until
its revision is accepted. Shared benchmark-log changes do not create samples.
The bounded window follows first acceptance of distinct IDs; updating a
retained ID does not renew its window position. It is not an attempt ledger.

Old `_champion_history.json` bytes are preserved but never used as samples:
bare numbers cannot be assigned retrospectively to the current protocol.
The `info.verified` compatibility field means the qualified current artifact
value, not the outcome of a second execution. History failures reject the
optional-policy check rather than pretending to have recorded a sample.

## Explicit work, not promotion side effects

Promotion no longer runs a shell `reproducer` command or extracts a guessed
number from stdout. Supplying one to the guard is rejected with
`reproducer_requires_explicit_evaluation_task`. Use an explicit evaluation or
replication task with an execution/output contract instead.

The controller emits an anomaly diagnostic when configured but does not
implicitly enqueue audit tasks. The low-level callback parameter is retained
for callers that explicitly supply their own audit action; it does not provide
exactly-once execution, and no such callback is supplied by the controller.

Champion selection versus objective improvement, duplicate finished events,
attempt fencing and scientific comparison remain separate contracts.
