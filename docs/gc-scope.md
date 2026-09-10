# Scoped garbage collection (V1-05C1b)

This slice covers the independent GC module, its CLI, periodic cleanup and
disk-pressure caller. Built-in `cleanup.patterns` has its separate
[contract](cleanup-patterns.md). Custom cleanup scripts are not made safe by
this contract.

## Explicit authority

`run_gc(..., *, cfg=..., lake=...)` requires an explicit configuration mapping.
`cfg=None` is a refusal, not an inferred offline project. An explicit `{}` can
describe genuinely legacy storage with no catalog or outstanding execution
authority. Callers must supply the actual project configuration, including
declared inputs and control roots, rather than an empty mapping for a native
project. The actual Lake, configured catalog, claims and current attempts must
agree. The GC does not create or migrate databases to establish authority.

The lower-level `gc_checkpoints`, `gc_results` and
`archive_to_cold_storage` now require a `scope` produced by
`orze.engine.gc_safety.gc_scope`. Calls without one refuse before writes.
This is an intentional destructive-API contract change, not unchanged legacy
API compatibility. Product callers pass their configuration and actual Lake.

Checkpoint/archive roots cannot overlap results, declared project inputs or
control storage, or equal/contain the project root. An otherwise independent
checkpoint subdirectory inside the project is allowed. Paths must have plain,
non-symlink directory ancestors. Checkpoints need a corresponding existing
`results/idea-*` task directory; orphaned storage does not authorize inventing
a task identity. Native running/launching/uncertain attempts and unresolved
effects or stops refuse reclamation. Legacy metrics, when present, must show
an exact closed status; absent metrics alone do not establish native authority.

Keep-top/recent/active sets are conservative retention hints, never deletion
authority. Declared source inputs, framework metadata, actual database storage
and launch-bound artifact/control roots remain protected even if a keep set
omits them. This does not discover undeclared dependencies.

## Detachment and failure

GC captures bounded, no-follow tree metadata before acquiring the task effect
guard. Symlinks, special files, multiply linked regular files, changing
identities, oversized scans and conflicting destinations refuse the candidate.
The guard rereads closed authority and the captured identity before atomic
same-root detachment into `_orze_gc_quarantine/<task>/<nonce>/content`.

The implementation requires Linux `renameat2(RENAME_NOREPLACE)`. An unavailable
primitive, unsupported filesystem or cross-device operation refuses; there is
no overwrite or implicit copy-and-delete fallback. Large reclamation and
archive movement happen outside the task guard, using captured identities.
An uncertain detach retains the effect owner. A later reclamation failure
retains the quarantine without a completion receipt. Subsequent GC reports
the pending quarantine as blocked even when the source no longer exists.

These are cooperative framework ownership guarantees, not an adversarial
same-UID sandbox, an atomic multi-file rollback, power-loss proof or automatic
recovery worker. Partial reclamation may already have removed files. Do not
clear unresolved quarantine/ownership markers merely to obtain a green run;
recovery/resolution is a separate unfinished V1 item.

## CLI and accounting

Use `python -m orze.agents.orze_gc -c /path/to/orze.yaml --dry-run`.
Configuration-relative paths are anchored to the selected configuration file;
explicit command-line path overrides are relative to the invocation directory.
Missing/invalid/over-1-MiB configuration is rejected before GC; diagnostics do
not echo YAML values. Explicit zero overrides are preserved. Blocked GC exits
with status 2.

An absent, unconfigured default database permits the explicit legacy route.
An explicitly selected missing database, or an existing/dangling redirected
default, is not silently downgraded to legacy. Neither route bootstraps a DB.

Dry-run creates no guards, markers, directories or payload changes. Its legacy
count keys describe planned candidates, not completed deletion. Delete
`freed_bytes` is logical file size removed, not measured physical blocks freed.
Same-filesystem archive reports `moved_bytes` and `freed_bytes=0`: moving files
on that filesystem does not free its space. A disk-free reading is a snapshot,
not proof that GC caused the difference.

Tests distinguish real temporary-file/SQLite operations, injected filesystem
failures and CLI subprocesses from consumer-spy wiring checks. None is a live
research campaign, provider call, GPU workload, generic CPU research loop or
claim of improved research quality. V1 is still unfinished.
