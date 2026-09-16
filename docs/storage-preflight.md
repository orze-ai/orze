# Storage admission preflight

This check refuses incompatible storage before new work is admitted. It does
not replace execution authority or certify every future filesystem operation.
Default atomic roles and GC require no-replace rename. Explicit
[`gc.storage_mode: guarded`](gc-guarded-storage.md) uses exclusive creation,
durable task retirement, copying and removal for GC on storage such as the
locally verified CephFS route. Explicit
[`roles.<name>.storage_mode: guarded`](guarded-role-storage.md) uses a persistent
role namespace and a captured lease for release without directory rename.

## Routes checked

- A non-CPU `Orze.run` checks the existing controller Stop/registration gate
  first. It then checks the configured locks directory for runnable, enabled
  roles, and the checkpoint route actually used by enabled emergency GC.
  Relative checkpoint paths use the existing `gc_scope` interpretation of the
  project/configuration root, not the caller's current working directory.
- Pro checks runnable role storage before receipt intake, role locks or budget
  reservation, and checks the actual captured locks parent again just before
  acquiring the role lock. Disabled and already active roles are not restarted
  or stopped by this check.
- Real GC checks the source storage before acquiring a task effect guard or
  creating quarantine. Archive preflight checks the existing destination
  parent, or its nearest existing ancestor when the destination is absent;
  it does not create an archive directory before native/effect authorization.
  Actual reclaim still opens/creates the exact route and enforces same-device
  and collision protection for the selected mode. Dry-run remains read-only and is not a
  filesystem capability certificate.
- The CPU-only loop does not use these role/GC routes and is not rejected just
  because its filesystem is named CephFS. Other operation-specific admission
  and storage checks remain in force.

## What the check does

Each probe creates a uniquely named private directory on the relevant route.
Atomic-mode probes test actual file and directory `RENAME_NOREPLACE`, including existing-target
collisions, identity/readback checks and directory synchronization. Renames use
captured directory FDs and single basenames; the probe does not reopen a replaced
pathname or route through `/proc`.

Guarded modes probe exclusive file/directory creation, existing-target collisions,
readback, synchronization and removal. Guarded GC's final operations create durable
retirement records under the short task guard, then perform bulk I/O outside it.
These retirements block fresh task effects until confirmed completion. An
uncertain probe leaves its private diagnostic objects and refuses admission.
Guarded roles instead require a captured process-local lease, closure/settlement
proofs and a non-stealable transition guard. Their namespace marker remains after
release. Mixed role configurations check each selected protocol.

Cleanup uses captured entries and owned FDs only, never a recursive deletion of
an arbitrary path. Unknown identities, synchronization or cleanup failures refuse
admission. Uncertain private leftovers may remain for diagnosis. They are not
execution receipts or permission to recover an existing owner. Administrative
locks/checkpoint directories may be created explicitly for a configured route;
no user data is moved and archive staging is deferred as described above.

The final atomic role-release and atomic GC operations still execute the protected rename.
Preflight success does not guarantee a later mount, permission or namespace has
not changed. Such later failures retain the existing HOLD/owner behavior; a
successful probe never clears Stop, refunds a reservation, or authorizes adoption.
There is no ordinary-rename fallback, check-then-rename substitute, automatic
state migration, or filesystem-name allowlist.

## Cost and deployment status

There is no capability cache. A successful existing-directory probe performs
four rename calls in atomic mode (two moves and two expected collisions), creates three tiny
files plus private directories, and performs synchronization and metadata reads.
The early Pro check can therefore perform probe I/O on an idle, statically
runnable role: existing cooldown logic depends on earlier receipt observation,
which itself can write. Actual role launch repeats the probe at its lock route;
GC can probe multiple source/destination parents per candidate. This fixed I/O
cost is intentional in this slice and is not claimed to be free or amortized.

The observed `/hot-data` CephFS mount rejects the no-replace operation;
the observed `/tmp` ext4 route supports it. Guarded GC has separate positive
CephFS coverage and a copy/readback cost. Guarded roles have separate actual CPU
coverage and measured acquire/release costs. Neither establishes production
support. Select and verify deployment storage and each mode explicitly; this
work does not switch existing services or supply rolling migration.

`StoragePreflightError` carries `path` and, where available, `probe` diagnostic
attributes. Not every existing caller prints both attributes; retained private
probe discovery may require inspecting the configured route. No automatic
cleanup or retry of uncertain leftovers is provided.

## Paired Core/Pro requirement

The updated Pro role runner requires this Core's `storage_preflight` module.
An identical package version string is not sufficient to establish compatibility.
Use the exact paired commits/wheel hashes in the final validation record. This
slice does not publish a package, upgrade a global environment, supply a license,
or silently fall back to an older Core without the admission check.
