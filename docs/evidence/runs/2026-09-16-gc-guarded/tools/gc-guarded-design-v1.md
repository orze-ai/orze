# GC without renameat2: explicit guarded storage mode

Original P2 requires positive CephFS GC and role-release support. Actual local
CephFS returns EINVAL for NOREPLACE with absent destinations; Linux v5.15 and
current upstream ceph_rename reject nonzero flags. No ordinary replacing rename
fallback is acceptable. This slice implements GC; role release remains open.

Explicit gc.storage_mode=guarded uses the same candidate exclusions, closed-task
authority and same-device restrictions as atomic GC. Default atomic mode stays.
Create a durable per-task retirement under the short task effect guard. Fresh
effect/claim admission checks this retirement before and after taking its guard.
Then release the short guard before bulk I/O. Pending retirements deny further
task effects, survive process loss and are never automatically adopted/retried.

Delete only previously captured regular single-link files/directories, through
no-follow directory descriptors with identity rechecks. Archive copies into
exclusively created destination entries, streaming bounded chunks with hashes,
fsync and identity/readback checks before deleting any source. Never replace an
existing target, cross a device, remove an uncaptured entry or erase a failure.
Unknown copy/delete/sync outcomes leave journal and retirement pending. Completion
is published only after confirmed payload mutation and durable GC completion.

All existing protected data, closed native/legacy authority, latches and no
implicit recovery semantics remain. Retirement metadata is bounded (1024 records
per task, matching existing GC journal bound); absence has a constant fast path.
No whole-tree copy/hashing/deletion runs while holding the short effect guard.

Validate actual CephFS checkpoint/result deletion and file/directory archiving;
negative keep/protected/active/target collision/link/identity/fsync/Stop cases;
real fresh-process admission during retirement; source/target preserved on copy
failure; unknown deletion blocks re-entry even after source disappearance.
Retain baseline, snapshots, failures, related two-repo regressions, raw artifacts,
costs and mechanical byte checks before ordinary paired commits/pushes.

No model/account/GPU/existing services/managers/deployment. No subagents.
This cannot close all P2, the original goal or prove research gains.
