# Guarded role storage

Set `roles.<name>.storage_mode: guarded` to run a managed role on storage that
supports exclusive creation and durable file/directory synchronization but lacks
no-replace directory rename. The default remains `atomic`.

```yaml
roles:
  engineer:
    mode: script
    script: scripts/research_worker.py
    storage_mode: guarded
```

The normal role preflight runs before receipt intake and budget reservation,
then checks the actual lock parent immediately before acquisition. Mixed role
configurations check every selected storage protocol. CPU-only work with no
runnable roles keeps its existing admission path.

## Ownership and release

The first guarded acquisition declares a persistent `<role>.role-lock` marker
beside the role directory. Current generic lock/unlock helpers cannot take over
or recursively remove that namespace. Proposal-source locks and guarded role
locks refuse each other's declared namespace; a conflicting declaration after
acquisition invalidates the current owner. An existing legacy owner is not migrated.
Guarded acquisition uses an exclusive directory creation and has no age/PID
takeover. Only the acquiring process's captured lease can bind a `RoleLaunch`.

Acquisition and release use a sibling `.<role>.role-transition` directory that
cannot be stolen on timeout. Its small record identifies the operation; release
records the owned directory identity and hashes of the exact lock and process
receipts. This record does not itself prove process closure or grant execution.

Release independently requires the existing strong proof of no execution or
`TREE_CLOSED`, and the matching trigger settlement when a trigger is present.
It checks the route, marker, directory identity and current receipt bytes, then
removes only the captured `lock.json` and `role-process.json` and the empty owned
directory. Unexpected contents cause HOLD. No replacement rename or recursive
deletion is used. A pre-launch budget/configuration refusal may cancel only the
same process's unbound lock, before any process receipt exists.

After confirmed release, another process may acquire a new lease. The namespace
marker remains. Startup reports leftover transition guards as unresolved;
unresolved v2 role receipts retain their existing recovery requirement. A
failed publication, removal or synchronization is not retried as a new owner.
Partial transitions retain their remaining files. If the final namespace
removal itself succeeded but its synchronization reply failed, the caller still
reports uncertainty; this is not a promise that every failed operation leaves
the same pathname present.

## Deployment boundaries

Use the matching Core/Pro versions that implement this protocol. Older binaries
do not understand its namespace marker, so this does not support rolling mixed
versions. Stop and prove closure of old controllers before enabling it. Changing
the option back to `atomic` does not erase the persistent marker or authorize
takeover. Migration, uncertain-owner recovery, marker removal and rollback need
separate operator procedures and are not automatic features of this option.

Validation uses isolated local CPU roles and the current CephFS workspace.
It does not establish cross-host fault tolerance, real model/Pro license
acceptance, machine reboot recovery or production deployment. See the
[paired validation and measured costs](plans/2026-09-16-role-guarded-results.zh-CN.md).
