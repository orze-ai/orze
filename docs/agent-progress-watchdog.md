# Managed-agent progress watchdog

Managed roles have both an absolute wall-clock timeout and a no-progress
watchdog. The watchdog is enabled by default at five minutes:

```yaml
role_stall_minutes: 5
roles:
  engineer:
    stall_minutes: 10
    stall_warmup_seconds: 60
```

A role counts as making progress when at least one content-free signal changes:

- its stdout/stderr log size;
- cumulative Linux CPU ticks for the role process and its live descendants; or
- `lstat` size/mtime metadata for declared output paths.

Research roles automatically monitor the configured `ideas_file`. Integrations
can pass additional project-bounded outputs through `RoleProcess.progress_paths`.
Orze never reads commands, prompts, environment values, log content, or artifact
content to compute this signal. When all available signals remain unchanged for
the configured interval, Orze logs `[ROLE STALL]`, terminates the role, and reaps
its process-group and escaped descendants. The existing timeout outcome is kept
for caller compatibility; the log marker distinguishes a progress stall from an
absolute timeout.

Normal completion, graceful shutdown, upgrade shutdown, and `atexit` use the
same exact-identity reaper as timeout handling. A managed role cannot
legitimize a detached child by exiting zero, and a child that called `setsid()`
remains bound to the live identity set observed by the controller. Dead
identities are pruned on every poll rather than accumulating across a long
campaign. A zero-exit role is classified as an error if its tracked descendants
cannot be proven stopped.

The warmup suppresses the watchdog before a role has emitted its first log byte.
The absolute timeout remains authoritative even while progress continues.

This is an efficiency signal, not proof that work is useful. A wrong-scope job
that consumes CPU still looks active; workspace/tool policy must reject such a
job before launch. Metadata can also be touched without producing a valid
artifact, so completion receipts and output validation remain separate gates.
