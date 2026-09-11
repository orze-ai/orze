# Controller stop refusal boundary

C2e1 is a joint Core/Pro safety-refusal slice. It removes guessed process
ownership from the scoped stop/restart/orphan paths and prevents an
unconfirmed stop from authorizing a restart, reallocation, or successful
Director action record. It is not a complete controller lifecycle or a positive
stop-confirmation implementation.

## Requests are not confirmation

Core `lifecycle.do_stop` publishes the existing `.orze_disabled` and
`.orze_stop_all` markers as a cooperative request. It returns a `StopOutcome`
with status `requested` only after request publication/readback completes;
publication uncertainty returns `hold`. The legacy timeout parameter remains
accepted, including zero, but this function no longer waits for or scans
processes. It does not promise that the requested stop has been consumed, that
the controller has exited, or that its former workers have closed.

`StopOutcome` has explicit `status` and `reason_code` fields and refuses
implicit boolean use. The reserved `confirmed` label is not a capability or
closure receipt. No current C2e1 consumer accepts that label alone, a legacy
boolean, a zero CLI return code, or a vanished PID as restart authority.

These scoped compatibility paths no longer signal by bare PID/PGID, inferred
parent-child relationship, command-name match, work-directory substring, or
GPU/UID/name overlap. Core startup/periodic `cluster.kill_orphans` and Pro's
GPU-name orphan cleanup no longer discover and kill presumed orphan processes.
Without an owned execution handle or a future qualified recovery protocol,
unknown processes are not automatically adopted or cleaned up.

## Stop markers persist

The shared read-only `require_controller_start_allowed(results_dir)` gate
checks all three existing markers:

- `.orze_disabled`
- `.orze_stop_all`
- `.orze_shutdown`

Any existing marker blocks start, including an empty file, stale-looking
content, or a dangling symlink. An unverifiable filesystem check also blocks.
The gate performs no marker writes, deletion, directory creation, process
discovery, or lease operation. Marker absence permits only continuation to the
other launch checks; it does not prove process-tree closure or GPU ownership.

Core `do_start`, direct controller `Orze.run`, normal CLI startup, and Pro's
task start apply this gate. They no longer automatically clear old stop or
shutdown markers. Existing managed-run, direct-launch, and runnable-config
checks already reject these markers and retain that behavior.

The historical `.orze_shutdown` file is shutdown bookkeeping, not a matched
all-workers-closed acknowledgement. This changes the old startup behavior:
old markers are no longer presumed stale merely because a new controller is
starting. Qualified positive resume/re-enable is deferred to C2e2.

`--enable` also refuses an existing marker instead of deleting it. This matters
when stop publication is partial: if `.orze_disabled` was written but writing
`.orze_stop_all` failed, the sole remaining latch must not disappear through
another CLI entry point. No new journal or additional stop marker is introduced.

## Caller behavior

Core `stop` and `--stop` return exit status 75 rather than report confirmed
closure. Both restart entry points remain held and do not start a replacement.
`cli_setup.stop_running_instance` requests a cooperative stop and then raises
`ControllerStopHOLD`; it no longer returns guessed boolean completion from PID
liveness. The CLI catches this refusal for the legacy `--restart`, `--upgrade`,
and `--reinstall` flags, returning 75 before package changes or replacement
launch. Independent controller-runtime validation remains in force; a
runtime-contract rejection can still return its separate error.

The separate public `orze upgrade` subcommand also calls the actual cooperative
`do_stop` and returns 75. Its old package-install, bare-daemon-PID signal,
PID-file removal and replacement-launch branch has been removed. Neither
`upgrade --no-reinstall` nor `upgrade --no-restart` bypasses this admission:
skipping installation does not authorize a restart, and skipping restart does
not make installation safe while old writers may remain. The legacy
`--reinstall --no-restart` route likewise stops at the helper's HOLD. These
options remain parseable, but none grants an unconfirmed upgrade or reinstall.

Pro's Director may request a cooperative stop, but restart, resume, pause and
GPU reallocation cannot claim completion without a future closure consumer.
These actions return blocked results. They do not start replacement controllers,
persist a new GPU assignment, or write the successful-action bookkeeping that
would suppress later handling as if the transition had completed. A missing PID
file, `running=False`, timeout, or successful stop CLI invocation is not enough.
Fresh task start no longer performs a GPU-name cleanup or removes stop markers.

## Preserved owned execution behavior

This slice does not replace or weaken the existing shutdown path for retained,
owned native training, evaluation and posthoc handles. Those paths continue to
check their exact attempt/protocol binding, request STOP through the existing
supervisor, require `TREE_CLOSED`, and validate closure again at publication.
Their uncertainty keeps the corresponding handle/state held. Their internal
stale-delivery no-op is not a controller-wide stop proof.

PID-file removal, a shutdown marker, controller exit, and release of a
controller's own bookkeeping are not substituted for descendant closure.
This slice adds no new process supervisor, durable controller registration,
matched stop ACK, adoption protocol, or authority to release another owner's
GPU lease.

## Deliberate limits and next slice

C2e2 must supply source-qualified positive controller registration and stop
acknowledgements before restart/resume/reallocation can be positively completed.
The refusal-only state here is not described as that finished product. It also
does not provide a filesystem sandbox, a general lifecycle transaction, or
protection against a hostile same-UID writer removing control files.

The scope is the Core stop/start/restart routes, their direct controller-start
gate, legacy stop-helper consumers and the public `upgrade` subcommand,
together with Pro Director stop/start and transition handling. Unrelated
uninstall and internal self-upgrade/cleanup paths were not comprehensively
changed or certified. This document must not be read as a claim that every CLI
or control-plane destructive path is now safe.
