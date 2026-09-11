# Bounded legacy executor supervision

This slice changes the synchronous, tokenless executor-fixer path. It does not
introduce a native repair action, persistent repair journal, or new provider.
Native task routing or execution-attempt history requires an explicit repair
authority that this path does not provide. Automatic native repair remains
disabled.

## Execution and completion

`failure._run_bounded_executor(cmd, timeout=..., env=..., cwd=...)` delegates to
`bounded_executor.run_bounded_executor`. The latter also accepts an optional
`before_start()` callback and a preparation injection used by focused tests.
The normal implementation uses the existing Linux supervisor without changing
its protocol or process implementation.

The helper copies command and environment inputs, validates a finite positive
timeout, resolves the executable, and captures the canonical working directory
and its device/inode. It prepares a blocked worker, validates READY against a
private invocation identity, invokes `before_start()`, then rechecks the READY
binding and directory witness before GO. The worker PID remains the real worker
PID. The private identity contains a random invocation ID and scope; it is not
an `AttemptRef` or durable native binding. Reusing this handle does not satisfy
the durable-bind-before-GO requirement of native adapters.

The admission callback checks the legacy/native boundary again using both the
captured original database-routing configuration and the current configuration.
It also checks the project scope, fixer/tool policy, attempt limit, and current
attempt counter. Policy `enabled` must still be the boolean `True`, not a value
that merely compares equal to it. These are short checks, not a reservation
held against another controller.

A result requires the actual supervisor's bound `TREE_CLOSED` receipt with
`ECHILD_WALL` and its normal exit, plus EOF on both output streams. Leader exit
alone is not completion. Descendants are allowed to finish naturally until the
execution deadline. STOP uses the owned supervisor protocol once; there is no
raw PID, process-group, or host-scan fallback in this helper.

The returned `BoundedExecutorResult` is a `subprocess.CompletedProcess` subclass.
It preserves the actual worker return code and adds `process_tree`,
`output_complete`, `stopped`, `stdout_bytes`, and `stderr_bytes`. A stop request
or forced cleanup sets `stopped=True`, even if the worker exited with code zero.
The fixer will not accept `FIX_APPLIED` from such a result.

## Output and time limits

The parent drains real stdout/stderr pipes using nonblocking reads, retaining
at most 65,536 tail bytes per stream and counting all bytes read. Reads are
bounded per loop so continuous output cannot prevent deadline checks. UTF-8
decoding uses universal-newline normalization. Truncation or invalid UTF-8 sets
`output_complete=False`; replacement characters and retained tails are
diagnostic only. Neither a surviving tail marker nor an exit code of zero can
qualify incomplete output. With complete output, marker and `UNFIXABLE` checks
use the complete captured stdout, not only the shorter log tail.

The configured execution deadline begins after GO. It is not an all-inclusive
wall-clock bound on executable lookup, supervisor preparation, arbitrary
admission callbacks, or provider-side spending. STOP confirmation has a
separate three-second allowance; closed-tree output EOF has a one-second
allowance. A confirmed timeout preserves the old `TimeoutExpired` interface
and carries the actual return code, closure receipt, and qualification flags.
An unconfirmed stop or missing EOF is HOLD, not an ordinary confirmed timeout.

## HOLD and retries

`BoundedExecutorHOLD` derives from `TerminationUnconfirmed`. The fixer and its
relaunch callers propagate it instead of returning ordinary failure or
continuing reset/retry. An observed missing executable before preparation can
still raise `FileNotFoundError`. The primitive's explicit
`SupervisionUnavailable` contract also denotes no process creation. Other
errors after entering preparation cannot be treated as proof of non-execution.

The current Python process keeps a bounded registry of at most 128 active or
held scopes. Canonical path is the primary key, with device/inode as an
additional witness. An active scope rejects concurrent execution without
poisoning the original owner. Unknown execution stays held with no age-based
eviction. Replacing a directory at the same canonical path or renaming the
same directory inode does not grant another invocation. Proven completion or
proven no-execution releases the slot. A callback refusal still raises HOLD;
only confirmed cleanup of the blocked worker allows its active slot to be
released. Latched supervisor uncertainty is not a second STOP opportunity.

This registry is process-local. It does not survive controller restart, adopt
orphans, coordinate independent controllers, or authorize recovery from a
persisted native action. No database/effect lock is held across worker or
provider waiting. Operator recovery must not interpret restarting the
controller as proof that an old worker has stopped.

## Scope and evidence limits

The legacy admission gate opens existing SQLite state read-only, without
bootstrap, migration, or writes. Declared task routing, a claim database route,
or any registered attempt history for this task prevents tokenless repair.
Genuinely absent default databases and readable pre-native databases without
this task's attempt history retain the legacy path; explicit missing or
unverifiable authority fails closed.

This is a process-completion boundary, not a filesystem sandbox, executable
content attestation, immutable patch transaction, verified patch-correctness
check, scientific observation, provider-cost budget, or security boundary
against a hostile same-UID process. Existing command/tool policy still applies.
`FIX_APPLIED` remains the fixer's textual attestation, not independent proof
that a patch is correct. No real provider, GPU training, or host process sweep
was used for this slice's focused acceptance tests.
