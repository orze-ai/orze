# Native post-script supervision

This contract covers only `run_post_scripts(..., source_event=CompletionEvent)`
and its source-bound native action adapter. It does not change the legacy
tokenless path, pre-scripts, artifact preflight, fixers or Director operations.
A post-script action is operational work, not a scientific observation.

## Identity and admission

An action retains the existing deterministic identity derived from the full
source AttemptRef and command. Different commands create separate post_script
generations. An already accepted historical action is not automatically rerun
because its timeout or environment changes. Historical records are not
retrofitted with process-tree proof.

Before creating LAUNCHING intent, the native adapter validates a strictly
positive, finite numeric timeout (default 3600 seconds; booleans and numeric
strings are invalid) and captures bounded, detached command/environment input.
Only their hashes, the normalized timeout and source/physical-resource identity
are persisted; environment values are not copied into the attempt record.
This pins the admitted invocation, not the contents of arbitrary script or data
paths, and it does not turn configuration into a scientific validity judgment.

The existing source completion and real catalog remain authoritative.
Unclosed prior actions prevent new actions. At the public caller, output-exists
and training-eligibility skip policies cannot silently resolve a pending native
action. A skip with no pending action remains a policy decision, not an execution
completion or output attestation.

## Process ownership and publication

The adapter uses the shared Linux subreaper protocol documented in
[native evaluation supervision](native-evaluation-supervision.md), with the
explicit execution phase `post_script`. It does not discover ownership from a
PID, process name, GPU or environment marker.

LAUNCHING is durable before process creation. The real worker is held behind
READY while its compute allocation and complete supervision binding are
registered. After a normal READY handoff, GPU lease release cannot precede the
READY compute-start record; an uncertain pre-READY handoff may have no such record.
GO follows fresh checks of the current action, accepted source, runtime,
launch/resource/campaign authority and unconfirmed-stop gate.

Neither a leader's exit code nor an output file authorizes terminal publication.
The exact current action must have the matching TREE_CLOSED receipt, checked
before and again inside the short terminal transaction. The complete closure
is bound into the effect plan and action terminal, together with actual exit
code and verified compute accounting. The source dependency is watched without
rewriting its lifecycle or manufacturing another evaluation.

A timeout requests owned termination once. STOP or forced cleanup cannot
produce a successful action solely because the worker exits zero. Unknown
prepare, supervision or stop results retain a nonterminal action/HOLD and do
not authorize blind retry. No blocking process waits or spawning occur while holding
the SQLite writer or task effect lock.

## Limits

This is not a script sandbox: user code can have external effects, and its
arbitrary output paths are not automatically isolated or registered as B1/B2
artifacts or observations. A closed script is not proof that its scientific
results are correct.

The timeout bounds active waiting, not controller downtime or all admission
latency. Restart adoption, detached-handle reconstruction and operator resolution
of uncertain outcomes remain separate work. Old native RUNNING actions without
supervision are not adopted by guessing their PID. No generic CPU research loop,
real GPU/model/provider workload or research-efficiency improvement is asserted
by these mechanism tests.
