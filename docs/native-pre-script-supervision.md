# Native CPU pre-script supervision

V1-05C2d2 covers project setup through the actual scheduler admission path.
It does not complete the generic CPU research loop or recovery/adoption work.

## Identity and continuation

A pre-script runs before a training attempt exists. It receives an independent
`pre_script` AttemptRef, bound to the exact claim ID, claim bytes, task scope
and global CLAIMED revision. It neither borrows the future training ID nor
manufactures a RUNNING training stage. One exact claim permits one invocation;
a safely rotated new claim permits a new generation. Changed command,
environment or timeout cannot silently reuse that claim's prior result.

The invocation captures a bounded detached command/environment and a finite,
strictly positive timeout. The attempt stores hashes rather than raw environment
values. CUDA/NVIDIA/HIP/ROCR visibility is cleared and the legacy `{gpu}`
argument is `-1`. The pre-script acquires no GPU lease and emits no GPU compute
start or terminal. This is cooperative CPU execution, not a GPU sandbox.

Native `run_pre_script(..., lake=lake)` returns a truth-valued
`PreScriptResult(success, attempt_ref)`. The phase passes that captured full
reference to the failure reporter; it does not look up the latest action to
guess what failed. A genuinely legacy no-Lake call retains its bool API.
An omitted optional results path is resolved from configured `results_dir`
before checking persisted native routing. Missing configuration or missing
in-memory wiring cannot erase an existing unresolved native execution.

## Owned process and terminal publication

LAUNCHING is persisted before process creation. The shared Linux subreaper
protocol holds the worker at READY; its full binding and actual PID are
registered before fresh claim/runtime/launch checks authorize GO. No process
creation or blocking wait occurs while holding the SQLite writer or effect lock.

A leader exit or an output file cannot authorize continuation. Normal completion
requires the exact owned TREE_CLOSED proof, rechecked inside the terminal effect
transaction. The result records the complete closure and actual return code,
without changing the task's CLAIMED state. Confirmed historical results may be
read back only with matching current claim, invocation and committed effect.
This is at-most-once invocation with replay of its confirmed result, not fresh
execution or scientific proof.

Timeout requests owned termination once. STOP or forced cleanup is never success
merely because the worker exits zero. Unknown preparation, lost supervision,
failed stop or partial publication raises HOLD; it cannot become an ordinary
False, authorize repair/reset, or publish a zero-GPU admission terminal.

Removing `pre_script` and direct training/posthoc launches still check existing
pre-script history. Launch entry points check before resource admission, and
native training/posthoc intent writers check again within their transaction.
Same-claim confirmed failure also blocks direct launch. No-history direct launch
retains compatibility; this slice does not add mandatory setup to every API.

## Confirmed failure and explicit repair

A confirmed failed native result is reported by a separate, exact-source,
once-only controller action. Under its effect transaction, the reporter verifies
the closed source and claim, writes/readbacks FAILED metrics and a zero-allocation
admission receipt against the original claim, and transitions CLAIMED to FAILED.
Duplicate delivery must match the committed action, current failed lifecycle and
claim; it reprojects the durable failure count without incrementing it again.
Stale delivered references do not acquire a replacement action's authority.

The native failure action records `repair_status=pending_explicit_action`.
It does not call the existing, not-yet-supervised automatic fixer. This follows
the native launch-failure boundary; legacy failure/fixer behavior is unchanged.
The existing prohibition on unqualified writes into native task directories
remains intact. Partial effect publication stays HOLD, without automatic repair.

## Limits

Standard output/error are discarded; persisted diagnostics are bounded outcome,
reason and exit code. Script contents, arbitrary inputs and outputs are not
isolated, immutable or scientifically validated by this contract. There is no
automatic B1 artifact or B2 observation registration for setup outputs.

Timeout is an active-wait budget, not a bound on controller downtime or every
admission operation. Restart adoption, detached-handle reconstruction, operator
resolution, artifact-preflight/fixer/director process ownership and the actual
general-purpose CPU research loop remain separate work. Mechanism tests use
synthetic CPU scripts or explicitly identified doubles, not GPU/model training,
paid providers or measured research-efficiency improvements.
