# Native training process-tree completion

V1-05C2b extends the [native evaluation contract](native-evaluation-supervision.md)
to newly launched native `kind=train` attempts. It uses the same Linux
subreaper/READY/GO/TREE_CLOSED protocol and shared, phase-qualified validation.
This is an execution-safety adapter, not a generic CPU research loop or a
domain-specific training optimization.

## Launch and publication

1. Persist LAUNCHING with `orze.linux_subreaper.v1` before creating processes.
2. Prepare a blocked worker. Its real PID remains the public/compute/lineage
   PID; the supervisor PID is separate. Verify the current claim, revision,
   protocol and READY before recording the allocation start inside the GPU
   lease. READY has consumed an allocation even if leaving that lease fails.
3. Construct the public holder. Revalidate replication and resume admission,
   then bind READY, claim identity and IN_PROGRESS in the existing transaction.
   Preserve artifact/replication bindings rather than reconstructing a subset.
4. Outside the writer, recheck captured executable/config identity and resume
   evidence. Recheck current ownership and dynamic runtime, pause/stop, GPU
   scope and campaign authorization before sending GO. READY is not continuing
   permission to execute changed inputs. This is a final local check, not an
   atomic filesystem snapshot or protection against external writes after GO.
5. Receive lineage attestation only after GO. The worker emits its nonce after
   boundary setup; waiting before GO would deadlock. `worker_only_fds` carries
   the lineage writer to the worker, while the supervisor closes its copy
   before READY. Shared `pass_fds` remain held by the supervisor until closure,
   so an early lineage EOF does not release an allocation with live descendants.
6. Natural worker exit leaves useful descendants running. Poll returns a
   terminal integer only after exact TREE_CLOSED and normal supervisor exit.
   Metrics reads, lineage finalization and B1 copy/registration require closure
   first. Recheck that exact receipt inside the short terminal transaction and
   include it in both the effect plan and terminal record.

Both descriptor groups are explicit lists/tuples, together at most 64 distinct
valid integer FDs >= 3. Closed FDs, duplicates, overlap and invalid values are
rejected before Popen. A failure closing worker-only copies after fork uses
owned cleanup, never READY or a guessed NOT_STARTED. The internal setup frame
is additive; persistent protocol and receipt schemas do not change. Temporary
FD numbers are not persistent ownership or recovery authority.

## Stops, failures and compatibility

Direct finish, failed-launch, requeue and shutdown paths verify the same
phase-qualified binding and closure; the polling entry is not the only gate.
STOP/forced cleanup preserves the actual worker return code. Even code zero
cannot become completed or trigger automatic `insufficient_vram` retry.
An explicitly requested, confirmed scheduler-slot rollback may still requeue;
it is not a measurement or an automatic retry inferred from trainer output.

Uncertain preparation can have created a supervisor even when the caller's
local process variable is None. It retains intent/reservation and cannot be
relabeled NOT_STARTED. Known initialization failures use owned stop before
closing an attempt. Changed runtime/inputs propagate rejection after stopping
the local tree and preserve intent for explicit recovery; no success or
automatic retry is granted. Lost supervision during monitoring keeps the
active handle and RUNNING ownership held.

A confirmed post-GO initialization failure is reported using its published
running lifecycle fence. Retained prelaunch history cannot misclassify it as
a stale CLAIMED request; an invalid running fence never falls back to that
older authority. Duplicate failure reporting retains once-only accounting.

The mutable `is_posthoc` flag cannot reroute a native training handle or active
native training history through legacy posthoc stop/publication. Historical
native rows without the protocol cannot gain it from a PID or callback.
Genuine pre-native/legacy-import completion keeps its explicitly limited old
compatibility; it is not process-tree proof.

The worker-PID CPU/GPU zombie and triple-signal watchdog heuristics do not cover
children adopted by the supervisor. They therefore do not infer inactivity
for supervised handles. Explicit timeout, configured log/progress stall,
fatal-log and administrative stop policies remain available. The public PID
and compute identity are not replaced merely to obtain telemetry.

## Evidence limits and remaining work

Actual tiny CPU workers test detached/double-fork/empty-environment descendants
holding output descriptors, direct publication, STOP with worker code zero,
shutdown and supervisor loss. Lease/accelerator boundaries in those fixtures
are simulated; a separate real flock test checks FD retention without a GPU.
Lineage pipe tests use actual nonce/receipt code with explicitly substituted
namespace setup. They prove GO ordering, EOF and worker PID, **not** actual
kernel isolation or training quality. Fake-Popen compatibility tests explicitly
install simulated supervision and retain their old business assertions.

This slice excludes posthoc workers (including their in-worker artifact
registration), pre/post scripts, director/startup OS ownership, crash adoption,
operator recovery of unknown effects, durable consumer ACK, observation-based
selection and the real generic CPU policy/domain/executor loop. HOLD is a safe
uncertainty disposition, not completion of recovery. No training job, paid
provider, GPU campaign, efficiency percentage or whole-V1 completion is claimed.
