# Native evaluation process-tree completion

V1-05C2a applies to newly launched native evaluations (an actual IdeaLake
execution attempt), including the legacy report adapter and JSON observations.
It does not change the meaning of a scientific observation or optimize a
particular research domain. An evaluator's leader exiting is insufficient to
authorize publication while an owned descendant can still change its output.

## Launch and completion contract

1. Commit the evaluation's LAUNCHING intent with the supervision protocol.
2. Start a dedicated Linux subreaper and a blocked worker. READY identifies
   the actual worker and supervisor, their start ticks, the full AttemptRef,
   absolute task scope, command digest and private nonce digest.
3. Persist the worker identity and compute start, then construct the public
   holder. Recheck prepared launch inputs and controller integrity before GO.
   READY is not permission to execute changed inputs.
4. The worker executes the evaluator. Its public PID remains the worker PID,
   not the supervisor PID; existing compute identity is not silently changed.
5. A terminal integer requires both an exact TREE_CLOSED receipt and normal
   supervisor exit. The supervisor reaps all direct/adopted descendants using
   `waitpid(-1, WNOHANG | __WALL)`; only ECHILD closes the tree. A pending wait,
   empty process scan, leader exit or surviving process-group number does not.
6. Native publication verifies the complete binding and receipt before reading
   results and again inside the short publication transaction. The accepted
   process-tree receipt is included in the effect plan and terminal record.

Useful descendants may finish naturally after the leader. Normal leader exit
does not automatically kill them. The existing evaluation timeout or explicit
shutdown instead requests STOP over the private control channel. Only the
supervisor signals its own direct/adopted children, through captured pidfds;
native evaluation never falls back to process-group discovery or numeric PID
signals. No blocking process wait or subprocess launch occurs in the publication writer.

An explicit stop or forced cleanup is not successful evaluation, even when
the actual leader return code is zero. The return code is preserved and the
operational failure/interruption is recorded separately; no successful
observation is manufactured from that zero. Direct publisher callbacks cannot
bypass this requirement. Shutdown requires the same binding and closure proof.

## Failure and compatibility

The parent's platform/API/pidfd probes reject unsupported capability before
process creation. Actually enabling and verifying subreaper behavior happens
inside the spawned supervisor. Once a supervisor may have been created,
preparation, subreaper setup, transport or supervisor failure
is uncertainty, not proof of NOT_STARTED. The intent remains held; neither an
ordinary failed integer nor a replacement execution is inferred. A known
blocked worker is stopped on initialization failure. If closure cannot be
confirmed, existing durable stop/HOLD rules apply.

Controller-integrity rejection is still propagated after stopping a known
blocked worker. It does not authorize publication under a rejected runtime;
the native intent is preserved for explicit recovery. Changed bound inputs
can likewise prevent failed-terminal publication after a confirmed stop.

Historical native RUNNING/LAUNCHING rows without the new protocol cannot be
enrolled using their PID or a completion callback. They require explicit
recovery handling. Genuine no-catalog legacy evaluations retain their prior
unsupervised behavior; they do not gain this guarantee. Native evaluations now
require supported Linux supervision, with no unsafe portability fallback.

The private control descriptor and gate do not survive evaluator exec. Supplied
lease descriptors remain inherited by the worker and retained by its supervisor
until closure. Persistent identity contains digests, not raw nonce, command or
environment. Setup frames are bounded to 1 MiB; supervision sub-bindings
are bounded to 8 KiB.

This is local descendant ownership, not a hostile same-UID sandbox, atomic
filesystem snapshot, container, or protection against arbitrary external
writers. It does not yet provide crash adoption, a durable supervisor-side
receipt store, or automatic recovery after parent loss. A parent that loses
proof must HOLD rather than rerun. Training, posthoc, pre/post scripts, director
control and the generic CPU research loop are separate unfinished work.

## Evidence interpretation

The regression uses actual CPU evaluators that double-fork, detach and exec
with an empty environment. Their authenticated socket peers provide exact
pidfds, and real publication writers are observed while the escaped writer is
alive. Training/GPU setup is simulated; no GPU campaign is run. The CPU process
tests are not evidence that the product's generic CPU research loop is complete.

Old tests that intentionally used fake Popen explicitly install a test-only
supervision fixture. Their original assertions are preserved; that simulation
is not process-closure evidence. Actual CPU fixtures restore the real supervisor
entry point. Baseline behavior failures, new protocol tests and draft defects
are recorded separately in the accompanying evidence, not combined into an
inflated historical defect count.

Linux semantics: [child subreapers](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html),
[pidfd lifetime](https://man7.org/linux/man-pages/man2/pidfd_open.2.html),
[pidfd signals](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html),
and [wait/ECHILD/__WALL](https://www.man7.org/linux/man-pages/man2/wait.2.html).
