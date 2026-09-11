# Native artifact preflight supervision

This V1-05C2d3 contract applies to the configured, domain-independent CPU resolver
called with an existing IdeaLake/claim. It is not a data/model-specific resolver,
a sandbox, or a scientific artifact validator. The no-Lake compatibility path
retains its old boolean and process-group limitations; removing native routing
or configuration cannot downgrade existing native history to that path.

## One invocation, one source

A native resolver has an independent `artifact_preflight` AttemptRef. Its action
ID derives from the task and captured claim ID; it is not the training claim ID.
The binding pins the claim bytes and CLAIMED lifecycle, scope, command, timeout,
working directory and the existing four preflight identity fields. A changed
claim can create a new generation; an uncertain current action cannot replay.

The resolver receives the existing network policy (`inherit`, `required` or
`offline`) and hidden accelerator environment. NVIDIA visibility is `none`;
CUDA, HIP and ROCR visibility are empty. This is CPU admission, not an allocation
or proof that arbitrary user scripts cannot access devices or networks.
Ordinary secret environment values are detached and rechecked locally before
GO, but are neither persisted nor fingerprinted in durable metadata. Reusing a
closed same-claim result proves that earlier invocation only; it does not prove
that secret values are identical across a controller restart.

## Closure precedes publication

Native execution uses the existing owned Linux supervisor with READY/GO. Actual
script/config hashing is outside the SQLite writer; short transactions check
current identity and file-stat witnesses. Inputs remain mutable configured files,
not globally locked executable snapshots. Before GO, the invocation and current
claim/runtime/launch authority are checked again.

The parent owns two output pipes with nonblocking read ends. It hashes actual stdout and stderr
bytes without retaining plaintext. It polls tree completion and a deadline while
draining bounded chunks. Timeout makes one explicit owned STOP request. A passed
receipt requires a complete TREE_CLOSED/ECHILD_WALL proof, real EOF on both pipes,
exit zero, and no STOP or forced-cleanup override. Missing EOF after closure has
a bounded grace period and then HOLD; no unbounded reader-thread join remains
in this native path. STOP followed by exit zero does not become success.

Unknown READY ownership, missing closure, altered identity or unconfirmed effects
raise HOLD, not False. No second invocation, ordinary failure report, later hook
or training launch is authorized by that exception. Confirmed configuration
refusal before prepare/READY can instead commit NOT_STARTED: it has no process
identity, fabricated tree proof or fabricated output hashes. Invalid native
budgets and malformed command/argument forms are rejected before execution intent.

The receipt preserves common audit/identity fields and adds the full AttemptRef.
Native timeouts carry the actual exit code; their budget is pinned in the attempt
binding, not a promise of exact field parity with the legacy timeout JSON.
Its exact bytes are read back and hash-bound to the owned effect transaction.
Current cached success/failure needs both that receipt and confirmed native
terminal/effect authority; a copied legacy passed JSON is not native authority.

## Consumers and confirmed failure

Training and posthoc capture a source outside the writer. Their launch intent
stores `artifact_preflight_source`: the full resolver reference, receipt hash,
claim ID/hash and preflight identity. The same reference is watched in the writer.
READY/GO rechecks the captured source and actual inputs; it never silently selects
a newer receipt. A started consumer legitimately adds trainer fields to the claim.
It must still have its own current RUNNING identity/lifecycle and exact original
source binding, rather than pretending the old CLAIMED claim bytes remain current.
Source rejection after READY stops the owned process but does not publish a failed
consumer terminal under rejected authority. Such intent remains for explicit
recovery; this slice does not implement adoption or automatic repair.

Native pre-script entry points also refuse unresolved or same-claim failed resolver
history. Disabled preflight with no history does not mandate a resolver or bind
unrelated configuration. Dropping a config switch, Lake argument or in-memory
route cannot erase existing native history.

A confirmed failed/NOT_STARTED resolver can be reported through an independent
controller action. The fixed-policy reporter shared with pre-script validates the
exact source and claim, writes/read-backs failure metrics and zero-GPU admission,
records the legal CLAIMED-to-FAILED edge and increments failure count once.
Duplicate delivery rechecks that action's FAILED fence; stale references are inert.
The existing in-memory dispatch backoff is set only after confirmed reporting.
There is no new durable scheduling/backoff or implicit fixer invocation.

## Evidence and remaining work

The evidence distinguishes preceding-source failures, first-green new mechanisms,
draft failures, explicit protocol doubles and real CPU process tests. The latter
use tiny temporary scripts/adapters, owned pidfds and real SQLite; accelerator
allocation/telemetry are replaced, and no models, datasets or providers run.

This closes native resolver supervision and its source consumers only. Legacy
execution, fixer/director ownership, crash adoption/repair, durable progress
consumption and the generic CPU autoresearch loop remain separate V1 work. Neither
local test counts nor tree closure establish scientific validity or research
throughput improvement. The independent holdout remains unread until interface freeze.
