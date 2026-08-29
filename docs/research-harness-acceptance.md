# Research harness acceptance contract

An efficient research harness maximizes valid decision information per unit of
wall time and compute. Process activity, queue depth, and raw experiment count
are not success metrics. Every accepted target below needs a durable artifact
and an executable verifier.

## Dimensions and targets

| Dimension | Target | Required evidence |
|---|---:|---|
| Time to reject invalid work | 0 GPU-seconds | Production-entry preflight receipt plus compute ledger showing no GPU lease |
| Time to first valid decision | Configured campaign SLA | Terminal decision receipt timestamps, not process logs |
| GPU ownership | 0 cross-scheduler collisions | Atomic campaign-lease contention fault test and lease receipts |
| GPU scope | 100% launches within allowlist | Launch receipts with physical GPU IDs; forbidden-ID negative tests |
| Evaluation code identity | 0 mixed-version executions | Content-addressed bundle manifest, receipt binding, concurrent-mutation fault test |
| Data separation | 0 forbidden sample identities | Signed exact-row manifests and overlap verifier output |
| Result honesty | 100% ranked rows and decisions contract-qualified | Benchmark receipt, exposure ledger, lineage, report qualification audit, and decision-input content identity |
| Model eligibility | 1 model, 1 pass, no routing | Receipt fields and artifact lineage verifier |
| Reproduction efficiency | No redundant replicas without declared question | Batch contract declaring seeds/replicate purpose and unique-artifact accounting |
| Recovery correctness | 0 live process / terminal-state contradictions | Process-bound FSM receipts and stale-process fault tests |
| Environment identity | 1 resolved dependency graph per run | Interpreter/package/path manifest verified in a fresh process |
| Operator visibility | Status update within configured SLA | Heartbeat containing last valid artifact, blocker, and next deadline |

## Phase99 baseline (2026-08-28)

The Phase99 incident is a failing baseline, not evidence of target attainment:

- Four jobs used about 126.7 GPU-minutes but produced one unique model hash.
- No complete benchmark receipt was produced.
- One legacy scheduler collision caused a GPU-5 OOM.
- A production import-path failure and a Switchboard loader-pin mismatch reached
  execution because the exact entry paths were absent from preflight.
- Editing a child proxy while an old evaluator remained alive produced an
  invalid mixed-version attempt.
- One process was gone while its FSM row remained `IN_PROGRESS`.

## Verification rule

A target is achieved only when its named verifier covers the production path,
passes a positive case, rejects the corresponding injected fault, and writes a
durable receipt. Unit tests of helper functions alone are insufficient for an
end-to-end claim. Any target without this evidence remains open.

The aggregate acceptance matrix is intentionally stricter than an individual
green gate. Its requirement universe is owned by Orze source, not by the input
manifest, so a project cannot become green by omitting recovery, campaign, or
official-result rows. Each declared proof binds an exact single-link JSON
receipt SHA-256, a receipt-owned status pointer, and scope assertions. Missing,
rewritten, redirected, hard-linked, contradictory, or concurrently changing
evidence is `UNVERIFIED`; a receipt-owned `FAILED` result remains `FAILED`.
Only when every required dimension is independently `VERIFIED` is the aggregate
status `VERIFIED`. In particular, a readiness receipt cannot prove research
yield or an official leaderboard outcome.

```bash
python -m orze.engine.acceptance_matrix \
  --manifest /path/to/acceptance-manifest.json \
  --output /path/to/acceptance-receipt.json
```

## Current implementation status

- Evaluation code identity: implementation present. The launcher executes a
  content-addressed bundle; bundle identity is bound into benchmark provenance
  and receipts. Positive launch, source-drift, symlink, bundle-tamper, missing
  receipt binding, and concurrent working-tree mutation cases are tested.
- GPU ownership primitive: implementation present. Orze controllers acquire
  kernel leases for their full physical-GPU scope, and external schedulers can
  participate through `orze gpu-lease-run --gpus 4,5 -- <command>`. Tests prove
  a contending external command never starts, a disjoint command executes, and
  a child retains ownership if its wrapper is killed. Project attainment stays
  open until every GPU-capable cron/Slurm entry is migrated to this boundary.
- Recovery correctness: staged lifecycle implementation present. Training and
  evaluation have separate current-state rows and immutable transition ledgers;
  launch and terminal stage writes share the global lifecycle transaction.
  Completed training remains globally `IN_PROGRESS` while evaluation is
  pending, evaluation failure preserves `training=COMPLETE`, and retry resets
  both stages. Fault injection proves a failed stage-audit write rolls back the
  global launch edge. The read-only recovery auditor verifies the shared SQLite
  policy, legacy/global state agreement, training/evaluation stage pairs, the
  final immutable transition for each current state, and nonce-bound local
  claim process liveness. It never runs startup reconciliation or changes a
  lifecycle row. Missing stage history or a remote/unreadable claim is
  `UNVERIFIED`; a proven live-process/terminal-state, dead-process/active-state,
  or ledger contradiction is `FAILED`.
- Campaign outcome efficiency: implementation present. A write-once
  preregistered campaign binds the exact decision receipts and targets. The
  analyzer validates resolved qualified-success identities, closed framework
  compute receipts, physical GPU scope, zero-GPU rejections, retry/duplicate
  counts, GPU-hours per success, decision latency, and current artifact lineage.
  Its required reproduction contract either gives a bounded not-applicable
  rationale or preregisters disjoint replica groups, the only scalar config
  paths allowed to vary, and an absolute metric tolerance. The analyzer checks
  every config against its immutable prospective canonical SHA-256, checks
  stored config integrity, rejects exact duplicates and undeclared drift, and
  accepts metric tolerance only from lifecycle-complete qualified evidence.
  Decision receipt schema 2 also records, for every terminal idea, the exact
  lifecycle state, qualification reason, finite primary value when qualified,
  and one content SHA-256 covering every file that affected qualification. Evidence is
  hashed before and after qualification, so concurrent rewrites cannot produce
  a decision. The closed-campaign audit requalifies those inputs and rejects a
  later metric, lifecycle, benchmark-receipt, provenance, or exposure-ledger
  rewrite as `UNVERIFIED`. Redirected or hard-linked evidence is rejected
  without following it. Legacy resolved receipts remain readable but cannot
  prove a current campaign outcome because they lack this input identity.
  The scheduler-efficiency analyzer scopes lifecycle latency to the exact
  preregistered idea universe, rejects any unexpected in-window allocation, and
  requires one physical host so cross-host GPU IDs cannot be conflated.
  Each scheduler observation is paired with a durable, operator-visible
  progress update containing the current qualified-artifact identity (or null),
  a categorical blocker, and the next deadline. Missing updates are
  `UNVERIFIED`; a complete campaign exceeding the 10-minute default SLA is
  `FAILED`.
  Missing evidence is `UNVERIFIED`; complete evidence that misses a target is
  `FAILED`. Synthetic production-path tests cover passing, zero-yield, missing
  terminal, scope, and artifact-relation cases. Project attainment remains open
  until a real resumed campaign produces its receipt.
- Official leaderboard outcome remains outside local verification and requires
  an accepted public submission under the leaderboard's complete rules.

## External scheduler integration

Every process capable of selecting or exposing a GPU must acquire the same
physical-device leases. For example:

```bash
orze gpu-lease-run --gpus 4,5,6,7 -- python scripts/drain_queue.py
```

The scope must cover every physical GPU the wrapped command may select. Lease
contention exits with status 75 and never starts the command, allowing cron or a
batch scheduler to retry safely. The wrapper passes the lease descriptors into
the child; killing only the wrapper cannot create an unleased detached job.
GPU-capable commands that bypass this boundary remain outside Orze's ownership
guarantee and must fail the project acceptance audit.

Kernel leases coordinate only participants in the Orze protocol. After acquiring
a new controller or external-wrapper lease, Orze also queries compute metadata on
exactly the leased physical IDs and rejects any pre-existing compute process before
starting a child. This catches Docker services and other schedulers that ignore the
lease files. The rejection exposes only the physical GPU number, never the foreign
PID, process name, command, or environment. A privileged scheduler can still race
after this attestation; whole-host ownership therefore remains failed unless every
scheduler participates in the lease protocol or an external platform enforces
exclusive allocation.

## Duplicate-training admission

The final training boundary hashes the effective recipe, base config, trainer,
Python command, extra arguments/environment, and data contracts, then atomically
reserves that identity before GPU telemetry or process creation. Code-owned
proposal metadata—titles, hypotheses, `_`-prefixed fields, and replication
labels—does not change the identity. Consequently, renaming an exact recipe or
changing only its replica index cannot bypass admission; concurrent equivalents
have exactly one winner. A future exact-reproduction exception must be backed by
an explicit preregistered contract rather than by weakening this default.

## GPU scope audit

The scope auditor checks the complete code-owned subprocess boundary universe,
requires every GPU-visible child environment to pass through the shared physical
allowlist guard, binds the exact source and project configuration, and launches
one trivial no-CUDA child under each permitted lease. Forbidden-ID faults stop
before a child is created. It does not query or open an accelerator:

```bash
python -m orze.engine.gpu_scope_audit \
  --config /path/to/orze.yaml \
  --physical-scope 4,5,6,7 \
  --forbidden-scope 0,1,2,3 \
  --output /path/to/gpu-scope-receipt.json
```
