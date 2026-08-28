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
| Result honesty | 100% ranked rows contract-qualified | Benchmark receipt, exposure ledger, lineage, and report qualification audit |
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
- Every other dimension: open until its required production-path evidence is
  implemented and verified.

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
