# Paired research comparison preparation

This offline application separates a fixed comparison plan from the workload
adapter that verifies actual research evidence. It does not run experiments,
call a model, authorize spending, or change Orze's autonomous stopping policy.
The only included adapter reads the **already published** 24 CPU pairs from
2026-09-12. Reanalysis is a measurement control, not new research evidence.

## Commands

Run from the repository root with its normal Python dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -m examples.research_comparison \
  replay-legacy \
  --evidence-dir docs/evidence/runs/2026-09-12-research-efficiency \
  --output-dir /absolute/new/comparison-reanalysis

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -m examples.research_comparison \
  check --protocol /absolute/new/comparison-reanalysis/protocol.json
```

The output directory must not already exist. Archive members are read without
extraction. Original archive, manifest and summary bytes have fixed hashes;
each archived `run.json` is rechecked against its original pointer. The adapter
recomputes closure, full attempt references, settlement, observations, artifact
bytes, independent confirmation and clocks using the existing qualifier. It
does not trust stored `metrics` or `quality_passed` fields. It records the
current verifier sources and requires all previously published group medians
and paired differences to match exactly.

The historical tasks had fixed deterministic data and no random seed. Their
seed entries are explicitly `null`. Provider calls, tokens, dollars and GPU
time have no measured usage ledger in these records, so those metrics and
limits remain `null`, even though the original declared execution scope was
CPU-only. They are not presented as measured zeros. This adapter is deliberately
limited to these pinned historical inputs.

## Protocol fields

`protocol.py` defines schema 1. Unknown fields, malformed hashes, duplicate task
identities, duplicate JSON keys, nonfinite numbers, and missing controls fail
validation. Numerical metrics use seconds, counts, and USD as named; they must
be finite and bounded by signed 64-bit magnitude. Counts are integers.

| Field | Required meaning |
| --- | --- |
| `comparison_id`, `mode` | Unique identifier; `prospective` or `retrospective_offline` |
| `repetitions`, `ordering` | Fixed number of pairs and AB/BA order: alternate by repetition, or by task and repetition |
| `arms.A`, `arms.B` | SHA-256 of the respective installed artifacts and complete treatment configuration |
| `shared` | SHA-256 of model specification, allowed tools, and environment shared by both arms |
| `verifier_sha256` | Digest of the complete evaluator/measurement adapter dependency inventory |
| `tasks[].id`, `domain`, `role` | Fixed task universe with both targets and negative controls |
| `tasks[].seeds` | One distinct integer seed per pair; both arms use the same seed |
| `tasks[].inputs` | SHA-256 of data, evaluator, instructions, initial history and initial memory |
| `tasks[].quality` | Metric direction, maximum allowed B-versus-A regression, minimum valid-observation count |
| `tasks[].limits` | Shared limits for calls, tokens, USD, reserved CPU seconds, GPU seconds and outer evaluation wall time |

Every prospective budget must be explicitly measurable, including zero where
that resource is prohibited. A null limit is allowed only when reanalyzing
historical evidence and means that endpoint is **not** budget-validated. The
model specification must include provider/model revision, sampling settings
and prompt configuration. Task instructions include task-specific prompts.
The treatment configuration identifies intended changes such as code version,
retrieval and memory use; other conditions must be shared. Input hashes alone
neither check file availability nor prove that execution consumed those inputs.

`check` validates schema and prints the fixed schedule and canonical digest.
It does **not** attest preregistration time, input availability, adapter
correctness, usage enforcement or permission to execute. Freeze actual input
files, adapter sources, artifacts, environment, seed list and stopping/cleanup
rules in a reviewed commit before running any prospective comparison. The
outer evaluator's finite budget is separate from product research decisions.

## Measurement contract

`report.compare(plan, records, verify=adapter)` is an in-process reducer. The
adapter is trusted application code and must derive values from original
evidence. It must check task/data/protocol and treatment identity, initial
state, actual execution order, current source qualification, independent
confirmation, complete failed/retried attempts, elapsed clocks and resource
ledgers. A saved success flag or model-written score is insufficient. The
included legacy adapter is not a general real-model adapter.

All scheduled runs remain in the denominator. Extra, duplicated, mislabeled,
wrong-seed, reordered or wrong-protocol records are rejected, not filtered.
A missing run or verifier failure remains missing/unknown. Known failure
costs are retained; unknown usage is never imputed as zero. For each cost,
`known_sum` is only a subtotal when `complete` is false. A verifier failure
makes that run's costs unknown rather than trusting unverifiable fields.

A pair qualifies only when both runs completed, have valid independently
confirmed quality, compatible comparison identities, sufficient declared
coverage and no unknown/exceeded required budget. Score B must be no worse
than score A within the **predeclared** tolerance. Coverage changes remain
visible, including fewer valid observations and unchanged/increased unknowns.
Tasks that require full candidate characterization need a coverage-aware
adapter; a minimum count alone cannot establish complete characterization.

Group medians require **every** planned pair to qualify. Each cost column also
requires measurements in every pair. Median paired differences are calculated
from individual B−A differences, not by subtracting arm medians. Raw per-run
outcomes remain in the report; groups/domains are not silently pooled. Reports
always set `new_research_evidence: false`: running this reducer creates no
new experiment, and passing its checks does not prove general research gains.

## Still required for real research

Select actual tasks and independent evaluators, freeze meaningful quality and
coverage criteria, implement and verify their raw-evidence adapters, instrument
provider billing including failed/retried calls, and define equal resource
limits and initial state. Then obtain the task/model/account/resource scope
needed for execution. The existing small CPU controls are public and already
known; they cannot serve as new held-out research tasks. A real model comparison
and production deployment remain open work.
