# Paired research comparison

This application separates a fixed comparison plan from the workload
adapter that verifies actual research evidence. Check/replay/audit commands
are read-only. The explicit `execute-campaign` command can run a frozen
scheduling campaign through normal Pro model calls and native CPU workers;
see the [execution and collection guide](../../docs/research-campaign.md).
No command authorizes spending or changes Orze's autonomous stopping policy.
The legacy adapter reads the **already published** 24 CPU pairs from
2026-09-12. The scheduling auditor verifies the task-quality and native-ledger
portion of a captured run. Neither reanalysis creates new research evidence.

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

## Scheduling task evidence

`scheduling.audit_scheduling(capture, **scope)` re-evaluates candidate artifact
bytes with the existing public scheduling evaluator. It accepts any instance
supported by that domain; it does not select a candidate or search for a better
schedule. The public examples are already known, not new held-out tasks.

The scope contains `instance`, `protocol`, `expected_attempt_refs`,
`selected_ref`, and optional `confirmation_ref` and `worker_command`. Each Ref
has exactly `task_id`, `phase`, `attempt_id`, and `generation`. Obtain the full
expected attempt list and selection from the experiment controller, including
failures. Do not infer the universe from whichever successful results remain.
The default command is the current Python executable and scheduling module;
an explicitly pinned historical command can be supplied without executing it.

The capture uses the existing `examples.holdout.testing` collection format:
configuration, complete database rows, artifact byte strings, ordered owned
controller calls with closure receipts, and before/after snapshots starting
before any native attempts. Source producers must close in an earlier
controller invocation. Multi-action controller invocations and existing-history
campaigns need a collector/adapter extension; this format does not silently
guess their internal execution order. The auditor reads no path embedded in
the capture and runs no captured command.

For a saved capture and scope, supply their independently retained SHA-256:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python3 -m examples.research_comparison \
  audit-scheduling --capture /absolute/capture.json --capture-sha256 CAPTURE_SHA256 \
  --scope /absolute/scope.json --scope-sha256 SCOPE_SHA256 \
  --output /absolute/new-audit.json
```

Both files are bounded to 64 MiB, must be regular files, and must match their
complete digests. Duplicate JSON fields are rejected. Output is create-only;
exit 0 means the evidence audit ran, including when the candidate is invalid.
An unavailable input or failed output exits 1; a failed sync can leave a file.
Neither the digest nor exit code authenticates the collector or certifies a
complete research campaign, provider bill, or scientific improvement.

The audit checks request/config/action/data identity, full native references,
process closure, settled reservations, all artifact and observation membership,
source producer/effect bindings, and controller clocks. It accepts the same
bounded YAML or JSON proposal configuration as the actual Core consumer. It
recomputes feasibility and objective value from candidate bytes and compares
the result with both the evaluation output and published observation. A model
score, internally consistent forged envelope, or producer label is insufficient.

Independent confirmation here means **another evaluator attempt on the same
candidate artifact**, not independent candidate production. Missing or invalid
confirmation prevents a confirmed-quality result; a valid zero remains valid.
Observations are separated by evaluation protocol. Only the requested protocol
counts toward the comparison's coverage gate; other protocols and failed
evaluation attempts remain visible. No global optimality is asserted.

`measurement` uses the comparison reducer's shape. Native attempt count,
reserved seconds, evaluator attempt count (including failures), summed native
elapsed time, and summed captured controller wall time are verified. Controller
wall time excludes gaps between invocations; the collector must capture every
relevant invocation before calling it a whole-project cost. Producer and failed
worker CPU/wall measurements are incomplete, so those totals stay null. So do
first evidence consumption, selection-decision latency, provider calls/tokens,
USD and GPU time; completed evaluation is not policy consumption. Separate
provider usage evidence may supply its own verified metrics, with unknowns kept.

`verify_scheduling(capture, task, scope=scope)` recomputes the audit and checks
`task.inputs.data == digest(instance)`, the evaluator identity from
`evaluator_identity(protocol)`, and maximize direction. This is the task part
of a workload adapter. It does **not** verify arm/model/tool/environment,
initial-state provenance, whole campaign coverage or external resource ledgers.
The output explicitly carries `campaign_identity_verified: false`. A complete
adapter must establish those missing facts separately before using a task audit
to claim an A/B research result. Prospective budgets remain unqualified while
required cost metrics are unknown.

### Keep the selection rule in one place

The optional [`MinimumLoss` example](loss_selection.py) provides the model's
selection instructions, the selector and an exact declared-bound check from one
project rule. Supply only verified, valid development candidates in their original
order. Ties retain the earlier candidate; secondary objectives do not change it.

```python
from examples.research_comparison.loss_selection import MinimumLoss

rule = MinimumLoss("error rate", lower_bound=0)
project_rules += "\n" + rule.instructions()
selected = rule.select(valid_development_candidates)
if rule.at_declared_bound(valid_development_candidates):
    confirm(selected)
else:
    request_next_experiments()
```

The callbacks above belong to the project; the helper grants no execution or
settlement authority. Declare a lower bound only when the metric definition
establishes it. The smallest observed value is not such a proof. Near-zero losses
do not trigger the check, and any loss below the declared bound raises an error.
With no bound, selection still works and the check never stops search. The helper
neither verifies observations nor claims convergence or heldout quality.

The [executed objective-alignment comparison](../../docs/plans/2026-09-17-research-selection.zh-CN.md)
reports both useful stops and worse heldout selections. Its two additional CPU
projects execute this helper and confirm full prediction identity. It is an
optional project example, not a default stop policy.

## Development coverage and group errors

`data_coverage.py` supplies measured facts that can accompany a project's
verified candidate history. Pass training and development features, explicit
group IDs, and any categorical feature names:

```python
from examples.research_comparison.data_coverage import (
    coverage_summary, group_error_summary,
)

facts = coverage_summary(
    train_X, development_X, feature_names, train_groups, development_groups,
    categorical_features=["origin"],
)
errors = group_error_summary(development_y, verified_predictions, development_groups)
```

The summary counts values outside each training range, unseen declared
categories, overlapping groups, and unequal group sizes. Per-group errors
retain every group and its row count; recover pooled MSE with row weights.
The concentration measure `1 / sum(group_row_fraction ** 2)` describes how
unevenly rows occupy groups. It does not estimate statistical independence.
Feature ranges alone cannot establish joint support or future generalization.

This dependency-free adapter fits no model and changes no selection rule.
Supply development inputs only and bind the facts to the same verified inputs
and predictions as the project history. An exposed
coverage gap does not justify automatic clipping or a default model change.

## Compute a two-model mixture from existing predictions

[`prediction_mixture.py`](prediction_mixture.py) finds the best convex pair in a
finite pool of verified development predictions, including every single model.
It fits no estimators and uses no numerical optimizer:

```python
from examples.research_comparison.prediction_mixture import best_pair

choice = best_pair(
    development_row_ids, development_targets,
    [{"id": candidate_id, "row_ids": row_ids, "prediction": predictions}, ...],
)
```

Each pair predicts `w * left + (1 - w) * right`. Squared error is a quadratic in
`w`, so its clipped analytic minimum suffices. Endpoints preserve predictions
exactly; strict improvements replace the current choice and supplied order
breaks exact ties. Candidate IDs must be unique, prediction rows must match in
order, and targets and predictions must be finite numbers. The work is
quadratic in the number of supplied models and linear in evaluation rows.

The caller verifies source artifacts, records this adaptive development
selection, and refits the selected components for independent confirmation.
The result proves optimality only among these pairs on these development rows.
It does not prove heldout gains, optimality among larger ensembles, or that
further code research is unnecessary. Use it when saved predictions answer the
current numeric question; it is an optional project helper.

The [executed multi-round action comparison](../../docs/plans/2026-09-17-research-action-choice.zh-CN.md)
shows why this distinction matters. On one QSAR partition, the exact pair
reduced development MSE by 15.32% but increased final refitted holdout MSE by
20.15% against the initial selected model. A separately generated training
program improved holdout MSE by 10.47%. The model-assisted pair trajectory
and a zero-model pair calculation produced identical final predictions;
that supported removing redundant calls for that result, without establishing
that pair-first research would choose a better result. Keep the helper optional
and confirm the actual selected training procedure.

## Measure a simplification through the complete CPU workflow

The [ensemble-weight follow-up](../../docs/plans/2026-09-17-research-weights.zh-CN.md)
publishes the [original winning program](../../docs/evidence/runs/2026-09-17-research-weights/weighted_ensemble.py)
and an [equal-weight version](../../docs/evidence/runs/2026-09-17-research-weights/uniform_ensemble.py).
Both use the same six estimators and feature transforms; the latter removes
repeated grouped out-of-fold prediction and ensemble-weight fitting. The
individual RidgeCV estimator still performs its own regularization selection.
These recipes expect the documented eight QSAR descriptor columns in order;
they are task examples, not generic feature definitions.

With the existing development-loss selector, 16 new cases produced one better
final result and 15 identical prediction vectors. Independently rerunning both
complete CPU evaluation/selection/refit workflows reduced summed project time
by 28.80%, compared with roughly 90% for the isolated program evaluation.
The original partition and some direct program comparisons regressed, including
all eight shuffled-target controls. Keep the simplified program as an optional
candidate. The repeated cases establish measured workflow costs, not new
independent quality samples or model-driven autoresearch speed.


## Separate the generated candidate from the final choice

The [Auto MPG study](../../docs/plans/2026-09-17-research-final-choice.zh-CN.md)
compares the initial winner, the development minimum after two actual code
experiments, and an explicit model choice from that same measured history.
A fixed native action accepts only an existing candidate ID, checks its identity
and predictions, and reuses its development result without another fit. Freeze
the selected training procedure before capturing holdout data, then refit it on
the declared final training pool. Preserve an explicit choice even when its
development score is worse; do not silently replace it with the automatic
minimum. Empty or invalid choice handling must be declared before evaluation.

On two overlapping new-task partitions, final model choices reduced holdout MSE
by 14.00% and 5.77% against the initial 128-candidate winner. The second required
selecting a slightly worse development-ranked program. The extra selection
steps cost 238.02 seconds across four projects; this is quality evidence with
additional decision cost, not an equal-quality speedup. Three negative-control
programs repeated an existing mean prediction on development rows. Complete
programs, failures, usage and confirmations are retained in the linked evidence.

The [early-finish follow-up](../../docs/plans/2026-09-17-research-early-finish.zh-CN.md)
exposes the same explicit choice from the first decision. Across four paired
workflows, final quality was better in two, identical in one and worse in one;
calls decreased from 12 to 10. The identical-prediction negative control chose
the existing candidate immediately and took 73.19% less complete project time
(54.57% less after excluding source-review pauses). The real Servo improvement
occurred with all three calls, while Forest Fires regressed with all three calls.
Changing the available actions can change the entire research trajectory; these
results do not isolate stopping from candidate generation or justify a default.
Keep direct selection optional, preserve confirmation, and measure whether
retrieving past evidence changes a later experiment or final choice.


## Check whether additional history changes a decision

The [native history view study](../../docs/plans/2026-09-17-research-history-read.zh-CN.md)
starts each workflow with 516 actual distinct candidate evaluations. Both arms
use the same 32 KiB snapshot. An experimental projection omits artifact records
and input binding metadata only after full qualification, while preserving all
observation values, validation, comparison scopes and evaluator identities.
These omitted fields include producer mappings, so the treatment does not
establish equivalence of every potentially useful provenance detail.

The first snapshot showed 24 versus 36 candidates. The model cited an added
record and then improved a generated program using measured positive residual
bias; its final Servo holdout MSE was 37.17% lower than the unchanged baseline
chosen by the control. Both negative controls regenerated an existing mean
prediction and finished with identical predictions, 3.85% worse than the initial
kNN. Calls were 5 per arm; total project time increased 2.68% in the experimental
arm. No explicit page or ID read occurred in any of the ten responses.

This is evidence of one local research-quality gain, not demonstrated later-page
retrieval, persistent-memory value or a complete speedup. Choose a consequential
unresolved comparison for the next retrieval evaluation; an already supplied
champion can make additional same-scope scores unhelpful. Reusing an already
measured complete configuration directly is also worth testing, since requiring
an opaque candidate ID can encourage another lookup or equivalent new code.
