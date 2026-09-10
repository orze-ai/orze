# Evaluation completion contract

Training completion, evaluation completion, and a rankable observation are
different facts. An evaluator may complete with zero observations when no
primary metric was declared. A ranking consumer still requires a qualified
declared objective and authoritative lifecycle completion.

## Entry points

Existing-output reconciliation, asynchronous completion, and blocking
`run_eval` use the same artifact validator. It checks the completed training
document, sealed files, the designated evaluation document, exact report
sources, metric validation/coverage, and configured lineage, clean-access and
benchmark policies. It does not manufacture a set of completed idea IDs to
authorize itself. Normal process completion additionally requires exit zero;
the blocking wrapper delegates its closure to the asynchronous implementation.

The designated output must be a JSON object when present. Its optional
`status` must be `COMPLETED`; missing status remains compatible. Missing output
is allowed unless declared metric sources or a benchmark contract require it.
Unrelated domain documents do not inherit this framework status convention.

An active `eval_script`, or an explicit report source referencing `eval_output`,
gives that document qualification semantics. Even when the score comes from
`metrics.json`, a present failed evaluation output vetoes ranking. This policy,
its output bytes, and its activation/path changes affect result-cache identity;
the policy changes also isolate champion-guard history.

## Admission and failure

Training eligibility is checked before considering an existing evaluation
output. Invalid existing output is preserved and reconciled as failure, not
silently overwritten. A configured evaluator returning no process is not a
successful completion: the scheduler requires matching terminal stage/global
state or retains pending work. Deduplication applies within one dispatch tick.

`eval_output=metrics.json` (including `./metrics.json`) does not make an existing
training artifact proof of completed evaluation. Evaluation still launches
when pending. Symlink, hard-link and escaping output paths are rejected. The
framework's failure fallback never overwrites training metrics or existing
output. This does not prevent a legacy evaluator itself from modifying its
in-place training file.

Normal closure writes a terminal compute receipt before publishing successful
lifecycle completion. If an owned process cannot be confirmed stopped after
timeout/observation failure, the error is explicit and no terminal receipt is
invented. This is not a claim of distributed exactly-once execution.

## Remaining boundaries

- Evaluation-only retry, immutable generation snapshots and attempt fencing
  remain separate work; ordinary whole-idea retry must not be presented as
  preserving a completed generation.
- Stage, filesystem and receipt updates are not one atomic transaction.
  Cross-controller recovery, stale-attempt protection and uncertain external
  effects require the subsequent state/handoff work.
- Legacy no-lake dispatch cannot establish completion from a `None` return.
  Backlog still respects known-project-config and configured checkpoint gates.
- Completion tests use temporary files and real SQLite/receipts, with GPU and
  subprocess boundaries replaced. They prove these mechanisms, not real GPU
  performance, research quality, paid-provider behavior or a complete CPU loop.
