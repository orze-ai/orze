# Prepare independent research proposals while evaluating ready work

Use `run_prepared` for the high-level research execution path. It prepares up to
two independent branch proposals concurrently and evaluates ready proposals one
at a time. Explicit `plan.workers` limits are preserved, and every preparation
uses one action from the existing call budget. The exploration policy remains
`ParallelRefine`; a caller may supply another policy explicitly.

```python
from orze.research.execution import run_prepared

# Your callbacks own model admission, scientific execution and goal acceptance.
trace = run_prepared(
    spec, prepare_proposal, evaluate_proposal,
    output=fresh_output_directory,
    finished=delivery_is_final,
    unused=record_unused_proposal,
)
```

`spec` follows the [online exploration contract](research-exploration.md).
`plan.workers` may be omitted on this path and defaults to two. The specification
is copied; the caller's object is unchanged. Each preparation sees the root and
its own branch ancestry, captured before that batch's results become available.
Pro exposes the same execution through
`run_discovery(spec, prepare=..., evaluate=..., output=..., finished=..., unused=...)`.
An explicit `execute_batch` callback continues to own its own scheduling.

## Callback contract

- `prepare(context)` admits and records one authorized model request and returns
  its response and accounting metadata. It must be safe to run concurrently.
- `evaluate(context, response)` checks the proposal format, executes it in the
  application's isolated numerical worker, and returns a native outcome.
  Expected proposal or execution failures become unscored `repairable` outcomes,
  retaining their cost and feedback. They consume attempts normally.
- `finished()` returns true when the application has finalized delivery. Publish
  the actual acceptance/rejection event before returning from the evaluator.
- `unused(context, response)` records a paid proposal that was submitted in the
  same batch but was no longer needed after delivery. Return a `blocked`, unscored
  native outcome with its actual cost or explicit unknown cost and provenance.
  The driver adds `feedback.unused_after_delivery = True`.

Supply `finished` and `unused` together, or omit both to use the declared budget.
On finalized delivery, the next batch is not started and ready surplus proposals
are not evaluated. Every already submitted preparation is joined and accounted
for. Actual delivery may precede cleanup; retain both timestamps. Uncertain
provider or orchestration exceptions stop the rollout without automatic retry.
Callbacks own child-process closure and outstanding cost reservations.

## Complete Python proposals

For executable methods, ask for complete Python directly and use
`parse_python_proposal` inside the evaluator. This avoids embedding long source
inside JSON. Check the provider completion state even if partial text compiles.
The following fragment belongs inside the application's evaluation callback:

```python
from orze.research.source import parse_python_proposal

try:
    proposal = parse_python_proposal(
        response["text"], complete=response["status"] == "complete")
except (ValueError, SyntaxError) as exc:
    return {
        "score": None, "status": "repairable", "artifact": {},
        "feedback": {"error": str(exc)},
        "cost": response["cost"], "seconds": elapsed_seconds,
    }
# Pass proposal to the application's isolated worker and score its real output.
```

The parser accepts raw Python or one Python fence. Methods define `build`,
`train`, and `predict`; diagnostics define `analyze`. It validates size and syntax
without executing source. The format check does not replace execution isolation.
The driver itself does not call a provider or import model-generated Python.

## Numerical inputs and custom batch integrations

`feature_tensor(values, device=..., dtype=...)` accepts NumPy arrays, read-only
memory maps, CPU tensors, and CUDA tensors. It copies array storage and transfers
tensors directly without passing CUDA through NumPy. NumPy and PyTorch are
optional imports. Worker APIs still own shapes, memory limits and evaluation.

Existing custom batch executors may use `prepare_batch(contexts, prepare,
workers=...)` directly. It yields completion order and joins all submitted work on
exit; it does not issue additional actions or change the declared budget.

## Evidence scope

The completed 16-episode comparison accepted 5/8 goals in each arm. Mean capped
time to final acceptance fell from 71.44 to 54.98 minutes (23.0%). The predeclared
mean/success-rate gate passed; prepared execution is the default integration path.
The four-problem difference interval spans zero. See the
[verified results and limitations](plans/2026-09-22-execution-efficiency-results.zh-CN.md).

The high-level entrypoint composes the bounded preparation helper tested in the
[paired real GPU protocol](plans/2026-09-22-execution-efficiency.zh-CN.md).
Independent verification of all 16 episodes, the full mean and success rate
rule, and the default decision must be retained in the experiment evidence.
The experiment concerns four known ASR corpora and one research model; it does
not establish a universal research optimum or attribute the bundle's effect to
a single component. CPU lifecycle checks additionally exercise native execution,
process closure, terminal delivery and surplus-proposal accounting.
