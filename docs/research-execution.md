# Prepare research proposals while the GPU is available

Research executors can overlap a bounded batch of independent proposals and
consume whichever finishes first. The native exploration controller already
captures branch-local contexts before submitting a batch; set `plan.workers`
to the number of preparations you explicitly authorize.

```python
from orze.research.execution import prepare_batch
from orze.research.source import parse_python_proposal


def execute_batch(contexts):
    outcomes = {}
    with prepare_batch(contexts, prepare, workers=2) as ready:
        for context, response in ready:
            proposal = parse_python_proposal(
                response.text, complete=response.status == "complete")
            # User-provided executor: generated Python runs in isolation.
            outcomes[context["action"]["id"]] = evaluate(proposal, context)
    return outcomes
```

`prepare`, `evaluate`, and the response object above are application callbacks,
not built-in providers. Admit every request against the existing call and money
budgets before submitting it. Convert proposal/execution errors into normal
unscored outcomes. Provider uncertainty must retain its reservation and stop an
unverifiable campaign. There is no automatic request retry.

`prepare_batch` yields completion order and joins every submitted callback on
exit, even after early delivery or an exception. The caller must close its own
subprocesses and record every call, including unused proposals. Record accepted
delivery immediately, before waiting for unrelated requests to settle; keep
cleanup time and charges in the campaign record. Return one outcome for every
requested action. GPU evaluation in this loop is serial, independently of model
request concurrency. The helper does not increase the controller's call budget
or share observations across branches.

`parse_python_proposal` accepts complete Python, optionally in a single code
fence. Methods define `build`, `train`, and `predict`; diagnostics define
`analyze`. It checks size and syntax without executing source. Provider
completion must be checked separately: syntactically valid partial output is
not a completed proposal. This is a formatting check, not a sandbox.

For numerical worker APIs, `feature_tensor(values, device=..., dtype=...)`
accepts NumPy arrays, read-only memory maps, CPU tensors, and CUDA tensors. It
copies array storage and transfers tensors directly without converting CUDA
through NumPy. NumPy and PyTorch are optional, loaded only when called. The
worker still owns shape checks, memory limits, and numerical evaluation.

## Evidence and status

Eight A100 conversion checks and four unchanged historical ASR prediction
failures passed after the input fix. These checks establish execution recovery,
not faster attainment of research goals. A prospective 16-episode paired study
compares the execution bundle with the old executor. Both use `ParallelRefine`;
the two-preparation path is explicit until the mean outcome is verified. See
[the study plan](plans/2026-09-22-execution-efficiency.zh-CN.md).
