# Executable exploration and historical replay

`orze.research.exploration` defaults to equal-depth `ParallelRefine` when the caller
supplies an executor and omits the policy. It separates **what the
researcher proposes** from **which research path receives the next attempt**.
It does not start a daemon, import generated code, grant execution permission,
or replace Orze's CPU/GPU executors.

## Online contract

```python
from orze.research.exploration import run_online, replay
from orze.research.exploration_policies import Portfolio  # optional alternative

spec = {
    "problem_id": "independent-problem-group-1",
    "protocol_id": "hash-of-fixed-model-data-evaluator-and-shared-context",
    "root": {"score": 0.0, "artifact": {}, "feedback": {}},
    "score_scale": 1.0,
    "plan": {"branches": 4, "depth": 6, "calls": 24, "workers": 2},
}
# Supply your already authorized research/execution integration:
# trace = run_online(spec, execute_batch=execute_batch, output=fresh_output_directory)
# portfolio = run_online(spec, Portfolio(), execute_batch, another_fresh_directory)
# alternative = replay(trace, my_policy, calls=12)
```

All four plan values are positive integers. `depth` counts attempts per branch,
including its first attempt. `calls` bounds the whole rollout, independently of
worker count. The replay cap may be smaller than the original declaration;
compare candidates with the same cap. Quality is higher-is-better; a loss can
be negated. Fix `score_scale > 0` before observing the trace. A trace's eventual
best score must not define its scale.

A policy implements `decide(view)` and returns:

```python
{"actions": ["b0-s0", "b1-s0"], "reason": "Test two independent directions."}
```

The view contains `root`, `score_scale`, `observed`, `legal`, `remaining_calls`,
and `workers`. `legal` contains unopened roots and one next action for each
opened branch. Each action has `id`, `branch`, `step`, and `parent`. Choose at
most `min(workers, remaining_calls)` distinct legal IDs. Empty actions stops
this discovery rollout. Every decision, including the stop reason, is retained.
Policies may implement arbitrary reviewed algorithms; the bundled policies are
starting points, not prescribed scientific methods.

`execute_batch(contexts)` receives a list of `{action, root, history}` objects.
Each history contains **only that branch's complete ancestry**, including
failed attempts. It returns exactly an ID-to-outcome mapping:

```python
{
    "b0-s0": {
        "status": "ok",                 # ok / repairable / blocked
        "score": 0.25,                  # None for an analysis without a score
        "artifact": {"source_id": "..."},
        "feedback": {"observations": []},
        "cost": None,                   # known nonnegative amount, or unknown
        "seconds": None,                # known duration, or unknown
    }
}
```

The executor must restore the corresponding parent program/workspace and keep
shared context fixed. A global mutable chat or access to later sibling results
would change the generation process and invalidate the replay interpretation.
Store full source or durable, verified artifact references, not just a summary
of the apparent winner. The integration owns provenance verification and must
include model, evaluator, input, environment and shared-context identities in
its protocol manifest. Archive policy source/config/dependencies alongside it.

Classify a **known** implementation failure as `repairable` only when that is
supported by the actual execution report. Failures cannot carry a quality
score. A successful analysis can return `score=None`: it counts as an attempt
and its findings remain available without inventing a quality measurement.
Unknown execution state is an executor error; do not turn it into a free failed
experiment. Exceptions stop the rollout and preserve the pre-dispatch request.

The driver passes batches; the existing executor owns actual scheduling and
limits. `parallel_seconds_estimate` is the sum of batch maxima, a scheduling
estimate, not a measured counterfactual elapsed time. Missing durations or costs
make the corresponding aggregate unknown. Replay does not call the executor.

`run_online` exclusively creates its output directory, writes each request
before dispatch and each response afterwards, and publishes `trace.json` only
after the rollout completes. Completed or failed output directories cannot be
reused. Incomplete directories must be reconciled by the existing executor;
this API does not resume or retry them.

## Replay support

`validate_trace` reconstructs recorded batches, ancestry, termination and
aggregate metrics. It checks structural integrity, not external scientific
truth or artifact availability. Only completed online traces enter a pool.
Supported replay results are marked `mode="replay"` and rejected as pool inputs.
The caller's evidence chain must still establish online origin; structural
validation does not authenticate arbitrary JSON files.

The policy receives only the revealed prefix. Legal actions derive from the
declared grid, so they do not reveal which hidden continuations were recorded.
If any requested continuation is missing, the entire batch remains unrevealed,
the result is `out_of_support`, and it cannot earn a selection score. Missing
results are neither scientific failures nor a reason to assume a branch stops.

Policy code is trusted. The detached views prevent accidental data mutation;
they are **not a sandbox** preventing a Python program from reading trace files.
Review model-written code using the integration's existing execution controls.

## Pro policy improvement

With matching Core and licensed Pro checkouts:

```python
from orze_pro.agents.exploration import Candidate, improve_exploration

# Candidate.factory creates a fresh trusted policy object for each world.
# Include complete source/config and dependency identities in its manifest.
# current = Candidate(factory=MyPolicy, manifest={"source": source, ...})
# selected, report = improve_exploration(
#     current, training_worlds, validation_worlds, develop,
#     revisions=3, calls=12, output=fresh_selection_directory,
# )
# fresh_trace = run_online(next_spec, selected.factory(), execute_batch, fresh_run)
```

`develop(current, training_feedback)` returns a new `Candidate`. It may invent
a different algorithm rather than change preset parameters. It receives the
training trajectories and errors from previous revisions. Each manifest is
archived before evaluation; duplicate or mutated manifests are rejected.
Bad policy decisions are retained as development feedback with no score.

`orze_pro.agents.meta_research.propose_exploration_revision` calls the existing
`research_llm.call_llm` provider path to write proposed Python source. Pass the
current manifest, training feedback, a fresh output directory and explicit
backend/model credentials. Existing usage journals and token envelopes apply.
It does **not** execute/import provider output; the development integration
reviews the source before supplying its factory. No scientific artifacts or
free-text experiment findings enter this meta-policy prompt.

Training and validation require disjoint `problem_id` groups. Different runs of
the same dataset/split are related, not independent held-out problems. Freeze
the validation set for an evaluation round; repeatedly developing against its
published results turns it into training and requires fresh validation.

Selection includes the incumbent and compares all frozen, training-supported
candidates on validation after development has finished. No validation feedback
enters another revision in that call. The supported candidate with the highest
mean normalized validation quality becomes the next default; ties retain the
incumbent (then the earliest candidate). A training runner-up can win validation.

First average repetitions within each `problem_id`, then weight problems equally.
This weighting is fixed before selection; running one problem more often cannot
increase its influence. Individual problem regressions are recorded and allowed
when the overall mean improves. Missing replay support is unknown evidence, so
it cannot earn a score or be dropped to inflate the mean. Lower costs alone do
not promote a policy.

`selection.json` records `default_policy`, `promoted_to_default`, all validation
candidate scores, and paired problem differences. The returned `selected` object
is the default to use in the next rollout, as in the example above. The earlier
`promoted_for_online_trial` field remains as a compatibility alias. This API does
not rewrite an unrelated daemon configuration.

These are empirical means over the chosen problem set, not proof of the true
expectation on all future tasks. Keep collecting fresh online evidence to assess
and update the default; the report records `online_improvement_proven=False`.

`ParallelRefine` is the default after the completed prospective comparison:
control mean confirmation gain 0.54954, learned Dream revision 4 mean 0.54638.
The difference interval spans zero; this is the higher observed mean, not proof
of a universal optimum. See the [results and limitations](plans/2026-09-19-dream-rsi-results.zh-CN.md).

`Portfolio` remains an explicit alternative with successful anchors, trajectory
evidence and bounded repairs. Short plateaus deprioritize paths but do not end
remaining exploration; `Portfolio(stop_on_plateau=True)` enables early stopping.
This bootstrap is distinct from the learned revision 4 used in the online study.
The exact tested source and identity are retained with the result evidence.

Pro's `run_discovery(spec, execute_batch, output=...)` uses the same default.
Supplying `training`, `validation`, and a reviewed `develop` callback runs replay
improvement first and directly executes its selected policy. It returns
`(selected_policy, trace)` so a campaign carries its learned default forward.
Every invocation requires a fresh output directory. These defaults apply to
these executable exploration entry points; existing projects must supply their
own branch-safe executor to use them. A legacy project with a single global
research conversation is not silently treated as a valid branch replay world.

The [long comparison protocol](plans/2026-09-18-dream-rsi-long.zh-CN.md) and
[completed results](plans/2026-09-19-dream-rsi-results.zh-CN.md) retain all failures
and separate the 100 extra development calls from the equal-call online arms.

Executor feedback should preserve typed provider failures alongside any parse
error (for example, `provider_result={"status": "error", "reason": "provider_transport_failed"}`).
An empty extracted response alone does not identify the cause. Preserve unknown
costs and actual usage status. The Core history carries the feedback supplied by
the executor; it cannot reconstruct omitted provider evidence. This study found
such an omission in its frozen adapter; it was not repaired during the trial.
