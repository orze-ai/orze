# V1-07 frozen CPU holdout challenge

Frozen after the V1 plan was published at core commit `1b3fc3e`.
This challenge is an independent acceptance input, not a change to the plan.
It contains no optimal answer, reference schedule, or recommended search policy.
It must remain outside the public core/pro repositories until the acceptance
workflow explicitly chooses to publish it.

## Task

Produce and evaluate a schedule for a small, fixed, resource-constrained job
selection instance. The objective is to **maximize the total value of scheduled
jobs**, subject to all feasibility constraints. A feasible low-value schedule is
a valid research result, not an infrastructure failure.

The executor runs an ordinary local CPU process. The instance's `capacity` and
each job's `demand` are simulated problem parameters, not physical CPU/GPU
reservations for the executor. No GPU, model weights, network, or real LLM is
needed.

## Input: instance `sched-holdout-001`, protocol `schedule-feasibility-v1`

```json
{
  "instance_id": "sched-holdout-001",
  "horizon": 12,
  "capacity": 2,
  "jobs": [
    {"id": "a", "release": 0, "deadline": 4, "duration": 2, "demand": 1, "value": 0, "required": true, "after": []},
    {"id": "b", "release": 0, "deadline": 6, "duration": 3, "demand": 2, "value": 8, "required": false, "after": []},
    {"id": "c", "release": 2, "deadline": 8, "duration": 4, "demand": 1, "value": 7, "required": false, "after": ["a"]},
    {"id": "d", "release": 1, "deadline": 9, "duration": 3, "demand": 1, "value": 6, "required": false, "after": ["a"]},
    {"id": "e", "release": 4, "deadline": 12, "duration": 3, "demand": 2, "value": 12, "required": false, "after": ["b"]},
    {"id": "f", "release": 5, "deadline": 12, "duration": 4, "demand": 1, "value": 8, "required": false, "after": ["c"]},
    {"id": "g", "release": 0, "deadline": 5, "duration": 2, "demand": 1, "value": 4, "required": false, "after": []},
    {"id": "h", "release": 7, "deadline": 12, "duration": 2, "demand": 1, "value": 5, "required": false, "after": ["d"]}
  ]
}
```

## Candidate output

A UTF-8 JSON artifact with exactly these top-level fields:

- `instance_id`: the input instance identifier.
- `schedule`: an array of objects with exactly `job_id` and `start` fields.
  `job_id` is a string. `start` is a nonnegative integer; booleans and floating
  point numbers are not accepted as integers by this contract.

The candidate does not report an authoritative score or validity flag. Those
are computed by the domain evaluator from the artifact and versioned input.
The order of entries has no semantic effect.

## Feasibility and objective

1. Unknown IDs, duplicate IDs, malformed JSON, or extra candidate fields are
   invalid candidate artifacts and receive a typed domain/schema rejection.
2. Every required job must occur exactly once. Other jobs may be omitted.
3. A scheduled job runs non-preemptively on the half-open interval
   `[start, start + duration)`. Release, deadline, and horizon constraints apply:
   `release <= start` and `start + duration <= min(deadline, horizon)`.
4. If a job is selected, every ID in its `after` list must also be selected and
   must finish no later than that job starts. Omitting a prerequisite is invalid.
5. At every integer tick, the sum of demand over active jobs must not exceed
   capacity. Jobs ending at time `t` do not overlap jobs starting at time `t`.
6. A valid observation has objective `scheduled_value`, direction `maximize`,
   and value equal to the sum of selected job values. An invalid artifact has
   no eligible objective value: do not encode invalidity as `0`, `999`, or another
   magic number. Zero is a legitimate valid objective value.
7. A successful evaluator invocation with an infeasible schedule is a completed
   evaluation with an invalid observation. A crashed evaluator is a distinct
   execution failure. Neither case implies that the producer source code should
   automatically be sent to a fixer.

## Required acceptance boundaries

- Run through the same public task/attempt/observation interfaces as the two
  design examples, using only a domain adapter, objective/protocol declarations,
  and policy configuration. Do not introduce scheduling-specific conditionals,
  job IDs, or objective names into core/pro.
- A deterministic baseline and any challenger may use the same framework entry
  point. Proving a global optimum or beating a threshold is **not** required for
  this framework acceptance challenge; do not manufacture a research-success
  claim from a successful integration test.
- A lower-scoring feasible candidate remains valid, is recorded with its input,
  artifact, and protocol identities, and does not invoke an infrastructure/code
  repair path. Two equal valid scores are numerical equality, not evidence of
  statistical equivalence.
- Test an exact deadline and an end/start boundary; then independently inject
  duplicate ID, missing prerequisite, resource overload, and malformed artifact.
  Domain validation must identify each condition without invoking GPU discovery
  or requiring a `train.py` entry point.
- Request an explicit replicate of one producer or evaluator action. It must be
  represented as an intentional additional attempt/run according to the frozen
  V1 contract, rather than silently dropped as an accidental duplicate.
- Re-evaluate an existing candidate artifact under protocol
  `schedule-feasibility-v2`, whose only semantic change is `capacity = 1`.
  Preserve both observations with different protocol identities, reuse the same
  producer artifact, and perform no new producer invocation. An artifact may
  change validity across protocols. Never silently compare v1 and v2 observations
  on one leaderboard or overwrite the v1 measurement with the v2 result.
- Inject evaluator failure before a complete observation is committed, restart,
  and recover the evaluation from the same artifact. A partial output must not
  become an eligible measurement. Record actual invocation counts and attempt
  transitions; a fixture asserting only helper return values is insufficient.
- When all requested work is measured and no new input/observation is available,
  the policy can return `wait` or `stop`. Repeated scheduler ticks alone must not
  force new proposals or a full cycle of agent calls.

## Evidence restrictions

All acceptance data are local and synthetic. Capture commit IDs, exact commands,
exit codes, task/attempt IDs, objective/protocol identities, artifact references,
and invocation counts. Actual GPU resource behavior and research productivity
remain untested by this challenge. Do not use simulated capacity, elapsed CPU
time, or completed-task counts to claim saved GPU hours or improved research
quality.
