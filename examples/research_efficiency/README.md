# Bounded research-efficiency comparison

This opt-in application compares the original acceptance policy with
`DominancePruningPolicy`. It uses the existing sorting/compression Domains and
an unchanged installed Core. It does not replace Orze's default policy.

The rule skips a validation-only analysis only when its measured cost is
strictly worse than an already valid comparable candidate, and its exact
protocol fingerprint is explicitly allowlisted by the application owner.
Unknown evidence remains unknown; the selected action still needs a real
independent-attempt replication. The tradeoff is less candidate-validation
coverage. Equal costs, missing capabilities and incompatible evidence fall
back to the original policy.

The policy currently adapts the finite acceptance application's candidate
layout. The decision rule is transferable; this example is not a ready-made
optimizer for arbitrary scientific tasks or model research.

Run from a clean committed checkout, using a fresh **Core-only**, non-editable
virtual environment and the exact local wheel installed in that environment:

```bash
/absolute/venv/bin/python -I examples/research_efficiency/run.py \
  --core-wheel /absolute/wheels/orze-4.6.2-py3-none-any.whl \
  --output-root /absolute/new-empty-parent/formal-01
```

The output directory must not already exist. The formal schedule is fixed at
24 pairs / 48 new CLI projects, with alternating AB/BA order. Each CLI uses one
CPU slot and a 10-second conservative action allowance; no GPU, provider or
Pro license is used. `--smoke` instead runs two development CLIs, never formal
speed evidence. Every run retains its database, logs, published artifact bytes,
process-closure records and decision trace. Failed or missing pairs stay in
the denominator; group medians require every planned pair to pass quality.

See [the preregistered protocol](../../docs/plans/2026-09-12-research-production-validation.zh-CN.md)
for endpoints, limits and production-deployment boundaries.
