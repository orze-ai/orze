# Compute an explicit paired comparison

`orze.research.paired_comparison` is an optional CPU analysis worker. It computes
configuration differences, pairs measurements by sample identity, and reports
both the mean sample difference and an equal-weight mean across groups. It also
counts overlap with supplied records of previous data use.

Use it when a research decision needs these facts. It does not choose candidates,
train a model, certify independent validation, or set a required experiment count.

## Input artifact

Capture a JSON document as an ordinary native artifact:

```json
{
  "reference": {
    "id": "baseline",
    "comparison_scope": "dataset-sha:protocol-v1:evaluation-split",
    "metric": "squared_error",
    "direction": "minimize",
    "configuration": {"learning_rate": 0.1, "l2": 1},
    "units": [
      {"id": "sample-1", "group": "condition-a", "value": 4},
      {"id": "sample-2", "group": "condition-b", "value": 2}
    ]
  },
  "candidate": {
    "id": "joint-change",
    "comparison_scope": "dataset-sha:protocol-v1:evaluation-split",
    "metric": "squared_error",
    "direction": "minimize",
    "configuration": {"learning_rate": 0.2, "l2": 3},
    "units": [
      {"id": "sample-2", "group": "condition-b", "value": 3},
      {"id": "sample-1", "group": "condition-a", "value": 1}
    ]
  },
  "prior_uses": [
    {"id": "earlier-model-selection", "unit_ids": ["sample-1"]}
  ]
}
```

Each record requires exactly the fields shown. IDs and labels are nonempty strings
up to 256 characters. `direction` is `minimize` or `maximize`. Each configuration
is a JSON object of at most 8,192 bytes and 16 levels of nesting. Each record has
1–10,000 uniquely identified units with finite numeric values. The document is
limited to 2 MiB; duplicate JSON fields are rejected by the worker.

Use a value whose arithmetic mean is meaningful under the declared protocol,
such as a squared error or a classification-error indicator per sample. This
worker does not combine ratios with unequal denominators: for example, an
unweighted mean of utterance WERs is not corpus WER. A domain adapter must preserve
the metric's actual units and aggregation. Scope strings and identities remain
declarations; the worker does not establish their scientific correctness.

`prior_uses` is `null` when no use history is supplied, or a list of at most 32
records, each with at most 10,000 unit IDs. A zero intersection only describes
those supplied records. It does not establish a complete history or independent
new evidence.

## Run through the native CPU path

Use CPU execution with `action_domain: {version: 1, kind: json_observations, config: {}}`.
Bind the captured artifact through the normal input-artifact contract:

```yaml
kind: native_cpu_action
domain_request:
  version: 1
  purpose: Compare measured candidate with its explicit reference
  inputs:
    source_id: REPLACE_WITH_CAPTURED_ARTIFACT_ID
  input_artifact_ids: [REPLACE_WITH_CAPTURED_ARTIFACT_ID]
  timeout_seconds: 5
  outputs:
    evaluation: {path: evaluation.json, max_bytes: 65536}
  payload:
    command: [python3, -m, orze.research.paired_comparison]
    specification: {analysis: paired_unit_means, version: 1}
    protocol: {id: paired-unit-means-v1}
    result_output: evaluation
```

The interpreter must have this Core version installed. Normal source, ownership,
budget and execution checks still apply. Reserve the declared action timeout as
well as any capture or subsequent research actions.

## Read the result

- `configuration_changes` lists all changed dictionary paths, including missing
  fields and numeric representation changes such as `1` versus `1.0`. Arrays are
  compared as whole values. More than 128 changes is rejected. These are declared
  JSON differences; implementation equivalence and causal effects are not inferred.
- `pairing` reports compatible declared scopes, shared and unmatched samples, and
  group-identity disagreements. All samples must match by ID and group, and both
  records must declare the same metric, direction and scope for a paired effect.
- `paired_effect.candidate_minus_reference_mean` is signed candidate minus
  reference. A negative value means lower average measurements. `improvement`
  also accounts for the declared optimization direction. Relative improvement
  is `null` when the reference mean is zero.
- `equal_group_candidate_minus_reference_mean` first averages within each group,
  then weights the groups equally. It can disagree in direction with the sample
  mean. Neither aggregation is automatically the project's preferred objective.
- `prior_use_overlap` identifies the supplied prior record and overlap count.
  Unknown or incomplete historical coverage remains unknown.

Incompatible records produce `paired_effect: null` and an invalid comparison
observation, while retaining coverage facts and separate descriptive means.
The worker never silently restricts a comparison to an overlapping subset.
An available source record can still contain this invalid scientific comparison;
source qualification does not turn it into a valid effect.

The module also exposes `compare(document)` for a domain adapter that already
has bound inputs. It performs no file access or execution. This arithmetic is a
component of a research decision, not evidence that the entire research policy
has improved.

## Return useful facts from a batch

When a project combines several comparisons, keep full computation details in
an artifact and project the needed fields into the model-facing observation.
Repeated configuration copies and interpretation text can exceed the Domain's
JSON structure bound or the research consumer's per-record capacity even when
the declared output file size is sufficient. Do not increase unrelated limits
before checking that representation.

The [executed research follow-up](plans/2026-09-17-research-edits.zh-CN.md)
retains a failed eight-candidate publication and verifies a smaller observation
containing configuration, measured loss, changed paths, pairing facts and known
reuse. Its four-candidate feedback appeared in the actual next model prompt.
Some larger initial records remained unavailable, so the study separately
verified the complete available-configuration and score table. Artifact retention
and source qualification do not by themselves establish model visibility.
