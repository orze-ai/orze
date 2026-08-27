# Benchmark contracts

Orze reports an internal comparison of experiments. It does not observe an
external leaderboard's hidden data, maintainer verification, eligible-model
rules, or current population. Consequently, every Orze ordering is a **local
rank**, never an official leaderboard rank.

Projects that need benchmark-comparable local results can enable a fail-closed
contract under `report.benchmark_contract`:

```yaml
eval_script: eval_exact.py
eval_output: eval_report.json
sealed_hashes:
  eval_exact.py: <sha256-of-eval_exact.py>

report:
  primary_metric: avg_wer
  sort: ascending
  min_datasets: 9
  columns:
    - {key: avg_wer, label: Average WER}
    - {key: wer_ami_cleaned, label: AMI-Cleaned}
    - {key: wer_earnings22_aa_chunked, label: Earnings22-AA}
    - {key: wer_gigaspeech_cleaned, label: GigaSpeech-Cleaned}
    - {key: wer_ls_clean, label: LS Clean}
    - {key: wer_ls_other, label: LS Other}
    - {key: wer_spgispeech, label: SPGISpeech}
    - {key: wer_voxpopuli_aa_cleaned, label: VoxPopuli-AA}
    - {key: wer_private_scripted, label: Private scripted}
    - {key: wer_private_conversational, label: Private conversational}
  benchmark_contract:
    benchmark_id: hf-audio/open_asr_leaderboard
    # Pin the source defining the view; branches such as `main` are rejected.
    revision: 873970213e211390bd43e9f6a3ad32818cdc3874
    view: default
    required_metrics:
      - wer_ami_cleaned
      - wer_earnings22_aa_chunked
      - wer_gigaspeech_cleaned
      - wer_ls_clean
      - wer_ls_other
      - wer_spgispeech
      - wer_voxpopuli_aa_cleaned
      - wer_private_scripted
      - wer_private_conversational
    receipt: benchmark_receipt.json
    model_form: single_model_single_pass
    # Exact local reproduction is not maintainer verification. Use
    # development_proxy for adaptive screens.
    evidence_scope: local_reproduction
    # Adaptive results are explicitly benchmark-fitted. Confirmation requires
    # zero prior exposure and is bounded by the same hard maximum.
    selection_mode: confirmation
    prior_exposures: 0
    max_evaluations: 1
    aggregate: macro_mean
    aggregate_tolerance: 1.0e-6
    evaluator_sha256: <sha256-of-eval_exact.py>
    dataset_manifest_sha256: <sha256-of-the-exact-ordered-sample-manifest>
    scorer_sha256: <sha256-of-the-normalizer-and-scoring-contract>
    # Optional: require the evaluated hash to match an Orze-managed training
    # artifact. This requires model_lineage.enabled: true.
    # managed_model_lineage: true
```

This example captures the Open ASR landing-page default at the pinned Space
revision; it is an example, not a permanently current built-in policy. Pin a
new immutable revision and update the exact metric set whenever the external
view changes. Because the private metrics are unavailable locally, an ordinary
local evaluator cannot satisfy this particular default-view contract. For a
public-only reproduction, declare the exact public view and manifest instead;
its scope is still `local_reproduction`, never an official rank.

## Evidence scope and exposure budgets

Every contract chooses one evidence scope:

- `development_proxy`: a predeclared local screen used to select or tune ideas.
- `local_reproduction`: an exact local reproduction of the declared benchmark
  view. This still does not imply maintainer verification or an official rank.

It also declares `selection_mode`:

- `adaptive`: the benchmark may influence later choices. Reports label every
  result benchmark-fitted rather than independent confirmation evidence.
- `confirmation`: an untouched, bounded check. It requires
  `prior_exposures: 0` and `max_evaluations: 1`; otherwise configuration fails
  closed. Any additional look must be declared adaptive.

`prior_exposures` records benchmark looks that occurred before Orze began
managing the campaign. `max_evaluations` is the hard total, including those
prior looks. Under-reporting prior exposure is a false attestation; Orze cannot
discover historical access that was never recorded.

Immediately before evaluator launch, Orze reserves an ordinal in
`.orze/_benchmark_exposures.jsonl` under a filesystem lock. This path is fixed
at project scope rather than the configured results directory, so rotating from
one campaign-results root to another cannot create fresh history. Ledgers from
the earlier result-local implementation are discovered, checked for conflicts,
and migrated before another reservation. Both `orze reset --scratch` and
`orze reset --full` preserve the project ledger. The append-only
record binds benchmark identity, scope, selection mode, idea, sealed evaluator,
and a hash of the fresh nonce. A failed launch conservatively consumes the
reservation. The sealed dataset-manifest digest is the accounting identity, so
renaming the benchmark or changing its scope, view, or selection mode cannot
reset recorded exposure. Corrupt or missing ledger evidence fails receipt
validation.

This is fail-closed local accounting, not an externally anchored audit log. A
user with write access can replace the ledger and all linked provenance, and
Orze cannot discover unrecorded historical access or overlap hidden behind a
different manifest. Copying the whole project without its `.orze` state also
requires declaring the copied history in `prior_exposures`. For public claims,
preserve and publish the ledger and artifacts or use an independent/WORM-backed
evaluation service.

## Evaluator receipt

At launch Orze hashes the sealed evaluator, generates a fresh random nonce,
writes `_benchmark_evaluation.json`, and passes the nonce and receipt path in
`ORZE_BENCHMARK_EVALUATION_NONCE` and `ORZE_BENCHMARK_RECEIPT`. The exposure
ordinal and record hash are passed in `ORZE_BENCHMARK_EXPOSURE_ORDINAL` and
`ORZE_BENCHMARK_EXPOSURE_RECORD_SHA256`. With managed lineage enabled, the
launcher also passes `ORZE_MANAGED_MODEL_LINEAGE_SHA256` and
`ORZE_MODEL_ARTIFACT_SHA256`. A successful evaluator writes JSON like:

```json
{
  "schema_version": 1,
  "benchmark_id": "hf-audio/open_asr_leaderboard",
  "benchmark_revision": "873970213e211390bd43e9f6a3ad32818cdc3874",
  "benchmark_view": "default",
  "evaluator_sha256": "<sha256-of-eval_exact.py>",
  "dataset_manifest_sha256": "<sha256-of-the-exact-ordered-sample-manifest>",
  "scorer_sha256": "<sha256-of-the-normalizer-and-scoring-contract>",
  "evidence_scope": "local_reproduction",
  "selection_mode": "confirmation",
  "prior_exposures": 0,
  "max_evaluations": 1,
  "exposure_ordinal": 1,
  "exposure_record_sha256": "<ORZE_BENCHMARK_EXPOSURE_RECORD_SHA256>",
  "evaluation_nonce": "<ORZE_BENCHMARK_EVALUATION_NONCE>",
  "model_form": "single_model_single_pass",
  "component_model_count": 1,
  "inference_passes_per_sample": 1,
  "dataset_specific_routing": false,
  "model_artifact_sha256": "<sha256-of-the-one-evaluated-model-artifact>",
  "managed_model_lineage_sha256": "<ORZE_MANAGED_MODEL_LINEAGE_SHA256>",
  "decoding_config_sha256": "<sha256-of-the-one-shared-decoding-config>",
  "metric_keys": ["<the exact required_metrics set>"]
}
```

The evaluator must write the receipt only after loading the one hashed model
artifact and completing the declared evaluation. A receipt that existed before
launch is rejected. Missing provenance, a nonce mismatch, evaluator drift,
multiple components or passes, dataset-specific routing, missing/extra metric
keys, dataset/scorer drift, an invalid shared-decoding identity, non-finite
metrics, an incorrect macro mean, or missing/corrupt exposure evidence fails
evaluation and keeps the row out of every local ranking.

When `managed_model_lineage: true`, the receipt's model digest must equal the
current managed artifact digest and its lineage digest must equal the parent
provenance. Orze validates the full managed attempt again before reserving a
benchmark exposure and when accepting the receipt. See
[managed model lineage](model-lineage.md). Leave the flag false for external
models; labeling an externally trained artifact as managed is not permitted.

The idea directory, metrics, configured metric sources, provenance, and receipt
must be ordinary non-redirected paths. Orze checks every path component and does
not follow symlinks while loading values or validating a receipt. The generated
leaderboard then applies the same completed-artifact, resolved-value validation,
finite-primary, and dataset-coverage rules used for non-contract local evidence
before it checks benchmark provenance. Its content-addressed cache cannot turn a
previously valid receipt or source into a stale rank after evidence changes.

This contract makes honest claims mechanically easier, but it does not turn a
local run into maintainer verification. An official rank must come from the
external leaderboard itself and is deliberately outside Orze's report schema.
