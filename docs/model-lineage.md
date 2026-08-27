# Managed model lineage

`model_lineage` binds one idea-local model artifact to the exact Orze training
attempt that produced it. It is optional because Orze cannot truthfully create
managed-training evidence for an external or pretrained artifact whose training
ran elsewhere.

```yaml
data_boundaries:
  forbidden_in_training:
    - /datasets/benchmark/held-out
  training_network: deny

data_separation:
  enabled: true
  # See data-separation.md for the full pinned-manifest contract.
  train_manifest: /project/manifests/train.jsonl
  train_manifest_sha256: <sha256>
  evaluation_manifest: /project/manifests/evaluation.jsonl
  evaluation_manifest_sha256: <sha256>
  fingerprint_namespace_sha256: <sha256>
  normalization_contract_sha256: <sha256>
  fields: [sample]
  max_overlap: {sample: 0}

model_lineage:
  enabled: true
  # Relative to results/<idea-id>/ and written by the trainer.
  artifact: model/model.safetensors
  max_files: 100000
  max_bytes: 107374182400
  attestation_timeout: 10
```

Enabling lineage requires all three controls: at least one hard
`forbidden_in_training` root, `training_network: deny`, and a passing
`data_separation` contract. Invalid policy rejects launch before GPU telemetry.

## Receipt chain

For each attempt, the launcher creates a private pipe and random nonce. The
child wrapper acknowledges that nonce only after the mandatory user, mount, and
network namespace setup has completed. It removes both pipe variables before
running user code. The parent then writes a one-shot `boundary.json` receipt
that binds the attempt, execution identity, boundary policy, and aggregate data
separation receipt. Missing, late, or incorrect acknowledgement terminates the
child before the attempt is advertised in progress.

After a trainer exits zero with `metrics.json.status: COMPLETED`, Orze hashes the
declared artifact twice. The two complete reads must agree, file identities must
remain stable, and no symlink, special file, empty artifact, or configured
resource-limit violation is accepted. A file artifact uses ordinary SHA-256. A
directory uses the documented `directory_tree_v1` digest over sorted relative
names, sizes, and bytes. Only then does Orze write `_model_lineage.json` and the
successful terminal compute receipt. A lineage failure preserves the trainer's
metrics under `metrics.lineage_invalid.*.json`, records a failed attempt, and
keeps the idea out of evaluation and local ranking.

Evaluation rechecks the boundary, separation, attempt, successful terminal
receipt, and a fresh two-pass artifact digest before GPU inspection. Repeated
report generation uses inode, size, mtime, and ctime metadata only as a cache
invalidation fast path; any metadata change triggers full revalidation. This
avoids rereading a large unchanged model on every report while same-size or
backdated normal rewrites still invalidate the cached row.

For a sealed benchmark evaluator, set
`report.benchmark_contract.managed_model_lineage: true`. Orze then passes
`ORZE_MANAGED_MODEL_LINEAGE_SHA256` and `ORZE_MODEL_ARTIFACT_SHA256` to the
evaluator and requires both exact values in its receipt. Substituting a
different syntactically valid model hash fails the contract.

## Claim boundary

These receipts prove only consistency of local Orze-managed evidence under the
configured controls. They do not prove that manifests enumerate every training
byte, that raw identifiers were fingerprinted honestly, that a pretrained base
model is uncontaminated, or that a machine administrator did not rewrite the
whole local evidence chain. They never prove an official leaderboard rank.
Preserve and publish the receipts and artifacts, and use independent evaluation
for public claims.

See [training data boundaries](data-boundaries.md),
[manifest separation](data-separation.md), and
[benchmark contracts](benchmark-contract.md).
