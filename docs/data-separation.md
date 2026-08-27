# Train/evaluation manifest separation

Filesystem and network isolation prevent access to declared locations. They
cannot detect a held-out sample copied into a training cache or the same speaker
appearing under a different filename. The optional `data_separation` contract
audits privacy-preserving manifests before GPU telemetry:

```yaml
data_separation:
  enabled: true
  train_manifest: /project/manifests/train.jsonl
  train_manifest_sha256: <sha256-of-exact-train-manifest>
  evaluation_manifest: /project/manifests/evaluation.jsonl
  evaluation_manifest_sha256: <sha256-of-exact-evaluation-manifest>
  fingerprint_namespace_sha256: <sha256-of-private-fingerprint-namespace>
  normalization_contract_sha256: <sha256-of-versioned-normalizer>
  fields: [sample, speaker, source]
  max_overlap: {sample: 0, speaker: 0, source: 0}
  max_records: 10000000
  max_bytes: 2147483648
  max_line_bytes: 4096
```

Each manifest is JSON Lines. Its first line is an exact header:

```json
{"schema_version":1,"role":"train","fingerprint_algorithm":"hmac-sha256","fingerprint_namespace_sha256":"<64 lowercase hex>","normalization_contract_sha256":"<64 lowercase hex>","fields":["sample","speaker","source"]}
```

Use `"role":"evaluation"` for the evaluation manifest. Every remaining line
contains exactly the configured fields, each as a lowercase 64-character keyed
HMAC-SHA-256 fingerprint:

```json
{"sample":"<hmac>","speaker":"<hmac>","source":"<hmac>"}
```

The HMAC key and raw identifiers must remain outside Orze. Both manifests must
use the same private namespace, canonical sample representation, speaker/source
normalization, and key. The namespace and normalization digests make that
declaration explicit; local software cannot prove that a dishonest manifest
generator followed it.

Orze rejects malformed headers or records, raw/extra fields, duplicate sample
fingerprints within either manifest, changed or redirected files, digest drift,
resource-limit violations, and cross-manifest overlap above any configured
dimension limit. `sample` is mandatory; speaker and source are optional policy
dimensions because some legitimate benchmarks permit those populations to
overlap.

The comparison streams bounded lines into an indexed SQLite file whose directory
entry is removed immediately after opening. This bounds process memory while
preventing the keyed fingerprints from remaining on disk even if the audit
process crashes. The durable project receipt stores only pinned manifest and
policy hashes, record/unique/overlap counts, and a change-resistant metadata
signature. Unchanged manifests reuse that receipt; normal edits, replacement,
deletion, redirection, or policy changes force a fresh exact-hash audit.

Passing this contract proves only that the two pinned manifests satisfy the
declared fingerprint-overlap limits. It does not prove that manifests enumerate
all bytes actually read by training, that a pretrained model is uncontaminated,
or that an evaluated artifact came from a particular managed training attempt.
Use it together with hard [training data boundaries](data-boundaries.md), pinned
artifact resolution, model lineage receipts, and independent leaderboard
evaluation. It never establishes an official rank by itself.
