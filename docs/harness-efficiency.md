# Harness efficiency contract

Orze treats harness efficiency as a set of separately verifiable properties.
A fast model run does not prove a fast scheduler, and a fast scheduler does not
prove high GPU utilization, research yield, or leaderboard rank.

For an overall claim, use `orze.engine.acceptance_matrix`. It aggregates exact
pinned receipts against Orze's complete, code-owned requirement universe. Its
status remains `UNVERIFIED` while any required campaign or external result is
missing, even when all CPU/readiness controls are green.

## CPU control-plane acceptance

Run the production APIs against a disposable database on the same filesystem
used by the project:

```bash
python -m orze.benchmarks.harness_efficiency \
  --work-dir /path/to/project/.orze \
  --ideas 50000 \
  --queue-limit 2000 \
  --iterations 20 \
  --output /path/to/receipt.json
```

The command does not import a model, launch training/evaluation, or probe an
accelerator. A receipt is `VERIFIED` only with at least 10,000 ideas and 20
steady-state samples, and only when every target and invariant passes. Twenty
is the minimum sample count at which nearest-rank p95 is not simply the single
maximum; the receipt also retains every raw latency observation:

| Dimension | Target |
|---|---:|
| Cold IdeaLake open p95 | <= 75 ms |
| Indexed admitted-identity batch lookup p95 | <= 50 ms |
| `get_queue(2000)` p95 | <= 60 ms |
| Warm parse/cache plus `get_unclaimed(2000)` p95 | <= 120 ms |
| Complete steady control tick p95 | <= 200 ms |
| Claim through completed training/evaluation ledger p95 | <= 500 ms |
| Bulk ingest | >= 1,000 rows/s |
| Config identity coverage | exact |
| Queue count and ordering | exact and stable |
| Final global/training/evaluation states | exact |
| Concurrent filesystem claim | exactly one winner |
| GPU/model/evaluation work | none |

Source-file SHA-256 values, runtime versions, filesystem location, all raw p95
summaries, and each target verdict are included in the JSON receipt. Runs below
the acceptance scale can be useful diagnostics but are labelled `DIAGNOSTIC`,
never `VERIFIED`.

## Read-only recovery acceptance

Audit the deployed database without invoking startup repair or opening an
accelerator:

```bash
python -m orze.engine.recovery_audit \
  --db /path/to/project/.orze/idea_lake.db \
  --results-dir /path/to/project/results \
  --output /path/to/recovery-receipt.json
```

`VERIFIED` requires the current and immutable global/stage ledgers to agree,
legacy status mirrors to agree with the global FSM, every local active claim to
name a live exact `(pid, start_ticks)` identity, and no terminal row to retain a
live owner or trainer. The `ideas` and `idea_state` tables must also contain the
exact same idea-ID universe in both directions; a missing or orphan current
state is a contradiction, even when no transition refers to it. A missing
deployed stage schema/history, redirected or
hard-linked database/claim, remote owner that cannot be checked locally, or
unstable evidence is `UNVERIFIED`. A directly observed state/process or
current/transition contradiction is `FAILED`. The receipt binds the database
and auditor SHA-256 values and explicitly proves no leaderboard rank.

## Why config identity is indexed

The scheduler only needs to ask whether the small batch of newly proposed
config identities already exists. Loading and reparsing every historical config
on every poll is O(lake) work and grows without bound. New writes persist a
canonical hash plus a source-YAML digest. Ingest performs an indexed O(batch)
lookup; a tick with no new proposals performs no identity SQL. A SQLite trigger
invalidates the derived identity if any caller updates the config source, and
legacy rows are backfilled once when a real lookup needs them.

## Dimensions requiring an active research campaign

The CPU receipt deliberately does **not** claim the following. They require a
separate preregistered run with real workload evidence:

| Dimension | Target/evidence needed |
|---|---|
| Eligible queue to claim latency | p95 <= two configured poll intervals while a permitted slot is free |
| Terminal result to next claim | p95 <= one poll interval while eligible work remains |
| GPU duty cycle | >= 90% while eligible work exists, measured only on explicitly permitted physical GPUs |
| Operator visibility | update gap <= 10 minutes with last valid artifact, blocker, and next deadline |
| Zero-compute rejection | 100% of invalid/ineligible proposals rejected before GPU allocation |
| Duplicate compute | zero duplicate config or claim launches |
| Time to decision | preregistered per campaign, measured from admission through valid screen/full decision |
| Research yield | independently verified improvement rate and GPU-hours per accepted improvement |
| Reproducibility | rerun within preregistered metric tolerance from sealed code/data/environment identities |
| Leaderboard outcome | accepted official submission/rank under the leaderboard's complete eligibility rules |

These rows remain open until their own receipts exist. Local proxy scores,
configuration inspection, or a CPU benchmark cannot close them.

## Preregistered campaign receipt

Before a campaign starts, write a JSON manifest with a unique `campaign_id`,
future `start_epoch` and `end_epoch`, the exact physical `physical_scope`, the
configured `poll_seconds`, the complete `expected_idea_ids` experiment
universe, minimum evidence counts, and all five targets from
`DEFAULT_CAMPAIGN_TARGETS`. Thresholds may be stricter than the defaults but
cannot be weakened. Register it once in the same IdeaLake used by the daemon:

```bash
python -m orze.engine.campaign_efficiency register \
  --db /path/to/project/.orze/idea_lake.db \
  --manifest /path/to/campaign-manifest.json
```

Registration stores a canonical manifest hash and database timestamp, rejects
a campaign whose start is not in the future, and never overwrites an existing
campaign ID through the API. This local SQLite ledger is auditable evidence,
not a cryptographic third-party attestation; externally published claims still
need the repository's sealed source/data identities and official evaluation
receipts. Then opt in to one local sample per ordinary scheduler iteration:

```yaml
campaign_efficiency:
  enabled: true
  campaign_id: the-same-unique-id
```

Each sample records only controller/host identity, timestamps, the exact
preregistered scheduler-demand partition, active physical GPU IDs, and scoped
hardware counters. It contains no configs, scores, model outputs, or dataset
content. The hardware query is always passed the controller's explicit GPU
list; an error or partial response is stored as incomplete evidence and is
never replaced by an all-GPU query.

After the registered window ends, generate the receipt:

```bash
python -m orze.engine.campaign_efficiency analyze \
  --db /path/to/project/.orze/idea_lake.db \
  --manifest /path/to/campaign-manifest.json \
  --output /path/to/campaign-receipt.json
```

`VERIFIED` requires an exact preregistered-manifest match, complete samples and
physical scope on exactly one host, no missing or unexpected
in-window lifecycle allocation, sufficient lifecycle evidence and demand, and
every target. Queue-to-claim and terminal-to-next-claim metrics use only the
preregistered idea IDs; unrelated work in the same Idea Lake cannot improve a
campaign receipt. Queue-to-claim retains the raw lifecycle latency and excludes
capacity-full time only when the receipt names an exact eligible scheduler
sample or a one-to-one demanded terminal release as the capacity opportunity;
without that evidence it conservatively uses the raw queue clock. Sequential
controller identities are retained as an
observation and allowed so a verified restart does not erase campaign evidence.
Every scheduler sample must have a matching immutable operator-progress update.
The latest update is published under `results/_campaign_progress/<campaign>/`
and in `status.json`; it carries the current qualified-result identity (or an
explicit null before one exists), a categorical blocker, and the next update
deadline. The default maximum update gap is 600 seconds and may only be
tightened.
Incomplete evidence is `UNVERIFIED`; complete evidence that misses a target is
`FAILED`. Mean hardware utilization is reported as an observation, not silently
used as a substitute for allocation duty cycle or official benchmark evidence.

For a research campaign, the same preregistered manifest may include an
`outcome_contract` containing the exact prospective decision-receipt SHA-256
identities, the expected qualified-artifact relation (`identical`, `distinct`,
or `any`), a `reproducibility_contract`, and every target in
`DEFAULT_OUTCOME_TARGETS`. A campaign with no reproduction question must say
`mode: not_applicable` and provide a bounded rationale. Otherwise, each
preregistered reproduction group names two or more exact idea IDs, the scalar
dotted config paths allowed to vary, and a maximum absolute primary-metric
delta. In both modes, `expected_config_identity_sha256` maps every expected
idea ID to the SHA-256 of its canonical JSON config, computed by
`orze.engine.reproducibility.config_identity_sha256`. Group membership is
disjoint and bound to `expected_idea_ids`.

The analyzer loads stored configs read-only, verifies their stored source and
canonical hashes, rejects exact duplicates, removes only the declared varying
paths, and requires the remaining configs to be identical. Every declared path
must actually vary. It then uses only lifecycle-complete, contract-qualified
metrics to check the preregistered tolerance. Missing or contradictory config,
lifecycle, or metric evidence is `UNVERIFIED`; observed non-reproduction or a
missed tolerance is `FAILED`. Neither result proves an official rank.

The default campaign outcome thresholds require:

Stage the prospective decision contracts to obtain their identities, register
the campaign manifest before its future start, and admit those exact contracts
only inside the registered window. Their combined idea IDs must exactly equal
the manifest's `expected_idea_ids`. This ordering binds the complete experiment
universe before any campaign allocation begins.

Resolved decision receipts bind more than a success count. Schema 2 records the
terminal lifecycle state and qualification reason for every idea, plus the exact
finite primary value and content SHA-256 for each lifecycle-complete result.
Qualification is bracketed by two evidence hashes; a concurrent rewrite leaves
the batch admitted instead of manufacturing a failure or success. The campaign
audit requalifies every recorded decision input and returns `UNVERIFIED` after
any metric, lifecycle, benchmark, provenance, or exposure-evidence drift. Schema
1 receipts remain loadable for operational continuity but cannot verify a new
campaign outcome.

| Outcome dimension | Maximum/minimum |
|---|---:|
| First valid decision | <= 4 hours |
| All declared decisions | <= 24 hours |
| Qualified success rate | >= 25% |
| GPU-hours per qualified success | <= 8 |
| Duplicate training attempts | 0 |
| Zero-GPU rejection rate | 100% |

These thresholds may be tightened before registration but not weakened. After
the window closes, combine the validated decision, compute-allocation, and
current artifact-lineage evidence:

```bash
python -m orze.engine.research_outcomes \
  --db /path/to/project/.orze/idea_lake.db \
  --results-dir /path/to/project/results \
  --config /path/to/project/orze.yaml \
  --manifest /path/to/campaign-manifest.json \
  --output /path/to/research-outcome-receipt.json
```

The outcome receipt distinguishes three states: missing/contradictory evidence
is `UNVERIFIED`, complete evidence that misses a preregistered target is
`FAILED`, and only complete passing evidence is `VERIFIED`. It explicitly sets
`rank_claim_proven: false`; local campaign success never proves an official
leaderboard rank. GPU hours come from framework allocation receipts rather than
trainer-reported utilization. The campaign audit independently derives every
paired allocation duration from its start and terminal timestamps, retains the
recorded duration separately, and makes any mismatch `UNVERIFIED`; rewriting a
long allocation to zero seconds therefore cannot improve
GPU-hours-per-qualified-success.
