# Harness efficiency contract

Orze treats harness efficiency as a set of separately verifiable properties.
A fast model run does not prove a fast scheduler, and a fast scheduler does not
prove high GPU utilization, research yield, or leaderboard rank.

## CPU control-plane acceptance

Run the production APIs against a disposable database on the same filesystem
used by the project:

```bash
python -m orze.benchmarks.harness_efficiency \
  --work-dir /path/to/project/.orze \
  --ideas 50000 \
  --queue-limit 2000 \
  --iterations 15 \
  --output /path/to/receipt.json
```

The command does not import a model, launch training/evaluation, or probe an
accelerator. A receipt is `VERIFIED` only with at least 10,000 ideas and 15
steady-state samples, and only when every target and invariant passes:

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
| Zero-compute rejection | 100% of invalid/ineligible proposals rejected before GPU allocation |
| Duplicate compute | zero duplicate config or claim launches |
| Time to decision | preregistered per campaign, measured from admission through valid screen/full decision |
| Research yield | independently verified improvement rate and GPU-hours per accepted improvement |
| Reproducibility | rerun within preregistered metric tolerance from sealed code/data/environment identities |
| Leaderboard outcome | accepted official submission/rank under the leaderboard's complete eligibility rules |

These rows remain open until their own receipts exist. Local proxy scores,
configuration inspection, or a CPU benchmark cannot close them.
