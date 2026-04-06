# Orze — Multi-Machine & Distributed Architecture

## Overview

Orze supports multi-machine training on shared filesystems using a **commander + worker** model. No database, message broker, or additional infrastructure is required beyond a POSIX-compliant shared filesystem (NFS, Lustre, EFS, GPFS, BeeGFS).

## Architecture

```
                    Shared Filesystem
                   ┌─────────────────┐
                   │  results/       │
                   │  ideas.md       │
                   │  idea_lake.db   │
                   │  configs/       │
                   │  orze.yaml      │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────┴─────┐ ┌────┴────┐ ┌─────┴─────┐
        │ Commander  │ │ Worker  │ │ Worker    │
        │ Machine 1  │ │ Machine │ │ Machine   │
        │            │ │ 2       │ │ 3         │
        │ Roles:     │ │         │ │           │
        │  research  │ │ Train   │ │ Train     │
        │  professor │ │ only    │ │ only      │
        │  fsm       │ │         │ │           │
        │  bug_fixer │ │         │ │           │
        │ + Training │ │         │ │           │
        └────────────┘ └─────────┘ └───────────┘
```

### Commander (1 per project)
- Runs all roles: research agent, professor, code evolution, bug fixer, FSM
- Generates ideas, reviews quality, manages strategy
- Also trains on its local GPUs
- Writes to ideas.md, idea_lake.db, report.md, status.json

### Workers (many)
- Only claim ideas and train
- No roles configured — just `orze -c orze.yaml --gpus <local_gpus>`
- Read from shared filesystem, write results to `results/{idea_id}/`

## Setup

```bash
# Machine 1 (commander) — full orze with roles
orze -c orze.yaml --gpus 0,1,2,3

# Machine 2 (worker) — training only, no roles
orze -c orze.yaml --gpus 0,1,2,3

# Machine 3 (worker)
orze -c orze.yaml --gpus 0,1,2,3
```

Workers use the same `orze.yaml` but only the commander has roles configured. If a worker's `orze.yaml` has roles, the filesystem lock dirs (`_{role_name}_lock`) prevent duplicate execution — only one machine runs each role at a time.

## Coordination Mechanisms

### Experiment Claiming
- Uses **atomic `mkdir()`** on `results/{idea_id}/`
- POSIX guarantees exactly one process succeeds
- Loser immediately picks the next idea
- No lock files, no database transactions

### Role Locks
- Each role uses a `_{role_name}_lock/` directory
- Atomic mkdir ensures single-holder
- `lock.json` inside records hostname, PID, timestamp
- Stale locks broken after 600 seconds
- Same-host stale locks broken immediately via PID check

### Idea Flow
```
Research Agent → ideas.md (append) → Orchestrator ingests to SQLite → Wipes ideas.md
```
- ideas.md is a transient inbox, not the database
- idea_lake.db (SQLite) is the persistent queue
- Only the commander writes to both

### Atomic Writes
- `report.md`, `status.json`, `metrics.json` use atomic write pattern:
  1. Write to temp file with hostname+PID in name
  2. `fsync()` the file
  3. `os.replace()` to final path (atomic on POSIX)
  4. `fsync()` the parent directory (Lustre safety)

## Scaling Limits

| Machines | Status | Notes |
|----------|--------|-------|
| 1 | Fully supported | Current setup, all features work |
| 2-10 | Well supported | Commander + workers, no issues expected |
| 10-20 | Supported | NFS attribute cache may cause brief stale reads (<3s) |
| 20-50 | Should work | Not extensively tested. Monitor for lock timeout issues |
| 50+ | Use with caution | ideas.md race window grows; consider idea-per-file pattern |

### Bottleneck Analysis

| Component | Write Frequency | Scales To |
|-----------|----------------|-----------|
| `results/{id}/` mkdir (claim) | 1 per GPU per experiment (hours) | **Unlimited** |
| `metrics.json` per idea | 1 write when training ends | **Unlimited** |
| `idea_lake.db` SQLite | ~1 write per 30s (commander only) | **1 writer** (safe) |
| `ideas.md` append | ~1 write per 30-120s (commander only) | **1 writer** (safe) |
| `report.md` / `status.json` | 1 write per 30s poll (commander only) | **1 writer** (safe) |
| Role lock dirs | 1 mkdir per role cycle | **Unlimited** (by design) |

The real limit is **not coordination** — it's shared filesystem bandwidth for reading training data across many machines.

## Known Limitations

### Only 1 Commander
- Roles (research, professor, FSM, bug fixer) run on a single machine
- If the commander goes down, workers continue training the existing queue but no new ideas are generated
- No automatic failover — restart the commander manually

### NFS Attribute Cache
- `os.path.exists()` can return stale results for 3-60 seconds on NFS
- Impact: a worker might briefly not see a newly claimed idea, but `mkdir()` atomicity prevents double-claiming
- Mitigation: not needed in practice — the race window is harmless

### Lock Timeout Race (Theoretical)
- ideas.md file lock has a 60-second stale timeout
- If SQLite ingestion takes >60s (very slow NFS), the research agent could append while the orchestrator is mid-parse
- Impact: potential loss of a few ideas (silent)
- Likelihood: very low — SQLite ingestion is <1s normally
- Mitigation: increase lock timeout if running on very slow NFS

### Config Mutation During Training
- Professor can edit `base.yaml` and `RESEARCH_RULES.md`
- Workers starting a new job during the edit may see partial config
- Impact: one experiment with wrong config (retried automatically)
- Mitigation: professor edits are rare and atomic writes could be added

### No Multi-Orchestrator Support
- Running two commanders on the same `results/` directory is **not supported**
- Both would race on ideas.md, role locks, and report generation
- The lock dirs prevent catastrophic failures but results may be inconsistent

## What's Safe to Claim

- "Scales training across multiple machines on shared storage"
- "One machine runs the research brain, others contribute GPUs"
- "No infrastructure beyond shared filesystem"
- "Atomic experiment claiming — no duplicate work"
- "Crash-safe — workers resume after restart"
- "Reliable up to ~20 machines"

## What's NOT Safe to Claim

- "Fully distributed roles" — roles are single-instance, not distributed
- "Multi-orchestrator" — only 1 commander per project
- "Production-grade high availability" — no automatic failover
- "Tested at 100+ nodes" — not validated at that scale

## Future Improvements

1. **Idea-per-file pattern**: Replace ideas.md append with a `ideas/` directory where each idea is a separate file. Eliminates the append race entirely. Easy to implement.

2. **Role failover**: If commander goes down for >N minutes, a worker could promote itself. Requires leader election via filesystem (e.g., atomic mkdir on a sentinel file).

3. **Config versioning**: Snapshot configs when an experiment starts, so mid-training config edits don't affect running experiments.

4. **Distributed roles**: Allow multiple machines to run roles with work-stealing. Requires replacing filesystem locks with a proper distributed lock (or just accepting the current single-holder design).
