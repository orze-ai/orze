# Determinism Hardening — Mechanical Glue Audit

Status: design / proposal
Owner: erik@boson.ai
Scope: `src/orze/engine/*`, `src/orze/agents/role_runner.py`, `src/orze/skills/triggers.py`
Out of scope: SOP markdown (`skills/*.skill.md`), LLM role prompts, executor auto-fix

## Problem

Recent recurring bugs (c1005, c1131, c1135, c1136, c1196, c1207) are not LLM
reasoning failures. They are mechanical-glue failures: filesystem-as-message-queue
races, silent skips with no instrumentation, and idea-lifecycle state inferred
from file presence rather than recorded explicitly.

The leverage is **not** in converting SOPs to LangGraph. The SOPs are
appropriately LLM-driven (research idea generation, role orchestration, executor
fix). The leverage is in hardening the mechanical layer underneath so that
restart, multi-host, and partial-failure scenarios are deterministic.

## Layer split (current state)

| Layer | Mode | Examples |
|---|---|---|
| Role SOPs | LLM-prompted | `core.skill.md`, `research.skill.md`, `release.skill.md` |
| Prompt composition | Mechanical Python | `skills/loader.py` |
| Role orchestration | LLM-driven via `role_runner` | `agents/research.py`, professor/engineer/thinker |
| FSM engine | Mechanical Python | `fsm/engine.py`, `fsm/runner.py` |
| Training lifecycle | Mechanical Python | `engine/launcher.py`, `engine/evaluator.py`, `engine/scheduler.py` |
| Idea state | **Implicit filesystem** | results_dir presence, claim.json, metrics.json |
| Trigger queue | **Filesystem + in-memory counters** | `skills/triggers.py`, `agents/role_runner.py` |
| Pause control | **Sentinel files (3 paths)** | `engine/launcher.py:_is_launcher_paused` |

The two bolded rows are the source of every recurring bug class listed below.

## Three failure modes

### F1. Filesystem-as-message-queue races

| Cycle | Symptom | Root cause |
|---|---|---|
| c1005 | Sibling triggers re-fired on daemon restart | consume-then-unlink not atomic |
| c1135 | Pause flag ignored when cwd shifts | 3 candidate paths checked in sequence |
| c1208 | Engine fixes inert until daemon restart | no in-process module reload signal |

### F2. Silent skips without instrumentation

| Cycle | Symptom | Root cause |
|---|---|---|
| c1196 | 6th silent-skip family (AMI gate) | `_ideas_were_modified` logs no credit signal |
| Stage-3 distill | 4 distinct root causes, identical observable | `evaluator.py:58` skips if output present, no marker |
| Various | Train script falls back silently | `launcher.py:882-893` exception → global script |

### F3. Implicit lifecycle state

Idea state is inferred per-iteration from filesystem (no results_dir = QUEUED,
claim.json present = CLAIMED, metrics.json present = COMPLETE). The
`idea_lake.db` exists but is best-effort and not transactionally tied to
filesystem state. Consequences:

- Multi-host claim stealing on Lustre (rare TOCTOU window)
- Orphan cleanup is not transactional — node crash mid-rmtree leaves limbo
- No audit trail for state transitions; forensics relies on log scraping

## Five hardening wins (ranked by ROI)

### #1 — Single canonical pause-flag path
**File**: `engine/launcher.py:805-825` (`_is_launcher_paused`)
**Effort**: S (≈30 LOC)
**Prevents**: c1135-class path mismatches

Today the function checks `results/_launcher_paused.flag`,
`results/../_launcher_paused.flag`, and `cwd/_launcher_paused.flag` in
sequence. Cwd changes flip pause detection unintentionally.

Fix: resolve a single canonical path from `Orze.config.results_dir` once at
startup, store on the orchestrator, log `[PAUSE_CHECK] path=<...> present=<bool>`
on every poll. Drop the other two sentinel paths.

### #2 — Instrument soft-failure credit signals
**File**: `engine/roles.py:240-280` (`_ideas_were_modified`)
**Effort**: S (≈20 LOC + 1 column in idea_lake)
**Prevents**: c1196-class silent skips

The function returns True if any of three credits fires (ideas_consumed counter,
ideas.md mtime, ideas.md size). None of the three are logged. Forensics cannot
distinguish "LLM ran and produced no idea" from "LLM crashed before parsing".

Fix: emit `[SOFT_FAILURE_REASON] role={} consumed={} mtime_delta={} size_delta={}`
on every soft-failure exit. Persist the triple to
`idea_lake.role_runs(soft_failure_reason JSONB)` for cross-cycle analysis.

### #3 — Atomic trigger consume via DB row
**File**: `engine/phases.py` (consume phase) + `skills/triggers.py`
**Effort**: M (DB schema + refactor consume loop)
**Prevents**: c1005 (DEC-009), all future restart-induced sibling duplication

Today: role appends ideas → in-memory `ideas_consumed_during_run++` →
unlink trigger file. Daemon restart between append and unlink re-fires the
trigger; producer re-generates ideas; sweep expander produces bit-identical
siblings.

Fix: add `idea_lake.triggers(trigger_id PK, payload, produced_at, consumed_at,
unlinked_at)`. Consume = `UPDATE triggers SET consumed_at=now() WHERE
trigger_id=? AND consumed_at IS NULL`. Only unlink the file after the
transaction commits. On startup, replay any rows where `consumed_at IS NULL`
AND file still exists; skip rows already marked consumed even if the file
survived a crash.

This is the right resolution of DEC-009 — the c1005 role_runner patch
(`read_text→env→unlink` at `role_runner.py:681-689`) is a band-aid that
papers over the race for the engineer channel only. See
[[project_dec_009_resolved_role_runner_patched]].

### #4 — Pre-launch GPU validation
**File**: `engine/launcher.py:828-950` (subprocess launch)
**Effort**: S-M (≈40 LOC, leverages `gpu_slots`)
**Prevents**: c1136 8-shards-on-1-GPU OOM cascade

Today: launcher sets `CUDA_VISIBLE_DEVICES=str(gpu)` from the scheduler's
claim and trusts it. The scheduler's view can be stale (5s nvidia-smi cache)
or wrong (exit code ignored, see c1136).

Fix: immediately before `Popen`, call
`gpu_slots.verify_gpu_available(gpu, min_free_vram=cfg.min_free_vram)`. On
mismatch: log `[GPU_VALIDATION_FAIL] claimed={} actual_free={} expected>={}`,
release the claim, requeue the idea, do not spawn. Pair with `set -o pipefail`
fix in `smart_dispatch.sh:43` so log() can never pollute stdout again (c1131).

### #5 — Explicit idea lifecycle FSM
**File**: `engine/scheduler.py` + `fsm/engine.py` + `idea_lake` schema
**Effort**: L (≈200 LOC, design doc warranted in addition to this one)
**Prevents**: multi-host desync, orphan limbo, untraceable state transitions

Today: state is inferred per-iteration from filesystem. No audit log of
transitions. Multi-host coordination relies on Lustre metadata, which is
asynchronous.

Fix: add `idea_lake.idea_state` column (enum: QUEUED, CLAIMED, TRAINING,
EVALUATING, COMPLETE, FAILED, ARCHIVED) and an `idea_transitions` table
(idea_id, from_state, to_state, reason, host, timestamp). Extend `fsm/engine.py`
with an `IdeaLifecycleFSM` plugin that owns the transition rules. Filesystem
remains the *artifact* store but is no longer the *truth* about state.

This is the only entry on the list big enough to justify its own design doc.
Recommend deferring until #1–#4 land — they reduce the bug rate enough to
think clearly about #5.

## Recommended sequencing

1. **Week 1**: #1, #2, #4 (each <1hr, each closes a known recurring bug class)
2. **Week 2**: #3 (closes DEC-009 properly, replaces the role_runner band-aid)
3. **Later**: #5 (separate design doc; requires migration plan for in-flight ideas)

## What this design explicitly is not

- **Not a LangGraph migration.** LangGraph would solve the state-machine gap
  (#5) but bring a heavy dependency and would not help #1–#4. A 50-line
  sqlite-backed FSM in `fsm/engine.py` covers the same ground.
- **Not an SOP rewrite.** The markdown SOPs in `skills/` are doing the right
  job — adaptive judgment in the role loops. Freezing them as graphs would
  lose the property that surfaced DEC-009-style course corrections.
- **Not a multi-host refactor.** #5 makes multi-host *possible* without
  filesystem TOCTOU but does not introduce new cross-host coordination
  primitives. `engine/cluster.py` remains opt-in.

## Open questions

- Does `idea_lake.db` get the new `triggers` and `idea_transitions` tables, or
  do they live in a sibling `orze_state.db`? (Argument for separate: triggers
  table is hot and high-write; idea_lake is read-mostly for forensics.)
- For #4, should requeuing on GPU validation failure increment a per-idea
  retry counter, and after N retries mark the idea FAILED rather than loop?
- For #5, do we migrate in-flight ideas (set state from filesystem on first
  boot) or quiesce, drain, and migrate cleanly?

## References

- DEC-009 resolution log: `results/_decisions/DEC-009-A.md`
- c1005 role_runner patch: `role_runner.py:681-689`
- c1131 smart_dispatch parser bug: `tools/smart_dispatch.sh:43`
- c1135 pause-flag mismatch: memory `[[project_pause_flag_path_mismatch]]`
- c1136 GPU detector exit-code: memory `[[project_smart_dispatch_min_count_bug_c1136]]`
- c1196 in-process AMI gate: memory `[[project_silent_skip_family_6_in_process_ami_gate]]`
- c1207 scheduler tiebreaker (fixed): memory `[[project_pf_idea_alphabetic_starvation]]`
