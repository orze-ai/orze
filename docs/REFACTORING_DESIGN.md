# Orze Refactoring Design: Smaller Research Runtime

Status: **design only — no runtime refactor is authorized by this document**

Scope: `orze` core and `orze-pro`

Date: 2026-08-17

## 1. Decision

Refactor Orze in place; do not rewrite it and do not turn it into a general
agent platform.

The public execution model remains:

```text
load config
  -> recover durable state
  -> claim one bounded experiment
  -> launch it in a process group
  -> validate the observed result
  -> commit one terminal state
  -> report and stop, or repeat within explicit limits
```

Pro remains an optional research-policy layer. It may propose experiments and
version research policy. Core remains the sole owner of process launch, the
running-experiment FSM, crash recovery, and post-launch `COMPLETE`/`FAILED`
transitions.

There is one deliberate legacy boundary: Pro's idea-verifier plugin may reject
a still-queued proposal by writing `ideas.status = skipped` and a `SKIPPED`
metrics record. That is a pre-launch policy disposition, not a running-process
transition. This refactor preserves it and does not falsely classify it as
core-owned. Replacing those raw writes is a separate correctness design, not a
side effect of moving code.

This design supersedes the structural-refactor portions of
`orze-pro/docs/ORZE_refactor_design.md`. That document is historical product
exploration; its proposed registries, ledgers, and capability expansion are not
authorized here.

## 2. Why refactor

The current runtime works, but a few functions carry too many unrelated
decisions. Measured on the current trees:

| Repository | Function | Lines | Mixed responsibilities |
|---|---|---:|---|
| core | `Orze.run` | 532 | startup, polling, process collection, scheduling, reporting, once-mode |
| core | `OrzePhaseMixin._launch_training` | 614 | queue policy, validation, dedup, retries, resource checks, launch |
| core | `launcher.check_active` | 347 | process observation, result validation, failure accounting, state commit |
| core | `launcher.launch` | 254 | command construction, isolation, claim identity, process spawn |
| Pro | `run_role_step` | 519 | triggers, cooldowns, budget, commands, receipts, environment, spawn |
| Pro | `run_all_roles` | 315 | completion, usage, alerts, retries, contribution accounting, relaunch |
| Pro | `_reserve_research_attempt` | 169 | migration, locking, accounting, durable append |

The problem is not file size by itself. `cli_setup.py`, reporting code, and
provider-specific prompts can be long without owning the critical state
transition. The refactor targets only functions where multiple failure domains
share one control path.

## 3. Goals

1. Make the experiment and role lifecycles readable top to bottom.
2. Give every durable transition one obvious owner.
3. Make crash, budget, and terminal-state invariants directly testable.
4. Preserve current CLI, configuration, file formats, extension loading, and
   on-disk state.
5. Reduce branching and duplicated cleanup without adding runtime work.
6. Make future deletion easier: optional behavior must remain outside core.

## 4. Non-goals

This refactor will not add:

- a workflow engine, event bus, service, daemon, or distributed transaction;
- a generic policy DSL, registry, approval system, or rollout platform;
- a provider router, price catalog, cache layer, or capability-negotiation tree;
- a new database or migration framework;
- a new plugin system or replacement extension API;
- multi-agent delegation, multi-GPU collectives, Kubernetes, or remote workers;
- an embedded OS sandbox; deployments may use an external container or
  scheduler when stronger isolation is required;
- new configuration keys, environment variables, dependencies, threads, or
  subprocesses;
- feature work disguised as refactoring.

Line-count reduction alone is not a reason to introduce a module, class, or
abstraction.

## 5. Constraints that must remain true

### 5.1 Core owns the running experiment lifecycle

Only core may perform these effects:

- claim an experiment;
- persist trainer PID, process group, and stable process identity;
- transition the experiment FSM;
- authorize retry after recovery;
- declare post-launch `COMPLETE` or `FAILED`.

Pro can request work through existing files and APIs. Before core claims an
idea, the existing Pro verifier may disposition it as `skipped`; it may not
transition a claimed or running experiment. The live plugin currently selects
queued rows and later updates by `idea_id` alone, leaving a claim race. The
first correctness slice must make both skip updates conditional on
`status = 'queued'` and write `metrics.json` only when that update succeeds.

### 5.2 No additional state model or store

Keep the existing Idea Lake/FSM and per-experiment files. The small Pro budget
JSONL remains a bounded campaign ledger. Do not introduce another authoritative
state store during refactoring.

### 5.3 Fail closed at the same boundaries

- Corrupt state must not authorize launch or retry.
- A durable attempt debit happens before role `Popen`.
- Recovery must prove the prior trainer group stopped before retry.
- A result becomes terminal only after metrics and postconditions validate.
- A policy proposal cannot become active without validation and versioning.

### 5.4 No extra work in hot paths

The refactor must not add directory scans, database passes, provider calls,
lock acquisitions, serialization, or polling iterations. Helper calls are fine;
new I/O is not.

## 6. Target dependency shape

```text
orze.cli
  -> engine.orchestrator       public runtime facade and loop
       -> engine.phases        queue/eval/train/report phase coordination
       -> engine.launcher      process construction and observation
       -> engine.idea_validation  config/method rules and file-backed lookup
       -> engine.lifecycle     startup, recovery, shutdown
       -> optional Pro extension facade

orze_pro.engine.role_runner    public Pro facade and scheduling
  -> engine.research_budget    durable campaign debit
  -> existing core process/fs/role primitives
```

Only two production modules are justified by this design:

1. `orze/engine/idea_validation.py`
2. `orze_pro/engine/research_budget.py`

Completion accounting stays local to `role_runner.py` because extracting it
would either create an import cycle or move a wide set of private helpers. A
third module requires a revised design that demonstrates an acyclic ownership
boundary. No additional class hierarchy is planned. Existing context
dictionaries and dataclasses remain until a reproduced bug requires changing
them.

## 7. Core design

### 7.1 `orchestrator.py`: imperative shell, not a god object

`Orze` remains the public class. Keep construction, signal handling, extension
bridging, and the outer loop there.

Split `run()` into named private steps in the same module:

- `_start_runtime()` — startup checks, recovery, leadership, canary;
- `_run_iteration()` — one ordered orchestration iteration;
- `_collect_finished_work()` — training/evaluation process observation;
- `_run_once_to_completion()` — the existing `--once` drain behavior;
- `_finish_runtime()` — final reporting and shutdown.

These are sequencing functions, not new services. They may call existing phase
methods but may not own result-validation rules. The body of `_run_iteration()`
should read as the execution outline and contain no provider- or idea-specific
branches.

### 7.2 `phases.py`: separate selection from effects

Keep the four existing phases. Break the large methods at their current
decision boundaries:

- `_sync_ideas`: ingestion, sweep expansion, and queue construction;
- `_launch_training`: candidate selection, candidate validation, retry/fix
  decision, and launch authorization;
- `_report_and_notify`: state snapshot, notifications, and durable save.

Pure decisions should return data. Effectful helpers should perform exactly one
kind of effect. Do not build a generic phase protocol or phase registry.

### 7.3 `idea_validation.py`: one narrow extraction

Move only the idea-config and method-validator logic currently embedded in
`launcher.py`:

- nested-config normalization and rejection;
- validator rule evaluation;
- method-validator loading and evaluation.

`launcher.py` temporarily re-exports existing names so imports and tests remain
compatible. The effectful `log_validator_rejection()` stays in `launcher.py`;
the new module does not write evidence or state. No validator registry, schema
framework, or plugin API is added.

The extraction does not merge the two validation sites. `phases.py` remains the
post-claim, pre-`Popen` authority: it applies normalization, rejects forbidden
control-plane overrides, records rejection evidence, writes the current
failure-shaped metrics, sets `ideas.status = skipped`, and updates failure
counts exactly as it does today.
The launch-time method check remains a last-moment guard for validators changed
after enqueue. Its current `RuntimeError` and downstream classification remain
unchanged in this slice. In particular:

- nested-config validation must not be removed from `phases.py`;
- a phase rejection must remain `SKIPPED`, not become a launch failure or an
  executor-fix attempt;
- idea data must not carry a validator-bypass switch at either site;
- validator names, rejection JSONL records, metrics text, and failure-count
  behavior must be characterized before code moves.

Unifying the two outcomes would be a semantic change and needs a later design.

### 7.4 `launcher.py`: launch and active-process boundary

After validation extraction, `launch()` should visibly perform:

1. resolve command and paths;
2. apply the existing data/isolation boundary;
3. verify GPU/resource preconditions;
4. rely on the already-durable `CLAIMED` intent;
5. spawn one process group;
6. capture PID, PGID, and process start ticks;
7. durably update `claim.json` with that identity;
8. persist `CLAIMED -> IN_PROGRESS`;
9. return `TrainingProcess`.

PID/PGID identity cannot exist before `Popen`. If identity capture, claim
persistence, or the FSM transition fails, `launch()` must kill and reap the
new child, close its log handle, and raise. It must never advertise
`IN_PROGRESS` before durable process identity exists.

`check_active()` should visibly perform:

1. observe process state;
2. classify one condition;
3. collect log/metrics evidence;
4. run the existing fix-and-relaunch decision when applicable;
5. commit success/failure once, or replace the active entry with the relaunched
   process;
6. release resources only when no replacement remains active.

Timeout, stall, zombie, fatal-log hang, reported failure, and nonzero exit all
currently reach direct retry paths. This refactor consolidates their duplicated
mechanics in one local helper but keeps retry ownership inside `launcher.py`;
it does not introduce a retry-result protocol or move relaunch into the
orchestrator. Each path needs a characterization test that proves whether it
relaunches, which evidence it writes, and which active GPU entry remains.

Small private helpers stay in `launcher.py`; a separate execution framework is
not justified. Existing posthoc dispatch, final method revalidation, and
launch-specific rank/resource guards also stay here; calling this a process
boundary does not authorize moving or deleting those policies.

## 8. Pro design

### 8.1 `research_budget.py`: isolate one correctness invariant

Move the durable attempt reservation intact, then simplify it behind one
function:

```python
reserve_attempt(
    *, cfg, role_states, active_roles, role_name, role_cfg, cycle_num
) -> str | None
```

It remains a locked, fsynced JSONL implementation. No database conversion,
refund/reconciliation service, provider price table, or background compaction
is part of this refactor.

### 8.2 `role_runner.py`: scheduling and launch sequencing

Keep the public exports unchanged:

- `RoleContext`
- `build_claude_cmd`
- `build_research_cmd`
- `run_role_step`
- `run_all_roles`
- `run_role_once`

Rewrite `run_role_step()` as a short sequence of private helpers:

1. reject already-active, malformed, paused, or not-due work without consuming
   a trigger;
2. inspect trigger presence/payload without acknowledging it;
3. prepare command, environment, and input snapshot;
4. acquire the existing per-role cross-machine lock;
5. durably `reserve_attempt`;
6. atomically `claim_trigger` for that reserved attempt;
7. `_spawn_role` while retaining the lock in the active-role record.

For scheduled roles, step 6 is absent. For trigger-driven roles, a trigger is
not consumed merely to discover that cooldown, role-lock, or budget eligibility
failed. Once consumed, it is attached to an already-reserved attempt; a later
spawn failure is recorded as that attempt rather than silently restoring work.
This preserves the existing debit-before-`Popen` invariant without adding a
second trigger store or acknowledgement protocol. Slice 0 must first capture
the exact manual-trigger cooldown bypass behavior so delayed consumption does
not change eligibility.

`claim_trigger()` can still return `None` after reservation when the inspected
file is unreadable, disappears, or is an orphan whose receipt was committed by
a prior cycle. In that case the reservation remains charged, no process is
spawned, and the role lock is released. This conservative no-refund outcome is
intentional: it avoids both budget overshoot and re-firing a possibly consumed
trigger without adding cross-store transactions. Slice 5 must test absent,
read-error, and orphan outcomes and make the no-spawn debit visible in existing
logs.

The authoritative reservation remains immediately before trigger claim and
spawn. Preparation may create scratch/log paths, but may not cause the role's
intended external effect. On every failure after acquiring the role lock, the
existing cleanup must release it.

Role auto-injection stays in this module. It is product policy, not a generic
discovery mechanism.

### 8.3 Completion accounting stays local

Create one private `_process_finished_roles()` helper in `role_runner.py` for
the daemon path. It keeps the completion-only behavior together:

- usage receipt append;
- declared-output receipt completion;
- stray-file handling;
- repeated-stub/intervention checks;
- success, timeout, rate-limit, and error counters;
- cooldown/backoff calculation;
- contribution summary data.

It processes the already-finished `(role, outcome)` list and returns the
existing summary information. It must not schedule or spawn a role. Keeping it
local preserves access to receipt, stray-file, timeout, cooldown, and
notification helpers without a circular dependency.

`run_role_once()` currently waits and logs but does not perform daemon
completion accounting. Preserve that behavior; routing it through the helper
would be a semantic change, not refactoring. Do not turn outcome handling into
events, handlers, or a plugin registry.

## 9. Compatibility plan

The following remain stable throughout the refactor:

- `orze.cli` behavior and flags;
- `orze.yaml` keys and defaults;
- Idea Lake schema and FSM state names;
- claim, recovery, metrics, status, report, and role-state formats;
- extension discovery through `orze.extensions`;
- public imports used by current tests;
- core `4.6.x` / Pro `0.13.x` compatibility line.

Moved functions are re-exported from their old modules for one minor release.
Internal underscore-prefixed wrappers with no callers may be deleted in the
same slice after repository-wide search and tests.

## 10. Implementation slices

Each slice is independently reviewable and revertible. Do not combine slices.

### Slice 0 — Characterization wall

- Add no implementation abstraction.
- Freeze focused tests for crash recovery, one terminal transition, budget
  reservation, all six relaunch conditions, trigger/cooldown/lock ordering,
  policy rollback, role outcome classes, and `--once` behavior.
- Record current hot-loop scan/query counts for one two-idea run.
- Record exact core/Pro commits, dirty patches, Python/dependency versions,
  test commands, and the AST measurements used in Section 2.

### Slice 1 — Guard the Pro verifier claim race

- In both Tier 1 and professor skip paths, use a conditional queued-to-skipped
  update and check the affected-row count.
- Write `SKIPPED` metrics only after that conditional update succeeds.
- Add a race test proving a concurrently claimed idea is left untouched.
- Add no API, module, schema, retry, or background reconciliation.

### Slice 2 — Pro budget extraction

- Move budget code to `research_budget.py` without semantic changes.
- Preserve adversarial tests for legacy import, concurrency, crash-before-spawn,
  token clamping, and cost-flag override.

### Slice 3 — Pro daemon completion helper

- Move daemon completion behavior to `_process_finished_roles()` in the same
  module.
- Reduce `run_all_roles()` to bootstrap, collect, summarize, and schedule.

### Slice 4 — Pro launch sequencing

- Split `run_role_step()` into local helpers without changing eligibility.

### Slice 5 — Pro trigger ordering correction

- Delay trigger consumption until eligibility, role lock, and budget
  reservation have succeeded.
- Prove triggered, manual-trigger, scheduled, budget-refused, and lock-refused
  cases; do not add a trigger schema or acknowledgement service.

### Slice 6 — Core validation extraction

- Move validator functions to `idea_validation.py`.
- Re-export old import names.
- Update imports only. Preserve the distinct phase rejection and last-moment
  launch revalidation outcomes described in Section 7.3.

### Slice 7 — Core active-process simplification

- Consolidate the six duplicated fix/relaunch mechanics inside `launcher.py`.
- Keep classification, retry eligibility, evidence, and active-GPU replacement
  behavior identical.

### Slice 8 — Core outer loop simplification

- Split `Orze.run()` in place.
- Keep call order, shutdown behavior, and I/O count identical.

### Slice 9 — Core phase simplification

- Refactor one large phase method per commit, starting with
  `_launch_training()`.
- Keep selection order and effect order identical.
- Delete wrappers and branches proven unused by repository search and tests.

Stop after any slice if the next slice lacks a measured complexity reduction.

## 11. Acceptance criteria

### Structural

- Add only the two production modules named in Section 6. Any further module
  requires revising and reviewing this design first.
- No new production classes, persistence formats, dependencies, or config keys.
- Every slice must reduce at least one recorded structural measure in the code
  it touches: top-level branches in the target function, repeated durable-write
  sites, repeated fix/relaunch blocks, or dependency edges. It must not increase
  the other recorded measures without written justification.
- Runtime LOC in a slice may grow only for a characterized correctness fix or a
  compatibility re-export; helper extraction by itself must be net-neutral or
  smaller after comments and blank lines are excluded.
- `run_role_step`, `run_all_roles`, `Orze.run`, `_launch_training`, `launch`,
  and `check_active` must read as ordered control flow; no line-count target is
  used as a proxy for clarity.
- Existing public imports continue to work.

### Behavioral

- All existing core tests pass.
- All Pro tests pass with only the commercial import gate substituted in the
  unlicensed development environment.
- Crash recovery still proves the old trainer group stopped before retry.
- Concurrent budget reservation cannot overshoot.
- Exactly one terminal transition is emitted per experiment.
- Policy promotion and rollback remain checksum-verified.
- A two-idea core run produces equivalent DB/FSM/report outcomes.

### Efficiency

- No additional filesystem scan, SQLite query pass, lock, subprocess, provider
  request, or polling cycle in the steady-state iteration.
- Compare startup and one-iteration wall time on the same machine and checkout
  using the pinned Slice 0 command. Timing is diagnostic: an apparent regression
  must be explained with I/O/call counts or profiled, not accepted or rejected
  from five noisy samples.
- Compare peak RSS for the same queue and active-process count; investigate a
  repeatable increase, but do not add a benchmark framework.

### Reproducible evidence

- Record the exact core and Pro commits plus any dirty patches and untracked
  source needed to reproduce a slice; a branch name alone is insufficient.
- Capture each verification command, exit code, and stdout/stderr in ordinary
  files that are included by normal Git staging.
- Evidence bundles must contain repository-relative paths only, have no
  host-local absolute symlinks, and pass their recorded checksums from a fresh
  checkout.
- Do not add an evidence service, schema, or runtime integration. This is a
  packaging requirement for review artifacts only.

## 12. Stop conditions

Stop and reconsider the design if any slice:

- requires an on-disk migration;
- changes retry, terminal-state, trigger, or budget semantics beyond the two
  characterized race/order corrections in Slices 1 and 5;
- adds a production module not named in Section 6;
- needs a generic interface with only one implementation;
- increases hot-path I/O;
- causes core to import Pro;
- requires broad test rewrites rather than compatibility shims;
- cannot state what code or branch becomes simpler or deleted.

## 13. Rollout and rollback

Land one slice per commit. Before each slice, capture the focused test list and
the current function/LOC measurements. After each slice:

1. run focused invariant tests;
2. run the full repository suite;
3. run the bounded two-idea acceptance workload where applicable;
4. compare hot-loop I/O counts;
5. inspect the diff for net abstraction growth.

Before accepting the slice, reproduce its evidence from a fresh checkout using
normal ignore rules. A local artifact that depends on unstaged files or `/tmp`
paths does not count as verification.

Rollback is `git revert` of that slice. There is no dual runtime, feature flag,
state migration, or compatibility mode.

## 14. First implementation recommendation

Begin with Slice 0, then Slice 1 only. The queued-to-skipped race is a small,
reproduced ownership violation that can be corrected without abstraction or
new state. It tests whether this work honors the architecture before moving
code. Do not begin extraction until Slice 1 is accepted.
