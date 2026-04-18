# Changelog

## 3.4.7

### Fixed

- **`cleanup_orphans` no longer leaves cross-host dead claims stuck
  forever.** The previous logic only reclaimed directories that had
  `claim.json` AND *no* `metrics.json`. A training that wrote partial
  metrics mid-run (e.g. per-epoch snapshots) and then crashed — exactly
  what happens when a second orze host dies mid-training — left
  `metrics.json` on disk, so the directory was skipped by cleanup even
  though the lake still said `status='running'`. Queue slots leaked
  silently.

  Cleanup now has a second branch: when `claim.json` exists AND
  `metrics.json` exists AND the lake reports `status='running'` AND
  last activity is older than `orphan_timeout_hours`, it removes just
  `claim.json` and resets the lake status to `queued`. The directory
  itself (partial metrics, checkpoints, logs) is preserved for
  inspection. Ideas with status `completed`/`failed`/`skipped` remain
  untouched.

  Observed impact: a second node that died 23h earlier had 9 claims
  stuck running; this release reclaims them without losing partial
  work.

## 3.4.6

### Breaking

- **`check_active_roles` now returns `list[tuple[role_name, Outcome]]`**
  instead of `list[tuple[role_name, str]]`. `Outcome` is a small enum
  (`OK`, `SOFT_FAILURE`, `TIMEOUT`, `ERROR`); use `is_success(outcome)`
  when the caller only cares whether the cycle produced useful output.
  Strategy roles that don't append to `ideas.md` (professor, thinker,
  data_analyst, engineer, code_evolution) no longer trip the
  ideas-modified soft-failure check — they set
  `rp.writes_ideas_file = False` at launch and are judged purely on
  exit code.

### Changed

- **`orze.agents.bug_fixer` renamed to `orze.agents.watchdog`.** The
  legacy `bug_fixer` role has been retired from scaffolding (`orze
  --init` no longer materializes `BUG_FIXER_RULES.md`) and from the
  scheduler role registry. Existing projects with `roles.bug_fixer:`
  in `orze.yaml` keep working but receive a one-time warning suggesting
  migration to the watchdog daemon.

## 3.4.5

### Added

- **`metric_harvest.inference_model` / `inference_timeout` / `claude_bin`**
  are now honored by the orchestrator's pattern_inferrer wiring.
  Previously the inferrer was called with hardcoded `model="haiku"`
  / `timeout=60`. Override in `orze.yaml`:
  ```yaml
  metric_harvest:
    inference_model: sonnet        # haiku (default) / sonnet / opus
    inference_timeout: 120         # seconds (default 60)
    claude_bin: /opt/claude        # (default: PATH lookup)
  ```

## 3.4.4

### Fixed

- **Tighter `_log_has_training_signal` gate.** 3.4.3's gate used two
  separate regexes — an eval-keyword check and a numeric check —
  anywhere in the log. That matched dataset-stats lines like
  `"Train: 1500, Test: 1344"` plus a completely unrelated `0.95`
  from a config echo, producing a false "ready for inference"
  signal. New `_EVAL_LINE_PATTERN` requires both signals on the
  same line and at least two decimal digits (filters out integers
  like sample counts).

- **`phases.py`: thread `flat_cfg` into SOP validator.** The dict-
  /list-stripped config (and the repair of LLM-collapsed keys like
  `"epochs: 40": None` → `{"epochs": 40}`) was being built but not
  handed to the SOP `validate_idea` step, which re-fetched the raw
  config and then flagged the collapsed key as unrecognized. Auto-
  patched by the engineer SOP on the live daemon at 16:20 UTC and
  carried into this release.

## 3.4.3

### Fixed

- **`metric_harvester` inference gating and empty-cache TTL.**
  3.4.2's harvester would invoke the LLM pattern_inferrer on any
  idea whose regex missed — including warmup-only logs that hadn't
  yet produced metrics. The inferrer correctly returned `[]` for
  those, but the empty result got cached permanently, blocking
  future (legitimate) inference for that train script forever.

  Two guards added:

  1. `_log_has_training_signal(log_text)` pre-check — the inferrer
     is only called on logs that are >= 50 bytes AND contain an
     eval/epoch marker AND contain a plausibly-metric-like number.
     Warmup-only logs skip inference entirely.

  2. Empty cache entries now expire after 30 min. Non-empty
     entries stick around until the train script mtime changes.
     A log that was too early on first harvest will get retried
     automatically on the next 5-min cycle after it grows.

  Existing poisoned `results/_metric_patterns_cache.json` files
  from 3.4.2 should be deleted once on upgrade; they'll regenerate
  cleanly from the fresh gating rules.

## 3.4.2

### Added

- **LLM pattern-inferrer hook in `metric_harvester`.** When the
  built-in regex patterns and user-configured `metric_harvest.patterns`
  all fail to extract the primary metric from a training script's log,
  the harvester can invoke an optional `pattern_inferrer` callable
  that proposes regex patterns for the specific script + metric.
  Results are cached keyed by `(train_script.name, mtime)` — one call
  per new script per edit, never again.

  orze's harvester stays pure regex by default; the LLM-backed
  inferrer ships in orze-pro (`orze_pro.agents.pattern_inference`)
  and is auto-wired when orze-pro is installed. Disable via
  `metric_harvest.llm_fallback: false`.

  Cached patterns are persisted to `results/_metric_patterns_cache.json`.

### Changed

- `harvest_running_ideas()` gained keyword args `pattern_inferrer` and
  `train_script`. Extraction now strips trailing punctuation
  (`.,;:`) from the captured number, so prose-style patterns like
  `NDCG came in at ([0-9.]+)` safely match values followed by a period.

## 3.4.1

### Added

- **`orze.engine.metric_harvester`**: periodic scan of
  `results/idea-*/train_output.log` every 20 iterations (~5 min).
  Extracts best-so-far for the configured `primary_metric` via
  regex patterns and writes `metrics.json`. Closes the telemetry
  gap for training scripts that log per-epoch metrics to stdout
  but never emit `metrics.json` — previously left the leaderboard
  blind to mid-run progress for the entire 4-12h training window.

  Default patterns cover `map`, `accuracy`, `auc`, `f1`, `loss`.
  Customize via:
  ```yaml
  metric_harvest:
    enabled: true            # default
    patterns:
      - "score\\s*=\\s*([0-9.]+)"
    maximize: true           # false for loss-like metrics
  ```

  Only rewrites files it authored (sentinel
  `_source: "harvested_from_log"`); genuine `metrics.json` from
  training scripts is left untouched.

### Fixed

- `fix(phases): don't crash on slot-race between launch and register`
  (merged from 733b80b).
- `fix(config): reject legacy rules_file key on role configs`
  (8cf5003 from 3.4.0 development).

## 3.4.0

### Breaking

- **`rules_file:` role config key removed.** Every `mode: claude` and
  `mode: research` role must now declare a `skills:` list. The
  loader's legacy `rules_file` fallback is gone. `orze --check` now
  surfaces `roles.<name>: mode 'claude' requires 'skills'` for any
  pre-refactor config.

  Migration: replace `rules_file: FOO.md` with `skills: [./FOO.md]`
  (treat the file as a dynamic SOP) or switch to bundled static
  SOPs with `skills: [@sop:<id>]`. See `orze sop list` for the
  available ids shipped by orze-pro.

- **`orze pro bootstrap-professor` subcommand removed.** Professor
  behavior now ships as bundled static SOPs in orze-pro (0.7.0+); no
  per-project LLM bootstrap is needed. Task-specific tailoring is
  authored as dynamic SOPs under `<project>/skills/`.

### Added

- **`orze sop` subcommand**: `list`, `check`, `status`.
  - `list`: enumerate every registered SOP — skills (static + dynamic),
    methods, validators, portfolios. Shows tier and role.
  - `check`: validate SOP wiring (`requires`, `consumed_by`,
    `overrides`). Exits non-zero on errors.
  - `status`: read execution receipts under `results/_receipts/` and
    show per-SOP last-cycle evidence (`yes`/`NO`).
  Handler is delegated to orze-pro via a guarded import so basic orze
  without orze-pro returns a helpful message.

- **`@sop:<name>` loader prefix** in `orze.skills.loader.compose_skills`.
  Resolves bundled SOPs from orze-pro's `orze_pro/sops/` directory.
  Unknown names log a warning and skip (fail-open).

- **`orze.skills.loader.compose_skills` composition features**: `order`
  frontmatter controls composition order (lower first); `trigger`
  frontmatter gates a skill in/out per cycle via the role's
  `_trigger_context` dict.

- **`orze.skills.triggers.evaluate_trigger`**: gate grammar supporting
  `always`, `periodic_research_cycles(N)`, `on_file(path)`, `on_plateau(N)`.
  Unknown expressions fail open.

- **`orze.skills.receipts`**: execution receipts recorded per role run.
  Captures pre-run mtimes of each declared `produces` path and
  computes which SOPs were evidenced (output files changed).

### Changed

- `orze --init` generates orze.yaml using `skills:` lists for every
  role. No more `rules_file:` entries or copies of bundled prompts
  to the project directory.
- `_check_pro_role` in the Pro Feature Check queries the SOP registry
  via `orze_pro.skills.registry.discover_skills` instead of looking
  for `*_RULES.md` files on disk. Reports `N static + M dynamic`.

### Removed

- Every `rules_file:` example from `RULES.md`, `AGENT.md`,
  `skills/ops.skill.md`, `skills/setup.skill.md`. Docs now show
  `skills:` throughout.
- The `agents.professor_bootstrap` extension registration in
  `orze.extensions` (module is deleted in orze-pro 0.7.0).
