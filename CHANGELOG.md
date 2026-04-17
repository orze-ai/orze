# Changelog

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
