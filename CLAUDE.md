# Orze — Development Guide

Read `SKILL.md` in this directory for the full operations & extension reference (CLI, API, architecture, diagnostics).

## Quick Reference

- **Run**: `orze -c orze.yaml` (auto-detect GPUs)
- **Stop**: `orze --stop`
- **Check**: `orze --check` (validate config)
- **Admin UI**: `orze --admin` → http://localhost:8787
- **MCP**: Claude Code connects via `.mcp.json` → `POST /mcp`

## Architecture (post-LOD refactor)

Every module in `src/orze/engine/` is <800 LOC with a calling spec at the top. Read the calling spec before reading the implementation.

```
engine/orchestrator.py    (669) — Orze class, run() recipe
engine/phases.py          (586) — main loop phases (mixin)
engine/role_runner.py     (563) — agent role lifecycle
engine/lifecycle.py       (390) — startup, shutdown, PID
engine/retrospection.py   (341) — signal detection + dispatch to evolution roles
engine/reporter.py        (338) — notifications + plateau
engine/launcher.py        (324) — training subprocess lifecycle
engine/evaluator.py       (298) — eval subprocess + sealed verification
engine/failure.py         (240) — failure tracking + LLM auto-fix
engine/upgrade.py         (223) — auto-upgrade pipeline
engine/failure_analysis.py(210) — structured failure classification
engine/family_guard.py    (180) — approach family taxonomy + repetition guard
engine/sealed.py          (143) — sealed file integrity + metric validation
engine/cluster.py         (140) — multi-machine coordination
engine/config_dedup.py     (85) — config hash dedup
skills/loader.py          (140) — composable prompt fragments
admin/mcp.py              (280) — MCP server for Claude Code
agents/research.py        (460) — research cycle orchestrator
agents/research_context.py(539) — context gathering for research
agents/research_llm.py    (250) — LLM backend implementations
agents/code_evolution.py  (320) — proactive code changes on plateau
agents/meta_research.py   (300) — research strategy adjustment
```

## Conventions

- LOD: max 800 LOC per file, calling spec at top, pure functions for tools
- Config: `orze.yaml` is the single source of truth
- Ideas: `ideas.md` is append-only, consumed into `idea_lake.db`
- Results: `results/{idea_id}/metrics.json` is the training contract
