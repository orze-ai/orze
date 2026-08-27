# orze

[![PyPI](https://img.shields.io/pypi/v/orze)](https://pypi.org/project/orze/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![orze-pro](https://img.shields.io/badge/orze--pro-private-blue)](https://orze.ai/pro)

A GPU experiment orchestrator for ML research.

Orze runs experiments on GPUs: **schedule ideas → train → evaluate → report → repeat**. It coordinates GPUs via filesystem locks, works across machines, and gives you a complete leaderboard, notifications, and analysis — out of the box.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
curl -sL https://orze.ai/install | bash
```

That's it. It installs orze, detects your GPUs and codebase, generates training scripts and experiment ideas, and starts running — all in one command.

Pass environment variables for additional options:

```bash
# LLM-powered setup
ANTHROPIC_API_KEY=sk-ant-... curl -sL https://orze.ai/install | bash

# With pro (autopilot)
ORZE_PRO_KEY=ORZE-PRO-xxx curl -sL https://orze.ai/install | bash

# Custom project path
curl -sL https://orze.ai/install | bash -s /nfs/my-project
```

## orze vs orze-pro

orze is a **complete, production-ready tool**. orze-pro adds **autopilot** — so experiments run while you sleep.

| Feature | orze (free) | + orze-pro |
|---------|:-----------:|:----------:|
| GPU scheduling & multi-node | ✓ | ✓ |
| Idea queue (ideas.md + SQLite) | ✓ | ✓ |
| Hyperparameter sweep (auto-expand grid) | ✓ | ✓ |
| Leaderboard report | ✓ | ✓ |
| Notifications (Telegram/Slack) | ✓ | ✓ |
| Admin dashboard & MCP server | ✓ | ✓ |
| Retrospection (plateau detection) | ✓ | ✓ |
| Cross-experiment regression analysis | ✓ | ✓ |
| Failure analysis & categorization | ✓ | ✓ |
| Checkpoint GC | ✓ | ✓ |
| Sealed eval protection | ✓ | ✓ |
| Service watchdog (auto-restart + containers) | ✓ | ✓ |
| **Autonomous research agents** (Gemini/GPT/Claude) | | ✓ |
| **The Professor** (paper lake, cross-domain search, strategy) | | ✓ |
| **Engineer** (implement ideas, fix bugs) | | ✓ |
| **Auto-fix failed experiments** | | ✓ |
| **Code evolution on plateau** | | ✓ |
| **Meta-research (strategy adjustment)** | | ✓ |
| **FSM orchestration** (7 procedures) | | ✓ |
| **Data analyst & thinker** (auto-injected) | | ✓ |

### Research Loop Comparison

| | orze free | + orze-pro |
|---|---|---|
| **How ideas are generated** | **Smart Suggestions** — rule-based: detects regressions, generates scale sweeps, perturbations | **Research Agents** — LLM-driven: reads all results, forms hypotheses, designs novel experiments |
| **How failures are handled** | You read the failure log | Auto-fix: LLM diagnoses and patches the error |
| **How plateaus are handled** | Smart Suggestions tries parameter variations | Code Evolution: LLM modifies your train script |
| **Does research stop?** | Continuous by default; stop via CLI or queue policy | Continuous by default; finite campaigns use `research_budget.max_agent_cycles` |
| **Requires API key?** | No | Yes (Gemini/OpenAI/Anthropic) |

### Compatibility

| orze | orze-pro | Notes |
|------|----------|-------|
| 4.6.x | 0.13.x | Current release |

## Quick Start

After install, orze auto-detects GPUs and starts running experiments.

**AI CLI users (Claude Code, Cursor, Codex):**
```bash
do @ORZE-AGENT.md
```

## CLI Reference

```bash
# Project lifecycle
orze init [path]              # initialize a new project
orze start                    # start as background daemon
orze stop                     # stop gracefully
orze restart                  # stop + start
orze --check                  # validate config, files, GPUs, API keys
orze --uninstall              # full cleanup, preserves research results

# Operations
orze run-idea IDEA --gpu N    # run one admitted queue item on one physical GPU
orze upgrade                  # reinstall from source + restart daemon
orze admin migrate            # migrate legacy layout to .orze/
orze service install          # auto-restart on crash (systemd)

# Pro
orze pro activate <key>       # activate license
orze pro status               # check license info
orze pro deactivate           # remove license
orze sop list                 # list available SOPs
```

## File Layout

```
your-project/
├── orze.yaml                 # Project config (single source of truth)
├── train.py                  # Your training script
├── ideas.md                  # Experiment queue
├── GOAL.md                   # Research objective
├── RESEARCH_RULES.md         # Agent constraints
├── configs/base.yaml         # Default hyperparameters
├── .env                      # API keys (gitignored)
├── ORZE-AGENT.md             # AI CLI instructions
├── ORZE-RULES.md             # Agent guardrails
├── venv/                     # Training dependencies
├── .orze/                    # Runtime state (gitignored)
│   ├── state/version.json    # Layout version
│   ├── logs/                 # Role logs
│   ├── locks/                # Filesystem locks
│   ├── rules/                # Migrated rule files
│   ├── mcp/                  # MCP server configs
│   ├── receipts/             # Execution evidence
│   ├── triggers/             # One-shot role triggers
│   ├── heartbeats/           # Per-host liveness
│   ├── backups/              # ideas.md backups
│   └── feedback/             # Failure feedback
├── procedures/               # User procedure overrides (pro)
├── fsm/runner.py             # FSM orchestrator (pro)
└── orze_results/             # Research outputs
    ├── idea-0001/metrics.json
    ├── methods/              # Generated code
    └── knowledge/            # Analysis insights
```

## Multi-node

Start orze in the same shared folder on any machine — nodes auto-join the research pool.

```bash
# Node 1
ssh node1 "cd /nfs/project && orze start"

# Node 2
ssh node2 "cd /nfs/project && orze start"
```

## Key Features

- **Scales to 1M+ Experiments** — SQLite-backed job queue with O(log N) scheduling
- **Config Inheritance** — Child ideas inherit parent configs; specify only what changes
- **HP Sweep** — `lr: [1e-4, 3e-4]` auto-expands into all combinations
- **Failure Protection** — Stops automatically when failure rates spike
- **Fail-closed Data Boundaries** — Kernel-isolated held-out paths and optional
  training-network denial plus keyed-fingerprint manifest separation
  ([configuration and limits](docs/data-boundaries.md)); managed runs can bind
  those controls to exact output bytes with
  [model lineage receipts](docs/model-lineage.md)
- **Cross-Experiment Analysis** — Detects regressions, tradeoffs, and suggests actions
- **Rich Notifications** — GPU VRAM, per-dataset breakdown, verified results, target/gap tracking
- **Admin Panel** — Real-time web dashboard at `http://localhost:8787`
- **Clean Uninstall** — `orze --uninstall` removes runtime files, preserves results

## The Contract

Your training script receives:
```bash
python train.py --idea-id idea-001 --results-dir orze_results --ideas-md ideas.md --config base.yaml
```

**Required output:** `orze_results/{idea_id}/metrics.json`:
```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

See [**SKILL.md**](SKILL.md) for the full technical specification.

## Admin Panel

Auto-launches at **http://localhost:8787**. No extra install needed.

<img width="900" height="674" alt="admin-panel" src="https://github.com/user-attachments/assets/b23879e3-d064-4e02-8251-6e8dbfad21f9" />
<img width="900" height="674" alt="admin-queue" src="https://github.com/user-attachments/assets/39747da2-7b7f-4a9f-ad4a-7cfaca41407b" />
<img width="900" height="551" alt="admin-leaderboard" src="https://github.com/user-attachments/assets/70e77941-efbf-4018-9200-93ea77998c5e" />

## Telegram Notifications

```yaml
notifications:
  enabled: true
  on: [completed, failed, new_best]
  channels:
    - type: telegram
      bot_token: "YOUR_BOT_TOKEN"
      chat_id: "YOUR_CHAT_ID"
```

<img width="521" height="341" alt="tg" src="https://github.com/user-attachments/assets/f931221d-b428-4b85-9a8e-af6d516cb5ad" />

## Service Management

```bash
orze service install -c orze.yaml    # auto-restart on crash + manage containers
orze service status                  # check health
orze service audit                   # verify effective unit + pinned runtime
orze service uninstall               # remove
```

The watchdog runs every minute (crontab) or every 5 minutes (systemd). It restarts orze on crash/stall and manages Docker containers defined in `orze.yaml`:

For systemd installations, the watchdog timer is the sole restart decision
owner. The main service does not unconditionally restart itself, so
`.orze_disabled` and `.orze_stop_all` remain authoritative during deliberate
stops and persistent startup failures cannot become restart storms.
Installation pins the imported Orze/Orze-Pro package trees. Before every
systemd start, Orze checks those hashes plus the effective unit properties
(including drop-ins) and fails closed on runtime drift or an active stop latch.
Reinstall the service intentionally after a reviewed package upgrade.

The watchdog also records repeated launch failures in a per-host, mode-0600
state file under `results_dir`. The second consecutive failure with the same
categorical reason emits a rate-limited `watchdog_failure_loop` alert; a changed
reason or restored service health resets the sequence. The receipt and alert
contain only a stable reason code, count, and SHA-256 fingerprint—not subprocess
output, command lines, paths, environment values, or notification secrets.

Direct/manual launches can opt into the same exact interpreter and package-tree
identity check with a `controller_runtime` block in `orze.yaml`. Run
`orze service capture-runtime` from the reviewed executable to print a
canonical block to paste under that key, then run `orze --check` to verify it
and see the active interpreter and import root. This in-process pin detects
drift only in contract-aware Orze
versions; it cannot prevent an older, pre-contract executable from ignoring an
unknown key. Use the managed systemd service and its independent `ExecStartPre`
check when downgrade resistance is required.

Every controller also takes a kernel-backed exclusive lease on each physical
GPU in its invocation scope before startup checks or GPU telemetry. Controllers
with disjoint scopes can coexist; any overlap fails closed. Training,
evaluation, pre/post, post-hoc, and GPU smoke-test children inherit the lease
descriptor, so a controller crash or clean detach cannot make a still-running
child's device appear available to another Orze controller. The startup smoke
test probes only the explicit invocation/`allowed_gpus` scope and uses CPU when
no GPU scope is declared. These host-local leases coordinate Orze processes
running as the same OS user; unrelated external launchers remain outside the
Orze control boundary.

`orze run-idea IDEA --gpu N` is the fail-closed manual campaign path. It
requires an exact runtime pin, an authoritative `QUEUED` lake row, a current
decision-admission receipt, and an allowed physical GPU. It honors every stop
and pause latch, runs no research roles or auto-fix agents, does not reconcile
or report unrelated queue entries, and restricts GPU telemetry and child CUDA
visibility to that one device. Exit status zero additionally requires a
`COMPLETE` lifecycle plus configured evaluation, post-script, benchmark, and
model-lineage evidence. It is not a ranking claim.

Public or benchmark-comparable campaigns should also make the evidence policy
explicit. These switches make configuration invalid—and admission impossible—
until the corresponding full contracts are configured:

```yaml
managed_run:
  require_data_separation: true
  require_model_lineage: true
  require_benchmark_contract: true
  require_explicit_untainted_metrics: true
```

With this policy, an absent or merely implicit no-leakage claim cannot produce
a successful command status. Data-separation evidence is revalidated at the
terminal boundary; model-lineage and benchmark receipts retain their existing
fail-closed validation. The defaults remain `false` for ordinary local runs
that make no public or benchmark-comparable claim.

```yaml
containers:
  paperdog:
    image: orzeai/paperdog:latest
    ports:
      - "8000:8000"
```

Containers are auto-pulled and recreated when a new image is available.

## Citation

```bibtex
@article{li2026autoresearching,
  title={Auto Researching, not hyperparameter tuning: Convergence Analysis of 10,000 Experiments},
  author={Li, Xiaoyi},
  journal={arXiv preprint arXiv:2603.15916},
  year={2026}
}
```

## License

Apache 2.0 — orze is and will always be free and open source.

[orze-pro](https://github.com/orze-ai/orze-pro) (autopilot features) is commercially licensed.
