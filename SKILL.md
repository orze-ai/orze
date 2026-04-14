# Orze — Complete Operations Guide (v3.2.29)

Orze is a filesystem-coordinated GPU experiment orchestrator. It runs the loop: **generate ideas → train → evaluate → learn → repeat**.

## Mental Model

```
ideas.md → [claim via mkdir] → train on GPU → metrics.json → leaderboard
     ↑                                                           |
     └──── research agent reads results, appends new ideas ──────┘
                         ↑
              _manual_results.json ← orze result add (external experiments)
```

- **ideas.md**: append-only experiment queue (markdown + YAML config blocks)
- **idea_lake.db**: SQLite archive (hot ideas consumed from ideas.md into DB)
- **results/{idea_id}/metrics.json**: the contract — `{"status": "COMPLETED", ...}`
- **results/_manual_results.json**: user-registered external results (merged into leaderboard)
- **Coordination**: atomic `mkdir` for claims, filesystem locks for roles — works across machines on shared storage (NFS/FSx/EFS)

## CLI Reference

### Lifecycle

```bash
orze start                         # start as background daemon (auto-detect GPUs)
orze start -c orze.yaml            # with explicit config
orze start --gpus 0,1,2            # specific GPUs
orze start --foreground            # run in foreground (for debugging)
orze stop                          # graceful stop
orze stop --timeout 120            # wait longer for children
orze restart                       # stop + start
orze restart --foreground          # restart in foreground
```

Legacy flags (still work):
```bash
orze --once                        # one cycle then exit
orze --stop                        # graceful stop
orze --disable                     # persistent disable (survives restarts)
orze --enable                      # re-enable
```

### Monitoring

```bash
orze --admin                       # web UI at :8787
orze --report-only                 # regenerate leaderboard
orze --check                       # validate config, files, API keys, GPUs
cat results/report.md              # human-readable leaderboard
cat results/status.json            # machine-readable status
```

### Run a Single Role

```bash
orze --role-only research          # run research agent once
orze --role-only professor         # run professor once
orze --role-only documenter        # any configured role
orze --research-only               # alias for --role-only research
```

### Register External Results

**Problem**: experiments run outside orze (manual launches, external repos) are invisible to the professor and research agents.

**Solution**: `orze result` writes to `results/_manual_results.json`, which is merged into the leaderboard context that professor and research agents see.

```bash
# Register a result
orze result add --name riskprop_ep10 --map 0.8337 --epoch 10 \
  --pipeline riskprop_slowonly_r50 \
  --notes "AdaLEA loss, 8 GPUs, val 300 samples"

# List all manual results
orze result list

# Remove a result
orze result rm old_experiment
```

The professor and research agents automatically see manual results in their leaderboard context on the next cycle — sorted by mAP alongside orze-tracked results.

### Reset Idea Lake

```bash
orze reset                         # show status summary
orze reset --failed                # purge all failed ideas
orze reset --all                   # purge all non-completed ideas
orze reset --full                  # wipe entire lake (backup created)
orze reset --full -y               # skip confirmation
```

### Service (Auto-Restart Watchdog)

```bash
orze service install -c orze.yaml  # install watchdog (crontab or systemd)
orze service status                # check health
orze service logs -n 100           # show watchdog logs
orze service uninstall
```

### Pro License

```bash
orze pro status                    # show license info
orze pro activate ORZE-PRO-xxx    # activate with key
orze pro deactivate               # remove license
orze pro bootstrap-professor      # generate PROFESSOR_RULES.md from GOAL.md
```

### Upgrade / Uninstall

```bash
orze --upgrade                     # upgrade from PyPI
orze --uninstall                   # full uninstall (keeps results)
```

## Multi-Machine Operation

Orze supports multiple instances on a shared filesystem. Each node:
- Gets a host-specific PID file: `results/.orze.pid.{hostname}`
- Gets a host-specific state file: `results/.orze_state_{hostname}.json`
- Claims ideas atomically via `mkdir` — no double-execution
- Shares the same `idea_lake.db`, `ideas.md`, and `results/`

**Starting on a remote node** (SSH doesn't preserve CWD):
```bash
ssh node2 "env -C /path/to/project orze start"
```

**Sentinel files (affect all nodes)**:
```bash
touch results/.orze_stop_all       # stop all nodes
touch results/.orze_disabled       # persistent disable
rm results/.orze_disabled          # re-enable
```

**Known limitation**: SQLite on network filesystems (Lustre, NFS) requires `flock` support. Journal mode `delete` (default) is safer than WAL on network FS.

## GPU Scheduling

```yaml
# orze.yaml
gpu_scheduling:
  mode: exclusive            # (default) 1 job per GPU — safe, predictable
  # mode: auto              # start exclusive, upgrade to VRAM packing if jobs are small
  # mode: vram              # pack multiple jobs per GPU by VRAM usage

  # vram mode settings:
  max_vram_pct: 85
  min_free_vram_mib: 2000
  max_jobs_per_gpu: 20

  # system throttling (both modes):
  max_load_per_cpu: 2.0
  min_free_ram_gb: 16
```

**Auto mode** starts exclusive, upgrades to VRAM packing if the first job uses <10% VRAM. Caution: if early jobs fail fast (0 VRAM), auto mode may pack too aggressively.

## Roles (AI Agents — requires orze-pro)

### Configured Roles
```yaml
roles:
  professor:
    mode: claude                   # spawns Claude CLI
    model: claude-opus-4-6
    rules_file: PROFESSOR_RULES.md
    cycle_interval: 600
    timeout: 1200
  research:
    mode: research                 # built-in multi-backend LLM
    backend: anthropic
    model: claude-opus-4-6
    num_ideas: 5
    cycle_interval: 300
    rules_file: RESEARCH_RULES.md
  research_gemini:
    mode: research
    backend: gemini
    model: gemini-3.1-pro-preview
```

### Auto-Enabled Roles (when professor is active)
- **data_analyst**: dataset auditing and error analysis
- **bug_fixer**: auto-fix failed experiments
- **thinker**: creative paradigm shifts on plateau
- **bot**: interactive Telegram/Slack bot

### Role Modes
- `mode: claude` — spawns `claude -p <rules_file>` as subprocess
- `mode: research` — built-in multi-backend LLM agent (gemini/openai/anthropic/ollama)
- `mode: script` — any Python script: `script: my_agent.py`, `args: [...]`

Template vars in rules_file and args: `{ideas_file}`, `{results_dir}`, `{cycle}`, `{gpu_count}`, `{completed}`, `{queued}`, `{role_name}`.

## Key Files

| File | Purpose | Who reads it |
|------|---------|--------------|
| `GOAL.md` | Research objective, dataset, target metric | Professor, thinker, data_analyst |
| `RESEARCH_RULES.md` | Hard constraints, dead techniques, active vectors | Research agents |
| `PROFESSOR_RULES.md` | Professor behavior, web search mandate | Professor |
| `ideas.md` | Idea queue (consumed into idea_lake.db) | Orze engine |
| `orze.yaml` | Project config | Everything |
| `results/report.md` | Human-readable leaderboard | You |
| `results/status.json` | Machine-readable status | Admin UI, monitoring |
| `results/_leaderboard.json` | Full leaderboard with metrics | Research context builder |
| `results/_manual_results.json` | User-registered external results | Research context builder (merged into leaderboard) |
| `results/_retrospection.txt` | Automated trend analysis | Research agents |
| `results/idea_lake.db` | SQLite idea archive | Orze engine |

## Training Script Contract

Input (provided by orze):
```bash
CUDA_VISIBLE_DEVICES=N python train.py \
  --idea-id idea-001 --results-dir results \
  --ideas-md ideas.md --config base.yaml
```

Output (required):
```json
// results/{idea_id}/metrics.json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

## Ideas Format

```markdown
## idea-abc123: My Experiment
- **Priority**: high
- **Category**: architecture
- **Approach Family**: optimization
- **Parent**: none
- **Hypothesis**: Why this might work.

\```yaml
model:
  type: transformer
training:
  lr: 0.001
\```
```

Ideas with IDs or titles matching prompt injection patterns (`PI_DIRECTIVE`, `SYSTEM_PROMPT`, `JAILBREAK`, etc.) are automatically filtered at parse time.

## Data Boundaries (Leakage Prevention)

```yaml
data_boundaries:
  forbidden_in_training: []         # hard abort if training reads these
  watch_paths:                      # log-only audit
    - /path/to/test/features
auto_seal_eval: true                # seal eval scripts (hash integrity)
```

## Notifications

Channels: telegram, slack, discord, webhook.

Events: `completed`, `failed`, `new_best`, `heartbeat`, `milestone`, `disk_warning`, `stall`, `plateau`, `role_summary`, `shutdown`, `started`, `upgrading`, `watchdog_restart`.

```yaml
notifications:
  enabled: true
  heartbeat_interval: 1800
  on: [completed, failed, new_best]
  channels:
    - type: telegram
      bot_token: "..."
      chat_id: "..."
      on: [new_best, heartbeat, milestone, stall, plateau, role_summary]
```

## Admin API

`orze --admin` → http://localhost:8787

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/api/status` | Pipeline status |
| GET | `/api/leaderboard` | Top models ranked by primary metric |
| GET | `/api/runs` | Active + recent runs (paginated) |
| GET | `/api/run/detail?idea_id=...` | Metrics for one run |
| GET | `/api/run/log?idea_id=...` | Tail of training log |
| GET | `/api/queue` | Filterable idea queue |
| GET | `/api/nodes` | Host heartbeats + GPU details |
| GET | `/api/alerts` | Active alerts |
| POST | `/api/ideas` | Add new idea |
| POST | `/api/actions/stop` | Stop all instances |
| POST | `/api/actions/kill` | Kill specific run |

## Diagnosing Problems

| Symptom | Check | Fix |
|---------|-------|-----|
| No experiments running | `orze --check`, `cat results/status.json` | Add ideas, check GPUs, check train_script exists |
| Training stuck | `results/{idea_id}/train_output.log` | Set `stall_minutes` in orze.yaml |
| Research agent no ideas | `results/_research_logs/cycle_*.log` | Check `rules_file`, API keys, model access |
| Professor not steering | `results/_professor_logs/cycle_*.log` | Update PROFESSOR_RULES.md with critical context |
| Manual results invisible | `orze result list` | `orze result add --name X --map 0.83` |
| Disk full | `df -h` | Set `gc.enabled: true`, `min_disk_gb` |
| Ideas duplicating | Config dedup cache | Normal — dedup auto-skips |
| Multi-machine conflict | `results/.orze.pid.*` files | Check shared FS mount, use `env -C` for SSH |
| "Already running" on SSH | pgrep self-detection with `-c` flag | Use `env -C /project/dir orze start` instead |
| GPUs idle after restart | VRAM packing escalated from early failures | Restart orze to reset GPU scheduling mode |

## Architecture (source: `src/orze/`)

```
engine/
├── orchestrator.py       — Orze class, run() recipe
├── phases.py             — main loop phases (sync, launch, eval, report)
├── launcher.py           — training subprocess launch
├── evaluator.py          — eval script launch & monitoring
├── gpu_slots.py          — GPU scheduling (exclusive/auto/VRAM modes)
├── scheduler.py          — idea claiming, GPU assignment
├── retrospection.py      — signal detection + dispatch
├── experiment_analysis.py— cross-experiment regression analysis
├── reporter.py           — notifications + plateau detection
├── health.py             — disk, stall, FS health monitoring
├── failure_analysis.py   — structured failure classification
├── sealed.py             — sealed file integrity
├── cluster.py            — multi-machine coordination
└── config_dedup.py       — config hash deduplication

core/
├── config.py     — config loading, validation, sanitization
├── ideas.py      — ideas.md parsing, sweep expansion, PI filtering
└── fs.py         — atomic filesystem ops, locks

lifecycle.py      — start/stop/restart, PID management, daemon fork
cli.py            — argparse + dispatch (result, reset, pro, service)

hardware/
└── gpu.py        — GPU detection, VRAM queries

reporting/
├── leaderboard.py    — report generation
├── state.py          — state persistence, heartbeats
└── notifications.py  — telegram/slack/discord/webhook
```

Each module has a calling spec at the top — read it before reading the implementation.

## orze.yaml Quick Reference

```yaml
# Required
train_script: train.py
ideas_file: ideas.md

# Paths
base_config: configs/base.yaml
results_dir: results

# Execution
timeout: 7200
poll: 15
stall_minutes: 10
max_idea_failures: 3

# GPU scheduling
gpu_scheduling:
  mode: auto                       # exclusive | auto | vram

# Data integrity
data_boundaries:
  forbidden_in_training: []
  watch_paths: []
auto_seal_eval: true

# Report
report:
  primary_metric: map
  columns:
    - {key: "map", label: "mAP"}
    - {key: "training_time", label: "Time (s)"}

# Notifications
notifications:
  enabled: true
  heartbeat_interval: 1800
  on: [completed, failed, new_best]
  channels:
    - type: telegram
      bot_token: "..."
      chat_id: "..."

# Roles (requires orze-pro)
roles:
  professor:
    mode: claude
    model: claude-opus-4-6
    rules_file: PROFESSOR_RULES.md
    cycle_interval: 600
    timeout: 1200
  research:
    mode: research
    backend: anthropic
    num_ideas: 5
    cycle_interval: 300
    rules_file: RESEARCH_RULES.md

# Bot
bot:
  type: telegram
  bot_token: "..."
  chat_id: "..."

# GC
gc:
  enabled: true
  keep_top: 50
```
