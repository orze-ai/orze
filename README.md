# orze

A GPU experiment orchestrator for ML research.

Orze runs experiments on GPUs: **schedule ideas → train → evaluate → report → repeat**. It coordinates GPUs via filesystem locks, works across machines, and gives you a complete leaderboard, notifications, and analysis — out of the box.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
pip install orze                   # complete experiment tool (free, open source)
pip install orze-pro               # + autopilot: autonomous research agents
```

## orze vs orze-pro

orze is a **complete, production-ready tool**. orze-pro adds **autopilot** — so experiments run while you sleep.

| Feature | orze (free) | + orze-pro |
|---------|:-----------:|:----------:|
| GPU scheduling & multi-node | ✓ | ✓ |
| Idea queue (ideas.md + SQLite) | ✓ | ✓ |
| Hyperparameter sweep (Cartesian product) | ✓ | ✓ |
| Leaderboard report | ✓ | ✓ |
| Telegram/Slack notifications (rich) | ✓ | ✓ |
| Admin dashboard & MCP server | ✓ | ✓ |
| Retrospection (plateau detection) | ✓ | ✓ |
| Cross-experiment regression analysis | ✓ | ✓ |
| Failure analysis & categorization | ✓ | ✓ |
| Checkpoint GC | ✓ | ✓ |
| Sealed eval protection | ✓ | ✓ |
| Service watchdog (auto-restart) | ✓ | ✓ |
| **Autonomous research agents** (Gemini/GPT/Claude) | | ✓ |
| **Auto-fix failed experiments** | | ✓ |
| **Code evolution on plateau** | | ✓ |
| **Meta-research (strategy adjustment)** | | ✓ |
| **Interactive Telegram/Slack bot** | | ✓ |

### Compatibility

| orze | orze-pro | Notes |
|------|----------|-------|
| 3.0.x | 0.1.x | Current release |
| 2.x | — | All features built-in (no pro needed) |

## User Journeys

### Free user — "I drive, orze manages"

```bash
orze init                          # creates orze.yaml, ideas.md, train.py
vim ideas.md                       # add your experiment ideas
orze -c orze.yaml                  # orze schedules them on GPUs
# → check leaderboard in report.md
# → get Telegram alerts on new bests
# → see regression analysis in _experiment_insights.txt
# → add more ideas to ideas.md based on what you learn
```

You analyze results. You decide what to try next. Orze handles the infrastructure.

### Pro user — "I sleep, orze researches"

```bash
pip install orze-pro               # one command, zero config change
orze -c orze.yaml                  # same command — pro features activate automatically
# → research agent reads results and proposes new ideas
# → failed experiments get auto-fixed and retried
# → when stuck on a plateau, code evolution kicks in
# → you wake up to a better model
```

Same `orze.yaml`. Same workflow. Pro just adds autonomy.

### The upgrade moment

You're using orze free. You see this in `_experiment_insights.txt`:

```
REGRESSIONS vs baseline:
  SPG: 2.91% → 3.57% (+0.66%)
  [pattern: All LoRA variants regress. Likely domain mismatch in training data.]

SUGGESTED ACTIONS:
  → PRIORITY: Fix SPG regression (+0.66%). Consider: adding SPG training data.
  → TRADEOFF detected — try per-dataset inference strategies.
```

You think: *"I wish orze would just propose the fix and run it."*

That's when you `pip install orze-pro`.

## Quick Start

**If you are in Claude/Gemini/Codex CLI:**
```bash
do @ORZE-AGENT.md
```

**If not:**
```bash
orze
```

That's it. Orze auto-detects GPUs and starts running experiments from `ideas.md`.

## Multi-node

Start orze in the same shared folder (e.g. `/nfs/project-52h/`) on any machine — the node automatically joins the research pool. **Orze can auto-update across nodes.**

## Key Features

- **Scales to 1M+ Experiments** — SQLite-backed job queue with O(log N) scheduling
- **Config Inheritance** — Child ideas inherit parent configs; specify only what changes
- **HP Sweep** — `lr: [1e-4, 3e-4]` auto-expands into Cartesian product sub-runs
- **Circuit Breaker** — Stops on failure spikes. Schema validation catches errors before they hit GPUs
- **Cross-Experiment Analysis** — Detects regressions, tradeoffs, and suggests actions
- **Rich Notifications** — GPU VRAM, per-dataset breakdown, verified results, target/gap tracking
- **Admin Panel** — Real-time web dashboard at `http://localhost:8787`
- **Clean Uninstall** — `orze --uninstall` removes runtime files, preserves results

## How It Works

Orze runs a continuous loop: pick an idea from the queue, train it on a free GPU, evaluate, record metrics. When ideas run out, orze generates variations of your best configs automatically — **the research never stops**, even without pro.

With **orze-pro**, LLM agents replace parameter variations with intelligent, hypothesis-driven ideas.

## Admin Panel

Auto-launches at **http://localhost:8787**. No extra install needed.

<img width="900" height="674" alt="admin-panel" src="https://github.com/user-attachments/assets/b23879e3-d064-4e02-8251-6e8dbfad21f9" />
<img width="900" height="674" alt="admin-queue" src="https://github.com/user-attachments/assets/39747da2-7b7f-4a9f-ad4a-7cfaca41407b" />
<img width="900" height="551" alt="admin-leaderboard" src="https://github.com/user-attachments/assets/70e77941-efbf-4018-9200-93ea77998c5e" />

## Telegram Notifications

Rich notifications with GPU VRAM, per-dataset breakdown, and target tracking:

```
📊 Orze Status — a100-41
✅ 20 completed | ❌ 0 failed | ⏳ 6 queued | 🔄 4 running
🎯 Verified: 5.43% avg WER | Target: 5.40% | Gap: +0.03%
  AMI=9.8 | E22=9.0 | GS=8.5 | LS-C=1.3 | SPG=3.6 | TED=2.7 | VP=6.1
🖥 GPU0:idle GPU1:18G/80G(51%) GPU3:17G/80G(48%)
🤖 Model: higgs-audio-v3-8b
⏱ Up 2h15m
```

Setup:
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

## Service Management (Watchdog)

```bash
orze service install -c orze.yaml    # auto-restart on crash
orze service status                  # check health
orze service uninstall               # remove
```

## The Contract

Your training script receives:
```bash
python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config base.yaml
```

**Required output:** `results/{idea_id}/metrics.json`:
```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

Orze writes `idea_config.yaml` to the results directory before launching, containing the merged base + idea config.

See [**SKILL.md**](SKILL.md) for the full technical specification.

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

[orze-pro](https://github.com/warlockee/orze-pro) (autopilot features) is commercially licensed by Boson AI.
