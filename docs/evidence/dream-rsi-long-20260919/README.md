# Prospective Dream-RSI comparison — completed 2026-09-19

Five synthetic problems, two paired data repetitions, one scientist model.
Equal-problem confirmation mean: control **0.5495436609**, learned Dream revision 4
**0.5463815683**; new-minus-old **−0.0031620926**, 95% interval
**[−0.0557328235, 0.0494086384]**. This does not establish either population optimum.
The higher observed mean retains `ParallelRefine` as the entry-point default.

Run from this directory with Python 3 and scipy (the study used scipy 1.15.3):

```sh
python3 rescore_public.py confirmation-evidence.json
```

This recomputes RMSE from all 600-row confirmation predictions, averages two
repetitions within each problem and then five problems equally, and checks every
reported primary value, the t interval and all 32 problem sign flips. It checks
arithmetic, not execution provenance or label isolation. `verification.json`
records the separate full raw-artifact check; the raw archive and frozen runtime
are in the private Pro repository at the same evidence path.

`learned_policy.py` is the exact tested generated policy, not the former Portfolio
bootstrap and not the current default. The study used a fresh `Policy()` for
**every decision** with the complete revealed prefix; it ran in a restricted
container. Reuse requires the caller's usual code review/execution controls.
No code is automatically imported from this evidence directory.

`operational-summary.json` retains failures, time, known usage and unknown-cost
reservations. `provider-error-sensitivity.json` is a posthoc hypothetical perfect
repair of two unique HTTP 529 affected control episodes, not observed performance.
`plot_results.py` regenerates the standalone PNG/PDF figure using matplotlib.

See [the full Chinese result report](../../plans/2026-09-19-dream-rsi-results.zh-CN.md)
for research-process evidence, limitations and the next research priorities.
