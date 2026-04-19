# Post-hoc adapters

Orze runs experiments by calling user-supplied training scripts; the
training step itself is generic. *Post-hoc adapters* are the thin glue
that turns a finished experiment's raw artifacts (NPZ predictions,
checkpoints, logs) into a metrics dict the leaderboard can rank.

Each consumer project (e.g. `nexar_collision`, `your_classifier`)
ships its own adapter. Orze itself stays generic.

## Adapter contract

An adapter is a single callable registered by name:

```python
from orze.engine.posthoc_runner import register_adapter

@register_adapter("my_adapter")
def run(idea_id: str, cfg: dict, idea_dir: Path) -> dict:
    """Return a metrics dict (will be merged into metrics.json)."""
    ...
```

The runner discovers built-in adapters automatically: any module under
`orze.engine.posthoc_adapters.<name>` is imported on first use, and any
`@register_adapter(...)` decorators fire as a side effect. To ship a
new adapter, drop a file in that package or import yours in your
project's startup hook before the orze daemon launches.

### Three logical responsibilities

A typical adapter has three responsibilities. Implement them as helper
functions inside the adapter module — they are not part of the public
orze API, but the convention keeps adapters readable:

| Helper           | Purpose                                                         |
| ---------------- | --------------------------------------------------------------- |
| `load_npz`       | Load per-frame / per-example predictions from `.npz` artifacts. |
| `tune_posthoc`   | Choose post-hoc hyperparameters (α, threshold, ensemble weights) on the **public/tune** split only. |
| `honest_report`  | Re-evaluate the chosen settings on **public**, **private**, and **all** splits and return the metric dict. |

The `nexar_collision` adapter is a worked example of all three.
The `example_classifier` adapter is a 60-line minimal version that
loads `logits/labels/split` from a single NPZ.

## Config keys

Adapters read everything they need from the orze config dict. Two
top-level keys are conventional; everything else is per-adapter:

```yaml
posthoc_defaults:
  project_root: /abs/path/to/your/project        # where artifacts live
  solution_csv: /abs/path/to/solution.csv        # public/private split labels
```

Per-adapter sections live alongside, namespaced by the adapter name:

```yaml
my_adapter:
  alpha_grid: [0.1, 0.2, 0.3]
  metric: pgm_ap
```

## Honest evaluation contract

Adapters **must** keep the public-tune / private split clean:

1. All hyperparameter selection happens on the public-tune split only.
2. The private split is touched **once**, at report time, with the
   already-selected hyperparameters.
3. The metrics dict reports both splits, so the leaderboard can
   surface tuning bias.

The runner does not enforce this — it trusts each adapter. The honest-
eval test (`test_posthoc_honest_eval.py`) verifies this for the
shipped adapters; mirror its structure if you add one.

## Registration

`@register_adapter(name)` is the only registration mechanism. Names
must be unique (the runner raises `KeyError` on collision). The cfg
key `posthoc.adapter` selects which adapter the runner invokes for a
given idea.

To list registered adapters at runtime:

```python
from orze.engine.posthoc_runner import list_adapters
print(list_adapters())
```

## Subprocess adapters

If your adapter needs to shell out to a project-specific Python
environment (e.g. a separate virtualenv with heavy ML deps), use the
`subprocess_adapter(...)` helper. It marshals the cfg dict via JSON
on stdin and parses a metrics dict from stdout. See
`posthoc_runner.subprocess_adapter` for the contract.
