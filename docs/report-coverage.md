# Explicit report coverage

Orze does not infer evidence validity from metric names. `report.columns`
declares exact value sources; `report.dataset_keys` selects which of those
measurements count toward `report.min_datasets`.

```yaml
report:
  primary_metric: objective
  sort: ascending
  min_datasets: 2
  dataset_keys: [part_a, part_b]
  columns:
    - {key: objective}
    - {key: part_a, source: evaluation.json:measurements.a}
    - {key: part_b, source: evaluation.json:measurements.b}
    - {key: elapsed_seconds}
```

Only the two declared parts count. Zero and negative finite numbers count;
booleans, missing values, NaN and infinity do not. A missing/null declared
source cannot be filled by a same-name proxy in `metrics.json`. The primary
metric, aggregate and elapsed time count only if explicitly selected. Display
fallbacks and `metric_harvest.columns` cannot change these evidence sources.

The declaration must be a list of unique, nonblank strings, at most 256 members
and 1024 UTF-8 bytes per key. Each member must resolve to exactly one report
column. An explicit empty list is legal, but cannot satisfy a positive minimum.
Invalid declarations return `dataset_coverage_declaration_invalid`.

If `dataset_keys` is absent, an existing benchmark contract's explicit
`required_metrics` is the fallback. If both are present, `dataset_keys` must
contain all required benchmark metrics; the benchmark's other validation rules
remain in force. Explicit null or an invalid list never invokes the fallback.

## Migrating existing configurations

Configurations without a coverage gate (`min_datasets` omitted or zero) still
work without this new field. Positive minima without either declaration now
fail closed with `dataset_coverage_not_declared`. Add the exact intended dataset
members and their source columns; keep the original minimum. Do not copy every
display column merely to satisfy the count. No user configuration is rewritten
automatically, and no WER-specific compatibility qualification path is retained.

Changing coverage invalidates report-cache and champion-history policy identity.
Legacy archive value inspection keeps its historical behavior, but cannot grant
current evidence, ranking or execution authority. These checks establish local
coverage, not independent scientific validity or an official leaderboard rank.
