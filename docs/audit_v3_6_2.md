# orze v3.6.2 feature audit (input to v3.7.0)

Goal: identify which of F8..F16 are (a) imported by live code, (b) covered
by passing tests, (c) exercised by a real run. Features that fail all
three get quarantined or deleted.

| Feature | Module | Imported by | Tests | Live exercise (v3.7.0) | Verdict |
|---|---|---|---|---|---|
| F8 `idea.kind` migration | `orze/idea_lake.py` | launcher, phases, search_role | `test_idea_kind.py` ✅ | yes — every insert | **keep** |
| F9 ArtifactCatalog | `orze/artifact_catalog.py` | launcher, search_role, ingest_champion, posthoc_runner | `test_artifact_catalog.py` ✅ | yes — `orze catalog scan` | **keep** |
| F10 InferenceBundle + bundle_combiner | `orze/engine/bundle_combiner.py` | adapter (indirectly), tests | `test_bundle_combiner.py` ✅ | yes — honest search run | **keep (hardened: splits=)** |
| F11 Aggregation registry | `orze/engine/aggregations.py` + `agg_search.py` | bundle_combiner, adapter | `test_aggregations.py` ✅, `test_agg_search.py` ✅ | **yes** — adapter now calls `make_calibrator('cv_mix_honest')` + `per_group_rank_normalize` | **keep (was dark-code in v3.6.2, now wired)** |
| F12 posthoc_runner | `orze/engine/posthoc_runner.py` | launcher `_launch_posthoc` | `test_posthoc_runner.py` ✅ | yes — bundle_combine/agg_search ideas both dispatched | **keep** |
| F13 Cross-host fleet scheduler | `orze/engine/cluster.py` + related | `test_fleet_scheduler.py` only | `test_fleet_scheduler.py` ✅ | **no** — no multi-host run this cycle | **keep (gated off by default; opt-in via cluster: block)** |
| F14 Champion-promotion guard | `orze/engine/champion_guard.py` | orchestrator (`check_promotion`) | `test_champion_guard.py` ✅ | yes — forged `honest=false` blocked end-to-end | **keep (hardened: honest flag)** |
| F15 search_role | `orze/agents/search_role.py` | CLI `python -m orze.agents.search_role` | `test_search_role.py` ✅ | yes — idempotent re-run verified | **keep** |
| F16 ingest-champion | `orze/agents/ingest_champion.py` | `orze ingest-champion` CLI | `test_ingest_champion.py` ✅ | yes — seeded sandbox ckpt_sha | **keep** |

## v3.7.0 decisions

* **F13 (fleet scheduler)** — kept but gated off by default (cluster
  is opt-in). Unit tests exercise it; live exercise is not required for
  the Nexar task. Not deleted.
* **F11 dark code** — previously instantiated by `agg_search` tests but
  not actually wired into the live evaluation path. The adapter now
  calls the registry directly, so F11 is no longer dead-code.
* **Legacy CV-OOF-on-test** (the pre-v3.7.0 `_CVMix` in aggregations) —
  kept for backward compatibility but downgraded to reference-only; any
  metrics.json emitted through that path must set `honest: false` and
  will be refused by champion_guard.
* **`scripts/eval_tta_npz.py` import from adapter** — **removed**. The
  helper still exists inside `nexar_collision/scripts/` for the manual
  reproducer (`eval_champion_0905_final.py`), but orze no longer
  depends on it.

## Dead code removed in v3.7.0

None — every F8..F16 module is live-exercised or test-covered. What
changed is WIRING: F11 is now the backbone of the nexar adapter rather
than a parallel island.

## Open follow-ups (non-blocking)

* F13: add a smoke-test that launches two orze daemons on the same box
  and has them claim disjoint ideas via the fleet lock.
* F15 plateau detection: currently triggers after 3 cycles with no new
  best; confirm this doesn't starve a legitimate slow-improvement run.
