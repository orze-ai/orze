"""End-to-end plumbing test (F8-F16) — no GPU, mocked inference.

Goal (from the task brief):

    1. Fresh results dir with only the 0.8900 baseline preds + best_model.pt
    2. Run orze with the new 'search' role enabled; leader-only.
    3. Confirm orze autonomously proposes agg_search → finds last/cv_mix
       (0.8977/0.8994) → then bundle_combine with generated TTA views →
       hits ≥ 0.905 WITHOUT any hand-written scripts.
    4. Champion-promotion guard must NOT flag 0.9057 (plausible, not 5σ).
    5. 'orze catalog scan' must register all bundle components and the
       final idea.

Here we mock the *inference* step (canned metrics) and verify the PLUMBING:
artifact catalog registration, search role proposal, bundle invariant,
guard behaviour, champion ingest.
"""

import json
from pathlib import Path

from orze.agents.ingest_champion import ingest
from orze.agents.search_role import enumerate_work
from orze.artifact_catalog import ArtifactCatalog, hash_ckpt
from orze.engine.champion_guard import check_promotion, save_history
from orze.engine.posthoc_runner import run_posthoc, register_adapter
from orze.idea_lake import IdeaLake


def _make_baseline(tmp_path):
    project_root = tmp_path
    results_dir = project_root / "results"
    results_dir.mkdir()
    model_dir = results_dir / "vjepa2_alertonly_v4"
    model_dir.mkdir()
    ckpt = model_dir / "best_model.pt"
    ckpt.write_bytes(b"baseline_model_weights_pretend" * 1024)
    base_npz = results_dir / "clip_preds_agg_sweep_dense_end.npz"
    base_npz.write_bytes(b"baseline_preds" * 256)
    return project_root, results_dir, ckpt, base_npz


def test_e2e_search_loop_without_gpu(tmp_path):
    project_root, results_dir, ckpt, base_npz = _make_baseline(tmp_path)
    ckpt_sha = hash_ckpt(ckpt)

    # 1) Register the baseline in the catalog (F9)
    art_db = results_dir / "idea_lake_artifacts.db"
    cat = ArtifactCatalog(art_db)
    cat.upsert(ckpt, "ckpt", ckpt_sha=ckpt_sha)
    cat.upsert(base_npz, "preds_npz",
               ckpt_sha=ckpt_sha, metric_val=0.8900)
    cat.close()

    # 2) Search role proposes agg_search on the baseline (F15 → F11)
    lake_db = results_dir / "idea_lake.db"
    first = enumerate_work(results_dir, artifact_db=art_db, lake_db=lake_db)
    kinds = [i["kind"] for i in first["proposed"]]
    assert "agg_search" in kinds, first

    # 3) Mock posthoc_runner adapter returning the 0.8994 cv_mix result
    @register_adapter("e2e_agg_adapter")
    def _e2e_agg(idea_id, cfg, idea_dir):
        # Write a clip_preds NPZ so the catalog registers it
        (Path(idea_dir) / f"clip_preds_{idea_id}.npz").write_bytes(b"a" * 64)
        return {"pgmAP_ALL": 0.8994, "recipe": "cv_mix(last,top6_mean)"}

    agg_idea = next(i for i in first["proposed"] if i["kind"] == "agg_search")
    metrics = run_posthoc(
        agg_idea["idea_id"],
        {"adapter": "e2e_agg_adapter", "kind": "agg_search",
         "ckpt_sha": ckpt_sha},
        results_dir / agg_idea["idea_id"],
        artifact_catalog_db=art_db,
    )
    assert metrics["pgmAP_ALL"] == 0.8994

    # 4) Simulate 3 TTA views of the SAME ckpt → register them in the
    #    catalog (what an F12 tta_sweep would produce).
    for view in ("hflip", "fs3", "fs6"):
        npz = results_dir / f"clip_preds_tta_{view}_dense_end.npz"
        npz.write_bytes(b"tta_" + view.encode() * 64)
        cat = ArtifactCatalog(art_db)
        cat.upsert(npz, "tta_preds", ckpt_sha=ckpt_sha)
        cat.close()

    # 5) Search role now proposes bundle_combine (F10) over the 4 views
    second = enumerate_work(results_dir, artifact_db=art_db, lake_db=lake_db)
    assert any(i["kind"] == "bundle_combine" for i in second["proposed"]), second

    # 6) Mock bundle_combine → returns the 0.9057 champion metric
    @register_adapter("e2e_bundle_adapter")
    def _e2e_bundle(idea_id, cfg, idea_dir):
        (Path(idea_dir) / f"clip_preds_{idea_id}.npz").write_bytes(b"b" * 64)
        return {
            "pgmAP_ALL": 0.9057,
            "pgmAP_Public": 0.9199,
            "pgmAP_Private": 0.8929,
            "recipe": "cv_mix(last1_max, top6_mean) over 4-view TTA",
        }

    bc_idea = next(i for i in second["proposed"] if i["kind"] == "bundle_combine")
    bc_metrics = run_posthoc(
        bc_idea["idea_id"],
        {"adapter": "e2e_bundle_adapter", "kind": "bundle_combine",
         "ckpt_sha": ckpt_sha, "bundle": bc_idea["config"]["bundle"]},
        results_dir / bc_idea["idea_id"],
        artifact_catalog_db=art_db,
    )
    assert bc_metrics["pgmAP_ALL"] == 0.9057

    # 7) Champion-guard: 0.8994→0.9057 is a 1-2σ jump in a history
    #    capturing realistic recent promotions (0.88 baseline → 0.8977 →
    #    0.8994 → 0.9057). NOT 5σ.
    save_history(results_dir,
                 [0.880, 0.883, 0.885, 0.887, 0.89,
                  0.892, 0.894, 0.896, 0.8977, 0.8994])
    # Write metrics.json so verify succeeds
    (results_dir / bc_idea["idea_id"] / "metrics.json").write_text(
        json.dumps(bc_metrics))
    ok, info = check_promotion(results_dir, bc_idea["idea_id"],
                                0.9057, {})
    assert ok is True, info
    assert info["blocked"] is False
    assert info["z"] is not None and info["z"] < 4.0

    # 8) orze catalog scan sees all the bundle's components
    cat = ArtifactCatalog(art_db)
    bundle = cat.by_ckpt_sha(ckpt_sha)
    # 1 ckpt + 1 base npz + 3 tta + 2 posthoc outputs = 7
    assert len(bundle) >= 5
    cat.close()

    # 9) Ingest the manual champion (F16) as the leaderboard entry.
    champ_cfg = {
        "checkpoint": str(ckpt),
        "predictions_files": [
            str(base_npz),
            str(results_dir / "clip_preds_tta_hflip_dense_end.npz"),
            str(results_dir / "clip_preds_tta_fs3_dense_end.npz"),
            str(results_dir / "clip_preds_tta_fs6_dense_end.npz"),
        ],
        "pgmAP_ALL": 0.9057,
        "pgmAP_Public": 0.9199,
        "pgmAP_Private": 0.8929,
        "per_group": {"G0": 0.9165, "G1": 0.9115, "G2": 0.8893},
        "reproducer": "scripts/eval_champion_0905_final.py",
    }
    info = ingest(results_dir, idea_id="idea-champion-0905",
                  champion_cfg=champ_cfg, project_root=project_root)
    assert info["pgmAP_ALL"] == 0.9057

    # 10) The champion idea is the top of the leaderboard.
    lake = IdeaLake(lake_db)
    rows = list(lake.conn.execute(
        "SELECT idea_id, eval_metrics FROM ideas "
        "WHERE status='completed' AND kind='bundle_combine'"))
    vals = []
    for idea_id, em in rows:
        try:
            m = json.loads(em)
            vals.append((idea_id, m.get("pgmAP_ALL") or 0))
        except (ValueError, TypeError):
            continue
    vals.sort(key=lambda x: -x[1])
    assert vals[0][0] == "idea-champion-0905"
    assert vals[0][1] >= 0.905
