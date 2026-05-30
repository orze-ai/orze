"""Regression tests for the config-dedup filename-mismatch bugs.

Root cause (both bugs): the code read ``resolved_config.yaml``, a file that
exists in 0 of 1605 idea dirs. The real on-disk file is ``idea_config.yaml``
(present in 1529 dirs) and it IS the canonical overrides key-set: empirically
``hash_config(idea_config.yaml)`` is byte-identical to the ingest-side hash of
``idea['config']`` (verified 400/400 ideas, 0 mismatches).

Bug #1 (integrity.rebuild_hashes): read idea_config.yaml -> cache populates.
Bug #3 (leaderboard._recover_overrides): read idea_config.yaml VERBATIM (no
base subtraction) -> completion save no longer skips.
"""
import json
from pathlib import Path

import yaml

from orze.core.integrity import hash_config, load_hashes, rebuild_hashes
from orze.reporting.leaderboard import NotificationProcessor


def _make_cfg(tmp_path: Path) -> dict:
    """Minimal cfg routing the dedup cache into tmp via orze_path.

    orze_path(cfg, "state", "config_hashes.json") resolves under _orze_dir,
    exactly like the live cfg-aware cache (mirrors test_config_dedup_wiring).
    """
    return {
        "_orze_dir": str(tmp_path / ".orze"),
        "_project_root": str(tmp_path),
        "_env_ORZE_RESULTS_DIR": str(tmp_path / "results"),
    }


def _write_idea(
    results_dir: Path,
    idea_id: str,
    overrides: dict,
    *,
    status: str = "COMPLETED",
) -> None:
    """Create an idea dir with metrics.json + idea_config.yaml and NO
    resolved_config.yaml (mirroring the real on-disk state: 0/1605 have it)."""
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(json.dumps({"status": status}))
    (idea_dir / "idea_config.yaml").write_text(yaml.safe_dump(overrides))


# --------------------------------------------------------------------------
# Bug #1 — rebuild_hashes must read idea_config.yaml (resolved never exists)
# --------------------------------------------------------------------------
def test_rebuild_hashes_reads_idea_config_yaml(tmp_path):
    cfg = _make_cfg(tmp_path)
    results_dir = Path(cfg["_env_ORZE_RESULTS_DIR"])
    idea_id = "idea-rebuild-0001"
    overrides = {"model.lr": 0.001, "data.domain": "ami", "seed": 7}
    _write_idea(results_dir, idea_id, overrides)  # NO resolved_config.yaml

    rebuild_hashes(results_dir, cfg)

    # rebuild_hashes returns None; assert against the persisted cfg-aware cache.
    cache = load_hashes(results_dir, cfg)
    expected_hash = hash_config(overrides)
    assert cache.get(expected_hash) == idea_id, (
        "rebuild did not key the idea by hash_config(idea_config.yaml); "
        f"cache={cache}"
    )


def test_rebuild_hashes_falls_back_to_resolved_config(tmp_path):
    """Back-compat: if only resolved_config.yaml exists, still index it."""
    cfg = _make_cfg(tmp_path)
    results_dir = Path(cfg["_env_ORZE_RESULTS_DIR"])
    idea_id = "idea-rebuild-legacy"
    cfg_data = {"model.lr": 0.002, "data.domain": "vp"}
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(json.dumps({"status": "COMPLETED"}))
    (idea_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg_data))
    # NO idea_config.yaml

    rebuild_hashes(results_dir, cfg)
    cache = load_hashes(results_dir, cfg)
    assert cache.get(hash_config(cfg_data)) == idea_id


def test_rebuild_hashes_skips_non_completed(tmp_path):
    """Gating preserved: non-COMPLETED ideas are not indexed."""
    cfg = _make_cfg(tmp_path)
    results_dir = Path(cfg["_env_ORZE_RESULTS_DIR"])
    overrides = {"model.lr": 0.5}
    _write_idea(results_dir, "idea-running", overrides, status="RUNNING")

    rebuild_hashes(results_dir, cfg)
    cache = load_hashes(results_dir, cfg)
    assert hash_config(overrides) not in cache


# --------------------------------------------------------------------------
# Bug #3 — _recover_overrides must return idea_config.yaml VERBATIM
# --------------------------------------------------------------------------
def _make_reporter(results_dir, cfg):
    rep = NotificationProcessor.__new__(NotificationProcessor)
    rep.results_dir = Path(results_dir)
    rep.cfg = cfg
    rep.lake = None
    return rep


def test_recover_overrides_reads_idea_config_verbatim(tmp_path):
    cfg = _make_cfg(tmp_path)
    results_dir = Path(cfg["_env_ORZE_RESULTS_DIR"])
    idea_id = "idea-recover-0001"
    overrides = {"model.lr": 0.001, "data.domain": "ami", "seed": 7}
    # A base config that SHARES a key, to prove we do NOT subtract it when
    # idea_config.yaml is present (data.domain must survive verbatim).
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"data.domain": "ami", "model.lr": 0.0}))
    _write_idea(results_dir, idea_id, overrides)  # NO resolved_config.yaml

    rep = _make_reporter(results_dir, cfg)
    recovered = rep._recover_overrides(idea_id, {"base_config": str(base_path)})

    assert recovered is not None, "_recover_overrides returned None"
    assert recovered == overrides  # verbatim, NOT base-subtracted
    assert hash_config(recovered) == hash_config(overrides)


def test_recover_overrides_falls_back_to_resolved_minus_base(tmp_path):
    """Back-compat: only resolved_config.yaml present -> subtract base."""
    cfg = _make_cfg(tmp_path)
    results_dir = Path(cfg["_env_ORZE_RESULTS_DIR"])
    idea_id = "idea-recover-legacy"
    base = {"data.domain": "ami", "model.lr": 0.0}
    overrides = {"model.lr": 0.003}
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(json.dumps({"status": "COMPLETED"}))
    resolved = dict(base)
    resolved.update(overrides)
    (idea_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved))
    # NO idea_config.yaml

    rep = _make_reporter(results_dir, cfg)
    recovered = rep._recover_overrides(idea_id, {"base_config": str(base_path)})
    assert recovered == overrides  # base keys subtracted out


def test_recover_overrides_none_when_no_config_files(tmp_path):
    cfg = _make_cfg(tmp_path)
    results_dir = Path(cfg["_env_ORZE_RESULTS_DIR"])
    idea_id = "idea-empty"
    idea_dir = results_dir / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    (idea_dir / "metrics.json").write_text(json.dumps({"status": "COMPLETED"}))
    # neither idea_config.yaml nor resolved_config.yaml

    rep = _make_reporter(results_dir, cfg)
    assert rep._recover_overrides(idea_id, {"base_config": None}) is None
