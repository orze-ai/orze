"""Regression test: the config-dedup cache never populated because the
completion-save and the ingest-load resolved to DIFFERENT files.

ROOT CAUSE (orze/engine/orchestrator.py::Orze._process_notifications): the
completion-side save was wired with an inline lambda

    save_config_hash_fn=lambda idea_id, config: save_hash(self.results_dir, idea_id, config)

that OMITTED ``self.cfg``. Without cfg, ``save_hash`` (orze/core/integrity.py)
falls back to the legacy ``results/_config_hashes.json``, while the ingest-side
load (``_load_config_hashes`` -> ``load_hashes(self.results_dir, self.cfg)``)
reads the cfg-aware ``<orze_dir>/state/config_hashes.json``. The two files never
meet, so the dedup cache stays empty and duplicate experiments are never skipped.

Live data at diagnosis: the legacy save target held 658 entries while the
cfg-aware load target ``.orze/state/config_hashes.json`` was ``{}`` (0 entries).

The fix routes the completion-save through the already-correct, cfg-aware
``self._save_config_hash`` method so save and load hit the same file.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orze.engine.orchestrator import Orze
from orze.core.integrity import hash_config


@pytest.fixture
def orze_obj(tmp_path):
    """A bare Orze instance with only the attributes _process_notifications,
    _save_config_hash and _load_config_hashes touch.

    cfg is a real config dict (the same shape load_project_config produces);
    orze_path(cfg, "state", "config_hashes.json") therefore resolves to
    tmp_path/.orze/state/config_hashes.json — exactly mirroring the live
    cfg-aware cache location.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cfg = {
        "_orze_dir": str(tmp_path / ".orze"),
        "_project_root": str(tmp_path),
        "_env_ORZE_RESULTS_DIR": str(results_dir),
    }
    obj = Orze.__new__(Orze)  # bypass __init__ (needs gpu_ids/heavy ctx)
    obj.results_dir = results_dir
    obj.cfg = cfg
    obj.active = {}
    obj._reporter = MagicMock()
    return obj


def test_completion_save_is_visible_to_ingest_load(orze_obj):
    """The save_config_hash_fn the orchestrator hands the reporter must write to
    the SAME cfg-aware cache file that _load_config_hashes reads.

    RED before fix: the inline lambda omits cfg -> writes the legacy
    results/_config_hashes.json -> _load_config_hashes (cfg-aware) returns {} and
    this assert fails.
    GREEN after fix: completion-save routes through _save_config_hash (cfg-aware)
    -> the saved hash is visible to the load.
    """
    # Drive the REAL wiring. _reporter is a mock so process() is a no-op, but it
    # records the exact save_config_hash_fn the orchestrator chose to hand it.
    orze_obj._process_notifications(
        finished=[], completed_rows=[], ideas={}, counts={})

    assert orze_obj._reporter.process.called, "reporter.process was not invoked"
    _, kwargs = orze_obj._reporter.process.call_args
    save_fn = kwargs["save_config_hash_fn"]

    # A completion persists its config hash through that fn.
    config = {"learning_rate": 1e-4, "epochs": 3}
    save_fn("idea-z", config)

    # The ingest-side (cfg-aware) load MUST see what the completion saved.
    loaded = orze_obj._load_config_hashes()
    assert hash_config(config) in loaded, (
        "completion-save wrote to a different file than ingest-load reads: the "
        "save path omitted cfg (-> legacy results/_config_hashes.json) while "
        "load uses cfg (-> .orze/state/config_hashes.json). Dedup never fires."
    )
    assert loaded[hash_config(config)] == "idea-z"


def test_completion_save_lands_in_cfg_aware_state_file(orze_obj):
    """Stronger invariant: the bytes land in <orze_dir>/state/config_hashes.json
    (the cfg-aware location), not the legacy results/_config_hashes.json.
    """
    orze_obj._process_notifications(
        finished=[], completed_rows=[], ideas={}, counts={})
    _, kwargs = orze_obj._reporter.process.call_args
    kwargs["save_config_hash_fn"]("idea-q", {"k": 1})

    state_file = Path(orze_obj.cfg["_orze_dir"]) / "state" / "config_hashes.json"
    legacy_file = Path(orze_obj.results_dir) / "_config_hashes.json"

    assert state_file.exists(), "cfg-aware state cache was not written"
    assert "idea-q" in state_file.read_text()
    assert not legacy_file.exists(), (
        "completion-save leaked into the legacy results/_config_hashes.json "
        "(cfg was omitted)"
    )
