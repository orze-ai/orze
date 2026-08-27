"""Launch invariants that idea data and direct callers cannot bypass."""

from pathlib import Path

import pytest
import yaml

from orze.engine.launcher import (
    LaunchIntegrityError,
    find_forbidden_launch_override,
    launch,
)


@pytest.fixture
def launch_case(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-test"
    idea_dir.mkdir(parents=True)
    train = tmp_path / "train.py"
    train.write_text("# trainer\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("{}\n", encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    cfg = {
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    }
    return results, idea_dir, cfg


@pytest.mark.parametrize(
    "sentinel", [".orze_disabled", ".orze_stop_all", ".orze_shutdown"])
def test_direct_launch_honors_every_stop_sentinel_before_gpu_check(
        launch_case, sentinel, monkeypatch):
    results, _, cfg = launch_case
    (results / sentinel).write_text("operator stop\n", encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )

    with pytest.raises(
            LaunchIntegrityError, match="launch_blocked_by_sentinel"):
        launch("idea-test", 4, results, cfg)
    assert gpu_checked == []


@pytest.mark.parametrize("pause_kind", ["config", "sentinel"])
def test_direct_launch_honors_pause_policy_before_gpu_check(
        launch_case, pause_kind, monkeypatch):
    results, _, cfg = launch_case
    if pause_kind == "config":
        cfg["launcher"] = {"paused": True}
    else:
        (results / "_launcher_paused.flag").write_text(
            "paused\n", encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )

    with pytest.raises(LaunchIntegrityError, match="pause_policy"):
        launch("idea-test", 4, results, cfg)
    assert gpu_checked == []


@pytest.mark.parametrize("value", [True, False, "yes", None])
def test_force_launch_key_is_forbidden_even_when_false(
        launch_case, value, monkeypatch):
    results, idea_dir, cfg = launch_case
    (idea_dir / "idea_config.yaml").write_text(yaml.safe_dump({
        "training": {"controls": [{"force_launch": value}]},
    }), encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )

    with pytest.raises(
            LaunchIntegrityError,
            match=r"forbidden_launch_override:config\.training\.controls\[0\]\.force_launch"):
        launch("idea-test", 4, results, cfg)
    assert gpu_checked == []


def test_forbidden_override_search_does_not_match_values():
    assert find_forbidden_launch_override({
        "note": "force_launch", "nested": [{"safe": True}],
    }) is None


def test_recursive_config_is_rejected_without_recursing_forever():
    value = []
    value.append(value)
    assert "recursive_reference" in find_forbidden_launch_override(value)


@pytest.mark.parametrize("idea_id", ["../escape", "a/b", ".", ""])
def test_direct_launch_rejects_unsafe_idea_ids_before_writing(
        launch_case, idea_id):
    results, _, cfg = launch_case
    with pytest.raises(LaunchIntegrityError, match="idea_id_invalid"):
        launch(idea_id, 4, results, cfg)
    assert not (results.parent / "escape").exists()


def test_malformed_idea_config_fails_closed_before_gpu_check(
        launch_case, monkeypatch):
    results, idea_dir, cfg = launch_case
    (idea_dir / "idea_config.yaml").write_text("broken: [\n", encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )

    with pytest.raises(
            LaunchIntegrityError, match="idea_config_validation_failed"):
        launch("idea-test", 4, results, cfg)
    assert gpu_checked == []


def test_broken_symlink_stop_latch_still_blocks_launch(launch_case):
    results, _, cfg = launch_case
    (results / ".orze_disabled").symlink_to(results / "missing-target")
    with pytest.raises(LaunchIntegrityError, match="orze_disabled"):
        launch("idea-test", 4, results, cfg)


def test_malformed_launcher_policy_pauses_fail_closed(launch_case):
    results, _, cfg = launch_case
    cfg["launcher"] = "not-a-mapping"
    with pytest.raises(LaunchIntegrityError, match="pause_policy"):
        launch("idea-test", 4, results, cfg)
