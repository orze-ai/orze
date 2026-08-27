"""Launch invariants that idea data and direct callers cannot bypass."""

from pathlib import Path

import pytest
import yaml

from orze.engine.launcher import (
    LaunchIntegrityError,
    _assert_gpu_authorized,
    _launch_min_free_vram,
    find_forbidden_launch_override,
    launch,
)
from orze.core.config import _validate_config
from orze.core.research_policy import validate_idea_against_research_policy
from orze.engine.evaluator import launch_eval


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


@pytest.mark.parametrize(
    "idea_cfg,expected",
    [
        ({"ensemble_components": ["a", "b"]}, "ensemble_components"),
        ({"decode": {"ensemble_weights": [0.5, 0.5]}},
         "ensemble_weights"),
        ({"inference_model_paths": ["a", "b"]},
         "inference_model_paths"),
        ({"strategy": "logit_ensemble"}, "strategy"),
        ({"decoder_method": "rover_consensus"}, "decoder_method"),
    ],
)
def test_single_model_policy_rejects_composition_keys(idea_cfg, expected):
    error = validate_idea_against_research_policy(
        idea_cfg,
        {"research_policy": {"model_form": "single_model_single_pass"}},
    )
    assert error == f"research_policy_composite_forbidden:config.{expected}"


def test_single_model_policy_allows_training_that_emits_one_artifact():
    assert validate_idea_against_research_policy(
        {
            "lora_path": "parent/checkpoint",
            "ema_enabled": True,
            "swa_enabled": True,
            "distillation_teacher": "teacher",
        },
        {"research_policy": {"model_form": "single_model_single_pass"}},
        approach_family="architecture",
    ) is None


def test_decision_contract_requires_explicit_sweep_arms():
    cfg = {"research_policy": {
        "require_batch_decision_contract": True,
        "max_decision_batch": 3,
        "min_decision_effect": 0.1,
    }}
    assert validate_idea_against_research_policy(
        {"learning_rate": [1e-4, 2e-4]}, cfg,
        approach_family="optimization",
    ) == "batch_decision_contract_implicit_sweep_forbidden"
    # Known bundled list keys remain one experiment rather than a sweep axis.
    assert validate_idea_against_research_policy(
        {"adapters": ["one", "two"]}, cfg,
        approach_family="architecture",
    ) is None
    recursive = {}
    recursive["nested"] = recursive
    assert validate_idea_against_research_policy(
        recursive, cfg, approach_family="architecture",
    ) == "batch_decision_contract_implicit_sweep_forbidden"


def test_single_model_policy_rejects_ensemble_family():
    assert validate_idea_against_research_policy(
        {},
        {"research_policy": {"model_form": "single_model_single_pass"}},
        approach_family="ensemble",
    ) == "research_policy_approach_family_forbidden:ensemble"


def test_benchmark_contract_automatically_activates_single_model_admission():
    assert validate_idea_against_research_policy(
        {"ensemble_models": ["model-a", "model-b"]},
        {
            "report": {"benchmark_contract": {
                "model_form": "single_model_single_pass",
            }},
        },
    ) == (
        "research_policy_composite_forbidden:config.ensemble_models"
    )


def test_unrestricted_research_policy_cannot_weaken_benchmark_contract():
    assert validate_idea_against_research_policy(
        {},
        {
            "research_policy": {"model_form": "unrestricted"},
            "report": {"benchmark_contract": {
                "model_form": "single_model_single_pass",
            }},
        },
        approach_family="ensemble",
    ) == "research_policy_approach_family_forbidden:ensemble"


def test_empty_family_extension_cannot_allow_ensemble():
    assert validate_idea_against_research_policy(
        {},
        {"research_policy": {
            "model_form": "single_model_single_pass",
            "forbidden_approach_families": [],
        }},
        approach_family="ensemble",
    ) == "research_policy_approach_family_forbidden:ensemble"


def test_direct_launch_rechecks_single_model_policy_before_gpu_telemetry(
        launch_case, monkeypatch):
    results, idea_dir, cfg = launch_case
    cfg["research_policy"] = {"model_form": "single_model_single_pass"}
    (idea_dir / "idea_config.yaml").write_text(yaml.safe_dump({
        "ensemble_models": ["model-a", "model-b"],
    }), encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )

    with pytest.raises(
            LaunchIntegrityError, match="research_policy_composite_forbidden"):
        launch("idea-test", 4, results, cfg)
    assert gpu_checked == []


def test_direct_launch_requires_admitted_decision_receipt_before_gpu_telemetry(
        launch_case, monkeypatch):
    results, _, cfg = launch_case
    cfg.update({
        "report": {"primary_metric": "score", "sort": "descending"},
        "research_policy": {
            "require_batch_decision_contract": True,
            "max_decision_batch": 1,
            "min_decision_effect": 0.1,
        },
    })
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )
    with pytest.raises(
            LaunchIntegrityError,
            match="decision_contract_launch_admission_missing"):
        launch("idea-test", 4, results, cfg)
    assert gpu_checked == []


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


def test_generic_launcher_does_not_impose_a_model_rank_policy(
        launch_case, monkeypatch):
    results, idea_dir, cfg = launch_case
    (idea_dir / "idea_config.yaml").write_text(yaml.safe_dump({
        "training_proposal": True,
        "lora_rank": 4096,
    }), encoding="utf-8")
    observed = {}

    class RunningProcess:
        pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free", lambda *args: None)

    def popen(cmd, **kwargs):
        observed["cmd"] = cmd
        return RunningProcess()

    monkeypatch.setattr("orze.engine.launcher.subprocess.Popen", popen)

    tp = launch("idea-test", 4, results, cfg)

    assert observed["cmd"]
    assert not (idea_dir / "_rank_guard_rejected.txt").exists()
    tp.close_log()


def test_project_validator_is_the_rank_policy_boundary(
        launch_case, monkeypatch):
    results, idea_dir, cfg = launch_case
    (idea_dir / "idea_config.yaml").write_text(yaml.safe_dump({
        "training_proposal": True,
        "lora_rank": 64,
    }), encoding="utf-8")
    validators = results / "_validators"
    validators.mkdir()
    (validators / "project_rank_budget.yaml").write_text(yaml.safe_dump({
        "name": "project_rank_budget",
        "severity": "error",
        "rules": [{
            "field": "lora_rank",
            "operator": "lte",
            "value": 32,
        }],
    }), encoding="utf-8")
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args: gpu_checked.append(True),
    )

    with pytest.raises(RuntimeError, match="queue_revalidation_validator"):
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


def test_configured_gpu_scope_blocks_other_physical_devices():
    cfg = {"gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]}}
    _assert_gpu_authorized(4, cfg)
    _assert_gpu_authorized(7, cfg)
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope:0"):
        _assert_gpu_authorized(0, cfg)


def test_direct_launch_enforces_gpu_scope_before_telemetry(
        launch_case, monkeypatch):
    results, _, cfg = launch_case
    cfg["gpu_scheduling"] = {"allowed_gpus": [4, 5, 6, 7]}
    gpu_checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: gpu_checked.append(True),
    )
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope:0"):
        launch("idea-test", 0, results, cfg)
    assert gpu_checked == []


def test_daemon_invocation_scope_narrows_configured_allowlist():
    cfg = {
        "_managed_gpu_ids": [4, 5],
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
    }
    _assert_gpu_authorized(5, cfg)
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope:6"):
        _assert_gpu_authorized(6, cfg)


def test_configured_allowlist_cannot_be_widened_by_internal_scope():
    cfg = {
        "_managed_gpu_ids": [0, 4],
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
    }
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope:0"):
        _assert_gpu_authorized(0, cfg)


def test_empty_daemon_scope_authorizes_no_gpu():
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope:4"):
        _assert_gpu_authorized(4, {"_managed_gpu_ids": []})


@pytest.mark.parametrize("scope", [["4"], [True], [4, 4], "4,5"])
def test_runtime_gpu_scope_requires_typed_unique_ids(scope):
    with pytest.raises(LaunchIntegrityError, match="managed_gpu_scope_invalid"):
        _assert_gpu_authorized(4, {"_managed_gpu_ids": scope})


def test_reserved_gpu_cannot_be_launched_directly():
    cfg = {
        "_managed_gpu_ids": [4, 5, 6, 7],
        "gpu_scheduling": {"reserved_gpus": [6]},
    }
    with pytest.raises(LaunchIntegrityError, match="gpu_is_reserved:6"):
        _assert_gpu_authorized(6, cfg)


@pytest.mark.parametrize("gpu", [True, -1, "4", None])
def test_invalid_gpu_identity_is_rejected(gpu):
    with pytest.raises(LaunchIntegrityError, match="gpu_id_invalid"):
        _assert_gpu_authorized(gpu, {})


def test_launch_vram_threshold_uses_nested_scheduler_policy():
    assert _launch_min_free_vram({
        "gpu_scheduling": {"min_free_vram_mib": 4321},
    }) == 4321
    assert _launch_min_free_vram({
        "launcher_min_free_vram_mib": 1234,
        "gpu_scheduling": {"min_free_vram_mib": 4321},
    }) == 1234


@pytest.mark.parametrize("gpu_cfg", [
    {"allowed_gpus": [4, 4]},
    {"allowed_gpus": [4, True]},
    {"min_free_vram_mib": -1},
])
def test_config_validation_rejects_ambiguous_gpu_policy(
        tmp_path, monkeypatch, gpu_cfg):
    train = tmp_path / "train.py"
    train.write_text("# idea_config.yaml\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    errors, _ = _validate_config({
        "train_script": "train.py", "gpu_scheduling": gpu_cfg,
    })
    assert any("gpu_scheduling" in error for error in errors)


def test_redundant_reservations_outside_allowlist_are_valid_and_denied():
    cfg = {
        "gpu_scheduling": {
            "allowed_gpus": [4, 5, 6, 7],
            "reserved_gpus": [0, 1, 2, 3],
        },
    }
    errors, _ = _validate_config(cfg)
    assert not any("gpu_scheduling" in error for error in errors)
    _assert_gpu_authorized(4, cfg)
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope"):
        _assert_gpu_authorized(0, cfg)


def test_eval_masks_to_one_authorized_gpu_and_uses_local_device_zero(
        launch_case, monkeypatch):
    results, idea_dir, cfg = launch_case
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    cfg.update({
        "eval_script": "/usr/bin/true",
        "eval_args": [
            "--gpu", "{gpu}", "--physical-gpu", "{physical_gpu}"],
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
    })
    observed = {}

    class RunningProcess:
        pid = 12345

        def poll(self):
            return None

    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free", lambda *args: None)

    def popen(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["env"] = kwargs["env"]
        return RunningProcess()

    monkeypatch.setattr("orze.engine.evaluator.subprocess.Popen", popen)
    ep = launch_eval("idea-test", 4, results, cfg)
    assert observed["cmd"][-4:] == [
        "--gpu", "0", "--physical-gpu", "4"]
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    ep.close_log()


def test_eval_rejects_out_of_scope_gpu_before_telemetry(
        launch_case, monkeypatch):
    results, idea_dir, cfg = launch_case
    (idea_dir / "metrics.json").write_text(
        '{"status":"COMPLETED"}', encoding="utf-8")
    cfg.update({
        "eval_script": "/usr/bin/true",
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
    })
    checked = []
    monkeypatch.setattr(
        "orze.engine.evaluator._verify_gpu_free",
        lambda *args: checked.append(True))
    with pytest.raises(LaunchIntegrityError, match="outside_managed_scope:0"):
        launch_eval("idea-test", 0, results, cfg)
    assert checked == []
