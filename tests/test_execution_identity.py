import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orze.engine.accounting import record_compute_terminal
from orze.engine.execution_identity import (
    DuplicateExecutionError,
    compute_execution_identity,
    release_execution_identity,
    reserve_execution_identity,
)
from orze.engine.launcher import GpuUnavailableError, LaunchIntegrityError, launch


def _inputs(tmp_path: Path, seed: int = 1):
    config = tmp_path / f"idea-{seed}.yaml"
    config.write_text(f"seed: {seed}\nlr: 0.001\n", encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text("model: example\n", encoding="utf-8")
    train = tmp_path / "train.py"
    train.write_text("# deterministic trainer\n", encoding="utf-8")
    return config, base, train


def _identity(config, base, train, *, env=None):
    return compute_execution_identity(
        config_path=config,
        base_config_path=base,
        train_script=train,
        python="python3",
        train_extra_args=["--epochs", "2"],
        train_extra_env=env or {},
        data_boundaries={},
    )


def test_identity_is_full_sha256_and_seed_sensitive(tmp_path):
    first = _inputs(tmp_path, seed=1)
    second_config = tmp_path / "idea-2.yaml"
    second_config.write_text("seed: 2\nlr: 0.001\n", encoding="utf-8")
    first_id = _identity(*first)
    second_id = _identity(second_config, first[1], first[2])

    assert len(first_id) == 64
    assert first_id != second_id


def test_identity_ignores_replica_labels_but_not_training_inputs(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model: example\n", encoding="utf-8")
    train = tmp_path / "train.py"
    train.write_text("# deterministic trainer\n", encoding="utf-8")
    first = tmp_path / "first.yaml"
    first.write_text(
        "seed: 7\nlr: 0.001\nreplication_role: fixed_recipe_reproduction\n"
        "replication_index: 1\n_replicate_of: idea-root\n"
        "title: first label\n",
        encoding="utf-8",
    )
    relabeled = tmp_path / "relabeled.yaml"
    relabeled.write_text(
        "seed: 7\nlr: 0.001\nreplication_role: rerun\n"
        "replication_index: 99\n_replicate_of: different-root\n"
        "title: changed label\nhypothesis: changed prose\n",
        encoding="utf-8",
    )
    changed_seed = tmp_path / "changed-seed.yaml"
    changed_seed.write_text(
        "seed: 8\nlr: 0.001\nreplication_index: 99\n",
        encoding="utf-8",
    )

    first_id = _identity(first, base, train)
    assert _identity(relabeled, base, train) == first_id
    assert _identity(changed_seed, base, train) != first_id


def test_replica_metadata_cannot_bypass_concurrent_reservation(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model: example\n", encoding="utf-8")
    train = tmp_path / "train.py"
    train.write_text("# deterministic trainer\n", encoding="utf-8")
    configs = []
    for index in (1, 2):
        path = tmp_path / f"replica-{index}.yaml"
        path.write_text(
            "seed: 7\nlr: 0.001\n"
            f"replication_index: {index}\n_replicate_of: idea-root\n",
            encoding="utf-8",
        )
        configs.append(path)
    identities = [_identity(path, base, train) for path in configs]
    assert identities[0] == identities[1]

    results = tmp_path / "results"
    results.mkdir()
    barrier = threading.Barrier(2)

    def admit(index):
        barrier.wait()
        try:
            reserve_execution_identity(
                results, {}, identities[index],
                f"idea-{index}", f"attempt-{index}",
            )
            return "admitted"
        except DuplicateExecutionError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(admit, (0, 1)))
    assert sorted(outcomes) == ["admitted", "rejected"]


def test_registry_never_serializes_environment_values(tmp_path):
    config, base, train = _inputs(tmp_path)
    secret = "credential-that-must-not-be-written"
    identity = _identity(config, base, train, env={"API_TOKEN": secret})
    results = tmp_path / "results"
    results.mkdir()

    reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")

    registry_bytes = b"".join(
        path.read_bytes()
        for path in (results / "_execution_identities").glob("*.json")
    )
    assert secret.encode() not in registry_bytes
    assert b"API_TOKEN" not in registry_bytes


def test_active_exact_replica_is_rejected(tmp_path):
    config, base, train = _inputs(tmp_path)
    identity = _identity(config, base, train)
    results = tmp_path / "results"
    results.mkdir()

    reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")
    with pytest.raises(
            DuplicateExecutionError, match="exact_execution_already_reserved:idea-a"):
        reserve_execution_identity(
            results, {}, identity, "idea-b", "attempt-b")


def test_concurrent_exact_replica_has_one_winner(tmp_path):
    config, base, train = _inputs(tmp_path)
    identity = _identity(config, base, train)
    results = tmp_path / "results"
    results.mkdir()
    barrier = threading.Barrier(2)

    def admit(index):
        barrier.wait()
        try:
            reserve_execution_identity(
                results, {}, identity, f"idea-{index}", f"attempt-{index}")
            return "admitted"
        except DuplicateExecutionError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(admit, (1, 2)))
    assert sorted(outcomes) == ["admitted", "rejected"]


def test_corrupt_or_redirected_registry_fails_closed(tmp_path):
    config, base, train = _inputs(tmp_path)
    identity = _identity(config, base, train)
    results = tmp_path / "results"
    registry = results / "_execution_identities"
    registry.mkdir(parents=True)
    (registry / f"{identity}.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(DuplicateExecutionError, match="owner_invalid"):
        reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")

    (registry / f"{identity}.json").unlink()
    registry.rmdir()
    target = tmp_path / "redirected"
    target.mkdir()
    registry.symlink_to(target, target_is_directory=True)
    with pytest.raises(DuplicateExecutionError, match="registry_symlink"):
        reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")


def test_release_cannot_remove_another_attempt(tmp_path):
    config, base, train = _inputs(tmp_path)
    identity = _identity(config, base, train)
    results = tmp_path / "results"
    results.mkdir()
    reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")

    release_execution_identity(
        results, {}, identity, "idea-a", "different-attempt")
    with pytest.raises(DuplicateExecutionError, match="already_reserved"):
        reserve_execution_identity(results, {}, identity, "idea-b", "attempt-b")


class _TerminalProcess:
    idea_id = "idea-a"
    gpu = 4
    attempt_id = "attempt-a"
    start_time = 0.0

    class process:
        pid = 123


@pytest.mark.parametrize("outcome", ["failed", "interrupted", "requeued"])
def test_noncompleted_terminal_attempt_can_be_replaced(tmp_path, outcome):
    config, base, train = _inputs(tmp_path)
    identity = _identity(config, base, train)
    results = tmp_path / "results"
    (results / "idea-a").mkdir(parents=True)
    reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")
    record_compute_terminal(
        _TerminalProcess(), results / "idea-a", outcome, "test_terminal",
        return_code=1,
    )

    reserve_execution_identity(results, {}, identity, "idea-b", "attempt-b")
    owner = json.loads(next(
        (results / "_execution_identities").glob("*.json")
    ).read_text(encoding="utf-8"))
    assert owner["idea_id"] == "idea-b"


def test_completed_exact_replica_remains_blocked(tmp_path):
    config, base, train = _inputs(tmp_path)
    identity = _identity(config, base, train)
    results = tmp_path / "results"
    (results / "idea-a").mkdir(parents=True)
    reserve_execution_identity(results, {}, identity, "idea-a", "attempt-a")
    record_compute_terminal(
        _TerminalProcess(), results / "idea-a", "completed", "test_terminal",
        return_code=0,
    )

    with pytest.raises(
            DuplicateExecutionError, match="exact_execution_already_completed:idea-a"):
        reserve_execution_identity(
            results, {}, identity, "idea-b", "attempt-b")


def test_launcher_rejects_replica_before_gpu_telemetry(tmp_path, monkeypatch):
    config, base, train = _inputs(tmp_path)
    results = tmp_path / "results"
    for idea_id in ("idea-a", "idea-b"):
        idea_dir = results / idea_id
        idea_dir.mkdir(parents=True)
        (idea_dir / "idea_config.yaml").write_text(
            config.read_text(encoding="utf-8"), encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    cfg = {
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
        "train_extra_args": ["--epochs", "2"],
    }
    identity = _identity(config, base, train)
    reserve_execution_identity(results, cfg, identity, "idea-a", "attempt-a")
    checked = []
    monkeypatch.setattr(
        "orze.engine.launcher._verify_gpu_free",
        lambda *args, **kwargs: checked.append(True),
    )

    with pytest.raises(
            LaunchIntegrityError, match="exact_execution_already_reserved:idea-a"):
        launch("idea-b", 4, results, cfg)
    assert checked == []


@pytest.mark.parametrize("failure_point", ["gpu", "popen"])
def test_preallocation_failure_releases_identity(
        tmp_path, monkeypatch, failure_point):
    monkeypatch.setattr(
        "orze.core.gpu_lease.assert_gpu_scope_idle", lambda ids: None,
    )
    config, base, train = _inputs(tmp_path)
    results = tmp_path / "results"
    idea_dir = results / "idea-a"
    idea_dir.mkdir(parents=True)
    (idea_dir / "idea_config.yaml").write_text(
        config.read_text(encoding="utf-8"), encoding="utf-8")
    ideas = tmp_path / "ideas.md"
    ideas.write_text("", encoding="utf-8")
    cfg = {
        "train_script": str(train),
        "base_config": str(base),
        "ideas_file": str(ideas),
        "python": "python3",
    }
    if failure_point == "gpu":
        monkeypatch.setattr(
            "orze.engine.launcher._verify_gpu_free",
            lambda *args: (_ for _ in ()).throw(GpuUnavailableError("busy")),
        )
        expected = GpuUnavailableError
    else:
        monkeypatch.setattr(
            "orze.engine.launcher._verify_gpu_free", lambda *args: None)
        monkeypatch.setattr(
            "orze.engine.launcher.subprocess.Popen",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("popen failed")),
        )
        expected = OSError

    with pytest.raises(expected):
        launch("idea-a", 4, results, cfg)
    assert list((results / "_execution_identities").glob("*.json")) == []
