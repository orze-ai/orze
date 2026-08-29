import ast
import os
from pathlib import Path

import pytest
import yaml

import orze.engine.gpu_scope_audit as scope_module
from orze.engine.gpu_scope_audit import (
    _cuda_writes,
    audit_gpu_scope,
    audit_source_boundaries,
)


def _config(tmp_path, *, allowed=None, reserved=None):
    path = tmp_path / "orze.yaml"
    path.write_text(yaml.safe_dump({
        "gpu_scheduling": {
            "allowed_gpus": allowed if allowed is not None else [4, 5, 6, 7],
            "reserved_gpus": reserved if reserved is not None else [0, 1, 2, 3],
        },
    }), encoding="utf-8")
    return path


def _copy_audited_sources(package):
    relative_paths = set(scope_module._BOUNDARIES)
    relative_paths.update(
        relative for relative, _function, _kind
        in scope_module._EXPECTED_CUDA_WRITES
    )
    for relative in relative_paths:
        source = Path(scope_module.__file__).resolve().parents[1] / relative
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def test_gpu_scope_audit_fixture_uses_process_scoped_synthetic_lease():
    mapped_gpu = 800_000 + (os.getpid() % 10_000) * 100 + 4
    with scope_module.gpu_execution_lease(4) as lease_fds:
        assert len(lease_fds) == 1
        target = os.readlink(f"/proc/self/fd/{lease_fds[0]}")
    assert target.endswith(f"/gpu-{mapped_gpu}.lock")
    assert not target.endswith("/gpu-4.lock")


def test_gpu_scope_audit_verifies_production_sources_and_no_cuda_children(
        tmp_path):
    receipt = audit_gpu_scope(
        _config(tmp_path),
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "VERIFIED"
    assert receipt["accelerator_access"] == "none"
    assert receipt["counts"] == {
        "production_boundaries": 4,
        "authorized_child_launches": 4,
        "forbidden_ids_rejected": 4,
    }
    assert receipt["forbidden_ids_rejected"] == [0, 1, 2, 3]
    assert receipt["rank_claim_proven"] is False


def test_gpu_scope_audit_rejects_weaker_configured_allowlist(tmp_path):
    receipt = audit_gpu_scope(
        _config(tmp_path, allowed=[4, 5, 6]),
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "gpu_scope_allowlist_mismatch"


def test_gpu_scope_audit_requires_every_forbidden_id_reserved(tmp_path):
    receipt = audit_gpu_scope(
        _config(tmp_path, reserved=[0, 1, 2]),
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "gpu_scope_forbidden_not_reserved"


def test_cuda_writer_scan_detects_new_dynamic_bypass():
    tree = ast.parse(
        "def bypass(env, gpu):\n"
        "    env['CUDA_VISIBLE_DEVICES'] = str(gpu)\n"
    )
    assert _cuda_writes(tree, "engine/rogue.py") == {
        ("engine/rogue.py", "bypass", "dynamic")
    }


def test_cuda_writer_single_pass_preserves_nested_attribution_semantics():
    tree = ast.parse(
        "def outer(env, gpu):\n"
        "    @env.update({'CUDA_VISIBLE_DEVICES': ''})\n"
        "    async def inner():\n"
        "        env['CUDA_VISIBLE_DEVICES']: str = str(gpu)\n"
        "        env.update(CUDA_VISIBLE_DEVICES='')\n"
        "    return inner\n"
    )

    assert _cuda_writes(tree, "engine/nested.py") == {
        ("engine/nested.py", "outer", "empty"),
        ("engine/nested.py", "outer", "dynamic"),
        ("engine/nested.py", "inner", "empty"),
        ("engine/nested.py", "inner", "dynamic"),
    }


def test_cuda_writer_scan_does_not_use_recursive_ast_walk(monkeypatch):
    tree = ast.parse(
        "def guarded(env):\n"
        "    env['CUDA_VISIBLE_DEVICES'] = ''\n"
    )
    monkeypatch.setattr(scope_module.ast, "walk", lambda _node: (
        _ for _ in ()
    ))

    assert _cuda_writes(tree, "engine/guarded.py") == {
        ("engine/guarded.py", "guarded", "empty")
    }


def test_source_boundary_audit_rejects_missing_guard(tmp_path):
    package = tmp_path / "orze"
    _copy_audited_sources(package)
    launcher = package / "engine/launcher.py"
    launcher.write_text(
        launcher.read_text().replace(
            "env = _authorized_gpu_environment(gpu, cfg, env)",
            "env['CUDA_VISIBLE_DEVICES'] = str(gpu)",
            1,
        ),
        encoding="utf-8",
    )
    rogue = package / "engine/rogue.py"
    rogue.write_text(
        "def bypass(env, gpu):\n"
        "    env['CUDA_VISIBLE_DEVICES'] = str(gpu)\n",
        encoding="utf-8",
    )
    try:
        audit_source_boundaries(package)
    except scope_module.GpuScopeAuditError as exc:
        assert str(exc) in {
            "gpu_scope_cuda_writer_universe_invalid",
            "gpu_scope_boundary_guard_missing",
        }
    else:
        raise AssertionError("new direct CUDA writer was accepted")


def test_source_boundary_audit_rejects_lease_without_idle_attestation(tmp_path):
    package = tmp_path / "orze"
    _copy_audited_sources(package)
    launcher = package / "engine/launcher.py"
    launcher.write_text(
        launcher.read_text().replace(
            "gpu_execution_lease(gpu, require_idle=True)",
            "gpu_execution_lease(gpu)",
            1,
        ),
        encoding="utf-8",
    )

    try:
        audit_source_boundaries(package)
    except scope_module.GpuScopeAuditError as exc:
        assert str(exc) == "gpu_scope_boundary_guard_missing"
    else:
        raise AssertionError("lease without idle attestation was accepted")


def test_source_boundary_audit_rejects_unaccounted_gpu_child(tmp_path):
    package = tmp_path / "orze"
    _copy_audited_sources(package)
    evaluator = package / "engine/evaluator.py"
    evaluator.write_text(
        evaluator.read_text().replace(
            "record_compute_start(", "unaccounted_compute_start(",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
            scope_module.GpuScopeAuditError,
            match="gpu_scope_boundary_guard_missing"):
        audit_source_boundaries(package)


def test_gpu_scope_audit_rejects_input_mutation(tmp_path, monkeypatch):
    config = _config(tmp_path)
    original = scope_module.capture_source_universe_identity

    def changed_after_start(root):
        report = original(root)
        report["python_source_universe_sha256"] = "0" * 64
        return report

    monkeypatch.setattr(
        scope_module, "capture_source_universe_identity", changed_after_start
    )
    receipt = audit_gpu_scope(
        config,
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "gpu_scope_inputs_changed_during_audit"


def test_gpu_scope_audit_uses_one_ast_pass_and_one_cryptographic_rehash(
        tmp_path, monkeypatch):
    original_audit = scope_module.audit_source_boundaries
    original_identity = scope_module.capture_source_universe_identity
    calls = {"ast_audit": 0, "identity_rehash": 0}

    def counted_audit(root):
        calls["ast_audit"] += 1
        return original_audit(root)

    def counted_identity(root):
        calls["identity_rehash"] += 1
        return original_identity(root)

    monkeypatch.setattr(scope_module, "audit_source_boundaries", counted_audit)
    monkeypatch.setattr(
        scope_module, "capture_source_universe_identity", counted_identity,
    )
    receipt = audit_gpu_scope(
        _config(tmp_path),
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )

    assert receipt["status"] == "VERIFIED"
    assert calls == {"ast_audit": 1, "identity_rehash": 1}


def test_gpu_scope_audit_rejects_parent_symlinked_config(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    config = _config(real)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    receipt = audit_gpu_scope(
        linked / config.name,
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "gpu_scope_source_redirected_or_unavailable"


def test_gpu_scope_audit_rejects_hardlinked_config(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "config-alias.yaml").hardlink_to(config)
    receipt = audit_gpu_scope(
        config,
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "UNVERIFIED"
    assert receipt["reason"] == "gpu_scope_source_identity_invalid"


def test_gpu_scope_audit_rejects_helper_that_authorizes_forbidden_id(
        tmp_path, monkeypatch):
    def permissive(gpu, _cfg, base_env):
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    monkeypatch.setattr(
        scope_module, "_authorized_gpu_environment", permissive
    )
    receipt = audit_gpu_scope(
        _config(tmp_path),
        physical_scope=[4, 5, 6, 7],
        forbidden_scope=[0, 1, 2, 3],
    )
    assert receipt["status"] == "FAILED"
    assert receipt["reason"] == "gpu_scope_forbidden_id_authorized"
