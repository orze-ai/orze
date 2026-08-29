"""Fail-closed physical GPU scope audit without accelerator access."""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, Sequence

from orze.core.config import load_project_config
from orze.core.fs import atomic_write
from orze.core.gpu_lease import gpu_execution_lease
from orze.engine.launcher import (
    LaunchIntegrityError,
    _authorized_gpu_environment,
)


_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_BOUNDARIES = {
    "engine/launcher.py": {
        "_launch_posthoc": frozenset({
            "_authorized_gpu_environment",
            "_assert_controller_runtime_attested",
            "_assert_campaign_evidence_authorized",
            "gpu_execution_lease",
            "record_compute_start",
        }),
        "launch": frozenset({
            "_authorized_gpu_environment",
            "_assert_controller_runtime_attested",
            "_assert_campaign_evidence_authorized",
            "gpu_execution_lease",
            "record_compute_start",
        }),
    },
    "engine/evaluator.py": {
        "launch_eval": frozenset({
            "_authorized_gpu_environment",
            "_assert_controller_runtime_attested",
            "_assert_campaign_evidence_authorized",
            "gpu_execution_lease",
            "record_compute_start",
        }),
        "run_post_scripts": frozenset({
            "_authorized_gpu_environment",
            "_assert_controller_runtime_attested",
            "_assert_campaign_evidence_authorized",
            "gpu_execution_lease",
            "record_compute_start",
            "record_compute_terminal",
        }),
    },
}
_EXPECTED_CUDA_WRITES = frozenset({
    ("benchmarks/launch_policy_latency.py", "run_benchmark", "empty"),
    ("engine/launcher.py", "_authorized_gpu_environment", "dynamic"),
    ("engine/process.py", "run_artifact_preflight", "empty"),
    ("engine/process.py", "run_pre_script", "empty"),
    ("engine/smoke_test.py", "run_smoke_test", "empty"),
})


class GpuScopeAuditError(RuntimeError):
    """GPU scope evidence is missing, ambiguous, or contradictory."""


class GpuScopeTargetError(GpuScopeAuditError):
    """Complete evidence proves a GPU scope target is missed."""


def _open_source(path: Path) -> tuple[int, Path]:
    absolute = Path(path).absolute()
    parts = absolute.parts
    flags_dir = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = None
    try:
        directory_fd = os.open(absolute.anchor, flags_dir)
        for component in parts[1:-1]:
            next_fd = os.open(component, flags_dir, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise GpuScopeAuditError(
            "gpu_scope_source_redirected_or_unavailable"
        ) from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    return descriptor, absolute


def _read_source(path: Path) -> tuple[str, str]:
    descriptor, absolute = _open_source(path)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 <= before.st_size <= _MAX_SOURCE_BYTES):
            raise GpuScopeAuditError("gpu_scope_source_identity_invalid")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(_MAX_SOURCE_BYTES + 1)
            after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    before_identity = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
        before.st_nlink,
    )
    after_identity = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        after.st_nlink,
    )
    if len(data) > _MAX_SOURCE_BYTES or before_identity != after_identity:
        raise GpuScopeAuditError("gpu_scope_source_changed_during_read")
    rebound_fd, _ = _open_source(absolute)
    try:
        rebound = os.fstat(rebound_fd)
    finally:
        os.close(rebound_fd)
    rebound_identity = (
        rebound.st_dev, rebound.st_ino, rebound.st_size,
        rebound.st_mtime_ns, rebound.st_nlink,
    )
    if rebound_identity != before_identity:
        raise GpuScopeAuditError("gpu_scope_source_namespace_changed")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GpuScopeAuditError("gpu_scope_source_encoding_invalid") from exc
    return text, hashlib.sha256(data).hexdigest()


def _called_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _lease_calls_require_idle(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == "gpu_execution_lease":
            calls.append(node)
    return bool(calls) and all(
        any(
            keyword.arg == "require_idle"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in calls
    )


def _subscript_key(node) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _value_kind(node) -> str:
    if isinstance(node, ast.Constant) and node.value == "":
        return "empty"
    return "dynamic"


def _cuda_writes(
    tree: ast.AST,
    relative: str,
) -> set[tuple[str, str, str]]:
    """Find direct CUDA visibility writes in one AST traversal.

    A write inside a nested function is attributed to that function and every
    enclosing function. This deliberately preserves the previous nested
    ``ast.walk(function)`` behavior without walking the same subtree once per
    ancestor function.
    """
    writes = set()

    class WriterVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_names: list[str] = []

        def _record(self, kind: str) -> None:
            for function_name in self.function_names:
                writes.add((relative, function_name, kind))

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            self.function_names.append(node.name)
            self.generic_visit(node)
            self.function_names.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            if any(
                    _subscript_key(target) == "CUDA_VISIBLE_DEVICES"
                    for target in node.targets):
                self._record(_value_kind(node.value))
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if _subscript_key(node.target) == "CUDA_VISIBLE_DEVICES":
                self._record(_value_kind(node.value))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"):
                for argument in node.args:
                    if not isinstance(argument, ast.Dict):
                        continue
                    for key, value in zip(argument.keys, argument.values):
                        if (isinstance(key, ast.Constant)
                                and key.value == "CUDA_VISIBLE_DEVICES"):
                            self._record(_value_kind(value))
                for keyword in node.keywords:
                    if keyword.arg == "CUDA_VISIBLE_DEVICES":
                        self._record(_value_kind(keyword.value))
            self.generic_visit(node)

    WriterVisitor().visit(tree)
    return writes


def capture_source_universe_identity(package_root: str | Path) -> dict:
    """Cryptographically reread the complete Python source universe.

    This is the post-audit stability pass. It deliberately retains the same
    no-follow, single-link, bounded-read, namespace-rebind, UTF-8, filename,
    and content-hash checks as the AST audit, but does not rebuild ASTs whose
    policy result has already been computed from those exact bytes.
    """
    root = Path(package_root).absolute()
    python_paths = sorted(
        root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not python_paths:
        raise GpuScopeAuditError("gpu_scope_source_universe_empty")
    universe_digest = hashlib.sha256()
    for path in python_paths:
        relative = path.relative_to(root).as_posix()
        _text, digest = _read_source(path)
        encoded = relative.encode()
        universe_digest.update(len(encoded).to_bytes(8, "big"))
        universe_digest.update(encoded)
        universe_digest.update(bytes.fromhex(digest))
    return {
        "python_source_count": len(python_paths),
        "python_source_universe_sha256": universe_digest.hexdigest(),
    }


def audit_source_boundaries(package_root: str | Path) -> dict:
    """Verify every GPU-visible subprocess path uses the shared boundary."""
    root = Path(package_root).absolute()
    source_sha256 = {}
    all_cuda_writes = set()
    boundary_checks = {}
    parsed_sources = {}
    universe_digest = hashlib.sha256()
    python_paths = sorted(
        root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not python_paths:
        raise GpuScopeAuditError("gpu_scope_source_universe_empty")
    for path in python_paths:
        relative = path.relative_to(root).as_posix()
        text, digest = _read_source(path)
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise GpuScopeAuditError("gpu_scope_source_syntax_invalid") from exc
        parsed_sources[relative] = (tree, digest)
        encoded = relative.encode()
        universe_digest.update(len(encoded).to_bytes(8, "big"))
        universe_digest.update(encoded)
        universe_digest.update(bytes.fromhex(digest))
        all_cuda_writes.update(_cuda_writes(tree, relative))
    for relative, functions in _BOUNDARIES.items():
        parsed = parsed_sources.get(relative)
        if parsed is None:
            raise GpuScopeTargetError(
                f"gpu_scope_boundary_source_missing:{relative}"
            )
        tree, digest = parsed
        source_sha256[relative] = digest
        definitions = {
            node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for function_name, required_calls in functions.items():
            function = definitions.get(function_name)
            if function is None:
                raise GpuScopeTargetError(
                    f"gpu_scope_boundary_missing:{relative}:{function_name}"
                )
            missing = sorted(required_calls - _called_names(function))
            idle_required = "gpu_execution_lease" in required_calls
            idle_guarded = (
                _lease_calls_require_idle(function) if idle_required else None
            )
            boundary_id = f"{relative}:{function_name}"
            boundary_checks[boundary_id] = {
                "required_calls": sorted(required_calls),
                "missing_calls": missing,
                "lease_idle_attestation": idle_guarded,
                "passed": not missing and idle_guarded is not False,
            }
    if all_cuda_writes != _EXPECTED_CUDA_WRITES:
        raise GpuScopeTargetError("gpu_scope_cuda_writer_universe_invalid")
    if not all(check["passed"] for check in boundary_checks.values()):
        raise GpuScopeTargetError("gpu_scope_boundary_guard_missing")
    return {
        "status": "VERIFIED",
        "boundary_count": len(boundary_checks),
        "boundaries": boundary_checks,
        "cuda_writes": [
            {"source": source, "function": function, "kind": kind}
            for source, function, kind in sorted(all_cuda_writes)
        ],
        "source_sha256": source_sha256,
        "python_source_count": len(python_paths),
        "python_source_universe_sha256": universe_digest.hexdigest(),
    }


def _validate_scope(values: Sequence[int], label: str) -> list[int]:
    if (isinstance(values, (str, bytes)) or not values
            or len(values) != len(set(values))
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in values)):
        raise GpuScopeAuditError(f"gpu_scope_{label}_invalid")
    return list(values)


def audit_gpu_scope(
    config_path: str | Path,
    *,
    physical_scope: Sequence[int],
    forbidden_scope: Sequence[int],
    package_root: str | Path | None = None,
) -> dict:
    """Audit code, config, and no-CUDA child propagation for exact IDs."""
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "status": "UNVERIFIED",
        "reason": "gpu_scope_evidence_incomplete",
        "accelerator_access": "none",
        "rank_claim_proven": False,
    }
    try:
        allowed = _validate_scope(physical_scope, "physical_ids")
        forbidden = _validate_scope(forbidden_scope, "forbidden_ids")
        if set(allowed) & set(forbidden):
            raise GpuScopeAuditError("gpu_scope_overlap")
        config_file = Path(config_path).absolute()
        config_text_before, config_digest = _read_source(config_file)
        cfg = load_project_config(str(config_file))
        config_text_loaded, config_digest_loaded = _read_source(config_file)
        if (config_digest_loaded != config_digest
                or config_text_loaded != config_text_before):
            raise GpuScopeAuditError(
                "gpu_scope_config_changed_during_load"
            )
        scheduling = cfg.get("gpu_scheduling")
        if not isinstance(scheduling, Mapping):
            raise GpuScopeAuditError("gpu_scope_config_missing")
        if scheduling.get("allowed_gpus") != allowed:
            raise GpuScopeTargetError("gpu_scope_allowlist_mismatch")
        reserved = scheduling.get("reserved_gpus") or []
        if not set(forbidden).issubset(set(reserved)):
            raise GpuScopeTargetError("gpu_scope_forbidden_not_reserved")
        scoped_cfg = dict(cfg)
        scoped_cfg["_managed_gpu_ids"] = allowed
        source_root = Path(package_root) if package_root is not None else (
            Path(__file__).resolve().parents[1]
        )
        source_before = audit_source_boundaries(source_root)

        child_observations = []
        for gpu in allowed:
            env = _authorized_gpu_environment(
                gpu, scoped_cfg, {"PATH": os.environ.get("PATH", "")}
            )
            with gpu_execution_lease(gpu) as lease_fds:
                completed = subprocess.run(
                    [
                        sys.executable, "-c",
                        "import os; print(os.environ.get("
                        "'CUDA_VISIBLE_DEVICES', '<missing>'))",
                    ],
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    pass_fds=lease_fds,
                    timeout=10,
                )
            observed = completed.stdout.strip()
            if completed.returncode != 0 or observed != str(gpu):
                raise GpuScopeTargetError("gpu_scope_child_propagation_failed")
            child_observations.append({
                "physical_gpu": gpu,
                "visible_value_sha256": hashlib.sha256(
                    observed.encode()
                ).hexdigest(),
            })

        rejected = []
        for gpu in forbidden:
            try:
                _authorized_gpu_environment(gpu, scoped_cfg, {})
            except LaunchIntegrityError as exc:
                if not str(exc).startswith((
                    "gpu_outside_managed_scope:", "gpu_is_reserved:",
                )):
                    raise GpuScopeTargetError(
                        "gpu_scope_forbidden_rejection_ambiguous"
                    ) from exc
                rejected.append(gpu)
            else:
                raise GpuScopeTargetError("gpu_scope_forbidden_id_authorized")

        source_after = capture_source_universe_identity(source_root)
        config_text_after, config_digest_after = _read_source(config_file)
        if (source_before["python_source_count"]
                != source_after["python_source_count"]
                or source_before["python_source_universe_sha256"]
                != source_after["python_source_universe_sha256"]
                or config_digest != config_digest_after
                or config_text_before != config_text_after):
            raise GpuScopeAuditError("gpu_scope_inputs_changed_during_audit")
        receipt.update({
            "status": "VERIFIED",
            "reason": "gpu_scope_verified",
            "physical_scope": allowed,
            "forbidden_scope": forbidden,
            "config_sha256": config_digest,
            "source_boundaries": source_before,
            "child_propagation": child_observations,
            "forbidden_ids_rejected": rejected,
            "counts": {
                "production_boundaries": source_before["boundary_count"],
                "authorized_child_launches": len(child_observations),
                "forbidden_ids_rejected": len(rejected),
            },
        })
    except GpuScopeTargetError as exc:
        receipt["status"] = "FAILED"
        receipt["reason"] = str(exc)
    except (OSError, TypeError, ValueError, subprocess.SubprocessError,
            LaunchIntegrityError, GpuScopeAuditError) as exc:
        receipt["reason"] = str(exc)
    return receipt


def write_gpu_scope_receipt(
    config_path: str | Path,
    output_path: str | Path,
    *,
    physical_scope: Sequence[int],
    forbidden_scope: Sequence[int],
) -> dict:
    receipt = audit_gpu_scope(
        config_path,
        physical_scope=physical_scope,
        forbidden_scope=forbidden_scope,
    )
    atomic_write(
        Path(output_path),
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _parse_ids(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated IDs") from exc
    try:
        return _validate_scope(values, "cli_ids")
    except GpuScopeAuditError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit exact GPU scope without accelerator access"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--physical-scope", type=_parse_ids, required=True)
    parser.add_argument("--forbidden-scope", type=_parse_ids, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = write_gpu_scope_receipt(
        args.config,
        args.output,
        physical_scope=args.physical_scope,
        forbidden_scope=args.forbidden_scope,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
