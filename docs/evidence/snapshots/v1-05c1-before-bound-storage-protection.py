"""Contained built-in disposable-file cleanup, separate from GC/user scripts.

CALLING SPEC:
    cleanup_pattern_files(results_dir, cfg, *, lake=None) -> dict
        Return status, deleted, skipped_tasks and content-free error codes.
        Validate the whole pattern batch before walking. Walk without following
        directory links; delete only captured regular single-link local leaves.
        Under each task's existing effect guard, open fresh read-only authority,
        reject active/unknown/HOLD tasks and preserve current declared outputs.

No database creation, bootstrap, migration or lifecycle updates occur. Native
publication/reset/claim use the same task guard; arbitrary external filesystem
writers are not sandboxed. Garbage collection and explicit user scripts are
not governed by this helper. Directory scans happen before the short guard.
"""
from __future__ import annotations

from contextlib import contextmanager
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import os
import stat

from orze.core.artifact_contract import get_artifact_contract, validate_artifact_publication_binding
from orze.core.execution_attempts import current_attempt
from orze.engine.attempt_effect_lock import attempt_effect_lock, require_effect_lease
from orze.engine.claim_authority import _closed, _lake_path, read_claim
from orze.engine.execution_catalog import declared_catalog
from orze.reporting.evidence import _open_authoritative_lifecycle, report_lifecycle_db_path


_NAMES = {
    "artifact_preflight.json", "interruption.json", "progress.json", "resume_request.json",
    "idea_config.yaml", "resolved_config.yaml", "sweep_config.yaml",
}
_MAX_ENTRIES = 16384
_MAX_DEPTH = 64


class CleanupRefused(ValueError):
    pass


def _patterns(cfg):
    cleanup = cfg.get("cleanup")
    if cleanup is None:
        return []
    if type(cleanup) is not dict:
        raise CleanupRefused("cleanup_configuration_invalid")
    values = cleanup.get("patterns", [])
    if type(values) is not list:
        raise CleanupRefused("cleanup_patterns_invalid")
    result = []
    for value in values:
        if (type(value) is not str or not value or "\0" in value or "\\" in value
                or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
                or str(PurePosixPath(value)) != value or value == "."):
            raise CleanupRefused("cleanup_patterns_invalid")
        result.append(tuple(PurePosixPath(value).parts))
    return result


def _matches(parts, pattern):
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return _matches(parts, pattern[1:]) or bool(parts and _matches(parts[1:], pattern))
    return bool(parts and fnmatchcase(parts[0], pattern[0]) and _matches(parts[1:], pattern[1:]))


def _absolute(path, root):
    value = Path(path)
    return Path(os.path.abspath(value if value.is_absolute() else root / value))


def _protected_name(parts):
    if any(part.startswith("_") or part.startswith(".orze") for part in parts):
        return True
    name = parts[-1]
    return (name in _NAMES or name.startswith("train_output") and name.endswith(".log")
            or name.endswith(".json") and name.startswith(("claim", "metrics", "recovery")))


def _declarations(folder, cfg):
    """Paths only: do not parse model/checkpoint/result bodies for cleanup."""
    root = _absolute(cfg.get("_project_root") or folder.parent.parent, Path.cwd())
    if not cfg.get("_project_root") and cfg.get("_config_path"):
        root = _absolute(cfg["_config_path"], Path.cwd()).parent
    exact = set()
    trees = {_absolute(cfg.get("_orze_dir") or ".orze", root)}
    for key in ("_config_path", "train_script", "base_config", "ideas_file", "pre_script",
                "post_script", "eval_script", "cleanup_script"):
        value = cfg.get(key)
        if value:
            if not isinstance(value, (str, Path)):
                raise CleanupRefused("cleanup_protected_configuration_invalid")
            exact.add(_absolute(value, root))
    db = _absolute(report_lifecycle_db_path(folder.parent, cfg), root)
    exact.update(Path(str(db) + suffix) for suffix in ("", "-wal", "-shm", "-journal"))
    for value in cfg.get("sealed_files") or []:
        exact.add(_absolute(value, root))
    report = cfg.get("report") or {}
    if type(report) is not dict:
        raise CleanupRefused("cleanup_protected_configuration_invalid")
    for column in report.get("columns") or []:
        if not isinstance(column, dict):
            raise CleanupRefused("cleanup_protected_configuration_invalid")
        source = column.get("source")
        if source:
            if type(source) is not str:
                raise CleanupRefused("cleanup_protected_configuration_invalid")
            exact.add(_absolute(source.split(":", 1)[0], folder))
    exact.add(_absolute(cfg.get("eval_output") or "eval_report.json", folder))
    benchmark = report.get("benchmark_contract") or {}
    if isinstance(benchmark, dict) and benchmark.get("receipt"):
        exact.add(_absolute(benchmark["receipt"], folder))
    resume = cfg.get("resume") or {}
    if isinstance(resume, dict):
        for key in ("progress_file", "checkpoint"):
            if resume.get(key):
                exact.add(_absolute(resume[key], folder))
    contract = get_artifact_contract(cfg)
    if contract is not None:
        exact.update(folder / output["path"] for output in contract["outputs"].values())
    return exact, trees


def _protected(path, folder, exact, trees):
    relative = path.relative_to(folder)
    return (_protected_name(relative.parts) or path in exact
            or any(path == tree or tree in path.parents for tree in trees))


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _candidates(folder, patterns, exact, trees):
    from orze.engine.artifact_publication import _open_directory
    found, entries = [], 0

    def walk(path, depth):
        nonlocal entries
        if depth > _MAX_DEPTH:
            raise CleanupRefused("cleanup_scan_limit")
        fd = _open_directory(path)
        try:
            with os.scandir(fd) as items:
                names = []
                for item in items:
                    entries += 1
                    if entries > _MAX_ENTRIES:
                        raise CleanupRefused("cleanup_scan_limit")
                    names.append(item.name)
            for name in sorted(names):
                candidate = path / name
                if _protected(candidate, folder, exact, trees):
                    continue
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    walk(candidate, depth + 1)
                elif (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                      and any(_matches(candidate.relative_to(folder).parts, pattern) for pattern in patterns)):
                    found.append((candidate, _identity(info)))
        finally:
            os.close(fd)
    walk(folder, 0)
    return found


@contextmanager
def _authority(folder, cfg, lake):
    """Fresh read snapshot inside the task effect guard, no implicit bootstrap."""
    if lake is not None and lake.conn.in_transaction:
        raise CleanupRefused("cleanup_caller_transaction_active")
    claim = read_claim(folder / "claim.json")
    declared = declared_catalog(folder)
    supplied = _lake_path(lake)
    routes = [value for value in (supplied, declared, (claim or {}).get("lifecycle_db")) if value is not None]
    configured = report_lifecycle_db_path(folder.parent, cfg).absolute()
    if "idea_lake_db" in cfg or configured.exists() or configured.is_symlink():
        routes.append(str(configured))
    if any(type(value) is not str or not value or not Path(value).is_absolute() for value in routes):
        raise CleanupRefused("cleanup_catalog_route_invalid")
    if routes and any(value != routes[0] for value in routes[1:]):
        raise CleanupRefused("cleanup_catalog_scope_mismatch")
    if not routes:
        # A legacy outstanding claim is not proof that its writer stopped.
        if claim is not None:
            raise CleanupRefused("cleanup_legacy_claim_unconfirmed")
        yield None
        return
    connection, reason = _open_authoritative_lifecycle(Path(routes[0]))
    if connection is None:
        raise CleanupRefused("cleanup_catalog_unavailable:" + reason)
    try:
        connection.execute("BEGIN")
        from orze.reporting.lifecycle_stages import validate_lifecycle_schema, stage_projection
        from orze.reporting.catalog import _agreed_state
        schema = validate_lifecycle_schema(connection)
        stages, joins = stage_projection(schema)
        rows = connection.execute(
            f"SELECT i.status,s.current_state,{stages} FROM ideas i "
            "LEFT JOIN idea_state s ON s.idea_id=i.idea_id COLLATE BINARY " + joins
            + " WHERE i.idea_id=? COLLATE BINARY LIMIT 2", (folder.name,),
        ).fetchall()
        if len(rows) != 1 or _agreed_state(*rows[0])[0] not in {"COMPLETE", "FAILED", "SKIPPED", "ARCHIVED"}:
            raise CleanupRefused("cleanup_task_not_closed")
        _closed(connection, folder.name)
        yield connection
    finally:
        connection.close()


def _native_outputs(connection, folder):
    if connection is None:
        return set()
    row = current_attempt(connection, folder.name, "training")
    if row is None or "artifact_publication" not in row["binding"]:
        return set()
    binding = validate_artifact_publication_binding(row["binding"]["artifact_publication"])
    if binding["scope"] != str(folder.parent):
        raise CleanupRefused("cleanup_artifact_scope_mismatch")
    return {folder / item["path"] for item in binding["contract"]["outputs"].values()}


def _unlink_local(folder, path, expected):
    from orze.engine.artifact_publication import _open_directory
    # Reopen every directory without following links; never resolve a link and
    # reinterpret the resolved file as belonging to this task.
    if folder not in path.parents or ".." in path.relative_to(folder).parts:
        raise CleanupRefused("cleanup_path_outside_task")
    parent = _open_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        if (_identity(os.fstat(fd)) != expected
                or _identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) != expected):
            raise CleanupRefused("cleanup_file_changed")
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)


def cleanup_pattern_files(results_dir, cfg, *, lake=None):
    """Delete only explicitly matched disposable files; refusals are observable."""
    report = {"status": "disabled", "deleted": 0, "skipped_tasks": 0, "errors": []}
    try:
        patterns = _patterns(cfg)
    except (ValueError, TypeError, AttributeError):
        report.update(status="invalid", errors=["cleanup_patterns_invalid"])
        return report
    if not patterns:
        return report
    report["status"] = "completed"
    root = Path(os.path.abspath(results_dir))
    from orze.engine.artifact_publication import _open_directory
    try:
        root_fd = _open_directory(root)
        try:
            with os.scandir(root_fd) as items:
                tasks = sorted(item.name for item in items
                               if item.name.startswith("idea-") and item.is_dir(follow_symlinks=False))
        finally:
            os.close(root_fd)
    except OSError:
        report.update(status="partial", errors=["cleanup_results_unavailable"])
        return report
    for name in tasks:
        folder = root / name
        try:
            exact, trees = _declarations(folder, cfg)
            candidates = _candidates(folder, patterns, exact, trees)
            if not candidates:
                continue
            with attempt_effect_lock(folder) as lease:
                with _authority(folder, cfg, lake) as connection:
                    exact.update(_native_outputs(connection, folder))
                    for path, expected in candidates:
                        require_effect_lease(lease, folder)
                        if not _protected(path, folder, exact, trees):
                            _unlink_local(folder, path, expected)
                            report["deleted"] += 1
        except Exception as exc:
            report["status"] = "partial"
            report["skipped_tasks"] += 1
            reason = str(exc) if isinstance(exc, CleanupRefused) else "cleanup_task_unavailable"
            if reason not in report["errors"]:
                report["errors"].append(reason)
    return report
