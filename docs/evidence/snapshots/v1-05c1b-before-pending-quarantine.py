"""Explicit project-scoped garbage collection, with short authority guards.

Keep sets only exclude candidates; they never authorize deletion. Native and
legacy authority are re-read inside the same guard used by claim/publication.
Directory reclamation occurs after atomic same-root quarantine detachment.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

from orze.engine.artifact_publication import _open_directory
from orze.engine.attempt_effect_lock import (
    AttemptEffectInDoubt, attempt_effect_lock, require_effect_lease,
)
from orze.engine.cleanup_patterns import (
    _absolute, _authority, _bound_storage, _declarations, _protected_name,
)
from orze.engine.claim_authority import _lake_path
from orze.reporting.evidence import report_lifecycle_db_path
from orze.engine.gc_tree import (
    GCRefused, identity, plain_directory, snapshot, prepare_quarantine,
    rename_no_replace, reclaim, require_no_pending,
)


@dataclass(frozen=True)
class GCScope:
    results_dir: Path
    project_root: Path
    checkpoints_dir: Path | None
    archive_dir: Path | None
    cfg: dict
    lake: object


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def gc_scope(results_dir, cfg, *, lake=None, lake_db_path=None,
             checkpoints_dir=None, archive_dir=None):
    """Read-only scope validation. cfg=None never means undeclared legacy."""
    if not isinstance(cfg, dict):
        raise GCRefused("gc_explicit_configuration_required")
    cfg = deepcopy(cfg)
    for name in ("gc", "report"):
        if cfg.get(name) is not None and not isinstance(cfg[name], dict):
            raise GCRefused("gc_configuration_invalid")
    results = _absolute(results_dir, Path.cwd())
    project = _absolute(cfg.get("_project_root") or
                        (Path(cfg["_config_path"]).parent if cfg.get("_config_path") else results.parent), Path.cwd())
    if "results_dir" in cfg and _absolute(cfg["results_dir"], project) != results:
        raise GCRefused("gc_results_scope_mismatch")
    cfg["_project_root"] = str(project)
    if lake_db_path is not None:
        db = _absolute(lake_db_path, project)
        if "idea_lake_db" in cfg and _absolute(report_lifecycle_db_path(results, cfg), project) != db:
            raise GCRefused("gc_catalog_scope_mismatch")
        cfg["idea_lake_db"] = str(db)
    supplied = _lake_path(lake)
    if supplied and "idea_lake_db" in cfg and str(report_lifecycle_db_path(results, cfg).absolute()) != supplied:
        raise GCRefused("gc_catalog_scope_mismatch")
    checkpoints = _absolute(checkpoints_dir, project) if checkpoints_dir is not None else None
    archive = _absolute(archive_dir, project) if archive_dir is not None else None
    plain_directory(project)
    plain_directory(results)
    protected, controls = _declarations(results / "idea-scope-probe", cfg)
    if supplied:
        protected.update(Path(supplied + suffix) for suffix in ("", "-wal", "-shm", "-journal"))
    for root in (checkpoints, archive):
        if root is None:
            continue
        plain_directory(root, missing=True)
        if root == project or root in project.parents or _overlap(root, results):
            raise GCRefused("gc_root_overlap")
        if any(_overlap(root, path) for path in protected | controls):
            raise GCRefused("gc_declared_source_overlap")
    if checkpoints is not None and archive is not None and _overlap(checkpoints, archive):
        raise GCRefused("gc_root_overlap")
    return GCScope(results, project, checkpoints, archive, cfg, lake)


def _require_scope(scope, root, mode):
    if not isinstance(scope, GCScope):
        raise GCRefused("gc_explicit_scope_required")
    expected = scope.checkpoints_dir if mode == "checkpoints" else scope.results_dir
    if expected is None or _absolute(root, scope.project_root) != expected:
        raise GCRefused("gc_operation_scope_mismatch")
    # Revalidate roots and config at each operation, before scanning or writes.
    gc_scope(scope.results_dir, scope.cfg, lake=scope.lake,
             checkpoints_dir=scope.checkpoints_dir, archive_dir=scope.archive_dir)
    return expected


def _entries(directory):
    """Capture each DirEntry identity before iteration can yield to a peer."""
    fd = _open_directory(directory)
    try:
        with os.scandir(fd) as iterator:
            entries = []
            for item in iterator:
                if len(entries) >= 16384:
                    raise GCRefused("gc_scan_limit")
                info = item.stat(follow_symlinks=False)
                entries.append((directory / item.name, identity(info)))
        return entries
    finally:
        os.close(fd)


def _base(task_id):
    return task_id.split("~", 1)[0].split("-ht-", 1)[0]


def _candidates(scope, mode):
    root = scope.checkpoints_dir if mode == "checkpoints" else scope.results_dir
    if root is None or not root.exists():
        return []
    found = []
    for path, captured in _entries(root):
        if not stat.S_ISDIR(captured[2]):
            continue
        if mode == "checkpoints":
            if path.name.startswith("idea-"):
                found.append((path.name, path, captured))
            elif not path.name.startswith(("_", ".")):
                # The inherited layout permits one grouping directory only.
                if identity(path.lstat()) != captured:
                    raise GCRefused("gc_parent_replaced")
                found.extend((child.name, child, info) for child, info in _entries(path)
                             if child.name.startswith("idea-") and stat.S_ISDIR(info[2]))
        elif path.name.startswith("idea-"):
            if identity(path.lstat()) != captured:
                raise GCRefused("gc_task_replaced")
            for child, info in _entries(path):
                if (child.suffix in {".pt", ".pth", ".ckpt", ".bin"}
                        or mode == "archive" and child.name == "overlays"):
                    found.append((path.name, child, info))
    return found


def _protected_candidate(path, folder, exact, trees):
    if folder in path.parents and _protected_name(path.relative_to(folder).parts):
        return True
    return any(_overlap(path, declaration) for declaration in exact | trees)


def _legacy_state(folder):
    metrics = folder / "metrics.json"
    if not metrics.exists():
        return
    from orze.engine.attempt_effect_receipts import _read
    try:
        value = json.loads(_read(metrics))
    except Exception as exc:
        raise GCRefused("gc_legacy_state_unknown") from exc
    if not isinstance(value, dict) or value.get("status") not in {"COMPLETED", "FAILED", "SKIPPED", "ARCHIVED"}:
        raise GCRefused("gc_legacy_state_unclosed")


def _destination(scope, task_id, path):
    destination = scope.archive_dir / task_id / path.name
    plain_directory(destination.parent, missing=True)
    if destination.exists() or destination.is_symlink():
        raise GCRefused("gc_archive_destination_exists")
    current = destination.parent
    while not current.exists():
        current = current.parent
    if current.stat().st_dev != path.parent.stat().st_dev:
        raise GCRefused("gc_cross_device_refused")
    return destination


def collect(scope, root, keep_ids, *, mode, dry_run=False):
    """Return old stat keys plus explicit HOLD/error reasons, never fake success."""
    root = _require_scope(scope, root, mode)
    if type(dry_run) is not bool or not isinstance(keep_ids, (set, frozenset)) or any(type(x) is not str for x in keep_ids):
        raise GCRefused("gc_arguments_invalid")
    key = {"checkpoints": "deleted", "results": "deleted_files", "archive": "archived_files"}[mode]
    stats = {key: 0, "freed_bytes": 0, "kept": 0, "errors": 0, "reasons": []}
    if mode == "archive":
        stats["moved_bytes"] = 0
    expanded = keep_ids | {_base(value) for value in keep_ids}
    try:
        candidates = _candidates(scope, mode)
    except Exception:
        stats.update(errors=1, reasons=["gc_scan_unavailable"])
        return stats
    for task_id, path, captured in candidates:
        if task_id in expanded or _base(task_id) in expanded:
            stats["kept"] += 1
            continue
        folder = scope.results_dir / task_id
        try:
            # Never manufacture a task directory/identity for orphaned storage.
            plain_directory(folder)
            exact, trees = _declarations(folder, scope.cfg)
            if _protected_candidate(path, folder, exact, trees):
                stats["kept"] += 1
                continue
            tree = snapshot(path, captured)
            destination = _destination(scope, task_id, path) if mode == "archive" else None
            require_no_pending(root, task_id)
            if dry_run:
                # No ownership directories, protection markers or output dirs.
                from orze.engine.attempt_effect_receipts import require_closed_effects
                from orze.engine.termination_hold import require_no_unconfirmed_stop
                require_closed_effects(folder)
                require_no_unconfirmed_stop(folder)
                if (folder / "_attempt_effect.lock").exists():
                    raise GCRefused("gc_effect_guard_busy")
                with _authority(folder, scope.cfg, scope.lake) as conn:
                    bound_exact, bound_trees = _bound_storage(conn, folder)
                    if conn is None:
                        _legacy_state(folder)
                    if _protected_candidate(path, folder, exact | bound_exact, trees | bound_trees):
                        stats["kept"] += 1
                        continue
                stats[key] += 1
                continue
            with attempt_effect_lock(folder) as lease:
                with _authority(folder, scope.cfg, scope.lake) as conn:
                    bound_exact, bound_trees = _bound_storage(conn, folder)
                    if conn is None:
                        _legacy_state(folder)
                    if _protected_candidate(path, folder, exact | bound_exact, trees | bound_trees):
                        stats["kept"] += 1
                        continue
                    if identity(path.lstat()) != captured:
                        raise GCRefused("gc_candidate_replaced")
                    require_effect_lease(lease, folder)
                    directory = prepare_quarantine(root, task_id, tree, destination)
                    try:
                        rename_no_replace(path, directory / "content")
                        from orze.engine.attempt_effect_receipts import _sync
                        _sync(path.parent)
                        _sync(directory)
                        if identity((directory / "content").lstat())[:6] != captured[:6]:
                            raise GCRefused("gc_detach_identity_changed")
                    except BaseException as exc:
                        raise AttemptEffectInDoubt("gc_detach_unconfirmed") from exc
            # No provider/large walk/deletion is held inside the task guard.
            reclaim(tree, directory, destination)
            stats[key] += 1
            stats["moved_bytes" if mode == "archive" else "freed_bytes"] += tree.size
        except Exception as exc:
            stats["errors"] += 1
            reason = str(exc) if isinstance(exc, (GCRefused, AttemptEffectInDoubt)) else "gc_task_unavailable"
            if reason not in stats["reasons"]:
                stats["reasons"].append(reason)
    return stats
