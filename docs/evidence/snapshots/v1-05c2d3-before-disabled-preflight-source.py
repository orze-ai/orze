"""Consume native resolver proof without reviving its prelaunch claim bytes.

Capture performs script/config hashing outside the SQLite writer. Verification
inside a writer checks bounded receipts, current rows, claim, policy and file
stat witnesses only (check_inputs=False); GO verification outside the writer
rehashes the actual inputs. Neither mode locks mutable script files globally.

A consumer RUNNING row carries the original source projection. Its own current
claim ID and lifecycle authorize use after started() legitimately adds trainer
fields to claim.json. The old preflight claim hash remains an immutable binding,
not an assertion that the current started claim still has its original bytes.
No source is inferred from a copied static receipt or a missing config switch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from orze.core.execution_attempts import AttemptRef, _json, current_attempt, require_current
from orze.engine.execution_authority import canonical_identity_equal as same, lifecycle_fence
from orze.engine.termination_hold import TerminationUnconfirmed
from orze.engine.training_attempts import _launch_state, _read


@dataclass(frozen=True)
class PreflightSourceCapture:
    """Detached local capture; .source is a fresh bounded JSON projection."""
    scope: str
    _source_json: str | None
    _policy_sha256: str
    _files: tuple

    @property
    def source(self):
        return None if self._source_json is None else json.loads(self._source_json)


def _native():
    from orze.engine import native_artifact_preflight
    return native_artifact_preflight


def _hold(reason):
    return _native().ArtifactPreflightHOLD("artifact_preflight_source_" + reason)


def _folder(idea_dir):
    return Path(os.path.abspath(idea_dir))


def _enabled(cfg):
    if type(cfg) is not dict:
        raise _hold("configuration_invalid")
    spec = cfg.get("artifact_preflight", {})
    if type(spec) is not dict or type(spec.get("enabled", False)) is not bool:
        raise _hold("configuration_invalid")
    return spec.get("enabled", False)


def _policy(cfg):
    # This signature stays in the transient capture, not in a durable receipt.
    # Bound JSON validation occurs before hashing; no arbitrary input is logged.
    from orze.engine.process import _OFFLINE_ENV_KEYS
    extra = cfg.get("train_extra_env") or {}
    if type(extra) is not dict or any(type(key) is not str for key in extra):
        raise _hold("environment_invalid")
    value = {"artifact_preflight": cfg.get("artifact_preflight", {}),
             "project_root": str(cfg.get("_project_root", ".")),
             "base_config": str(cfg.get("base_config", "configs/base.yaml")),
             "python": str(cfg.get("python", "")),
             "extra_environment": {key: str(val) for key, val in extra.items()},
             "ambient_offline": {key: os.environ.get(key, "") for key in _OFFLINE_ENV_KEYS}}
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _input_paths(folder, cfg):
    spec = cfg["artifact_preflight"]
    root = Path(cfg.get("_project_root", "."))
    script = Path(str(spec.get("script") or ""))
    if not script.is_absolute():
        script = root / script
    config = folder / "idea_config.yaml"
    if not config.exists():
        config = Path(cfg.get("base_config", "configs/base.yaml"))
        if not config.is_absolute():
            config = root / config
    return tuple(Path(os.path.abspath(path)) for path in (script, config))


def _stat(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise _hold("input_file_invalid")
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _files(folder, cfg):
    return tuple((str(path), _stat(path)) for path in _input_paths(folder, cfg))


def _identity(folder, cfg):
    from orze.engine.process import _artifact_preflight_identity
    before = _files(folder, cfg)
    value = _artifact_preflight_identity(folder.name, folder.parent, cfg)
    if before != _files(folder, cfg):
        raise _hold("input_changed_during_read")
    if (set(value) != {"network_policy", "script_sha256", "config_sha256", "contract_sha256"}
            or value["network_policy"] not in ("inherit", "required", "offline")
            or any(type(value[key]) is not str or len(value[key]) != 64
                   or any(char not in "0123456789abcdef" for char in value[key])
                   for key in ("script_sha256", "config_sha256", "contract_sha256"))):
        raise _hold("input_identity_invalid")
    return value, before


def _history(lake, folder, cfg):
    native = _native()
    native._scope(lake, folder, cfg)
    if lake is None:
        return None
    row = current_attempt(lake.conn, folder.name, "artifact_preflight")
    if row is not None:
        if row["state"] not in ("TERMINAL", "NOT_STARTED"):
            raise _hold("previous_action_unclosed")
        success = native._cached(row, folder)
        claim, _ = _read(folder / "claim.json", 8192)
        if row["binding"].get("claim_attempt_id") == claim.get("attempt_id") and not success:
            raise _hold("same_claim_failed")
    return row


def require_preflight_history_closed(lake, idea_dir, cfg):
    """Existing-history gate only; absence does not mandate a new invocation."""
    try:
        _history(lake, _folder(idea_dir), cfg)
    except TerminationUnconfirmed:
        raise
    except Exception as exc:
        raise _hold("history_unconfirmed") from exc


def _projection(row):
    ref = AttemptRef(row["task_id"], "artifact_preflight", row["attempt_id"], row["generation"])
    return {"attempt_ref": asdict(ref), "receipt_sha256": row["terminal"]["receipt_sha256"],
            "claim_attempt_id": row["binding"]["claim_attempt_id"],
            "claim_sha256": row["binding"]["claim_sha256"],
            "preflight_identity": row["binding"]["preflight_identity"]}


def capture_preflight_source(lake, idea_dir, cfg):
    """Read current source/input hashes outside any SQLite writer."""
    try:
        folder = _folder(idea_dir)
        if lake is not None and lake.conn.in_transaction:
            raise _hold("capture_requires_outside_writer")
        enabled = _enabled(cfg)
        row = _history(lake, folder, cfg)
        policy = _policy(cfg)
        if not enabled and row is None:
            return PreflightSourceCapture(str(folder), None, policy, ())
        if not enabled or lake is None or row is None:
            raise _hold("native_receipt_required")
        if _native()._cached(row, folder) is not True:
            raise _hold("not_passed")
        identity, files = _identity(folder, cfg)
        source = _projection(row)
        if not same(identity, source["preflight_identity"]):
            raise _hold("inputs_changed")
        capture = PreflightSourceCapture(str(folder), _json(source), policy, files)
        # Bounded authority checks after the potentially long file reads.
        _verify(lake, folder, cfg, capture, consumer_ref=None, check_inputs=False)
        return capture
    except TerminationUnconfirmed:
        raise
    except Exception as exc:
        raise _hold("capture_unconfirmed") from exc


def _verify(lake, folder, cfg, capture, *, consumer_ref, check_inputs):
    row = _history(lake, folder, cfg)
    enabled = _enabled(cfg)
    if capture is None:
        if not enabled and row is None:
            return None
        raise _hold("capture_required")
    if (type(capture) is not PreflightSourceCapture or capture.scope != str(folder)
            or capture._policy_sha256 != _policy(cfg)):
        raise _hold("capture_changed")
    source = capture.source
    if source is None:
        if not enabled and row is None and capture._files == ():
            return None
        raise _hold("native_receipt_required")
    if not enabled or lake is None or row is None:
        raise _hold("native_receipt_required")
    if (not same(_projection(row), source) or _native()._cached(row, folder) is not True
            or row["binding"].get("origin") != "native_artifact_preflight"
            or row["binding"].get("scope") != str(folder)):
        raise _hold("receipt_changed")
    if _files(folder, cfg) != capture._files:
        raise _hold("input_stat_changed")
    if check_inputs:
        identity, files = _identity(folder, cfg)
        if files != capture._files or not same(identity, source["preflight_identity"]):
            raise _hold("inputs_changed")
    claim, claim_sha = _read(folder / "claim.json", 8192)
    if claim.get("attempt_id") != source["claim_attempt_id"]:
        raise _hold("claim_changed")
    if consumer_ref is None:
        if (claim_sha != source["claim_sha256"]
                or not same(_launch_state(lake, folder.name), row["binding"].get("launch_lifecycle"))):
            raise _hold("launch_authority_changed")
    else:
        if (not isinstance(consumer_ref, AttemptRef) or consumer_ref.task_id != folder.name
                or consumer_ref.phase not in ("training", "posthoc")
                or consumer_ref.attempt_id != source["claim_attempt_id"]):
            raise _hold("consumer_reference_invalid")
        consumer = require_current(lake.conn, consumer_ref, states=("RUNNING",))
        binding = consumer["binding"]
        fence = lifecycle_fence(lake, folder.name, "training")
        supervision = binding.get("supervision")
        if (binding.get("origin") != "native_" + consumer_ref.phase
                or type(supervision) is not dict
                or not same(supervision.get("identity"),
                            {"attempt_ref": asdict(consumer_ref), "scope": str(folder)})
                or binding.get("claim_sha256") != source["claim_sha256"]
                or not same(binding.get("artifact_preflight_source"), source)
                or fence["global_state"] != "IN_PROGRESS"
                or not same(binding.get("lifecycle"), fence)):
            raise _hold("consumer_changed")
    return source


def verify_preflight_source(lake, idea_dir, cfg, capture, *, consumer_ref=None, check_inputs=True):
    """Default GO verification rehashes; only a writer may request stat-only."""
    try:
        if type(check_inputs) is not bool:
            raise _hold("check_policy_invalid")
        in_writer = lake is not None and lake.conn.in_transaction
        if check_inputs == bool(in_writer):
            raise _hold("verification_writer_mode_invalid")
        return _verify(lake, _folder(idea_dir), cfg, capture,
                       consumer_ref=consumer_ref, check_inputs=check_inputs)
    except TerminationUnconfirmed:
        raise
    except Exception as exc:
        raise _hold("verification_unconfirmed") from exc
