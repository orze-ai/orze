"""Native posthoc launch: private bounded input, blocked worker, exact GO.

Existing adapters keep their dict-returning API and run in an attempt-local
work directory. Configuration identity is not a content digest of arbitrary
adapter data paths; it grants neither model lineage nor scientific validity.
This adapter has no automatic restart adoption or replica/resume enrollment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

from orze.core.execution_attempts import _json
from orze.engine.attempt_effect_lock import AttemptEffectBusy
from orze.engine import posthoc_attempts as attempts, posthoc_supervision as proof
from orze.engine.sealed_payload import sealed_payload
from orze.engine.supervised_process import SupervisionUncertain
from orze.engine.termination_hold import TerminationUnconfirmed, require_no_unconfirmed_stop


def _configuration(path, cfg, kind):
    """Read a bounded stable YAML mapping; never silently replace invalid input."""
    import yaml
    from orze.engine.artifact_publication import _path_identities, _verify_identities
    value = {}
    if path.exists() or path.is_symlink():
        captured = _path_identities(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 65536:
                raise AttemptEffectBusy("posthoc_configuration_invalid")
            raw = os.read(fd, 65537)
            if len(raw) != before.st_size or len(raw) > 65536:
                raise AttemptEffectBusy("posthoc_configuration_changed")
            _verify_identities(captured)
        finally:
            os.close(fd)
        value = yaml.safe_load(raw)
        if value is None:
            value = {}
    if type(value) is not dict:
        raise AttemptEffectBusy("posthoc_configuration_not_mapping")
    value.setdefault("kind", kind)
    if value["kind"] != kind:
        raise AttemptEffectBusy("posthoc_kind_changed")
    defaults = cfg.get("posthoc_defaults") or {}
    if type(defaults) is not dict:
        raise AttemptEffectBusy("posthoc_defaults_not_mapping")
    if not value.get("adapter"):
        value["adapter"] = cfg.get("posthoc_adapter") or "null"
    for key, item in defaults.items():
        value.setdefault(key, item)
    if (type(value["adapter"]) is not str or not value["adapter"]
            or len(value["adapter"]) > 128):
        raise AttemptEffectBusy("posthoc_adapter_invalid")
    return json.loads(_json(value))


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _timeout(cfg):
    """Reject unusable execution budgets before creating intent or a process."""
    from orze.engine.launcher import LaunchIntegrityError
    value = cfg.get("posthoc_timeout", 3600)
    if type(value) not in (int, float):
        raise LaunchIntegrityError("posthoc_timeout_invalid")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise LaunchIntegrityError("posthoc_timeout_invalid") from exc
    if not math.isfinite(value) or value <= 0:
        raise LaunchIntegrityError("posthoc_timeout_invalid")
    return value


def _require_supported_request(lake, idea_id):
    """A durable training replica reservation is not posthoc authority."""
    from orze.core.replication_requests import request_for_task
    from orze.engine.launcher import LaunchIntegrityError
    if request_for_task(lake.conn, idea_id) is not None:
        raise LaunchIntegrityError("posthoc_replication_not_supported")


def artifact_binding(cfg, folder, execution_identity):
    """Use the same file contract, with a distinct legacy-posthoc specification."""
    from orze.core.artifact_contract import artifact_publication_binding
    bound = artifact_publication_binding(cfg, folder, execution_identity)
    if bound is not None:
        bound["spec_fingerprint"] = _sha({
            "specification_schema": "orze.legacy_posthoc_artifacts.v1",
            "execution_identity": execution_identity, "artifact_contract": bound["contract"]})
    return bound


def _create_work(folder, attempt_id):
    """Create only a fresh attempt directory, never adopt old shared outputs."""
    from orze.engine.artifact_publication import _open_directory
    parent = _open_directory(folder / "_posthoc_attempts", create=True)
    child = None
    try:
        os.mkdir(attempt_id, 0o700, dir_fd=parent)
        os.fsync(parent)
        child = os.open(attempt_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.mkdir("work", 0o700, dir_fd=child)
        os.fsync(child)
    except FileExistsError as exc:
        raise AttemptEffectBusy("posthoc_work_requires_resolution") from exc
    finally:
        try:
            if child is not None:
                os.close(child)
        finally:
            os.close(parent)


def launch(idea_id, gpu, results_dir, cfg, *, kind, idea_cfg_path, lake):
    from orze.engine import launcher
    from orze.engine.accounting import record_compute_start
    from orze.engine.execution_authority import canonical_identity_equal
    folder = Path(os.path.abspath(results_dir)) / idea_id
    attempts.require_catalog(lake, folder, cfg)
    if lake is None:
        raise AttemptEffectBusy("posthoc_native_catalog_required")
    timeout = _timeout(cfg)
    _require_supported_request(lake, idea_id)
    launcher._assert_launch_authorized(idea_id, folder.parent, cfg)
    launcher._assert_gpu_authorized(gpu, cfg)
    launcher._assert_campaign_evidence_authorized(cfg, lake)
    require_no_unconfirmed_stop(folder)
    if not launcher.verify_artifact_preflight_receipt(idea_id, folder.parent, cfg):
        raise launcher.LaunchIntegrityError("artifact_preflight_receipt_missing_or_stale")
    claim, _ = attempts._read(folder / "claim.json", 8192)
    attempt_id = claim.get("attempt_id")
    if type(attempt_id) is not str or not attempt_id:
        raise launcher.LaunchIntegrityError("posthoc_claim_attempt_missing")
    configuration = _configuration(Path(idea_cfg_path), cfg, kind)
    python = cfg.get("python", sys.executable)
    execution_identity = _sha({"schema": "orze.legacy_posthoc_execution.v1",
        "kind": kind, "configuration": configuration, "python": python,
        "timeout_seconds": timeout})
    work = folder / "_posthoc_attempts" / attempt_id / "work"
    packet = {"schema": 1, "task_id": idea_id, "attempt_id": attempt_id,
              "kind": kind, "work_dir": str(work), "configuration": configuration}
    raw = _json(packet).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    inputs = {"schema": 1, "kind": kind, "adapter": configuration["adapter"],
              "config_sha256": _sha(configuration), "payload_sha256": digest,
              "work_dir": str(work), "execution_identity": execution_identity,
              "timeout_seconds": timeout}
    bound = artifact_binding(cfg, folder, execution_identity)
    tp = SimpleNamespace(idea_id=idea_id, gpu=gpu, process=None,
        start_time=time.time(), attempt_id=attempt_id, is_posthoc=True,
        execution_identity=execution_identity, attempt_ref=None)
    log_fh = None
    try:
        with sealed_payload(raw) as payload_fd:
            tp.attempt_ref = attempts.begin(lake, tp, folder,
                launch_inputs=inputs, artifact_binding=bound)
            _create_work(folder, attempt_id)
            log_path = folder / "train_output.log"
            log_fh = open(log_path, "a", encoding="utf-8")
            launcher._assert_controller_runtime_attested(cfg)
            with launcher.gpu_execution_lease(gpu, require_idle=True) as lease_fds:
                launcher._verify_gpu_free(gpu, launcher._launch_min_free_vram(cfg))
                env = launcher._authorized_gpu_environment(gpu, cfg, os.environ.copy())
                command = [python, "-m", "orze.engine.posthoc_worker", str(payload_fd),
                           digest, str(tp.attempt_ref.generation)]
                try:
                    tp.process = launcher.prepare_supervised(command,
                        identity=proof.identity(tp, folder), env=env,
                        stdout=log_fh, stderr=subprocess.STDOUT,
                        pass_fds=lease_fds, worker_only_fds=(payload_fd,))
                except SupervisionUncertain:
                    # Preserve uncertainty even if the enclosing FD close
                    # also fails and replaces the original exception.
                    tp._termination_unconfirmed = True
                    raise
                tp.start_time = time.time()
                attempts.record_ready_start(lake, tp, folder, record_compute_start)
            # Constructor failure retains the already populated provisional tp.
            tp = launcher.TrainingProcess(idea_id=idea_id, gpu=gpu, process=tp.process,
                start_time=tp.start_time, log_path=log_path,
                timeout=timeout,
                attempt_id=attempt_id, attempt_ref=tp.attempt_ref,
                execution_identity=execution_identity, _log_fh=log_fh)
            tp.is_posthoc = True
            attempts.started(lake, tp, folder, launcher.capture_process_identity(tp.process.pid))
            fresh = _configuration(Path(idea_cfg_path), cfg, kind)
            if (not canonical_identity_equal(configuration, fresh)
                    or cfg.get("python", sys.executable) != python
                    or _timeout(cfg) != timeout
                    or not canonical_identity_equal({"value": bound},
                        {"value": artifact_binding(cfg, folder, execution_identity)})):
                raise launcher.LaunchIntegrityError("posthoc_execution_inputs_changed")
            if not attempts.current(lake, tp, folder):
                raise launcher.LaunchIntegrityError("posthoc_launch_authority_changed")
            launcher._assert_controller_runtime_attested(cfg)
            launcher._assert_launch_authorized(idea_id, folder.parent, cfg)
            launcher._assert_gpu_authorized(gpu, cfg)
            launcher._assert_campaign_evidence_authorized(cfg, lake)
            _require_supported_request(lake, idea_id)
            require_no_unconfirmed_stop(folder)
            tp.process.start()
        return tp
    except SupervisionUncertain as exc:
        tp._termination_unconfirmed = True
        launcher._close_launch_log(log_fh)
        raise TerminationUnconfirmed("posthoc_supervision_unconfirmed") from exc
    except launcher.LaunchIntegrityError:
        try:
            if tp.process is not None:
                from orze.engine.termination_hold import terminate_execution
                terminate_execution(tp, folder, phase="posthoc",
                                    reaper=launcher._terminate_and_reap, timeout=3)
        finally:
            launcher._close_launch_log(log_fh)
        # Rejected live authorization never grants a failure publication.
        raise
    except BaseException as error:
        if getattr(tp, "_termination_unconfirmed", False) is True:
            launcher._close_launch_log(log_fh)
            raise TerminationUnconfirmed("posthoc_supervision_unconfirmed") from error
        if tp.process is not None:
            launcher._cleanup_failed_launch(tp, folder, "posthoc", log_fh, lake=lake)
        else:
            launcher._close_launch_log(log_fh)
            if tp.attempt_ref is not None:
                attempts.failed_launch(lake, tp, folder, None, not_started=True)
        from orze.engine.launch_failure_report import bind_launch_error
        bind_launch_error(error, tp.attempt_ref)
        raise
