"""Explicit replication admission; never starts workers or discovers GPUs.

The control service owns authorization and durable task admission. This thin
CLI adapter resolves the selected project's paths, opens existing authority,
and returns one JSON result. A storage/close error is not proof of rollback:
retry the same request ID to inspect the idempotent outcome.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3

import yaml


def run_replication(args) -> int:
    from orze.core.config import load_project_config
    from orze.core.evaluation_retry_state import open_existing_lake
    from orze.engine.replication import request_replication
    from orze.engine.termination_hold import TerminationUnconfirmed
    from orze.reporting.evidence import report_lifecycle_db_path
    from orze.service.runtime_contract import (
        RuntimeContractError, require_controller_runtime_contract,
    )

    lake = None
    result = None
    error = None
    try:
        config_path = Path(args.config_file or "orze.yaml").absolute()
        if not config_path.is_file():
            raise ValueError("replication_config_missing")
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if document is not None and not isinstance(document, dict):
            raise ValueError("replication_config_not_mapping")
        caller_cwd = Path.cwd()
        try:
            # Shared config loading is cwd-sensitive. Resolve the exact selected
            # project before restoring the caller, including path-only launch
            # inputs. Do not replace a bare interpreter command with a resolved
            # executable: its configured spelling is part of launch identity.
            os.chdir(config_path.parent)
            cfg = load_project_config(str(config_path))
            cfg["_config_path"] = str(config_path)
            from orze.core.cpu_execution import cpu_execution
            path_keys = (("results_dir", "ideas_file") if cpu_execution(cfg) is not None
                         else ("results_dir", "train_script", "base_config", "ideas_file"))
            for key in path_keys:
                cfg[key] = str(Path(cfg[key]).absolute())
            results_dir = Path(cfg["results_dir"])
            cfg["_env_ORZE_RESULTS_DIR"] = str(results_dir)
            cfg["_env_ORZE_IDEAS_FILE"] = cfg["ideas_file"]
            db_path = report_lifecycle_db_path(results_dir, cfg).absolute()
            cfg["idea_lake_db"] = str(db_path)
        finally:
            os.chdir(caller_cwd)
        require_controller_runtime_contract(cfg.get("controller_runtime"))
        lake = open_existing_lake(db_path)
        result = request_replication(
            args.source_task_id, results_dir, cfg, lake,
            request_id=args.request_id, reason=args.reason,
        )
    except (ValueError, TypeError, OSError, sqlite3.Error, yaml.YAMLError, RuntimeContractError,
            TerminationUnconfirmed) as exc:
        error = str(exc)
    finally:
        if lake is not None:
            try:
                lake.close()
            except (OSError, sqlite3.Error):
                error = "replication_database_close_unconfirmed"
    if error is not None:
        print(json.dumps({"error": error}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
