"""Direct launches enforce controller identity before GPU discovery."""

import pytest

import orze.cli as cli
import orze.lifecycle as lifecycle
from orze.service.runtime_contract import RuntimeContractError


def test_controller_runtime_drift_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_controller_runtime_contract",
        lambda contract: {
            "schema_version": 1,
            "contract_ok": False,
            "errors": ["runtime_package_root_drift:orze"],
        },
    )

    with pytest.raises(SystemExit, match="runtime_package_root_drift:orze"):
        cli._require_controller_runtime({"controller_runtime": {"pin": True}})


def test_controller_runtime_unconfigured_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        "orze.hardware.gpu.detect_all_gpus",
        lambda: pytest.fail("runtime no-op must not inspect GPUs"),
    )

    cli._require_controller_runtime({})


def test_main_checks_controller_runtime_before_gpu_discovery(monkeypatch):
    monkeypatch.setattr("sys.argv", ["orze", "--gpus", "4"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: {
        "controller_runtime": {"pin": True},
    })
    monkeypatch.setattr(
        cli, "_require_controller_runtime",
        lambda cfg: (_ for _ in ()).throw(SystemExit("runtime drift")),
    )
    monkeypatch.setattr(
        cli, "detect_all_gpus",
        lambda: pytest.fail("GPU discovery must happen after runtime check"),
    )

    with pytest.raises(SystemExit, match="runtime drift"):
        cli.main()


def test_main_uses_configured_gpu_scope_without_global_discovery(monkeypatch):
    observed = {}

    class Runner:
        def __init__(self, gpu_ids, cfg, once=False):
            observed["gpu_ids"] = gpu_ids
            observed["once"] = once

        def run(self):
            observed["ran"] = True

    monkeypatch.setattr(
        "sys.argv", ["orze", "--once", "--no-admin"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: {
        "gpu_scheduling": {"allowed_gpus": [4, 5, 6, 7]},
    })
    monkeypatch.setattr(cli, "_require_controller_runtime", lambda cfg: None)
    monkeypatch.setattr(
        cli, "detect_all_gpus",
        lambda: pytest.fail("must not inventory GPUs outside allowlist"),
    )
    monkeypatch.setattr("orze.engine.orchestrator.Orze", Runner)

    cli.main()

    assert observed == {
        "gpu_ids": [4, 5, 6, 7],
        "once": True,
        "ran": True,
    }


def test_lifecycle_start_runtime_drift_preserves_stop_latch(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    latch = results / ".orze_disabled"
    latch.write_text("operator stop\n", encoding="utf-8")
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_controller_runtime_contract",
        lambda contract: {
            "schema_version": 1,
            "contract_ok": False,
            "errors": ["runtime_package_sha256_drift:orze"],
        },
    )
    monkeypatch.setattr(
        lifecycle, "_read_pid",
        lambda *args: pytest.fail("runtime check must precede process checks"),
    )

    with pytest.raises(
            RuntimeContractError, match="runtime_package_sha256_drift:orze"):
        lifecycle.do_start({
            "results_dir": str(results),
            "controller_runtime": {"contract_version": 1},
        })

    assert latch.read_text(encoding="utf-8") == "operator stop\n"


def test_lifecycle_restart_stops_but_cannot_reenable_under_runtime_drift(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    latch = results / ".orze_disabled"

    def safe_stop(cfg, timeout):
        latch.write_text("restart stop\n", encoding="utf-8")

    monkeypatch.setattr(lifecycle, "do_stop", safe_stop)
    monkeypatch.setattr(
        "orze.service.runtime_contract.audit_controller_runtime_contract",
        lambda contract: {
            "schema_version": 1,
            "contract_ok": False,
            "errors": ["runtime_package_root_drift:orze"],
        },
    )

    with pytest.raises(RuntimeContractError, match="runtime_package_root_drift"):
        lifecycle.do_restart({
            "results_dir": str(results),
            "controller_runtime": {"contract_version": 1},
        })

    assert latch.read_text(encoding="utf-8") == "restart stop\n"


def test_enable_runtime_drift_preserves_persistent_disable_latch(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    latch = results / ".orze_disabled"
    latch.write_text("operator stop\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["orze", "--enable"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: {
        "results_dir": str(results),
        "controller_runtime": {"contract_version": 1},
    })
    monkeypatch.setattr(
        cli, "_require_controller_runtime",
        lambda cfg: (_ for _ in ()).throw(SystemExit("runtime drift")),
    )

    with pytest.raises(SystemExit, match="runtime drift"):
        cli.main()

    assert latch.read_text(encoding="utf-8") == "operator stop\n"
