"""Direct launches enforce controller identity before GPU discovery."""

import pytest

import orze.cli as cli


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
