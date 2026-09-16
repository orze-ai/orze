"""One explicit configuration selects every hosted service management target."""
import hashlib
from pathlib import Path
import re
import subprocess
import time


def unit_names(svc):
    if "service_owner" not in svc:
        return "orze.service", "orze-watchdog.timer", "orze-watchdog.service"
    from orze.service.host import PROFILE
    declaration = svc["service_owner"]
    path = svc.get("service_config_file")
    if (type(declaration) is not dict or declaration != PROFILE
            or type(declaration.get("version")) is not int
            or type(path) is not str or not Path(path).is_absolute()):
        raise ValueError("service_scope_invalid")
    name = "orze-" + hashlib.sha256(path.encode()).hexdigest()[:32]
    return name + ".service", name + "-watchdog.timer", name + "-watchdog.service"


def selected(path):
    from orze.service.host import _service
    return _service(path)[3]


def _quote(value, *, command=False):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("service_unit_value_invalid")
    if re.fullmatch(r"[-A-Za-z0-9_./:]+", value):
        return value
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    value = "".join(f"\\x{ord(char):02x}" if ord(char) < 32 or ord(char) == 127 else char for char in value)
    return '"' + value + '"'


def _command(*values):
    return " ".join(_quote(value, command=True) for value in values)


def render_units(svc):
    main, timer, watchdog = unit_names(svc)
    if "service_owner" not in svc or svc.get("method") != "systemd":
        raise ValueError("service_host_systemd_required")
    python, config = svc["python"], svc["service_config_file"]
    from orze.service.runtime_contract import _RUNTIME_ENVIRONMENT_KEYS
    unset = " ".join(sorted(_RUNTIME_ENVIRONMENT_KEYS))
    common = ("WorkingDirectory=" + _quote(svc["workdir"]) + "\n"
              + "UnsetEnvironment=" + unset + "\n"
              + "StandardOutput=" + _quote("append:" + svc["log_file"]) + "\n"
              + "StandardError=" + _quote("append:" + svc["log_file"]) + "\n")
    return {
        main: "[Unit]\nDescription=Orze project lifecycle owner\nAfter=network.target\n\n[Service]\n"
              + "Type=simple\nKillMode=control-group\nRestart=no\nTimeoutStopSec=75\n" + common
              + "ExecStartPre=" + _command(python, "-m", "orze.service.runtime_contract", "--startup-check", "--service-config", config) + "\n"
              + "ExecStart=" + _command(python, "-m", "orze.service.host", "--service-config", config)
              + "\n\n[Install]\nWantedBy=default.target\n",
        watchdog: "[Unit]\nDescription=Orze project watchdog client\n\n[Service]\nType=oneshot\nKillMode=control-group\nRestart=no\n" + common
                  + "ExecStart=" + _command(python, "-m", "orze.service.watchdog", "--service-config", config) + "\n",
        timer: "[Unit]\nDescription=Orze project watchdog timer\n\n[Timer]\nOnBootSec=60\nOnUnitActiveSec=300\n"
               + "Unit=" + watchdog + "\n\n[Install]\nWantedBy=timers.target\n",
    }


def _checked(*args):
    result = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        raise RuntimeError("service_manager_operation_unconfirmed")
    return result


def _wait_ready(svc):
    from orze.service.host import request
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            return request(svc["service_config_file"], "status", timeout=min(3, deadline - time.monotonic()))
        except Exception:
            time.sleep(.1)
    raise RuntimeError("service_host_readiness_unconfirmed")


def _require_contract(svc, *, starting=False):
    from orze.service.runtime_contract import audit_runtime_contract
    report = audit_runtime_contract(svc)
    if starting:
        allowed = report.get("startup_allowed") is True
    else:
        # Existing operator latches deny startup, not verified shutdown.
        errors = set(report.get("errors", ())) - {
            "latched_systemd_unit_active", "latched_systemd_unit_enabled"}
        allowed = not errors and (report.get("contract_ok") is True or bool(report.get("errors")))
    if not allowed:
        raise RuntimeError("service_host_runtime_rejected")


def install_units(svc, directory):
    units = render_units(svc)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() or (directory / name).is_symlink() for name in units):
        raise RuntimeError("service_unit_already_present")
    for name, value in units.items():
        with (directory / name).open("x") as stream:
            stream.write(value)
    _checked("daemon-reload")
    _require_contract(svc, starting=True)
    main, timer, watchdog = unit_names(svc)
    # Initial bootstrap remains owned by this persistent unit. The timer only
    # requests a handoff from it; it cannot launch another initial controller.
    try:
        _checked("enable", "--now", main)
        _wait_ready(svc)
        _require_contract(svc, starting=True)
        _checked("enable", "--now", timer)
    except Exception:
        # A manager error alone is not proof that its child never started.
        # Close the verified host before disabling this project's units. If
        # closure is unknown, preserve the owner and report the uncertainty.
        from orze.service.host import request
        proof = request(svc["service_config_file"], "stop")
        if proof.get("kind") != "stopped":
            raise RuntimeError("service_host_rollback_unconfirmed")
        for unit in (timer, main):
            _require_contract(svc)
            _checked("disable", "--now", unit)
        raise


def uninstall_units(svc, directory):
    units = render_units(svc)
    directory = Path(directory)
    from orze.engine.controller_session import _file_witness
    witnesses = {}
    for name, value in units.items():
        path = directory / name
        witness, raw = _file_witness(path, 65536)
        if raw != value.encode():
            raise RuntimeError("service_unit_changed")
        witnesses[name] = witness
    def check_files():
        if any(_file_witness(directory / name, 65536)[0] != witness for name, witness in witnesses.items()):
            raise RuntimeError("service_unit_changed")
    _require_contract(svc)
    from orze.service.host import request
    proof = request(svc["service_config_file"], "stop")
    if proof.get("kind") != "stopped":
        raise RuntimeError("service_host_stop_unconfirmed")
    for name in unit_names(svc):
        check_files()
        _require_contract(svc)
        _checked("disable", "--now", name)
    check_files()
    for name, value in units.items():
        path = directory / name
        if path.is_symlink() or path.read_text() != value:
            raise RuntimeError("service_unit_changed")
        path.unlink()
    _checked("daemon-reload")
    # Keep configuration, boot intention, closed proof and research state.
    # Removing a service is not permission to clear controller ownership.
