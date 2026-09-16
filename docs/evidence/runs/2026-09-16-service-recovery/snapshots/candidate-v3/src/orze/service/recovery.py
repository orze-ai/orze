"""Explicit new host for a completely closed, unchanged local CPU service.

Old closure records remain informational. This operation additionally requires
an explicit pinned recovery instruction, current closed history and absence of
both exact old processes. It never signals or adopts a historical PID, removes
an ownership record, repairs incomplete history or resets a resource budget.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from orze.engine.controller_control import ControllerHOLD, current_instance, _head, _path, _token
from orze.engine.controller_handoff import (
    _Coordinator, _Route, _history_observer, _plain, _process, _run_coordinator,
)
from orze.engine.controller_session import _file_witness, _row
from orze.engine.supervisor_worker import process_identity


def _absent(process, hostname, boot_id):
    """A negative observation, never a process-control handle or permission."""
    process = _process(process)
    if hostname != socket.gethostname():
        raise ControllerHOLD('service_recovery_other_host')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot_id != boot:
        return True
    try:
        observed = process_identity(process['pid'])[0]
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return True
        raise ControllerHOLD('service_recovery_process_unknown') from exc
    return observed != process


def _inactive(svc):
    if svc['method'] != 'systemd':
        return
    from orze.service.scoped import _require_inactive, unit_names
    # A wholly uninstalled source is also acceptable. A partially changed set
    # remains unavailable; the operator can finish its explicit uninstall.
    units = unit_names(svc)
    properties = []
    for unit in units:
        keys = ['Id', 'LoadState', 'ActiveState', 'UnitFileState']
        if unit != units[1]:
            keys += ['MainPID', 'ControlGroup']
        result = subprocess.run(['systemctl', '--user', 'show', unit, '--all', '--property=' + ','.join(keys)],
                                capture_output=True, text=True, timeout=10)
        values = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition('=')
            if not separator or key in values:
                raise ControllerHOLD('service_recovery_manager_state_unknown')
            values[key] = value
        if result.returncode != 0 or not set(keys) <= values.keys():
            raise ControllerHOLD('service_recovery_manager_state_unknown')
        properties.append(values)
    if all(p.get('Id') == unit and p.get('LoadState') == 'not-found'
           and p.get('ActiveState') == 'inactive'
           and (unit == units[1] or p.get('MainPID') == '0' and p.get('ControlGroup') == '')
           for unit, p in zip(units, properties)):
        return
    if not all(p['LoadState'] == 'loaded' and p['UnitFileState'] in {'disabled', 'static'}
               for p in properties):
        raise ControllerHOLD('service_recovery_source_units_enabled_or_changed')
    _require_inactive(svc)


class _ClosedSource:
    def __init__(self, cfg, source, *, declaration=None):
        from orze.core.cpu_execution import cpu_handoff_profile
        from orze.service.closed_state import read_closed
        from orze.service.host import _service, _read, state_directory, _sha
        if not cpu_handoff_profile(cfg):
            raise ControllerHOLD('service_recovery_cpu_handoff_required')
        self.route = _Route(cfg)
        path, _, raw, self.svc = _service(source)
        self.proof = read_closed(path)
        if (self.svc['config_file'] != str(self.route.config_file)
                or self.svc['results_dir'] != str(self.route.scope)):
            raise ControllerHOLD('service_recovery_source_scope_changed')
        state = state_directory(path)
        self.state_identity = _path(state, directory=True)
        self.files = {p: _file_witness(p, 65536)[0] for p in
                      [path, *[state / n for n in ('owner.json', 'boot.json', 'ready.json', 'closed.json', 'lock.json')],
                       state.with_name(state.name + '.source-lock')]}
        self.owner = _read(state / 'owner.json')[2]
        with self.route.connection() as conn:
            current = current_instance(conn, self.route.scope)
            self.session = _row(conn, current[0])
            self.historical = _history_observer(self.route, current[2], self.session[1])
            self.registration = self.historical.check(conn)[0]
            if _head(conn, self.route.scope)[2] is not None:
                raise ControllerHOLD('service_recovery_pending_grant')
        self.identity = self.historical.identity
        self.identity_json = self.historical.identity_json
        self.controller_id = self.historical.controller_id
        self.source = {'version': 1, 'source_service_config': str(path),
                       'source_service_sha256': _sha(raw), 'source_snapshot_sha256': self.proof['snapshot_sha256'],
                       'source_controller_id': self.controller_id}
        if declaration is not None:
            expected = {**self.source, 'request_id': declaration.get('request_id')}
            if declaration != expected or type(declaration.get('version')) is not int:
                raise ControllerHOLD('service_recovery_source_changed')
            _token(declaration['request_id'], 'service_recovery_request_id_invalid')
        self.check()

    def exited(self):
        self.route.check()
        if (_path(self.state_identity[0], directory=True) != self.state_identity
                or any(_file_witness(path, 65536)[0] != witness for path, witness in self.files.items())):
            raise ControllerHOLD('service_recovery_source_files_changed')
        for name in ('.orze_disabled', '.orze_stop_all', '.orze_shutdown'):
            path = self.route.scope / name
            if path.exists() or path.is_symlink():
                raise ControllerHOLD('service_recovery_stop_latch')
        _inactive(self.svc)
        return all(_absent(process, self.identity['host'], self.identity['boot_id'])
                   for process in (self.owner['process'], self.identity['process']))

    def check(self, conn=None):
        if not self.exited():
            raise ControllerHOLD('service_recovery_source_alive')
        if conn is None:
            with self.route.connection() as connection:
                return self.check(connection)
        observed = self.historical.check(conn)
        if observed != (self.registration, self.session):
            raise ControllerHOLD('service_recovery_source_history_changed')
        return observed

    def close(self):
        self.historical.close()


class _RecoveryCoordinator(_Coordinator):
    def __init__(self, host):
        self.host = host
        self.declaration = host.svc['recovery']
        super().__init__(host.cfg, self.declaration['request_id'], 60, host.children)

    def prepare_source(self):
        self.observer = _ClosedSource(self.route.cfg, self.declaration['source_service_config'],
                                      declaration=self.declaration)

    def check(self, **kwargs):
        super().check(**kwargs)
        self.host._check_owner()
        self.host._check_inputs()


def launch(host):
    """Only a new host invokes this; it retains the actual successor handles."""
    return _run_coordinator(_RecoveryCoordinator(host))


def prepare(source, destination, request_id):
    from orze.core.config import load_project_config
    from orze.service.host import _service, _create, state_directory
    request_id = _token(request_id, 'service_recovery_request_id_invalid')
    path, _, _, svc = _service(source)
    destination = Path(destination).absolute()
    if (destination == path or destination.exists() or destination.is_symlink()
            or state_directory(destination).exists()
            or Path(svc['results_dir']) in destination.parents):
        raise ControllerHOLD('service_recovery_destination_unavailable')
    if str(Path.cwd()) != svc['workdir']:
        raise ControllerHOLD('service_recovery_workdir_changed')
    cfg = load_project_config(svc['config_file'])
    closed = _ClosedSource(cfg, path)
    try:
        declaration = {**closed.source, 'request_id': request_id}
        result = {**svc, 'service_config_file': str(destination), 'recovery': declaration}
        closed.check()
        _create(destination, result)
        return {'kind': 'prepared_recovery', 'service_config': str(destination), 'recovery': declaration}
    finally:
        closed.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-service-config', required=True)
    parser.add_argument('--service-config', required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--install', action='store_true', help='Also install/start the new systemd service')
    args = parser.parse_args(argv)
    from orze.service.host import _service
    try:
        source = Path(args.source_service_config).absolute()
        destination = Path(args.service_config).absolute()
        _, _, _, svc = _service(source)
        if args.install and svc['method'] != 'systemd':
            raise ControllerHOLD('service_recovery_install_requires_systemd')
        os.chdir(svc['workdir'])
        result = prepare(source, destination, args.request_id)
        if args.install:
            from orze.service.install import _SYSTEMD_DIR
            from orze.service.scoped import install_units
            _, _, _, prepared = _service(destination)
            install_units(prepared, _SYSTEMD_DIR)
            result['kind'] = 'installed_recovery'
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        print('HOLD: service_recovery_unconfirmed', file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
