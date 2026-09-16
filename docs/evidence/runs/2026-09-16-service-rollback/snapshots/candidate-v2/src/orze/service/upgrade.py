"""Explicit runtime transition for a closed CPU service in its existing scope.

The supported adapter preserves the current local CPU handoff state format.
It checks the old frozen packages, a pinned closed-service backup, and the
current closed history. The existing one-shot database handoff transaction
grants the only successor. No schema conversion, copied scope or PID adoption.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

from orze.engine.controller_control import ControllerHOLD, _path, _token
from orze.engine.controller_handoff import _Coordinator, _plain, _run_coordinator
from orze.engine.controller_session import _file_witness
from orze.service.recovery import _ClosedSource, _RecoveryCoordinator

ADAPTER = 'local_cpu_state_v1'


def _declaration(value):
    if (type(value) is not dict or set(value) != {'version', 'compatibility', 'source', 'backup', 'manifest_sha256'}
            or type(value['version']) is not int or value['version'] != 1 or value['compatibility'] != ADAPTER
            or type(value['source']) is not dict or set(value['source']) != {
                'version', 'source_service_config', 'source_service_sha256', 'source_snapshot_sha256',
                'source_controller_id', 'request_id'}
            or type(value['source']['version']) is not int or value['source']['version'] != 1
            or type(value['backup']) is not str or not Path(value['backup']).is_absolute()
            or type(value['manifest_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', value['manifest_sha256']) is None):
        raise ControllerHOLD('service_upgrade_declaration_invalid')
    _token(value['source']['request_id'], 'service_upgrade_request_id_invalid')
    return value


def _recorded_runtime(svc):
    """Hash the retained source tree without importing it into the new runtime."""
    from orze.service.runtime_contract import _hash_package_tree
    packages = svc.get('runtime_packages')
    if type(packages) is not list or not 1 <= len(packages) <= 2 or svc['python'] != sys.executable:
        raise ControllerHOLD('service_upgrade_source_runtime_invalid')
    names, current, cli = set(), [], None
    for record in packages:
        if (type(record) is not dict or set(record) != {'name', 'root', 'sha256', 'file_count'}
                or type(record['name']) is not str or record['name'] not in {'orze', 'orze_pro'}
                or record['name'] in names or type(record['root']) is not str
                or not Path(record['root']).is_absolute() or type(record['file_count']) is not int
                or record['file_count'] <= 0 or type(record['sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}', record['sha256']) is None):
            raise ControllerHOLD('service_upgrade_source_runtime_invalid')
        names.add(record['name'])
        root, _ = _path(Path(record['root']), directory=True)
        digest, count = _hash_package_tree(root)
        observed = {'name': record['name'], 'root': str(root), 'sha256': digest, 'file_count': count}
        if observed != record:
            raise ControllerHOLD('service_upgrade_source_runtime_changed')
        current.append(observed)
        if record['name'] == 'orze':
            cli = _file_witness(root/'cli.py', 1048576)[0]
    if cli is None:
        raise ControllerHOLD('service_upgrade_source_core_missing')
    binary = _path(Path(sys.executable).resolve(strict=True), directory=False)
    return current, _plain({'executable': sys.executable, 'binary': binary, 'cli': cli})


class _CompatibleSource(_ClosedSource):
    def __init__(self, cfg, source, backup, digest, *, declaration=None):
        self.backup, self.digest = Path(backup).absolute(), digest
        self.recorded = None
        super().__init__(cfg, source, declaration=declaration)

    def _read_closed(self, path):
        from orze.service.backup import _verify
        from orze.service.closed_state import _read_closed
        packages, runtime = _recorded_runtime(self.svc)
        proof = _read_closed(path, packages, historical_runtime=runtime)
        manifest = _verify(self.backup, self.digest)
        source = manifest['source']
        if (type(source) is not dict or set(source) != {
                'service', 'database', 'controller_id', 'closure', 'runtime_packages', 'absent_sources'}
                or source['service'] != str(path) or source['database'] != str(self.route.db)
                or source['controller_id'] != proof['closure']['controller_id']
                or source['closure'] != proof or source['runtime_packages'] != packages):
            raise ControllerHOLD('service_upgrade_backup_source_changed')
        self.recorded = packages, runtime
        if _recorded_runtime(self.svc) != self.recorded:
            raise ControllerHOLD('service_upgrade_source_runtime_changed')
        return proof

    def exited(self):
        from orze.service.backup import _verify
        if _recorded_runtime(self.svc) != self.recorded:
            raise ControllerHOLD('service_upgrade_source_runtime_changed')
        _verify(self.backup, self.digest)
        return super().exited()


class _UpgradeCoordinator(_RecoveryCoordinator):
    def __init__(self, host):
        self.host = host
        self.upgrade = _declaration(host.svc['upgrade'])
        self.declaration = self.upgrade['source']
        _Coordinator.__init__(self, host.cfg, self.declaration['request_id'], 60, host.children)

    def prepare_source(self):
        self.observer = _CompatibleSource(self.route.cfg, self.declaration['source_service_config'],
            self.upgrade['backup'], self.upgrade['manifest_sha256'], declaration=self.declaration)


def launch(host):
    return _run_coordinator(_UpgradeCoordinator(host))


def prepare(source, destination, request_id, backup, digest):
    from orze.core.config import load_project_config
    from orze.service.host import _service, _create, state_directory
    from orze.service.runtime_contract import capture_runtime_packages, require_controller_runtime_contract
    request_id = _token(request_id, 'service_upgrade_request_id_invalid')
    path, _, _, svc = _service(source)
    destination = Path(destination).absolute()
    if (destination == path or destination.exists() or destination.is_symlink()
            or state_directory(destination).exists() or Path(svc['results_dir']) in destination.parents):
        raise ControllerHOLD('service_upgrade_destination_unavailable')
    if str(Path.cwd()) != svc['workdir']:
        raise ControllerHOLD('service_upgrade_workdir_changed')
    cfg = load_project_config(svc['config_file'])
    require_controller_runtime_contract(cfg.get('controller_runtime'))
    closed = _CompatibleSource(cfg, path, backup, digest)
    try:
        declaration = _declaration({'version': 1, 'compatibility': ADAPTER,
            'source': {**closed.source, 'request_id': request_id},
            'backup': str(Path(backup).absolute()), 'manifest_sha256': digest})
        result = {**svc, 'service_config_file': str(destination), 'upgrade': declaration,
                  'runtime_packages': capture_runtime_packages()}
        result.pop('recovery', None)
        closed.check()
        _create(destination, result)
        return {'kind': 'prepared_upgrade', 'service_config': str(destination), 'upgrade': declaration}
    finally:
        closed.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-service-config', required=True)
    parser.add_argument('--service-config', required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--backup', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--install', action='store_true', help='Also install/start the new systemd service')
    args = parser.parse_args(argv)
    from orze.service.host import _service
    previous = Path.cwd()
    try:
        source, destination, backup = (Path(p).absolute() for p in
                                      (args.source_service_config, args.service_config, args.backup))
        _, _, _, svc = _service(source)
        if args.install and svc['method'] != 'systemd':
            raise ControllerHOLD('service_upgrade_install_requires_systemd')
        os.chdir(svc['workdir'])
        result = prepare(source, destination, args.request_id, backup, args.manifest_sha256)
        if args.install:
            from orze.service.install import _SYSTEMD_DIR
            from orze.service.scoped import install_units
            _, _, _, prepared = _service(destination)
            install_units(prepared, _SYSTEMD_DIR)
            result['kind'] = 'installed_upgrade'
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        print('HOLD: service_upgrade_unconfirmed', file=sys.stderr)
        return 75
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    raise SystemExit(main())
