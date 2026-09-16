"""Read-only verification of a service closure record, never process authority."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from orze.engine.controller_control import ControllerHOLD, current_instance, _head, _path
from orze.engine.controller_handoff import _Route, _validate_history, _schema, _plain, _encode
from orze.engine.controller_session import _row, _file_witness, CompletedControllerStop


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_controller_record(cfg, proof):
    """Validate current closed history without looking up or signalling PIDs."""
    route = _Route(cfg)
    with route.connection() as conn:
        version = conn.execute('PRAGMA data_version').fetchone()[0]
        conn.execute('BEGIN')
        current = current_instance(conn, route.scope)
        head = _head(conn, route.scope)
        if head[2] is not None:
            raise ControllerHOLD('service_closed_pending_handoff')
        if conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='controller_handoffs'").fetchone():
            _schema(conn)
            if conn.execute("SELECT 1 FROM main.controller_handoffs WHERE state!='STARTED' OR hold_reason IS NOT NULL LIMIT 1").fetchone():
                raise ControllerHOLD('service_closed_pending_handoff')
        session = _row(conn, current[0])
        history = _validate_history(route, head[0], head[1], connection=conn)
        identity = json.loads(current[2])
        expected = {'kind': 'stopped', **asdict(CompletedControllerStop(
            current[0], current[4], _sha(session[3].encode()), route.fingerprint,
            str(route.scope), str(route.db), identity['process']))}
        if proof != expected:
            raise ControllerHOLD('service_closed_record_mismatch')
        conn.rollback()
        if conn.execute('PRAGMA data_version').fetchone()[0] != version:
            raise ControllerHOLD('service_closed_history_changed')
    return {'closure': expected, 'history_sha256': history,
            'scope_identity': _plain(route.scope_witness), 'database_identity': _plain(route.db_witness)}


def read_closed(service_config):
    """Informational record only; it does not establish that a host is dead."""
    from orze.core.config import load_project_config, find_dotenv
    from orze.service.host import _service, state_directory, _read
    from orze.service.runtime_contract import _runtime_errors, capture_runtime_packages
    path, witness, raw, svc = _service(service_config)
    if str(Path.cwd()) != svc['workdir'] or svc['python'] != sys.executable:
        raise ControllerHOLD('service_closed_runtime_mismatch')
    if _runtime_errors(svc.get('runtime_packages'), capture_runtime_packages()):
        raise ControllerHOLD('service_closed_runtime_changed')
    state = state_directory(path)
    directory = _path(state, directory=True)
    witnesses, documents = {}, {}
    for name in ('owner.json', 'boot.json', 'ready.json', 'closed.json', 'lock.json'):
        witnesses[name], _, documents[name] = _read(state / name)
    marker = state.with_name(state.name + '.source-lock')
    marker_witness, marker_raw = _file_witness(marker, 4096)
    if marker_raw != b'orze-idea-source-lock-v1\n':
        raise ControllerHOLD('service_closed_owner_changed')
    owner, boot, lock = (documents[n] for n in ('owner.json', 'boot.json', 'lock.json'))
    if (type(owner.get('schema')) is not int or owner.get('schema') != 1
            or type(boot.get('schema')) is not int or owner.get('service_sha256') != _sha(raw)
            or owner.get('service_config_file') != str(path)
            or boot != {'schema': 1, 'service_sha256': _sha(raw)}
            or owner.get('nonce') != lock.get('owner_nonce')
            or owner.get('process', {}).get('pid') != lock.get('pid')
            or lock.get('protocol') != 'orze-idea-source-lock-v1'):
        raise ControllerHOLD('service_closed_owner_changed')
    config_witness = _file_witness(Path(svc['config_file']), 65536)[0]
    dotenv = find_dotenv(svc['config_file'])
    env_witness = None if dotenv is None else _file_witness(dotenv, 65536)[0]
    cfg = load_project_config(svc['config_file'])
    if str(Path(cfg['results_dir']).resolve()) != svc['results_dir']:
        raise ControllerHOLD('service_closed_scope_changed')
    verified = verify_controller_record(cfg, documents['closed.json'])
    route = _Route(cfg)
    with route.connection() as conn:
        initial = conn.execute('SELECT controller_id FROM controller_instances WHERE scope=? AND generation=0',
                               (str(route.scope),)).fetchall()
        if initial != [(documents['ready.json'].get('controller_id'),)]:
            raise ControllerHOLD('service_closed_initial_controller_changed')
    if (_path(state, directory=True) != directory or _file_witness(path, 65536)[0] != witness
            or _file_witness(marker, 4096)[0] != marker_witness
            or _file_witness(Path(svc['config_file']), 65536)[0] != config_witness
            or find_dotenv(svc['config_file']) != dotenv
            or dotenv is not None and _file_witness(dotenv, 65536)[0] != env_witness
            or any(_file_witness(state / name, 65536)[0] != saved for name, saved in witnesses.items())):
        raise ControllerHOLD('service_closed_inputs_changed')
    snapshot = {'service': witness, 'state': directory, 'documents': witnesses,
                'marker': marker_witness, 'config': config_witness, 'dotenv': env_witness,
                'verified': verified}
    return {'kind': 'closed_record', 'closure': verified['closure'],
            'snapshot_sha256': _sha(_encode(_plain(snapshot)).encode())}


def inspect_closed(service_config):
    """Load the project in a separate interpreter at its declared workdir."""
    from orze.service.host import _service
    path, witness, _, svc = _service(service_config)
    result = subprocess.run([sys.executable, '-m', 'orze.service.closed_state', '--service-config', str(path)],
                            cwd=svc['workdir'], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    if (result.returncode != 0 or len(result.stdout.encode()) > 65536
            or _file_witness(path, 65536)[0] != witness):
        raise ControllerHOLD('service_closed_state_unconfirmed')
    proof = json.loads(result.stdout)
    if (type(proof) is not dict or set(proof) != {'kind', 'closure', 'snapshot_sha256'}
            or proof.get('kind') != 'closed_record' or type(proof['closure']) is not dict
            or proof['closure'].get('kind') != 'stopped'
            or type(proof['snapshot_sha256']) is not str or len(proof['snapshot_sha256']) != 64):
        raise ControllerHOLD('service_closed_state_unconfirmed')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-config', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(read_closed(args.service_config), sort_keys=True))
        return 0
    except Exception:
        print('HOLD: service_closed_state_unconfirmed', file=sys.stderr)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
