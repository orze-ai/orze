"""Consistent closed-service data backups; copied records never grant execution.

Capture serializes against a new controller through the existing database
writer gate. Verification and restoration only read the pinned backup, never
contact a saved process, rewrite an execution scope, or replenish a budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import sys

from orze.engine.controller_control import ControllerHOLD
from orze.engine.artifact_publication import _open_directory
from orze.engine.gc_tree import identity, plain_directory

MAX_ENTRIES = 16384
MAX_MANIFEST = 16 * 1024 * 1024
DEFAULT_BYTES = 4 * 1024 ** 3
CHUNK = 1024 * 1024


def _encoded(value):
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(data) > MAX_MANIFEST:
        raise ControllerHOLD('service_backup_manifest_limit')
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _limit(value):
    if type(value) is not int or not 0 < value <= 1024 ** 5:
        raise ControllerHOLD('service_backup_capacity_invalid')
    return value


def _write(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError('service_backup_short_write')
        view = view[count:]


def _publish(parent, name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    try:
        _write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(parent)


class _Output:
    def __init__(self, path):
        self.path = Path(path).absolute()
        plain_directory(self.path.parent)
        self.route = [(p, identity(p.lstat())[:3]) for p in self.path.parents]
        parent = _open_directory(self.path.parent)
        try:
            os.mkdir(self.path.name, 0o700, dir_fd=parent)
            os.fsync(parent)
            self.fd = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        self.ident = identity(os.fstat(self.fd))[:3]
        try:
            self.check()
        except BaseException:
            os.close(self.fd)
            raise

    def check(self):
        if (identity(self.path.lstat())[:3] != self.ident
                or any(identity(p.lstat())[:3] != saved for p, saved in self.route)):
            raise ControllerHOLD('service_backup_destination_changed')

    def publish(self, name, value):
        self.check()
        _publish(self.fd, name, _encoded(value))
        self.check()

    def close(self):
        os.close(self.fd)


def _roots(cfg, service):
    from orze.core.config import find_dotenv
    from orze.service.host import state_directory
    paths = [Path.cwd(), Path(cfg['results_dir']), Path(cfg['idea_lake_db']),
             Path(cfg['_config_path']), Path(service), state_directory(service),
             state_directory(service).with_name(state_directory(service).name + '.source-lock')]
    paths.extend(Path(cfg[key]).absolute() for key in ('_orze_dir', 'ideas_file', 'goal_file') if cfg.get(key))
    dotenv = find_dotenv(cfg['_config_path'])
    if dotenv is not None:
        paths.append(Path(dotenv).absolute())
    selected, missing = [], []
    for path in sorted(set(p.absolute() for p in paths), key=lambda p: (len(p.parts), str(p))):
        if any(path == prior or prior in path.parents for prior in selected):
            continue
        plain_directory(path.parent, missing=True)
        try:
            path.lstat()
        except FileNotFoundError:
            missing.append(path)
        else:
            selected.append(path)
    return selected, missing


def _require_absent(paths):
    for path in paths:
        plain_directory(path.parent, missing=True)
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise ControllerHOLD('service_backup_missing_source_appeared')


def _scan(roots, database, max_bytes):
    entries, total = [], 0
    omitted = ({Path(str(database) + suffix) for suffix in ('-wal', '-shm', '-journal')}
               if database is not None else set())

    def visit(root, path, relative, depth):
        nonlocal total
        if path in omitted:
            return
        if depth > 64 or len(entries) >= MAX_ENTRIES:
            raise ControllerHOLD('service_backup_entry_limit')
        info = path.lstat()
        row = {'root': root, 'path': relative, 'mode': stat.S_IMODE(info.st_mode) & 0o777,
               'identity': list(identity(info))}
        if stat.S_ISDIR(info.st_mode):
            row['kind'] = 'directory'
            entries.append(row)
            fd = _open_directory(path)
            try:
                if identity(os.fstat(fd)) != identity(info):
                    raise ControllerHOLD('service_backup_source_changed')
                names = []
                with os.scandir(fd) as iterator:
                    for item in iterator:
                        if len(names) + len(entries) >= MAX_ENTRIES:
                            raise ControllerHOLD('service_backup_entry_limit')
                        names.append(item.name)
                for name in sorted(names):
                    visit(root, path/name, name if relative == '.' else relative+'/'+name, depth+1)
                if identity(os.fstat(fd)) != identity(info):
                    raise ControllerHOLD('service_backup_source_changed')
            finally:
                os.close(fd)
        elif stat.S_ISLNK(info.st_mode):
            row.update(kind='symlink', target=os.readlink(path))
            entries.append(row)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            row['kind'] = 'database' if path == database else 'file'
            # SQLite's consistent image replaces the main file and sidecars.
            if path == database:
                row['identity'] = row['identity'][:3]
            else:
                total += info.st_size
            if total > max_bytes:
                raise ControllerHOLD('service_backup_byte_limit')
            entries.append(row)
        else:
            raise ControllerHOLD('service_backup_special_or_linked_file')
    for number, path in enumerate(roots):
        visit(f'root-{number:03}', path, '.', 0)
    return entries


def _stream(source, destination_fd, expected=None, capacity=DEFAULT_BYTES):
    plain_directory(Path(source).parent)
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        captured = identity(info)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or expected is not None and list(captured) != expected):
            raise ControllerHOLD('service_backup_file_changed')
        digest, size = hashlib.sha256(), 0
        while True:
            data = os.read(fd, min(CHUNK, capacity - size + 1))
            if not data:
                break
            size += len(data)
            if size > capacity:
                raise ControllerHOLD('service_backup_byte_limit')
            digest.update(data)
            if destination_fd is not None:
                _write(destination_fd, data)
        if captured != identity(os.fstat(fd)) or captured != identity(Path(source).lstat()):
            raise ControllerHOLD('service_backup_file_changed')
        if destination_fd is not None:
            os.fsync(destination_fd)
        return {'bytes':size, 'sha256':digest.hexdigest()}
    finally:
        os.close(fd)


def capture(service_config, destination, *, max_bytes=DEFAULT_BYTES):
    from orze.core.config import load_project_config
    from orze.service.recovery import _ClosedSource
    from orze.service.host import _service
    max_bytes = _limit(max_bytes)
    service, _, _, svc = _service(Path(service_config).absolute())
    if str(Path.cwd()) != svc['workdir']:
        raise ControllerHOLD('service_backup_workdir_changed')
    cfg = load_project_config(svc['config_file'])
    closed = _ClosedSource(cfg, service)
    output, objects = None, None
    try:
        roots, missing = _roots(cfg, service)
        destination = Path(destination).absolute()
        if any(destination == p or p in destination.parents or destination in p.parents for p in [*roots, *missing]):
            raise ControllerHOLD('service_backup_destination_overlaps_source')
        # No source mutation is committed. The writer gate also prevents a
        # competing recovery grant from becoming current during the copy.
        with closed.route.connection(write=True) as writer:
            closed.check(writer)
            reader = sqlite3.connect(closed.route.db.as_uri()+'?mode=ro', uri=True, timeout=.25)
            try:
                reader.execute('BEGIN')
                reader.execute('SELECT count(*) FROM main.sqlite_master').fetchone()
                initial = _scan(roots, closed.route.db, max_bytes)
                output = _Output(destination)
                output.publish('intent.json', {'version':1, 'kind':'closed_service_backup', 'source':str(service)})
                os.mkdir('objects', 0o700, dir_fd=output.fd)
                objects = os.open('objects', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=output.fd)
                entries, total = [], 0
                for index, captured in enumerate(initial):
                    row = dict(captured)
                    if row['kind'] in {'file', 'database'}:
                        name = f'{index:06}'
                        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=objects)
                        source = roots[int(row['root'][5:])] / row['path']
                        try:
                            output.check()
                            if row['kind'] == 'database':
                                path = destination/'objects'/name
                                image = sqlite3.connect(path)
                                try:
                                    if identity(path.lstat())[:3] != identity(os.fstat(fd))[:3]:
                                        raise ControllerHOLD('service_backup_destination_changed')
                                    reader.backup(image)
                                    if image.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                                        raise ControllerHOLD('service_backup_database_invalid')
                                finally:
                                    image.close()
                                os.fsync(fd)
                                pinned = _stream(path, None, list(identity(os.fstat(fd))), max_bytes-total)
                            else:
                                pinned = _stream(source, fd, row['identity'], max_bytes-total)
                            row.update(object=name, **pinned)
                            total += pinned['bytes']
                        finally:
                            os.close(fd)
                    entries.append(row)
                os.fsync(objects)
                if initial != _scan(roots, closed.route.db, max_bytes):
                    raise ControllerHOLD('service_backup_source_changed')
                _require_absent(missing)
                closed.check(writer)
                manifest = {'version':1, 'kind':'closed_service_backup', 'activation':'requires_separate_migration',
                    'source':{'service':str(service), 'database':str(closed.route.db),
                              'controller_id':closed.controller_id, 'closure':closed.proof,
                              'runtime_packages':svc['runtime_packages'],
                              'absent_sources':[str(p) for p in missing]},
                    'roots':[{'name':f'root-{i:03}', 'source':str(p)} for i,p in enumerate(roots)],
                    'entries':entries, 'total_bytes':total}
                output.publish('manifest.json', manifest)
                digest = _sha(_encoded(manifest))
                _verify(destination, digest, require_complete=False)
                _require_absent(missing)
                closed.check(writer)
                output.publish('complete.json', {'version':1, 'manifest_sha256':digest})
                return {'kind':'closed_service_backup', 'backup':str(destination),
                        'manifest_sha256':digest, 'files':len(entries), 'bytes':total, 'activation':False}
            finally:
                reader.close()
    finally:
        if objects is not None:
            os.close(objects)
        if output is not None:
            output.close()
        closed.close()


def _document(path):
    from orze.engine.controller_session import _file_witness
    return _file_witness(path, MAX_MANIFEST)[1]


def _verify(path, expected, *, require_complete=True):
    path = Path(path).absolute()
    plain_directory(path)
    if type(expected) is not str or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ControllerHOLD('service_backup_digest_required')
    raw = _document(path/'manifest.json')
    if _sha(raw) != expected:
        raise ControllerHOLD('service_backup_manifest_changed')
    manifest = json.loads(raw)
    if (type(manifest) is not dict or set(manifest) != {'version','kind','activation','source','roots','entries','total_bytes'}
            or type(manifest['version']) is not int or manifest['version'] != 1
            or manifest['kind'] != 'closed_service_backup' or manifest['activation'] != 'requires_separate_migration'
            or type(manifest['entries']) is not list or not 0 < len(manifest['entries']) <= MAX_ENTRIES
            or type(manifest['roots']) is not list or not 0 < len(manifest['roots']) <= MAX_ENTRIES
            or type(manifest['total_bytes']) is not int or manifest['total_bytes'] < 0):
        raise ControllerHOLD('service_backup_manifest_invalid')
    if require_complete:
        complete = json.loads(_document(path/'complete.json'))
        if (type(complete) is not dict or type(complete.get('version')) is not int
                or complete != {'version':1, 'manifest_sha256':expected}):
            raise ControllerHOLD('service_backup_incomplete')
    expected_names = {'intent.json','manifest.json','objects'} | ({'complete.json'} if require_complete else set())
    if set(p.name for p in path.iterdir()) != expected_names:
        raise ControllerHOLD('service_backup_contents_changed')
    roots = {f'root-{i:03}' for i in range(len(manifest['roots']))}
    if any(type(row) is not dict or set(row) != {'name','source'} or row['name'] != f'root-{i:03}'
           or type(row['source']) is not str for i,row in enumerate(manifest['roots'])):
        raise ControllerHOLD('service_backup_roots_invalid')
    seen, objects, total, databases, previous = {}, set(), 0, 0, None
    plain_directory(path/'objects')
    for row in manifest['entries']:
        if type(row) is not dict or row.get('root') not in roots or type(row.get('path')) is not str:
            raise ControllerHOLD('service_backup_entry_invalid')
        rel = PurePosixPath(row['path'])
        key = (row['root'], row['path'])
        order = (row['root'], *rel.parts)
        if (rel.is_absolute() or '..' in rel.parts or str(rel) != row['path'] or '\0' in row['path']
                or key in seen or type(row.get('mode')) is not int or not 0 <= row['mode'] <= 0o777
                or row.get('kind') not in {'directory','file','database','symlink'}
                or row['path'] != '.' and seen.get((row['root'], str(rel.parent))) != 'directory'
                or previous is not None and order <= previous):
            raise ControllerHOLD('service_backup_entry_invalid')
        previous = order
        seen[key] = row['kind']
        base = {'root','path','kind','mode','identity'}
        if row['kind'] in {'file','database'}:
            if (set(row) != base | {'object','bytes','sha256'} or type(row['object']) is not str
                    or re.fullmatch('[0-9]{6}', row['object']) is None or row['object'] in objects
                    or type(row['bytes']) is not int or row['bytes'] < 0):
                raise ControllerHOLD('service_backup_object_invalid')
            objects.add(row['object'])
            if _stream(path/'objects'/row['object'], None, capacity=row['bytes']) != {k:row[k] for k in ('bytes','sha256')}:
                raise ControllerHOLD('service_backup_object_changed')
            total += row['bytes']
            databases += row['kind'] == 'database'
        elif set(row) != base | ({'target'} if row['kind'] == 'symlink' else set()):
            raise ControllerHOLD('service_backup_entry_invalid')
        elif row['kind'] == 'symlink' and (type(row['target']) is not str or '\0' in row['target']):
            raise ControllerHOLD('service_backup_link_invalid')
    if (set(seen).intersection((r,'.') for r in roots) != {(r,'.') for r in roots}
            or total != manifest['total_bytes'] or databases != 1
            or set(p.name for p in (path/'objects').iterdir()) != objects
            or _document(path/'manifest.json') != raw):
        raise ControllerHOLD('service_backup_contents_changed')
    return manifest


def verify(path, expected):
    manifest = _verify(path, expected)
    return {'kind':'verified_service_backup', 'manifest_sha256':expected,
            'files':len(manifest['entries']), 'bytes':manifest['total_bytes'], 'activation':False}


def _verify_restored(destination, manifest):
    roots = [destination/row['name'] for row in manifest['roots']]
    before = _scan(roots, None, manifest['total_bytes'])
    if len(before) != len(manifest['entries']):
        raise ControllerHOLD('service_backup_restored_tree_changed')
    for actual, expected in zip(before, manifest['entries']):
        kind = 'file' if expected['kind'] == 'database' else expected['kind']
        if ({key:actual[key] for key in ('root','path','mode')} != {
                key:expected[key] for key in ('root','path','mode')} or actual['kind'] != kind):
            raise ControllerHOLD('service_backup_restored_tree_changed')
        path = destination/actual['root']/actual['path']
        if kind == 'file':
            pinned = _stream(path, None, actual['identity'], expected['bytes'])
            if pinned != {key:expected[key] for key in ('bytes','sha256')}:
                raise ControllerHOLD('service_backup_restored_file_changed')
        elif kind == 'symlink' and actual['target'] != expected['target']:
            raise ControllerHOLD('service_backup_restored_link_changed')
    if (before != _scan(roots, None, manifest['total_bytes'])
            or set(p.name for p in destination.iterdir()) != {'restore-intent.json', *[p.name for p in roots]}):
        raise ControllerHOLD('service_backup_restored_tree_changed')


def restore(path, expected, destination):
    """Materialize historical data in a new tree; retain original scope bindings."""
    path = Path(path).absolute()
    manifest = _verify(path, expected)
    destination = Path(destination).absolute()
    if destination == path or path in destination.parents or destination in path.parents:
        raise ControllerHOLD('service_backup_restore_overlap')
    output = _Output(destination)
    try:
        output.publish('restore-intent.json', {'version':1, 'manifest_sha256':expected, 'activation':False})
        # Directory descriptors remain owned throughout restoration. Symlinks
        # are leaves in the verified manifest and never become path parents.
        descriptors, modes = {None:output.fd}, {}
        def finish(key):
            fd = descriptors.pop(key)
            try:
                os.fchmod(fd, modes.pop(key))
                os.fsync(fd)
            finally:
                os.close(fd)
        try:
            for row in manifest['entries']:
                rel = PurePosixPath(row['path'])
                parent_key = None if row['path'] == '.' else (row['root'], str(rel.parent))
                needed = {None} if row['path'] == '.' else {
                    None, *((row['root'], str(parent)) for parent in rel.parents)}
                for key in reversed(list(descriptors)):
                    if key not in needed:
                        finish(key)
                parent = descriptors[parent_key]
                name = row['root'] if row['path'] == '.' else rel.name
                output.check()
                if row['kind'] == 'directory':
                    os.mkdir(name, 0o700, dir_fd=parent)
                    key = (row['root'],row['path'])
                    descriptors[key] = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    modes[key] = row['mode']
                elif row['kind'] == 'symlink':
                    os.symlink(row['target'], name, dir_fd=parent)
                else:
                    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                    try:
                        actual = _stream(path/'objects'/row['object'], fd, capacity=row['bytes'])
                        if actual != {k:row[k] for k in ('bytes','sha256')}:
                            raise ControllerHOLD('service_backup_object_changed')
                        os.fchmod(fd, row['mode'])
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                os.fsync(parent)
            _verify(path, expected)
            for key in reversed(list(descriptors)):
                if key is not None:
                    finish(key)
            _verify_restored(destination, manifest)
            output.publish('restored.json', {'version':1, 'manifest_sha256':expected,
                'activation':False, 'roots':manifest['roots'], 'kind':'inactive_service_data'})
        finally:
            for key, fd in descriptors.items():
                if key is not None:
                    os.close(fd)
        return {'kind':'inactive_service_data', 'destination':str(destination),
                'manifest_sha256':expected, 'activation':False}
    finally:
        output.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    create = sub.add_parser('create')
    create.add_argument('--service-config', required=True)
    create.add_argument('--destination', required=True)
    create.add_argument('--max-bytes', type=int, default=DEFAULT_BYTES)
    for operation in ('verify','restore'):
        command = sub.add_parser(operation)
        command.add_argument('--backup', required=True)
        command.add_argument('--manifest-sha256', required=True)
        if operation == 'restore':
            command.add_argument('--destination', required=True)
    args = parser.parse_args(argv)
    previous = Path.cwd()
    try:
        if args.operation == 'create':
            from orze.service.host import _service
            service = Path(args.service_config).absolute()
            destination = Path(args.destination).absolute()
            svc = _service(service)[3]
            os.chdir(svc['workdir'])
            result = capture(service, destination, max_bytes=args.max_bytes)
        elif args.operation == 'verify':
            result = verify(args.backup, args.manifest_sha256)
        else:
            result = restore(args.backup, args.manifest_sha256, args.destination)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print('HOLD: service_backup_unconfirmed: '+str(exc), file=sys.stderr)
        return 75
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    raise SystemExit(main())
