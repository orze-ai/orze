"""Primitive storage probe in a new owned directory, never an existing scope.

No roles, services, providers or cleanup actions. Every probe preserves its
source/target bytes on rejection; outputs are retained, never extracted/reused.
"""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'ceph-primitives-v1'


def snapshot(root):
    result = {}
    for path in sorted(root.rglob('*')):
        info = path.lstat()
        result[str(path.relative_to(root))] = {'device': info.st_dev, 'inode': info.st_ino,
            'directory': path.is_dir(), 'sha256': None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def main():
    BASE.mkdir()
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    results = []
    for kind in ('file', 'directory'):
        for collision in (False, True):
            root = BASE / (kind + ('-collision' if collision else '-absent'))
            root.mkdir()
            source, target = root / 'source', root / 'target'
            if kind == 'directory':
                source.mkdir();(source / 'data').write_bytes(b'owned-source')
                if collision:
                    target.mkdir();(target / 'data').write_bytes(b'owned-target')
            else:
                source.write_bytes(b'owned-source')
                if collision:
                    target.write_bytes(b'owned-target')
            before = snapshot(root)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
                ctypes.set_errno(0)
                code = rename(fd, b'source', fd, b'target', 1)
                error = ctypes.get_errno() if code else 0
                os.fsync(fd)
            finally:
                os.close(fd)
            after = snapshot(root)
            if code:
                assert before == after
            else:
                assert not collision and not source.exists()
                assert (target / 'data' if kind == 'directory' else target).read_bytes() == b'owned-source'
                assert target.stat().st_ino == before['source']['inode']
            results.append({'kind': kind, 'collision': collision, 'flag': 'RENAME_NOREPLACE',
                            'returncode': code, 'errno': error, 'error_name': errno.errorcode.get(error),
                            'before': before, 'after': after})
    root = BASE / 'exchange'
    root.mkdir()
    (root / 'source').write_bytes(b'owned-source');(root / 'target').write_bytes(b'owned-target')
    before = snapshot(root)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        ctypes.set_errno(0)
        code = rename(fd, b'source', fd, b'target', 2)
        error = ctypes.get_errno() if code else 0
        os.fsync(fd)
    finally:
        os.close(fd)
    after = snapshot(root)
    assert (before == after) if code else ((root / 'source').read_bytes() == b'owned-target'
                                          and (root / 'target').read_bytes() == b'owned-source')
    results.append({'kind': 'file', 'collision': True, 'flag': 'RENAME_EXCHANGE', 'returncode': code,
                    'errno': error, 'error_name': errno.errorcode.get(error), 'before': before, 'after': after})
    from orze.engine.gc_tree import rename_no_replace
    root = BASE / 'product-no-replace'
    root.mkdir();source = root / 'source';source.mkdir();(source / 'data').write_bytes(b'owned-source')
    before = snapshot(root)
    error = None
    try:
        rename_no_replace(source, root / 'target')
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    after = snapshot(root)
    if error:
        assert before == after
    report = {'filesystem': subprocess.check_output(['stat', '-f', '-c', '%T', str(BASE)], text=True).strip(),
              'kernel': platform.release(), 'primitive_probes': results,
              'product': {'error': error, 'before': before, 'after': after},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Primitive probe only; no production storage compatibility claim'}
    with (BASE / 'result.json').open('x') as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'filesystem': report['filesystem'],
                     'primitives': [(p['kind'], p['collision'], p['flag'], p['error_name']) for p in results],
                     'product_error': error}))


if __name__ == '__main__':
    main()
