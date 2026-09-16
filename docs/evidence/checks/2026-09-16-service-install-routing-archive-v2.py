"""Create-only archives of closed service routing baselines, failures and CPU product fixtures."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile

CORE = Path(__file__).resolve().parents[3]
OUT = CORE / 'docs/evidence/runs/2026-09-16-service-install-routing'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
ROOTS = {'baseline': Path('/tmp/si-b1'),
'old_feature': Path('/tmp/si-o1'),
'targeted_v1': Path('/tmp/si-t1'),
'probe_old_v1': Path('/tmp/orze-service-routing-old-v1'),
'probe_v1': Path('/tmp/orze-service-routing-v1'),
'product_old_v1': Path('/tmp/orze-service-routing-product-old-v1'),
'product_controls_v1': Path('/tmp/orze-service-routing-product-controls-v1'),
'product_v1': Path('/tmp/orze-service-routing-product-v1')}
RUNS = [('baseline', 0), ('old-feature', 1), ('targeted-v1', 0),
('probe-old-v1', 1), ('probe-v1', 0), ('product-old-v1', 1),
('product-controls-v1', 0), ('product-v1', 0)]


def main():
    assert not (OUT / 'archives.json').exists()
    for name, code in RUNS:
        run = json.loads((OUT / name / 'run.json').read_text())
        assert run['frozen'] and run['exit_code'] == code
    records = {}
    for name, root in ROOTS.items():
        assert root.is_dir() and not root.is_symlink()
        target = OUT / (name + '-v2.tar.gz')
        files = []
        for directory, dirs, names in os.walk(root, followlinks=False):
            files.extend(Path(directory) / child for child in names + [child for child in dirs if (Path(directory) / child).is_symlink()])
        members = {}
        with tarfile.open(target, 'x:gz', dereference=False) as archive:
            for path in sorted(files):
                # Runtime identity tests intentionally author synthetic cache bytes; retain them as raw fixture evidence.
                relative = path.relative_to(root).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    entry = {'kind': 'symlink', 'target': os.readlink(path)}
                elif stat.S_ISFIFO(info.st_mode):
                    entry = {'kind': 'fifo', 'device': info.st_dev, 'inode': info.st_ino,
                             'nlink': info.st_nlink, 'mode': info.st_mode}
                else:
                    assert stat.S_ISREG(info.st_mode), path
                    entry = {'kind': 'file', 'bytes': info.st_size, 'sha256': sha(path.read_bytes()),
                             'device': info.st_dev, 'inode': info.st_ino, 'nlink': info.st_nlink}
                members[relative] = entry
                archive.add(path, arcname=relative, recursive=False)
        records[name] = {'root': str(root), 'archive': target.name, 'sha256': sha(target.read_bytes()),
                        'members': members, 'script_sha256': sha(Path(__file__).read_bytes())}
        print(json.dumps({'archive': name, 'entries': len(members)}), flush=True)
    with (OUT / 'archives.json').open('x') as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
        handle.write('\n')


if __name__ == '__main__':
    main()
