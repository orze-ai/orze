"""Read-only final evidence check; no execution, process control or manager access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    root, output = map(Path, sys.argv[1:])
    relative = Path('docs/evidence/runs/2026-09-16-service-recovery')
    public, private = root / 'orze' / relative, root / 'pro' / relative
    counts = {}
    for label, directory in [('public', public), ('private', private)]:
        index = read(directory / 'files.json')
        for name, expected in index.items():
            path = directory / name
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], path
        assert set(index) == {str(p.relative_to(directory)) for p in directory.rglob('*')
                              if p.is_file() and p != directory / 'files.json'}
        counts[label] = len(index)
    summary = read(public / 'summary.json')
    assert read(private / 'public-link.json')['summary_sha256'] == sha(public / 'summary.json')
    entries = 0
    for row in summary['archives']:
        path = next(p for p in [public / row['path'], private / row['path']] if p.is_file())
        assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
        manifest = read(path.with_suffix('.manifest.json'))
        assert manifest['sha256'] == row['sha256']
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            assert len(members) == row['entries'] == len(manifest['entries'])
            for member, expected in zip(members, manifest['entries']):
                assert member.name == expected['path'] and member.size == expected['bytes']
                if member.isfile() or member.islnk():
                    assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == expected['sha256']
        entries += len(members)
    for path in (public / 'runs').glob('*/projection.json'):
        record = read(path)
        assert set(record['before']) == {'core'} and record['before'] == record['after']
        assert not any(name.startswith('src/orze_pro/') for name in record['before']['core'])
    recorder = root / 'orze/docs/evidence/runs/2026-09-14-budget-normalization/run_frozen.py'
    spec = importlib.util.spec_from_file_location('recorder', recorder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('core-full', 'pro-full', 'paired', 'fixture-regression'):
        run = read(private / 'runs' / name / 'run.json')
        assert run['exit_code'] == (1 if name == 'core-full' else 0)
        assert run['frozen'] and run['before'] == run['after']
        for label, repo in run['repositories'].items():
            current, previous = module.fingerprint(Path(repo)), run['after'][label]
            assert set(current) == set(previous)
            changed = {key for key in current if current[key] != previous[key]}
            expected = ({'tests/test_runtime_lease_publication.py'}
                        if name != 'fixture-regression' and Path(repo).resolve() == (root / 'orze').resolve() else set())
            assert changed == expected, (name, label, changed)
    for name in ('product-audit-v1.json', 'full-product-audit-v1.json'):
        assert read(public / name)['summary'] == summary['product']
    result = {'files': counts, 'archives': len(summary['archives']), 'archive_entries': entries,
              'script_sha256': sha(Path(__file__)),
              'validation': 'Full Core retained one fixture failure; fixture-only correction subsequently targeted. Product sources unchanged.',
              'scope': 'Static byte/source check, not a new research experiment'}
    with output.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
