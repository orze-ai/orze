"""Archive recovery evidence only after all frozen test processes finish.

Uses the retained byte-comparing tar helper from the prior campaign slice.
Full Pro inventories and full Pro temporary fixtures remain private.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'service-recovery'
REL = Path('docs/evidence/runs/2026-09-16-service-recovery')
PUBLIC, PRIVATE = ROOT / 'orze' / REL, ROOT / 'pro' / REL
spec = importlib.util.spec_from_file_location('retained_campaign_packager', ROOT / 'package_research_campaign_v1.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
copy, write, sha, archive = helper.copy, helper.write, helper.sha, helper.archive


def selected_archive(source, destination):
    prefixes = ('test_closed_host_recovers_', 'test_competing_prepared_', 'test_prepared_recovery_',
                'test_closed_record_', 'test_recovery_preparation_', 'test_short_lease_autonomously_')
    roots = [path for path in sorted(source.iterdir())
             if path.is_dir() and not path.is_symlink() and path.name.startswith(prefixes)]
    assert len(roots) == 12
    with tarfile.open(destination, 'x:gz', dereference=False) as output:
        for path in roots:
            output.add(path, arcname=path.name)
    entries = []
    with tarfile.open(destination, 'r:gz') as stored:
        for member in stored:
            row = {'path': member.name, 'kind': member.type.decode(), 'bytes': member.size}
            if member.isfile() or member.islnk():
                raw = stored.extractfile(member).read()
                assert raw == (source / member.name).read_bytes()
                row['sha256'] = sha(raw)
            if member.issym() or member.islnk():
                row['linkname'] = member.linkname
            entries.append(row)
    write(destination.with_suffix('.manifest.json'), {'sha256': sha(destination.read_bytes()), 'entries': entries})
    return {'path': 'raw/' + destination.name, 'bytes': destination.stat().st_size,
            'sha256': sha(destination.read_bytes()), 'entries': len(entries)}


def main():
    PUBLIC.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    names = ('old-feature-v1', 'targeted-v1', 'targeted-v2', 'targeted-v3', 'targeted-v4',
             'core-full', 'pro-full', 'paired', 'fixture-regression')
    final = json.loads((RAW / 'targeted-v4/run.json').read_bytes())['before']
    results = []
    for name in names:
        folder = RAW / name
        record = json.loads((folder / 'run.json').read_bytes())
        assert record['frozen'] and record['before'] == record['after']
        core = next(key for key, path in record['repositories'].items()
                    if Path(path).resolve() == (ROOT / 'orze').resolve())
        pro = next(key for key in record['repositories'] if key != core)
        if name in ('targeted-v4', 'core-full', 'pro-full', 'paired'):
            assert record['before'][core] == final['primary'] and record['before'][pro] == final['peer']
            assert record['exit_code'] == (1 if name == 'core-full' else 0)
        if name == 'fixture-regression':
            assert record['exit_code'] == 0 and record['before'][pro] == final['peer']
            assert {path for path in record['before'][core]
                    if record['before'][core][path] != final['primary'].get(path)} == {'tests/test_runtime_lease_publication.py'}
            assert set(record['before'][core]) == set(final['primary'])
        for filename, pinned in record['files'].items():
            raw = (folder / filename).read_bytes()
            assert len(raw) == pinned['bytes'] and sha(raw) == pinned['sha256']
        for path in folder.iterdir():
            if path.is_file():
                copy(path, PRIVATE / 'runs' / name / path.name)
                if path.name not in ('run.json', 'before.json', 'after.json'):
                    copy(path, PUBLIC / 'runs' / name / path.name)
        cases = list(ET.parse(folder / 'junit.xml').getroot().iter('testcase'))
        counts = {'tests': len(cases), 'failed': sum(c.find('failure') is not None for c in cases),
                  'errors': sum(c.find('error') is not None for c in cases),
                  'skipped': sum(c.find('skipped') is not None for c in cases)}
        if name == 'core-full':
            assert [c.attrib['name'] for c in cases if c.find('failure') is not None] == [
                'test_short_lease_autonomously_closes_term_zero_and_escaped_child_without_parent_poll']
            assert counts == {'tests': 5461, 'failed': 1, 'errors': 0, 'skipped': 6}
        projection = {key: record[key] for key in
                      ('command', 'exit_code', 'frozen', 'started_unix', 'finished_unix', 'recorder')}
        projection.update(schema=1, scope='Explicit Core-only projection; complete original wrapper remains private',
                          original_run_sha256=sha((folder / 'run.json').read_bytes()), counts=counts,
                          before={'core': record['before'][core]}, after={'core': record['after'][core]})
        write(PUBLIC / 'runs' / name / 'projection.json', projection)
        results.append({'name': name, 'counts': counts, 'exit_code': record['exit_code']})
    for folder in sorted(RAW.glob('*-v*')):
        if (folder / 'manifest.json').exists():
            manifest = json.loads((folder / 'manifest.json').read_bytes())
            for name, pinned in manifest.items():
                raw = (folder / name).read_bytes()
                assert len(raw) == pinned['bytes'] and sha(raw) == pinned['sha256']
            shutil.copytree(folder, PUBLIC / 'snapshots' / folder.name)
    for name in ('snapshot_service_recovery_v1.py', 'snapshot_service_recovery_v2.py',
                 'audit_service_recovery_v1.py', 'verify_service_recovery_v1.py',
                 'package_service_recovery_v1.py', 'package_research_campaign_v1.py',
                 'service-recovery-design-v1.md'):
        copy(ROOT / name, PUBLIC / 'tools' / name)
    for name in ('product-audit-v1.json', 'product-audit-v1.stdout', 'product-audit-v1.stderr',
                 'full-product-audit-v1.json', 'full-product-audit-v1.stdout', 'full-product-audit-v1.stderr'):
        copy(RAW / name, PUBLIC / name)
    (PUBLIC / 'raw').mkdir()
    (PRIVATE / 'raw').mkdir()
    archives = []
    for name in ('sr-o1', 'sr-t1', 'sr-t2', 'sr-t3', 'sr-t4', 'sr-i1', 'sr-p1', 'sr-f1'):
        destination = (PRIVATE if name == 'sr-p1' else PUBLIC) / 'raw' / (name + '.tar.gz')
        archives.append(archive(Path('/tmp') / name, destination))
    archives.append(selected_archive(Path('/tmp/sr-c1'), PUBLIC / 'raw/core-full-recovery-and-failure-fixtures.tar.gz'))
    summary = {'schema': 1, 'runs': results, 'archives': archives,
               'product': {'scopes': 11, 'positive_scopes': 3, 'controllers': 16, 'native_actions': 13},
               'scope': 'Same-runtime closed CPU host recovery; no real systemd, reboot, migration or research-gain claim',
               'validation': 'Core full: 5454 passed, one test fixture JSON publication race failed, six skipped; fixture-only fix then targeted regression. No second full-suite run. Product source is unchanged from full runs.',
               'core_full_raw_scope': 'Archive contains new recovery fixtures and original failed lease fixture; other full-suite temporary fixtures remain local'}
    write(PUBLIC / 'summary.json', summary)
    write(PRIVATE / 'public-link.json', {'public_relative_path': str(REL),
                                        'summary_sha256': sha((PUBLIC / 'summary.json').read_bytes())})
    for destination in (PUBLIC, PRIVATE):
        write(destination / 'files.json', {str(p.relative_to(destination)): {'bytes': p.stat().st_size,
              'sha256': sha(p.read_bytes())} for p in sorted(destination.rglob('*')) if p.is_file()})
    print(json.dumps({'runs': len(results), 'archives': len(archives),
                      'entries': sum(row['entries'] for row in archives)}))


if __name__ == '__main__':
    main()
