"""Retain frozen runs, source snapshots, and raw test trees without extraction.

Full paired source inventories and Pro fixture trees stay in the private repo.
Public wrappers are explicit Core-only projections, never raw-wrapper rewrites.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'research-campaign'
REL = Path('docs/evidence/runs/2026-09-16-research-campaign')
PUBLIC, PRIVATE = ROOT / 'orze' / REL, ROOT / 'pro' / REL


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())


def archive(source, destination):
    with tarfile.open(destination, 'x:gz', dereference=False) as output:
        output.add(source, arcname=source.name)
    entries = []
    with tarfile.open(destination, 'r:gz') as stored:
        for member in stored:
            row = {'path': member.name, 'kind': member.type.decode('ascii'), 'bytes': member.size}
            original = source.parent / member.name
            if member.isfile() or member.islnk():
                raw = stored.extractfile(member).read()
                assert raw == original.read_bytes()
                row['sha256'] = sha(raw)
            if member.issym() or member.islnk():
                row['linkname'] = member.linkname
            entries.append(row)
    write(destination.with_suffix('.manifest.json'), {'sha256': sha(destination.read_bytes()), 'entries': entries})
    return {'path': str(destination.relative_to(destination.parents[1])), 'entries': len(entries),
            'bytes': destination.stat().st_size, 'sha256': sha(destination.read_bytes())}


def main():
    PUBLIC.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    runs = []
    names = ('old-feature-v1', 'targeted-v1', 'model-binding-gap-v1', 'targeted-v2', 'targeted-v3',
             'targeted-v4', 'targeted-v5', 'targeted-v6', 'pro-regression', 'paired')
    frozen_core = json.loads((RAW / 'targeted-v6/run.json').read_bytes())['before']['primary']
    for name in names:
        source = RAW / name
        record = json.loads((source / 'run.json').read_bytes())
        assert record['frozen'] and record['before'] == record['after']
        key = next(k for k, value in record['repositories'].items() if Path(value).resolve() == (ROOT / 'orze').resolve())
        for filename, pinned in record['files'].items():
            raw = (source / filename).read_bytes()
            assert len(raw) == pinned['bytes'] and sha(raw) == pinned['sha256']
        if name in ('targeted-v6', 'pro-regression', 'paired'):
            assert record['before'][key] == frozen_core and record['exit_code'] == 0
        for path in source.iterdir():
            if path.is_file():
                copy(path, PRIVATE / 'runs' / name / path.name)
                if path.name not in ('run.json', 'before.json', 'after.json'):
                    copy(path, PUBLIC / 'runs' / name / path.name)
        junit = ET.parse(source / 'junit.xml').getroot()
        cases = list(junit.iter('testcase'))
        counts = {'tests': len(cases), 'failed': sum(c.find('failure') is not None for c in cases),
                  'errors': sum(c.find('error') is not None for c in cases),
                  'skipped': sum(c.find('skipped') is not None for c in cases)}
        projection = {k: record[k] for k in ('command', 'exit_code', 'frozen', 'started_unix', 'finished_unix', 'recorder')}
        projection.update(schema=1, scope='Core-only projection; unmodified paired wrapper in private repository',
                          original_run_sha256=sha((source / 'run.json').read_bytes()), counts=counts,
                          before={'core': record['before'][key]}, after={'core': record['after'][key]})
        write(PUBLIC / 'runs' / name / 'projection.json', projection)
        runs.append({'name': name, 'counts': counts, 'exit_code': record['exit_code']})
    for source in sorted(RAW.glob('*-v*')):
        if (source / 'manifest.json').is_file():
            manifest = json.loads((source / 'manifest.json').read_bytes())
            for name, pinned in manifest.items():
                raw = (source / name).read_bytes()
                assert len(raw) == pinned['bytes'] and sha(raw) == pinned['sha256']
            shutil.copytree(source, PUBLIC / 'snapshots' / source.name)
    for name in ('snapshot_research_campaign_v1.py', 'snapshot_research_campaign_v2.py',
                 'snapshot_research_campaign_v3.py', 'audit_research_campaign_v1.py',
                 'package_research_campaign_v1.py', 'research-campaign-design-v1.md'):
        copy(ROOT / name, PUBLIC / 'tools' / name)
    copy(RAW / 'product-audit.json', PUBLIC / 'product-audit.json')
    archives = []
    for name in ('rc-t1', 'rc-g1', 'rc-t2', 'rc-t3', 'rc-t4', 'rc-t5', 'rc-t6', 'rc-i1', 'rc-p1'):
        destination = (PRIVATE if name == 'rc-p1' else PUBLIC) / 'raw' / (name + '.tar.gz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        archives.append(archive(Path('/tmp') / name, destination))
    summary = {'schema': 1, 'runs': runs, 'archives': archives,
               'validation_scope': 'Core targeted 88; Pro research/usage regression 636; paired 31. No new full-suite run.',
               'product': {'outer_runs': 18, 'native_actions': 50},
               'real_model_calls': 0, 'new_research_gain_evidence': False}
    write(PUBLIC / 'summary.json', summary)
    write(PRIVATE / 'public-link.json', {'public_relative_path': str(REL),
                                        'summary_sha256': sha((PUBLIC / 'summary.json').read_bytes())})
    for destination in (PUBLIC, PRIVATE):
        write(destination / 'files.json', {str(p.relative_to(destination)): {'bytes': p.stat().st_size,
              'sha256': sha(p.read_bytes())} for p in sorted(destination.rglob('*')) if p.is_file()})
    print(json.dumps({'runs': len(runs), 'archives': len(archives), 'public': str(PUBLIC), 'private': str(PRIVATE)}))


if __name__ == '__main__':
    main()
