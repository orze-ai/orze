"""Create-only frozen-run packaging after every referenced process is terminal."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'gc-guarded'
REL = Path('docs/evidence/runs/2026-09-16-gc-guarded')
PUBLIC, PRIVATE = ROOT / 'orze' / REL, ROOT / 'pro' / REL
spec = importlib.util.spec_from_file_location('retained_packager', ROOT / 'package_research_campaign_v1.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
copy, write, sha, archive = helper.copy, helper.write, helper.sha, helper.archive


def selected_core_archive():
    cases = ET.parse(RAW / 'core-full/junit.xml').getroot().iter('testcase')
    names = [case.attrib['name'] for case in cases if case.attrib.get('classname', '').endswith('test_gc_guarded')]
    prefixes = {re.sub(r'[\W]', '_', name)[:30] for name in names}
    source = Path('/tmp/gg-c1')
    roots = [p for p in sorted(source.iterdir()) if p.is_dir() and not p.is_symlink()
             and any(p.name.startswith(prefix) for prefix in prefixes)]
    assert len(roots) == len(names), (len(roots), len(names))
    destination = PUBLIC / 'raw/core-full-guarded-fixtures.tar.gz'
    with tarfile.open(destination, 'x:gz', dereference=False) as output:
        for path in roots:
            output.add(path, arcname=path.name)
    entries = []
    with tarfile.open(destination) as stored:
        for member in stored:
            row = {'path':member.name, 'kind':member.type.decode(), 'bytes':member.size}
            if member.isfile() or member.islnk():
                raw = stored.extractfile(member).read()
                assert raw == (source / member.name).read_bytes()
                row['sha256'] = sha(raw)
            if member.issym() or member.islnk():
                row['linkname'] = member.linkname
            entries.append(row)
    write(destination.with_suffix('.manifest.json'), {'sha256':sha(destination.read_bytes()), 'entries':entries})
    return {'path':'raw/' + destination.name, 'entries':len(entries), 'bytes':destination.stat().st_size,
            'sha256':sha(destination.read_bytes()), 'fixture_roots':len(roots)}


def main():
    names = ('baseline-product','product-v1','targeted-v1','targeted-v2','targeted-v3',
             'targeted-v4','cost-v1','product-v2','core-full','pro-full','paired',
             'boundary-gap-v1','boundary-gap-v2','boundary-tests-v1','boundary-fixed','targeted-v5','cost-v2','product-v3','pro-targeted','paired-v2')
    final_names = {'targeted-v5','cost-v2','product-v3','pro-targeted','paired-v2'}
    prior_names = {'targeted-v4','cost-v1','product-v2','core-full','pro-full','paired'}
    records = {name:json.loads((RAW / name / 'run.json').read_bytes()) for name in names}
    final = records['targeted-v5']['before']
    prior = records['targeted-v4']['before']
    for name, record in records.items():
        assert record['frozen'] and record['before'] == record['after']
        core = next(key for key,path in record['repositories'].items() if Path(path).resolve() == (ROOT / 'orze').resolve())
        pro = next(key for key in record['repositories'] if key != core)
        if name in final_names:
            assert record['exit_code'] == 0 and record['before'][core] == final['primary'] and record['before'][pro] == final['peer']
        if name in prior_names:
            assert record['exit_code'] == 0 and record['before'][core] == prior['primary'] and record['before'][pro] == prior['peer']
        for file, expected in record['files'].items():
            raw = (RAW / name / file).read_bytes()
            assert len(raw) == expected['bytes'] and sha(raw) == expected['sha256']
    assert json.loads((RAW / 'product-audit-v3.json').read_bytes())['final_ceph_candidates'] == 4
    PUBLIC.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    results = []
    for name, record in records.items():
        directory = RAW / name
        core = next(key for key,path in record['repositories'].items() if Path(path).resolve() == (ROOT / 'orze').resolve())
        for path in directory.iterdir():
            if path.is_file():
                copy(path, PRIVATE / 'runs' / name / path.name)
                if path.name not in ('run.json','before.json','after.json'):
                    copy(path, PUBLIC / 'runs' / name / path.name)
        counts = None
        if (directory / 'junit.xml').is_file():
            cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
            counts = {'tests':len(cases), 'failed':sum(c.find('failure') is not None for c in cases),
                      'errors':sum(c.find('error') is not None for c in cases), 'skipped':sum(c.find('skipped') is not None for c in cases)}
        projection = {key:record[key] for key in ('command','exit_code','frozen','started_unix','finished_unix','recorder')}
        projection.update(schema=1, scope='Explicit Core-only projection; complete paired wrapper retained privately',
                          original_run_sha256=sha((directory / 'run.json').read_bytes()), counts=counts,
                          before={'core':record['before'][core]}, after={'core':record['after'][core]})
        write(PUBLIC / 'runs' / name / 'projection.json', projection)
        results.append({'name':name, 'counts':counts, 'exit_code':record['exit_code']})
    for directory in sorted(RAW.glob('candidate-v*')):
        manifest = json.loads((directory / 'manifest.json').read_bytes())
        for name, expected in manifest.items():
            raw = (directory / name).read_bytes()
            assert sha(raw) == expected['sha256'] and len(raw) == expected['bytes']
        shutil.copytree(directory, PUBLIC / 'snapshots' / directory.name)
    baseline = {}
    for name in ('gc_safety','storage_preflight','attempt_effect_lock'):
        path = 'src/orze/engine/' + name + '.py'
        raw = subprocess.check_output(['git','show','e60d8ff694bcdb03146a021d5ccf74de6067df1d:' + path], cwd=ROOT / 'orze')
        assert sha(raw) == records['baseline-product']['before']['primary'][path]
        target = PUBLIC / 'baseline' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(raw)
        baseline[path] = {'bytes':len(raw), 'sha256':sha(raw)}
    write(PUBLIC / 'baseline/manifest.json', baseline)
    for name in ('gc-guarded-design-v1.md','snapshot_gc_guarded_v1.py','probe_gc_guarded_v1.py',
                 'cost_gc_guarded_v1.py','audit_gc_guarded_v1.py','audit_gc_guarded_v2.py',
                 'package_gc_guarded_v1.py','package_research_campaign_v1.py','ceph_atomic_probe_v1.py',
                 'probe_gc_guarded_boundaries_v1.py','probe_gc_guarded_boundaries_v2.py','audit_gc_guarded_v3.py','verify_gc_guarded_v1.py','gc_guarded_regressions_v1.txt'):
        copy(ROOT / name, PUBLIC / 'tools' / name)
    for path in sorted(RAW.glob('product-audit-v*.*')):
        copy(path, PUBLIC / path.name)
    for destination in (PUBLIC, PRIVATE):
        (destination / 'raw').mkdir()
    sources = [(Path('/tmp') / name, name in {'gg-p1','gg-p2'}) for name in
               ('gg-t1','gg-t2','gg-t3','gg-t4','gg-i1','gg-p1','gg-b1','gg-g1','gg-g2','gg-g3','gg-r1','gg-t5','gg-b2','gg-p2','gg-i2')]
    sources += [(RAW / name, False) for name in ('baseline-product-fixture','product-v1-fixture',
                'product-v2-fixture','ceph-targeted-v1','ceph-targeted-v2','ceph-targeted-v3',
                'ceph-targeted-v4','ceph-core-full','cost-v1-fixture','ceph-targeted-v5','cost-v2-fixture','product-v3-fixture')]
    sources.append((ROOT / 'ceph-primitives-v1', False))
    archives = [archive(source, (PRIVATE if private else PUBLIC) / 'raw' / (source.name + '.tar.gz'))
                for source,private in sources]
    archives.append(selected_core_archive())
    summary = {'schema':1, 'runs':results, 'archives':archives,
               'product':json.loads((RAW / 'product-audit-v3.json').read_bytes()),
               'validation_scope':'Full Core/Pro and paired suites on candidate-v5; final candidate-v8 fixes FD cleanup, bounded directory enumeration, and stop/identity checks while publishing archive manifests, covered by final targeted Core/Pro, paired, boundary probe, product and cost runs. No repeated full suites after these fixes.',
               'core_full_raw_scope':'Only new guarded GC fixtures archived from full Core tmp; other temporary fixtures remain local',
               'remaining':'CephFS role release, actual services/deployment, migration/backup/rollback, real research gains remain open'}
    write(PUBLIC / 'summary.json', summary)
    write(PRIVATE / 'public-link.json', {'public_relative_path':str(REL), 'summary_sha256':sha((PUBLIC / 'summary.json').read_bytes())})
    for destination in (PUBLIC, PRIVATE):
        write(destination / 'files.json', {str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
              for p in sorted(destination.rglob('*')) if p.is_file()})
    print(json.dumps({'runs':len(results), 'archives':len(archives), 'entries':sum(a['entries'] for a in archives)}))


if __name__ == '__main__':
    main()
