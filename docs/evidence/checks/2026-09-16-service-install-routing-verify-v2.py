"""Independent byte, Git, closed-run, SQL, native artifact and archive checks."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

CORE=Path(__file__).resolve().parents[3]
OUT=CORE/'docs/evidence/runs/2026-09-16-service-install-routing'
CHECKS=CORE/'docs/evidence/checks'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
load=lambda path:json.loads(path.read_text())
def git(*args):return subprocess.check_output(['git','-C',str(CORE),*args])
def sources():
    manifest=load(OUT/'baseline-inputs/source.json');head=manifest['commits']['orze']
    result={}
    for name,record in manifest['files'].items():
        raw=git('show',head+':'+name)
        assert raw==(OUT/'baseline-inputs'/name).read_bytes()
        assert sha(raw)==record['sha256'] and len(raw)==record['bytes']
        assert git('rev-parse',head+':'+name).decode().strip()==record['git_blob']
    for folder in ('src','tests','examples'):
        for name in git('ls-tree','-r','--name-only',head,'--',folder).decode().splitlines():
            if name=='src/orze/service/install.py':continue
            raw=git('show',head+':'+name);assert (CORE/name).read_bytes()==raw,name
            result[name]=git('rev-parse',head+':'+name).decode().strip()
    for candidate,name in [('install-v1.py','src/orze/service/install.py'),('tests-v1.py','tests/test_service_install_routing.py')]:
        assert (OUT/'candidate-inputs'/candidate).read_bytes()==(CORE/name).read_bytes()
    old=ast.parse((OUT/'baseline-inputs/src/orze/service/install.py').read_text())
    new=ast.parse((CORE/'src/orze/service/install.py').read_text())
    old.body=[n for n in old.body if not isinstance(n,ast.FunctionDef) or n.name!='install']
    new.body=[n for n in new.body if not isinstance(n,ast.FunctionDef) or n.name!='install']
    assert ast.dump(old)==ast.dump(new)
    return {'baseline':manifest,'unchanged_git_blobs':result,'candidate_sha256':sha((CORE/'src/orze/service/install.py').read_bytes())}

def run(name,code,counts=None):
    folder=OUT/name;value=load(folder/'run.json')
    assert value['frozen'] and value['exit_code']==code and value['before']==value['after']
    assert value['repositories']=={'primary':str(CORE)}
    assert value['environment']=={'PYTHONPATH':str(CORE/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
    assert value['before']==load(folder/'before.json') and value['after']==load(folder/'after.json')
    assert sha(Path(value['recorder']['path']).read_bytes())==value['recorder']['sha256']
    for file,record in value['files'].items():
        raw=(folder/file).read_bytes();assert len(raw)==record['bytes'] and sha(raw)==record['sha256']
    if counts is not None:
        suites=ET.parse(folder/'junit.xml').getroot().findall('testsuite')
        totals={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
        assert tuple(totals.values())==counts,(name,totals,counts)
    return {k:value[k] for k in ('started_unix','finished_unix','exit_code','frozen')}

def native(case,row):
    connection=sqlite3.connect((case/'project/lake.db').as_uri()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    try:tables={name:[dict(r) for r in connection.execute('SELECT * FROM '+name)] for name in ('ideas','execution_attempts','cpu_action_reservations','research_artifacts')}
    finally:connection.close()
    assert tables==row['tables']
    idea,=tables['ideas'];assert idea['idea_id']=='idea-route' and idea['status']=='completed'
    attempt,=tables['execution_attempts'];terminal= json.loads(attempt['terminal_json'])
    assert attempt['state']=='TERMINAL' and terminal['outcome']=='completed'
    tree=terminal['process_tree'];assert tree['event']=='TREE_CLOSED' and tree['wait_proof']=='ECHILD_WALL'
    ref=tree['binding']['identity']['attempt_ref'];assert all(attempt[k]==v for k,v in ref.items())
    binding=json.loads(attempt['binding_json']);results=Path(row['results'])
    assert Path(binding['scope'])==results/'idea-route'
    assert Path(binding['work_dir']).is_relative_to(results/'idea-route')
    assert binding['artifact_publication']['scope']==str(results)
    reservation,=tables['cpu_action_reservations']
    assert reservation['state']=='SETTLED' and json.loads(reservation['ref_json'])==ref
    assert reservation['terminal_sha256']==sha(attempt['terminal_json'].encode())
    record,=tables['research_artifacts'];artifact=json.loads(record['record_json'])
    assert artifact==row['artifact'] and artifact['producer']==ref
    assert artifact['artifact_id'] in terminal['artifact_ids']
    raw=Path(artifact['path']).read_bytes();assert sha(raw)==artifact['content_sha256']
    assert json.loads(raw)==row['artifact_body']==json.loads((Path(binding['work_dir'])/'route-proof.json').read_text())
    return binding

def products():
    result={}
    script=CHECKS/'2026-09-16-service-install-routing-product-v1.py'
    for name,total,variant in [('product-old-v1',1,'old'),('product-controls-v1',2,'old'),('product-v1',4,'new')]:
        root=Path('/tmp/orze-service-routing-'+name);passed=name!='product-old-v1'
        report=load(root/('report.json' if passed else 'progress.json'))
        assert report['script_sha256']==sha(script.read_bytes()) and report['variant']==variant
        assert report['baseline_sha256']==sha((OUT/'baseline-inputs/src/orze/service/install.py').read_bytes())
        assert len(report['cases'])==total
        if passed:assert report['passed']
        invocations=0
        for row in report['cases']:
            case=root/row['name'];binding=native(case,row)
            initial=load(case/'installer-cwd/install.json');saved=initial['saved']
            recorded=row['initial_install']['result']
            assert {k:v for k,v in recorded.items() if k!='saved'}=={k:v for k,v in initial.items() if k!='saved'}
            expected_saved=dict(saved)
            if passed:expected_saved['review_witness']='existing service metadata must remain'
            assert recorded['saved']==expected_saved
            assert load(case/'after-controller.json')['initial_install']['result']==initial
            assert initial['error'] is None and initial['manager_calls']==0
            assert initial['calls']==[[saved['method'],saved]]
            assert saved==load(case/'initial-service.json')
            assert sha((case/'initial-service.json').read_bytes())==row['initial_service_raw_sha256']
            package,=saved['runtime_packages'];assert package['name']=='orze' and package['root']==str(CORE/'src/orze')
            from orze.service.runtime_contract import _hash_package_tree
            digest,count=_hash_package_tree(Path(package['root']))
            assert (digest,count)==(package['sha256'],package['file_count'])
            for call in row['invocations']:
                assert call['exit_code']==0 and call['cwd']==str(case/'project')
            invocations+=len(row['invocations'])
            observation=row['routing_observation']
            assert observation['same_route']==passed and observation['same_log']==passed
            assert observation['work_inside_saved_results']==passed
            assert Path(saved['results_dir'])==(Path(row['results']) if passed else case/'installer-cwd/results')
            log=Path(row['log_path']);assert log==Path(saved['log_file'])
            if passed:
                assert len(row['invocations'])==2 and row['passed'] and row['stopped_metadata_unchanged']
                assert sha(log.read_bytes())==row['log_after_replay_sha256']
                assert (case/'target-stop-before.json').read_bytes()==(case/'target-stop-after.json').read_bytes()
                stop=row['target_stop']['result'];assert stop['calls']==[] and 'stop latch' in stop['error']
                allowed=row['unrelated_stop']['result'];assert allowed['error'] is None and len(allowed['calls'])==1
                assert (case/'installer-cwd/results/.orze_shutdown').read_text()=='unrelated project stop'
                assert not (Path(row['results'])/'.orze_stop_all').exists()
                assert sha((case/'service.json').read_bytes())==row['final_service_raw_sha256']
            else:
                assert len(row['invocations'])==1 and sha(log.read_bytes())==row['log_after_controller_sha256']
        result[name]={'cpu_workers':total,'cli_invocations':invocations,'report_sha256':sha((root/('report.json' if passed else 'progress.json')).read_bytes())}
    for name,truth in [('old-v1',False),('v1',True)]:
        root=Path('/tmp/orze-service-routing-'+name);observation=load(root/'observation.json')
        assert observation['same_route']==truth and observation['service_manager_calls']==0
        assert observation['actual_controller_processes']==0
        assert observation['script_sha256']==sha((CHECKS/'2026-09-16-service-install-routing-probe-v1.py').read_bytes())
    return result

def archive_checks():
    records=load(OUT/'archives.json');counts=Counter()
    for name,record in records.items():
        root=Path(record['root']);path=OUT/record['archive'];assert sha(path.read_bytes())==record['sha256']
        assert record['script_sha256']==sha((CHECKS/'2026-09-16-service-install-routing-archive-v2.py').read_bytes())
        actual=[]
        for directory,dirs,files in os.walk(root,followlinks=False):
            actual.extend(str((Path(directory)/child).relative_to(root)) for child in files+[c for c in dirs if (Path(directory)/c).is_symlink()])
        assert set(actual)==set(record['members'])
        with tarfile.open(path,'r:gz') as archive:
            members=archive.getmembers();assert len(members)==len(record['members'])
            assert {m.name for m in members}==set(record['members'])
            for member in members:
                entry=record['members'][member.name];original=root/member.name;info=original.lstat()
                counts[entry['kind']]+=1
                if entry['kind']=='file':
                    assert member.isfile() and stat.S_ISREG(info.st_mode)
                    raw=original.read_bytes();assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes']
                    assert archive.extractfile(member).read()==raw
                    assert info.st_ino==entry['inode'] and info.st_dev==entry['device'] and info.st_nlink==entry['nlink']
                elif entry['kind']=='symlink':
                    assert member.issym() and original.is_symlink() and os.readlink(original)==member.linkname==entry['target']
                else:
                    assert entry['kind']=='fifo' and member.isfifo() and stat.S_ISFIFO(info.st_mode)
    partial=OUT/'baseline.tar.gz'
    with tarfile.open(partial,'r:gz') as archive:
        members=archive.getmembers();assert members
        for member in members:
            original=Path('/tmp/si-b1')/member.name
            assert member.isfile() and archive.extractfile(member).read()==original.read_bytes()
    return {'archives':len(records),'regular_files':counts['file'],'symlinks':counts['symlink'],'fifos':counts['fifo'],
            'manifest_sha256':sha((OUT/'archives.json').read_bytes()),'failed_partial_archive_sha256':sha(partial.read_bytes())}

def main():
    preflight='--preflight' in sys.argv
    result={'script_sha256':sha(Path(__file__).read_bytes()),'preflight':preflight,'sources':sources()}
    specs=[('baseline',0,(78,0,0,0)),('old-feature',1,(24,18,0,0)),('targeted-v1',0,(102,0,0,0)),
        ('probe-old-v1',1,None),('probe-v1',0,None),('product-old-v1',1,None),('product-controls-v1',0,None),
        ('product-v1',0,None),('archive-v1',1,None),('archive-v2',0,None),('verify-preflight',1,None)]
    if not preflight:specs.append(('full-core',0,(5202,0,0,7)))
    result['runs']={name:run(name,code,counts) for name,code,counts in specs}
    candidate=load(OUT/'targeted-v1/before.json')
    for name in ('probe-v1','product-old-v1','product-controls-v1','product-v1','archive-v1','archive-v2','verify-preflight'):
        assert load(OUT/name/'before.json')==candidate
    if not preflight:assert load(OUT/'full-core/before.json')==candidate
    expected_old=dict(candidate['primary']);expected_old['src/orze/service/install.py']=result['sources']['baseline']['files']['src/orze/service/install.py']['sha256']
    assert load(OUT/'old-feature/before.json')=={'primary':expected_old}
    expected_old.pop('tests/test_service_install_routing.py')
    assert load(OUT/'baseline/before.json')==load(OUT/'probe-old-v1/before.json')=={'primary':expected_old}
    errors=(OUT/'product-old-v1/stderr.log').read_text()
    assert "assert row['routing_observation']['same_route'] and row['routing_observation']['same_log']" in errors
    assert 'AssertionError' in (OUT/'archive-v1/stderr.log').read_text()
    result['products']=products();result['archives']=archive_checks();result['passed']=True
    target=OUT/('verification-preflight.json' if preflight else 'verification.json')
    with target.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'passed':True,'preflight':preflight,'sha256':sha(target.read_bytes()),'archives':result['archives']}))

if __name__=='__main__':main()
