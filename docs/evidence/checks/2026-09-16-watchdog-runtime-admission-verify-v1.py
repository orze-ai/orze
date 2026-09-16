"""Independent Git/source, closed-run, child, SQL, CPU artifact and archive verification."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tarfile

CORE=Path(__file__).resolve().parents[3]
OUT=CORE/'docs/evidence/runs/2026-09-16-watchdog-runtime-admission'
CHECKS=CORE/'docs/evidence/checks'
HELPER=CHECKS/'2026-09-16-service-install-routing-verify-v3.py'
spec=importlib.util.spec_from_file_location('prior_independent_checks',HELPER)
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);prior.OUT=OUT
sha=lambda raw:hashlib.sha256(raw).hexdigest()
load=lambda p:json.loads(p.read_text())
def git(*args):return subprocess.check_output(['git','-C',str(CORE),*args])
def sources():
    manifest=load(OUT/'baseline-inputs/source.json');head=manifest['commits']['orze']
    assert head=='93aaa00ffe255b24dffebc76d5526be787272208'
    records={};changed={'src/orze/service/watchdog.py','tests/test_watchdog_failure_loop.py'}
    for name,record in manifest['files'].items():
        raw=git('show',head+':'+name);assert raw==(OUT/'baseline-inputs'/name).read_bytes()
        assert len(raw)==record['bytes'] and sha(raw)==record['sha256']
        assert git('rev-parse',head+':'+name).decode().strip()==record['git_blob']
    for folder in ('src','tests','examples'):
        for name in git('ls-tree','-r','--name-only',head,'--',folder).decode().splitlines():
            if name in changed:continue
            raw=git('show',head+':'+name);assert raw==(CORE/name).read_bytes(),name
            records[name]=git('rev-parse',head+':'+name).decode().strip()
    candidates={}
    for candidate,name in [('watchdog-v1.py','src/orze/service/watchdog.py'),('failure-loop-tests-v1.py','tests/test_watchdog_failure_loop.py'),('tests-v1.py','tests/test_watchdog_runtime_admission.py')]:
        raw=(CORE/name).read_bytes();assert raw==(OUT/'candidate-inputs'/candidate).read_bytes();candidates[name]=sha(raw)
    for name,allowed in [('src/orze/service/watchdog.py',{'_require_runtime_contract','_launch_orze','check_and_restart'}),('tests/test_watchdog_failure_loop.py',{'test_watchdog_escalates_repeated_failure_without_logging_raw_output'})]:
        old=ast.parse((OUT/'baseline-inputs'/name).read_text());new=ast.parse((CORE/name).read_text())
        for tree in (old,new):tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in allowed]
        assert ast.dump(old)==ast.dump(new)
    return {'baseline':manifest,'unchanged_git_blobs':records,'candidates':candidates}

def probes():
    result={}
    for name,root,old in [('old-order','/tmp/orze-watchdog-admission-order-baseline-v1',True),('order-v1','/tmp/orze-watchdog-admission-order-v1',False)]:
        value=load(Path(root)/'observation.json')
        assert value['script_sha256']==sha((CHECKS/'2026-09-16-watchdog-runtime-admission-order-v1.py').read_bytes())
        assert value['actual_runtime_audit']['startup_allowed'] is False
        assert value['actual_runtime_audit']['errors']==['runtime_package_sha256_drift:orze']
        assert value['watchdog_error']=='runtime_contract_rejected'
        assert value['events']==([['requested_stale_kill',424242]] if old else [])
        assert value['actual_signals']==value['actual_controllers']==value['actual_managers']==0
        result[name]=value
    for name,root,old in [('old-canary','/tmp/orze-watchdog-owner-canary-baseline-v1',True),('canary-v1','/tmp/orze-watchdog-owner-canary-v1',False)]:
        root=Path(root);value=load(root/'observation.json');cleanup=load(root/'cleanup.json')
        assert value['script_sha256']==sha((CHECKS/'2026-09-16-watchdog-runtime-admission-canary-v1.py').read_bytes())
        identity,parent=value['own_canary_identity'];pid=identity['pid']
        assert type(identity['start_ticks']) is int and parent!=pid
        assert cleanup=={'child_pid':pid,'reaped':True,'exit_code':-15}
        assert value['audit_before']['startup_allowed'] is False and value['watchdog_error']=='runtime_contract_rejected'
        assert value['canary_exit_before_cleanup']==(-15 if old else None)
        assert value['signals_before_cleanup']==([{'target':'own_isolated_pgid','pid':pid,'signal':sig} for sig in (15,9)] if old else [])
        assert value['actual_controllers']==value['actual_managers']==0
        result[name]={'observation':value,'cleanup':cleanup}
    return result

def native_failed(case,mode):
    connection=sqlite3.connect((case/'project/lake.db').as_uri()+'?mode=ro',uri=True);connection.row_factory=sqlite3.Row
    try:tables={name:[dict(r) for r in connection.execute('SELECT * FROM '+name)] for name in ('ideas','execution_attempts','cpu_action_reservations','research_artifacts')}
    finally:connection.close()
    record,=tables['research_artifacts'];artifact=json.loads(record['record_json'])
    body={'status':'COMPLETED','runtime_case':mode,'scope':'isolated watchdog admission'}
    row={'results':str(case/'project/results'),'tables':tables,'artifact':artifact,'artifact_body':body}
    native(case,row)
    return {'artifact_id':artifact['artifact_id'],'artifact_sha256':artifact['content_sha256'],'terminal_sha256':sha(tables['execution_attempts'][0]['terminal_json'].encode())}

def products():
    result={}
    for name,mode in [('product-old-v1','drift'),('product-control-v1','matching'),('product-v1','drift')]:
        root=Path('/tmp/orze-watchdog-runtime-'+name);case=root/mode
        assert not (root/'report.json').exists() and not (case/'after-watchdog.json').exists()
        assert 'FileNotFoundError' in (OUT/name/'stderr.log').read_text()
        cleanup=load(case/'controller-reaped.json');assert cleanup['reaped'] and cleanup['exit_code']==0
        result[name]={'cpu_workers':1,'cli_invocations':1,'failure':'missing native CPU legacy PID marker','native':native_failed(case,mode)}
    for name,total,baseline in [('product-old-v2',1,True),('product-control-v2',1,True),('product-v2',2,False)]:
        root=Path('/tmp/orze-watchdog-runtime-'+name);passed=name!='product-old-v2'
        path=root/('report.json' if passed else 'progress.json');report=load(path)
        assert report['script_sha256']==sha((CHECKS/'2026-09-16-watchdog-runtime-admission-product-v2.py').read_bytes())
        assert report['helper_sha256']==sha((CHECKS/'2026-09-16-service-install-routing-product-v1.py').read_bytes())
        assert report['baseline_watchdog_sha256']==sha((OUT/'baseline-inputs/src/orze/service/watchdog.py').read_bytes())
        assert report['baseline']==baseline and len(report['cases'])==total
        if passed:assert report['passed']
        for row in report['cases']:
            case=root/row['mode'];native(case,row)
            expected={'status':'COMPLETED','runtime_case':row['mode'],'scope':'isolated watchdog admission'}
            assert row['artifact_body']==expected
            assert len(row['invocations'])==2 and all(call['exit_code']==0 for call in row['invocations'])
            assert row['controller_alive_after_watchdog'] and row['pid_file_unchanged']
            origin=load(case/'pid-marker-origin.json')
            assert origin['origin']=='fixture-authored legacy service PID marker'
            assert origin['controller_identity']==row['controller_identity']
            assert int(Path(origin['path']).read_text())==row['controller_identity'][0]['pid']
            cleanup=load(case/'controller-reaped.json');assert cleanup=={'pid':row['controller_identity'][0]['pid'],'exit_code':0,'reaped':True}
            initial=row['initial_install']['result'];saved=load(case/'service.json')
            assert initial['saved']==saved and initial['calls']==[['crontab',saved]] and initial['error'] is None and initial['manager_calls']==0
            expected_cfg=json.loads(json.dumps(saved))
            if row['mode']=='drift':expected_cfg['runtime_packages'][0]['sha256']='0'*64
            assert row['watchdog_service']==load(case/'watchdog-service.json')==expected_cfg
            package,=saved['runtime_packages'];assert package['name']=='orze' and package['root']==str(CORE/'src/orze')
            from orze.service.runtime_contract import _hash_package_tree
            assert _hash_package_tree(Path(package['root']))==(package['sha256'],package['file_count'])
            assert row['audit']['startup_allowed']==(row['mode']=='matching')
            if row['mode']=='drift' and not baseline:
                assert row['watchdog_error']=='runtime_contract_rejected' and row['watchdog_events']==[]
            else:
                assert row['watchdog_error'] is None and row['watchdog_events']==['read_pid','own_controller_kill0']
            original=load(case/'after-replay.json')
            assert {k:v for k,v in row.items() if k!='passed'}==original
        result[name]={'cpu_workers':total,'cli_invocations':2*total,'report_sha256':sha(path.read_bytes())}
    assert 'runtime drift was not rejected before controller inspection' in (OUT/'product-old-v2/stderr.log').read_text()
    return result

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
    assert json.loads(raw)==row['artifact_body']==json.loads((Path(binding['work_dir'])/'proof.json').read_text())
    return binding

def archive_checks():
    records=load(OUT/'archives.json');counts=Counter()
    for name,record in records.items():
        root=Path(record['root']);path=OUT/record['archive'];assert sha(path.read_bytes())==record['sha256']
        assert record['script_sha256']==sha((CHECKS/'2026-09-16-watchdog-runtime-admission-archive-v1.py').read_bytes())
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
                    assert (member.isfile() or member.islnk()) and stat.S_ISREG(info.st_mode)
                    if member.islnk():
                        target=record['members'][member.linkname]
                        assert target['kind']=='file' and target['inode']==entry['inode'] and target['device']==entry['device']
                    raw=original.read_bytes();assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes']
                    assert archive.extractfile(member).read()==raw
                    assert info.st_ino==entry['inode'] and info.st_dev==entry['device'] and info.st_nlink==entry['nlink']
                elif entry['kind']=='symlink':
                    assert member.issym() and original.is_symlink() and os.readlink(original)==member.linkname==entry['target']
                else:
                    assert entry['kind']=='fifo' and member.isfifo() and stat.S_ISFIFO(info.st_mode)
    return {'archives':len(records),'regular_files':counts['file'],'symlinks':counts['symlink'],'fifos':counts['fifo'],
            'manifest_sha256':sha((OUT/'archives.json').read_bytes())}

def main():
    preflight='--preflight' in sys.argv
    result={'script_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(HELPER.read_bytes()),'preflight':preflight,'sources':sources()}
    assert git('show',result['sources']['baseline']['commits']['orze']+':'+str(HELPER.relative_to(CORE)))==HELPER.read_bytes()
    specs=[('baseline',0,(102,0,0,0)),('old-feature',1,(38,28,0,0)),('targeted-v1',0,(140,0,0,0)),
        ('old-order',1,None),('order-v1',0,None),('old-canary',1,None),('canary-v1',0,None),
        ('product-old-v1',1,None),('product-control-v1',1,None),('product-v1',1,None),
        ('product-old-v2',1,None),('product-control-v2',0,None),('product-v2',0,None),('archive-v1',0,None)]
    if not preflight:specs.append(('full-core',0,(5240,0,0,7)))
    result['runs']={name:prior.run(name,code,counts) for name,code,counts in specs}
    candidate=load(OUT/'targeted-v1/before.json')
    for name,_,_ in specs:
        if name not in ('baseline','old-feature','old-order','old-canary'):
            assert load(OUT/name/'before.json')==candidate,name
    old=dict(candidate['primary'])
    for name in ('src/orze/service/watchdog.py','tests/test_watchdog_failure_loop.py'):
        old[name]=result['sources']['baseline']['files'][name]['sha256']
    assert load(OUT/'old-feature/before.json')=={'primary':old}
    old.pop('tests/test_watchdog_runtime_admission.py')
    for name in ('baseline','old-order','old-canary'):assert load(OUT/name/'before.json')=={'primary':old}
    result['probes']=probes();result['products']=products();result['archives']=archive_checks();result['passed']=True
    target=OUT/('verification-preflight.json' if preflight else 'verification.json')
    with target.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'passed':True,'preflight':preflight,'sha256':sha(target.read_bytes()),'archives':result['archives']}))

if __name__=='__main__':main()
