"""Independently inspect fixed source, raw runs, artifacts and archive bytes."""
import ast, hashlib, importlib.util, json, statistics, subprocess, sys
from pathlib import Path
import xml.etree.ElementTree as ET
CORE=Path(__file__).resolve().parents[3]
OUT=CORE/'docs/evidence/runs/2026-09-16-service-route-audit'
CHECKS=CORE/'docs/evidence/checks'
HELPER=CHECKS/'2026-09-16-watchdog-runtime-admission-verify-v1.py'
spec=importlib.util.spec_from_file_location('prior_native_and_archive_checks',HELPER)
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);prior.OUT=OUT
sha=lambda raw:hashlib.sha256(raw).hexdigest()
load=lambda p:json.loads(p.read_bytes())
def git(*args):return subprocess.check_output(['git','-C',str(CORE),*args])
def run(name,code,counts=None):
    folder=OUT/name;value=load(folder/'run.json')
    assert value['frozen'] and value['exit_code']==code
    assert value['before']==value['after']==load(folder/'before.json')==load(folder/'after.json')
    assert value['repositories']=={'primary':str(CORE)}
    assert value['environment']=={'PYTHONPATH':str(CORE/'src')+':'+str(CORE.parent/'pro/src'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
    assert sha(Path(value['recorder']['path']).read_bytes())==value['recorder']['sha256']
    for name,record in value['files'].items():
        raw=(folder/name).read_bytes();assert len(raw)==record['bytes'] and sha(raw)==record['sha256']
    if counts:
        suites=ET.parse(folder/'junit.xml').getroot().findall('testsuite')
        actual=tuple(sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped'))
        assert actual==counts,(folder,actual,counts)
    return {k:value[k] for k in ('frozen','exit_code','started_unix','finished_unix')}
def sources():
    manifest=load(OUT/'baseline-inputs/manifest.json');head=manifest['commit']
    assert head=='a3058f952d57c5c61d4c611d4003920c320b99d0'
    for name,entry in manifest['files'].items():
        raw=git('show',head+':'+name)
        assert raw==(OUT/'baseline-inputs'/name).read_bytes()
        assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes']
        assert git('rev-parse',head+':'+name).decode().strip()==entry['git_blob']
    changed={'src/orze/core/config.py','src/orze/service/runtime_contract.py'}
    unchanged={}
    for name in git('ls-tree','-r','--name-only',head,'--','src','tests','examples').decode().splitlines():
        if name not in changed:
            raw=git('show',head+':'+name);assert raw==(CORE/name).read_bytes(),name
            unchanged[name]=sha(raw)
    candidates=load(OUT/'candidate-v1/manifest.json')
    for name,digest in candidates.items():
        assert (CORE/name).read_bytes()==(OUT/'candidate-v1'/name).read_bytes()
        assert sha((CORE/name).read_bytes())==digest
    before=ast.parse((OUT/'baseline-inputs/src/orze/core/config.py').read_text())
    after=ast.parse((CORE/'src/orze/core/config.py').read_text())
    changed_functions={'_expand_env_vars','find_dotenv','_apply_dotenv','_load_dotenv','resolve_project_results'}
    for tree in (before,after):
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in changed_functions]
    assert ast.dump(before)==ast.dump(after)
    final=load(OUT/'targeted-v1/before.json')['primary']
    for name,digest in final.items():assert sha((CORE/name).read_bytes())==digest,name
    old=dict(final)
    for name in changed:old[name]=manifest['files'][name]['sha256']
    assert load(OUT/'old-behavior/before.json')=={'primary':old}
    old.pop('tests/test_service_route_audit.py')
    assert load(OUT/'baseline/before.json')==load(OUT/'old-route-probe/before.json')=={'primary':old}
    return {'baseline_commit':head,'candidates':candidates,'unchanged':unchanged,'current':final}
def probes():
    result={}
    for name,path,allowed in [('old-route-probe','/tmp/orze-service-route-audit-old-v2',True),('fixed-route-probe','/tmp/orze-service-route-audit-fixed-v1',False)]:
        value=load(Path(path)/'observation.json')
        assert value['script_sha256']==sha((CHECKS/'2026-09-16-service-route-audit-probe-v1.py').read_bytes())
        assert value['initial_audit']['startup_allowed']
        assert value['after_config_change_audit']['startup_allowed'] is allowed
        assert value['target_stop_present'] and value['service_metadata_unchanged']
        assert value['saved_results']!=value['current_config_results']
        assert value['actual_controllers']==value['actual_managers']==0
        result[name]=value
    root=Path('/tmp/orze-service-route-canary-v1');value=load(root/'observation.json');cleanup=load(root/'cleanup.json')
    assert value['script_sha256']==sha((CHECKS/'2026-09-16-service-route-audit-canary-v1.py').read_bytes())
    assert value['canary_exit_before_cleanup'] is None and value['signals_before_cleanup']==[]
    assert not value['audit_before']['startup_allowed'] and value['watchdog_error']=='runtime_contract_rejected'
    assert cleanup=={'child_pid':value['own_canary_identity'][0]['pid'],'reaped':True,'exit_code':-15}
    result['canary']={'observed':value,'cleanup':cleanup}
    return result
def products():
    root=Path('/tmp/orze-service-route-product-v1');report=load(root/'report.json')
    assert report['passed'] and len(report['cases'])==3
    assert report['script_sha256']==sha((CHECKS/'2026-09-16-service-route-audit-product-v1.py').read_bytes())
    assert report['helper_sha256']==sha((CHECKS/'2026-09-16-service-install-routing-product-v1.py').read_bytes())
    assert report['baseline_runtime_sha256']==sha((OUT/'baseline-inputs/src/orze/service/runtime_contract.py').read_bytes())
    for row in report['cases']:
        case=root/row['mode'];prior.native(case,row)
        assert row['controller_alive_after_watchdog'] and row['pid_file_unchanged']
        assert len(row['invocations'])==2 and all(x['exit_code']==0 for x in row['invocations'])
        before_watchdog=load(case/'after-watchdog.json')
        assert before_watchdog['invocations']==[]
        assert all(row[k]==v for k,v in before_watchdog.items() if k!='invocations')
        assert {k:v for k,v in row.items() if k!='passed'}==load(case/'after-replay.json')
        cleanup=load(case/'controller-reaped.json')
        assert cleanup=={'pid':row['controller_identity'][0]['pid'],'exit_code':0,'reaped':True}
        initial=row['initial_install']['result'];saved=load(case/'service.json')
        assert initial['saved']==saved and initial['calls']==[['crontab',saved]]
        assert initial['error'] is None and initial['manager_calls']==0
        assert row['watchdog_service']==saved==load(case/'watchdog-service.json')
        package,=saved['runtime_packages']
        from orze.service.runtime_contract import _hash_package_tree
        assert package['name']=='orze' and _hash_package_tree(Path(package['root']))==(package['sha256'],package['file_count'])
        inputs=load(case/'route-inputs.json');assert Path(inputs['target_stop']).exists()
        assert (case/'project/orze.yaml').read_text()==inputs['config_before']
        assert (case/'project/.env').read_text()==inputs['dotenv_before']
        if row['mode']=='matching':
            assert row['audit']['startup_allowed'] and row['watchdog_error'] is None
            assert row['watchdog_events']==['read_pid','own_controller_kill0']
            assert inputs['config_before']==inputs['config_at_audit'] and inputs['dotenv_before']==inputs['dotenv_at_audit']
        else:
            assert not row['audit']['startup_allowed'] and 'service_results_route_drift' in row['audit']['errors']
            assert row['watchdog_error']=='runtime_contract_rejected' and row['watchdog_events']==[]
            field='config' if row['mode']=='config-drift' else 'dotenv'
            assert inputs[field+'_before']!=inputs[field+'_at_audit']
    return {'workers':3,'cli_invocations':6,'report_sha256':sha((root/'report.json').read_bytes())}
def costs():
    root=Path('/tmp/orze-service-route-cost-v1');report=load(root/'report.json')
    assert report['passed'] and report['script_sha256']==sha((CHECKS/'2026-09-16-service-route-audit-cost-v1.py').read_bytes())
    for padding,row in zip((0,65536,262144),report['cases'],strict=True):
        raw=(root/str(padding)/'orze.yaml').read_bytes()
        assert len(raw)==row['config_bytes'] and sha(raw)==row['config_sha256']
        assert row==load(root/str(padding)/'cost.json') and row['outputs_identical']
        assert [s['variant'] for s in row['samples']]==['old','new','new','old','new','old','old','new']
        for variant in ('old','new'):
            for key in ('wall_ns','cpu_ns'):
                assert row['median'][variant][key]==statistics.median(s[key] for s in row['samples'] if s['variant']==variant)
    return report
def main():
    preflight='--preflight' in sys.argv
    specs=[('baseline',0,(140,0,0,0)),('old-route-probe',1,None),('old-behavior',1,(18,17,0,0)),
           ('targeted-v1',0,(169,0,0,0)),('fixed-route-probe',0,None),('canary-v1',0,None),
           ('product-v1',0,None),('cost-v1',0,None),('archive-v1',0,None)]
    if not preflight:specs.append(('core-full',0,(5344,0,0,6)))
    result={'script_sha256':sha(Path(__file__).read_bytes()),'preflight':preflight,'sources':sources(),
            'runs':{name:run(name,code,counts) for name,code,counts in specs}}
    final=load(OUT/'targeted-v1/before.json')
    for name,_,_ in specs:
        if name not in ('baseline','old-route-probe','old-behavior'):assert load(OUT/name/'before.json')==final,name
    result.update(probes=probes(),products=products(),costs=costs(),archives=prior.archive_checks(),passed=True)
    with (OUT/('verification-preflight.json' if preflight else 'verification.json')).open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'passed':True,'preflight':preflight,'archives':result['archives']}))
if __name__=='__main__':main()
