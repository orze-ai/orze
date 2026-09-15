"""Complete ingress with many unparsed headings before one retained valid proposal."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'docs/evidence/runs/2026-09-15-sidecar-scan-windows/baseline'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

helper=load('ingress_measure',ROOT/'docs/evidence/checks/2026-09-15-ingress-benchmark.py')
sha=helper.sha

def main(root):
    from orze.core import ideas
    from orze.engine import idea_ingress,sidecar_prefix
    root.mkdir(parents=True,exist_ok=False)
    old_ideas=load('old_ideas',BASE/'ideas.py')
    old_prefix=load('old_prefix',BASE/'sidecar_prefix.py')
    old_prefix._iter_sidecar_text=old_ideas._iter_sidecar_text
    old_prefix._iter_sidecar_ideas=old_ideas._iter_sidecar_ideas
    paths=[ROOT/'src/orze/core/ideas.py',ROOT/'src/orze/engine/sidecar_prefix.py',ROOT/'src/orze/engine/idea_ingress.py',BASE/'ideas.py',BASE/'sidecar_prefix.py']
    sources={str(p):sha(p.read_bytes()) for p in paths}
    report={'baseline':'a952321b33e6973d6c580f5593139bebdce19e30','sources':sources,
        'script_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(Path(helper.__file__).read_bytes()),
        'repetitions':3,'rows':[],
        'limits':['Headings without YAML are skipped before a valid tail proposal; imported metadata, no worker.',
                  'Complete real ingress retains exact title conflict through one current writer/rollback.',
                  'Each call starts a new traversal, freshly reads the complete bounded file and scans every heading.',
                  'Python tracing is a separate pass; three alternating ordinary samples after warmup.',
                  'No scientific, production, or universal latency claim; source-size limit remains unchanged.']}
    for count in (500,5000,50000):
        project=root/str(count);project.mkdir();side=project/'ideas.d';side.mkdir()
        source=project/'ideas.md';source.write_text('# Ideas\n');results=project/'results';results.mkdir()
        text=''.join(f'## idea-empty-{i:05d}: Empty\n' for i in range(count))+'## idea-tail: Authored title\n```yaml\nseed: 1\n```\n'
        assert len(text.encode())<=4*1024*1024
        tail=side/'all.md';tail.write_text(text)
        cfg={'ideas_file':str(source),'results_dir':str(results),'idea_lake_db':str(project/'lake.db'),'_orze_dir':str(project/'.orze')}
        engine=helper.Orze.__new__(helper.Orze);engine.cfg,engine.results_dir,engine.active_roles=cfg,results,{}
        engine.lake=helper.IdeaLake(cfg['idea_lake_db'])
        try:
            engine.lake.insert('idea-tail','Stored metadata','seed: 1\n','',status='completed')
            before=sha('\n'.join(engine.lake.conn.iterdump()).encode())
            def call(module):
                engine._idea_ingress_cursor=None
                with patch.object(sidecar_prefix,'SidecarPrefix',module.SidecarPrefix):
                    raw,inserted=idea_ingress.ingest_ideas_source(engine,cfg)
                assert list(raw)==['idea-tail'] and raw['idea-tail']['config']=={'seed':1} and not inserted
                assert source.read_text()=='# Ideas\n' and engine._idea_ingress_cursor[2]==0
                return {'raw_sha256':sha(json.dumps(raw,sort_keys=True).encode()),'inserted':inserted}
            modules={'old':old_prefix,'new':sidecar_prefix}
            with patch.object(idea_ingress.logger,'warning',lambda *a,**k:None):
                measured=helper.measure({name:lambda module=module:call(module) for name,module in modules.items()},3,memory=True)
                counters={}
                for name,module in modules.items():
                    counted={'begins':0,'rollbacks':0,'heading_matches':0,'yaml_loads':0}
                    parser=old_ideas if name=='old' else ideas
                    compile_re,parse_yaml=parser.re.compile,parser.yaml.safe_load
                    class Pattern:
                        def __init__(self,value):self.value=value
                        def finditer(self,text):
                            for match in self.value.finditer(text):
                                counted['heading_matches']+=1;yield match
                    def compiled(pattern,*args,**kwargs):
                        value=compile_re(pattern,*args,**kwargs)
                        return Pattern(value) if isinstance(pattern,str) and pattern.startswith('^## (') else value
                    def parsed(value,*args,**kwargs):
                        counted['yaml_loads']+=1;return parse_yaml(value,*args,**kwargs)
                    def statement(sql):
                        counted['begins']+=int(sql=='BEGIN IMMEDIATE');counted['rollbacks']+=int(sql=='ROLLBACK')
                    engine.lake.conn.set_trace_callback(statement)
                    try:
                        with patch.object(parser.re,'compile',compiled),patch.object(parser.yaml,'safe_load',parsed):
                            assert call(module)==measured['value']
                    finally:engine.lake.conn.set_trace_callback(None)
                    assert counted=={'begins':1,'rollbacks':1,'heading_matches':count+1,'yaml_loads':2},counted
                    counters[name]=counted
            assert before==sha('\n'.join(engine.lake.conn.iterdump()).encode()) and tail.read_text()==text
            report['rows'].append({'unparsed_headings':count,'source_bytes':len(text.encode()),'source_sha256':sha(text.encode()),
                'database_sha256':before,'counters':counters,**measured})
            (root/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
            print(json.dumps({'unparsed_headings':count,'ms':{k:v['median_seconds']*1000 for k,v in measured['arms'].items()},
                'python_peak':{k:v['traced_peak_bytes'] for k,v in measured['arms'].items()}}),flush=True)
        finally:engine.lake.close()
    assert sources=={str(p):sha(p.read_bytes()) for p in paths};report['passed']=True
    (root/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
