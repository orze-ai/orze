"""One real Haiku cycle with observational response capture; no parser changes."""
import hashlib
import json
import os
from pathlib import Path
import yaml
import real_memory_pilot_v2 as pilot
from examples.research_comparison.scheduling_campaign import _configuration, _native, _initialize_memory

pilot.credentials()
root = pilot.OUTPUT/'memory-response-diagnostic-v1'
root.mkdir()
pilot.write(root/'budget.json', {'authorized_total_usd':5000,'prior_reserved_upper_bound_usd':179,
                               'additional_upper_bound_usd':1,'basis':'one 100000-token Haiku envelope x $5/M maximum rate, doubled'})
spec=json.loads((pilot.OUTPUT/'haiku45-specification.json').read_bytes())
slot={'task_id':'memory-target','seed':17,'arm':'B','repetition':0}
request={'inputs':{**spec['shared'],**spec['tasks']['memory-target'],'treatment':spec['arms']['B']['treatment']},
         'protocol':spec['protocol'],'slot':slot}
cfg=_configuration(request,root)
config=root/'orze.yaml';config.write_text(yaml.safe_dump(cfg))
(root/'ideas.md').write_text('# Ideas\n')
rules=root/'instructions.md';rules.write_text(request['inputs']['instructions'])
run={'root':str(root),'cfg':cfg,'calls':[],'admissions':[],'snapshots':[],'campaign':{}}
_native(run,'initialize')
_initialize_memory(run,request)
cfg['_config_path']=str(config)
cfg['_research_config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
os.environ['ORZE_LLM_USAGE_LOG']=str(root/'usage.jsonl')
os.environ['ORZE_LLM_TOKEN_ENVELOPE']='100000'
from orze_pro.agents import research
original=research.call_llm
trace=[]
def observe(prompt,*args,**kwargs):
    response=original(prompt,*args,**kwargs)
    record={'prompt':prompt,'response':response,'result':dict(kwargs.get('result_out') or {})}
    pilot.write(root/('response-'+str(len(trace)+1)+'.json'),record)
    trace.append(record)
    return response
research.call_llm=observe
outcome={}
research.run_research_cycle('anthropic',1,root/'ideas.md',root/'results',cfg['report'],num_ideas=2,
    model='claude-haiku-4-5-20251001',endpoint='https://api.anthropic.com/v1',
    lake_db_path=root/'lake.db',project_cfg=cfg,rules_file=str(rules),
    rules_sha256=hashlib.sha256(rules.read_bytes()).hexdigest(),result_out=outcome)
record={'outcome':outcome,'responses':len(trace)}
if trace:
    response=trace[0]['response']
    parsed=research._try_parse_json(response)
    record.update(response_prefix=response[:50],ordinary_parser_ideas=len(parsed) if isinstance(parsed,list) else None)
pilot.write(root/'result.json',record)
print(json.dumps(record))
