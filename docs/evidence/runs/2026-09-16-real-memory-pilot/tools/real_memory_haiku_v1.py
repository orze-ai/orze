"""A separately frozen Haiku replication after DeepSeek's HTTP 402 refusal."""
import copy
import json
import os
from pathlib import Path
import time

import real_memory_pilot_v2 as pilot
from examples.research_comparison import batch, campaign
from examples.research_comparison.protocol import digest

pilot.credentials()
root = pilot.OUTPUT
reservation = {'total_authorized_usd': 5000, 'prior_reserved_upper_bound_usd': 146,
               'new_upper_bound_usd': 33, 'model': 'claude-haiku-4-5-20251001',
               'arithmetic': '8 projects x 4 processes x 100000-token envelope x $5/M = $16; reserve $32 plus $1 probe',
               'prices_per_million': {'input': 1, 'output': 5, 'cache_read': .1, 'cache_write_5m': 1.25, 'cache_write_1h': 2},
               'price_source': 'https://platform.claude.com/docs/en/about-claude/pricing',
               'reason': 'DeepSeek discovery succeeded but all 8 generation attempts returned HTTP 402; preserve that batch and run a separately identified available model.'}
assert reservation['prior_reserved_upper_bound_usd'] + reservation['new_upper_bound_usd'] <= reservation['total_authorized_usd']
pilot.write(root / 'haiku45-budget-reservation.json', reservation)
spec = json.loads((root / 'sonnet5-specification.json').read_bytes())
spec['shared']['model']['model'] = reservation['model']
plan = spec['protocol']
plan['comparison_id'] = 'real-memory-haiku45'
plan['shared']['model'] = digest(spec['shared']['model'])
for task in plan['tasks']:
    task['limits']['provider_cost_usd'] = 4
batch.validate(spec)
pilot.write(root / 'haiku45-specification.json', spec)
probe = root / 'haiku45-probe'
probe.mkdir()
from orze_pro.agents.research_llm import call_anthropic
os.environ['ORZE_LLM_USAGE_LOG'] = str(probe / 'usage.jsonl')
os.environ['ORZE_LLM_TOKEN_ENVELOPE'] = '2000'
result = {}
started = time.monotonic()
answer = call_anthropic('Reply with the single word READY.', os.environ['ANTHROPIC_API_KEY'],
                       model=reservation['model'], max_tokens=32, result_out=result)
pilot.write(probe / 'result.json', {'answer': answer, 'result': result, 'seconds': time.monotonic()-started})
if result.get('status') != 'complete':
    raise RuntimeError('Haiku connectivity check failed; batch not started')
os.environ.pop('ORZE_LLM_USAGE_LOG')
os.environ.pop('ORZE_LLM_TOKEN_ENVELOPE')
receipt = batch.execute(spec, root / 'haiku45', env=dict(os.environ))
pilot.write(root / 'haiku45-receipt.json', receipt)
report = batch.audit(root / 'haiku45', receipt['specification_sha256'])
pilot.write(root / 'haiku45-report.json', report)
print(json.dumps({'receipt': receipt, 'counts': report['counts']}))
