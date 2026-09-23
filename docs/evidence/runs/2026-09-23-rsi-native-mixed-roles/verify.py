"""Independently check saved native role routing, exact prompts and settlement."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


closures = []
for suffix in ('', '-v2', '-v3', '-v4'):
    folder = ROOT / ('rsi-native-mixed-roles-20260923' + suffix)
    started = read(folder / 'started.json')
    closed = read(folder / 'process-closed.json')
    assert closed['closure']['binding'] == started['binding']
    assert closed['closure']['event'] == 'TREE_CLOSED'
    assert closed['closure']['wait_proof'] == 'ECHILD_WALL'
    assert closed['exit_code'] == (0 if suffix == '-v4' else 1)
    for identity in started['binding']['worker'], started['binding']['supervisor']:
        path = Path(f'/proc/{identity["pid"]}/stat')
        assert not path.exists() or int(path.read_text().rsplit(')', 1)[1].split()[19]) != identity['start_ticks']
    closures.append({'run': folder.name, 'exit_code': closed['exit_code'], 'tree_closed': True})

old = ROOT / 'rsi-native-mixed-roles-20260923-v3'
new = ROOT / 'rsi-native-mixed-roles-20260923-v4'
assert (old / 'run.py').read_bytes() == (new / 'run.py').read_bytes()
before_source = subprocess.check_output(['git', 'show', '518c51f:src/orze_pro/agents/research.py'], cwd=ROOT / 'pro')
for folder in (old, new):
    for filename, expected in read(folder / 'plan.json')['source_sha256'].items():
        source = before_source if folder == old and filename == 'pro/src/orze_pro/agents/research.py' else (ROOT / filename).read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected, filename

rows = []
all_bindings = []
for label, folder in [('before', old), ('after', new)]:
    for arm, expected in [('incumbent-routing', {'research': ('sonnet', 3)}),
                          ('mixed-routing', {'research_opus5': ('claude-opus-5', 1),
                                             'research_opus55': ('claude-opus-5-5', 1)})]:
        project = folder / arm
        state = project / '.orze/state'
        http = read(project / 'http-capture.json')
        assert not http['errors']
        requests = http['requests']
        assert len(requests) == len(expected)
        if len(expected) == 2:
            assert max(r['received'] for r in requests) <= min(r['response_started'] for r in requests)
        usage = lines(state / 'agent_usage.jsonl')
        reservations = [r for r in lines(state / 'agent_attempts.jsonl') if r['event'] == 'reserved']
        llm = lines(state / 'llm_usage.jsonl')
        assert len(usage) == len(reservations) == len(expected)
        assert {r['role'] for r in usage} == set(expected)
        with sqlite3.connect(f'file:{project / "lake.db"}?mode=ro', uri=True) as conn:
            conn.row_factory = sqlite3.Row
            terminals = [dict(r) for r in conn.execute(
                "SELECT * FROM trigger_delivery_transitions WHERE to_state='TERMINAL'")]
            assert len(terminals) == len(expected)
            assert conn.execute('SELECT COUNT(*) FROM ideas').fetchone()[0] == 0
        queued = re.findall(r'^## (idea-[a-z0-9]+):', (project / '.orze/ideas.md').read_text(), re.M)
        accepted = []
        for record in usage:
            aid = record['attempt_id']
            model, requested = expected[record['role']]
            assert record['model'] == model
            assert sum(r['attempt_id'] == aid for r in reservations) == 1
            terminal, = [r for r in terminals if r['attempt_id'] == aid]
            assert terminal['exit_code'] == 0 and terminal['cleanup_verified'] == 1
            result = read(state / f'agent_results/{aid}.json')
            prompt = read(state / f'agent_results/{aid}.prompt.json')
            assert result['identity'] == prompt['identity']
            assert result['identity']['role_name'] == record['role']
            assert result['result'] == record['native_result']
            assert record['new_ideas'] == result['result']['accepted_count']
            request, = [r for r in requests if r['request']['model'] == model]
            contents = [m['content'] for m in request['request']['messages']]
            assert any(hashlib.sha256(c.encode()).hexdigest() == prompt['manifest']['final_sha256']
                       and len(c.encode()) == prompt['manifest']['final_utf8_bytes'] for c in contents)
            bound = [r for r in llm if r.get('binding', {}).get('role_attempt_id') == aid]
            assert [r['event'] for r in bound] == ['call_started', 'attempt_prepared', 'attempt_finished', 'call_finished']
            assert all(r['model'] == model and r['binding']['role_name'] == record['role']
                       and r['binding']['prompt_manifest_sha256'] == digest(prompt['manifest']) for r in bound)
            completion = bound[2]
            assert completion['response_model'] == f'fixture-only:{model}'
            assert completion['response_id'] == f'fixture-{arm}-{request["request_number"]}'
            assert completion['total_tokens'] == 0 and completion['status'] == 'complete'
            assert bound[-1]['status'] == 'complete' and bound[-1]['attempts'] == 1
            assert all(r['pid'] == bound[0]['pid'] for r in bound)
            assert not Path(f'/proc/{bound[0]["pid"]}').exists()
            accepted.extend(result['result']['accepted_ids'])
            all_bindings.append({'run': label, 'arm': arm, 'role': record['role'], 'model': model,
                'attempt_id': aid, 'accepted_count': record['new_ideas'],
                'reason': result['result']['reason'], 'terminal': terminal,
                'provider_requests': 1, 'provider_binding_verified': True})
            if label == 'after' or arm == 'incumbent-routing':
                assert record['new_ideas'] == requested
                assert terminal['outcome'] == 'ok'
        assert sorted(queued) == sorted(accepted) and len(set(queued)) == len(queued)
        rows.append({'run': label, 'arm': arm, 'local_fixture_requests': len(requests),
                     'accepted_unexecuted_proposals': len(accepted), 'scientific_acceptances': 0})
assert [r['accepted_unexecuted_proposals'] for r in rows] == [3, 1, 3, 2]
blocked = [r for r in all_bindings if r['reason'] == 'queue_append_unavailable']
assert len(blocked) == 1 and blocked[0]['run'] == 'before'
before_states = read(old / 'mixed-routing/final-role-state.json')
assert before_states[blocked[0]['role']]['cooldown_override'] == 1800

checks = []
for name in ('core-tests.xml', 'pro-tests.xml'):
    suites = list(ET.parse(BASE / name).getroot().iter('testsuite'))
    totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    assert totals['tests'] > 0 and totals['failures'] == totals['errors'] == totals['skipped'] == 0
    checks.append({'file': name, **totals})

report = {'valid': True, 'finished': time.time(), 'rows': rows, 'native_bindings': all_bindings,
          'closure_checks': closures, 'tests': checks,
          'research_model_requests': 0, 'api_cost_usd': 0,
          'gpu_executions': 0, 'scientific_acceptances': 0,
          'candidate_bound_to_native_runtime': False,
          'efficiency_30_percent_proven': False,
          'remaining': ['Exclusive GPU native execution and independent acceptance',
                        'Actual native binding of candidate branch/refinement policy',
                        'Fresh frozen 32-episode prospective comparison'],
          'limitation': 'This fixes one real concurrent proposal loss. Local HTTP responses do not attest paid model identity or measure research capability. Separate native roles do not reproduce ParallelRefine.'}
with (BASE / 'verification.json').open('x') as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps({'valid': True, 'native_requests_checked': len(all_bindings), 'tests': checks}))
