"""New history requirements with real V2 Lake/registration, not a live grant.

The host is explicitly the existing Session-unit stand-in. No restart or
pidfd-based issuance is claimed by a direct historical verifier test.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest

import test_controller_session as fixture


BOOT = fixture.BOOT.replace("scope=pathlib.Path(cfg['results_dir']); scope.mkdir()", r'''
from orze.core.controller_profile import profile_fingerprint
cfg['controller_control']={'version':2,'profile':'local_handoff_v1'}
cfg['_config_path']=str(root/'orze.yaml')
(root/'orze.yaml').write_text(json.dumps({k:v for k,v in cfg.items() if not k.startswith('_')}))
cfg['_controller_profile_fingerprint']=profile_fingerprint(cfg)
scope=pathlib.Path(cfg['results_dir']); scope.mkdir()
''')


@pytest.mark.parametrize('fault', ['global_unknown', 'stage_unknown', 'projection', 'schema_bool', 'identity_float', 'generation_float'])
def test_unknown_or_noncanonical_history_never_grants_a_successor(tmp_path, fault):
    import json
    code = BOOT + "\nfault=" + repr(fault) + r'''
from orze.engine import controller_handoff as handoff
from orze.engine.supervisor_worker import canonical
if fault in {'global_unknown','stage_unknown','projection'}:
    host.lake.conn.execute("INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES ('task','title','{}','','queued')")
    host.lake.conn.execute("INSERT INTO idea_state(idea_id,current_state) VALUES (?,?)",('task','MYSTERY' if fault=='global_unknown' else 'QUEUED'))
    if fault=='stage_unknown':
        host.lake.conn.execute("INSERT INTO idea_stage_state(idea_id,stage,current_state,updated_at) VALUES ('task','training','MYSTERY','fixture')")
    if fault=='projection':
        host.lake.conn.execute("UPDATE ideas SET status='archived' WHERE idea_id='task'")
    host.lake.conn.commit()
ack=session.finish()
route=handoff._Route(cfg)
if fault in {'schema_bool','identity_float','generation_float'}:
    with route.connection(write=True) as conn:
        identity=json.loads(conn.execute('SELECT identity_json FROM controller_instances').fetchone()[0])
        binding=json.loads(conn.execute('SELECT binding_json FROM controller_sessions').fetchone()[0])
        if fault=='schema_bool': ack['schema']=True
        elif fault=='identity_float': identity['schema']=2.0
        else: identity['generation']=0.0
        binding['identity']=identity
        encoded=canonical(binding).decode()
        conn.execute('UPDATE controller_instances SET identity_json=?',(canonical(identity).decode(),))
        conn.execute('UPDATE controller_sessions SET binding_json=?',(encoded,))
        request=json.loads(conn.execute('SELECT request_json FROM controller_sessions').fetchone()[0])
        request['binding_sha256']=handoff._sha(encoded.encode())
        request_json=canonical(request).decode()
        ack['binding_sha256']=handoff._sha(encoded.encode())
        ack['request_sha256']=handoff._sha(request_json.encode())
        conn.execute('UPDATE controller_sessions SET request_json=?,ack_json=?',(request_json,canonical(ack).decode()))
replay=os.environ.get('ORZE_TEST_HANDOFF_HISTORY_REPLAY')
if replay:
    exec(compile(pathlib.Path(replay).read_bytes(),replay,'exec'),handoff.__dict__)
try: handoff._validate_history(route,session.ctx.controller_id,0)
except ControllerHOLD: pass
else: raise AssertionError('unknown or noncanonical history accepted')
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path), json.dumps(fixture._config(tmp_path))],
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONPATH':fixture.SOURCE, 'CUDA_VISIBLE_DEVICES':''},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pidfd = os.pidfd_open(child.pid)
    try:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stdout + stderr
    finally:
        fixture._close(child, pidfd)
