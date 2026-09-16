"""Recompute scheduling quality from a frozen native execution capture.

This audits the task/ledger portion of a comparison. It does not authenticate
an experiment collector, establish complete campaign or treatment identity,
read paths mentioned inside the capture, or authorize execution. A controller
must supply the expected attempt universe independently of surviving results.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys

from examples.holdout import scheduling as domain
from orze.core.cpu_action_contract import action_fingerprint
from orze.core.cpu_observation_contract import cpu_observation_binding, cpu_observation_records
from orze.core.execution_attempts import AttemptRef
from orze.core.research_artifacts import _record as artifact_record
from orze.core.research_observations import validate_observation_record

from .protocol import digest, keys, number, read_json
from .report import METRICS

MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_ATTEMPTS = 10000
_REF = ('task_id', 'phase', 'attempt_id', 'generation')


def _require(condition, reason):
    if not condition:
        raise ValueError('scheduling audit: ' + reason)


def _same(left, right):
    return digest(left) == digest(right)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _ref(value):
    keys(value, _REF, 'attempt reference')
    _require(all(type(value[k]) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value[k])
                 for k in _REF[:3]) and value['phase'] == 'action'
             and number(value['generation'], integer=True) and value['generation'] > 0, 'invalid attempt reference')
    return dict(value)


def _row_ref(row):
    return _ref({k: row[k] for k in _REF})


def _mapping(rows, field, label):
    _require(type(rows) is list and len(rows) <= MAX_ATTEMPTS * 4, 'invalid ' + label + ' inventory')
    result = {}
    for row in rows:
        _require(type(row) is dict and type(row.get(field)) is str and row[field] not in result,
                 'duplicate or invalid ' + label)
        result[row[field]] = row
    return result


def _closed(closure, binding, return_code):
    _require(type(return_code) is int and type(closure) is dict
             and closure.get('event') == 'TREE_CLOSED' and closure.get('wait_proof') == 'ECHILD_WALL'
             and closure.get('worker_returncode') == return_code
             and closure.get('stop_requested') is False and closure.get('forced_cleanup') is False
             and _same(closure.get('binding'), binding), 'unconfirmed process closure')


def _clocks(record, attempts):
    snapshots = _mapping(record['snapshots'], 'label', 'snapshot')
    ordered = {name: i for i, name in enumerate(snapshots)}
    _require(snapshots and not next(iter(snapshots.values()))['database']['execution_attempts'],
             'capture must start before native attempts')
    _require(_same(next(reversed(snapshots.values()))['database'], record['database']), 'final snapshot mismatch')
    calls = record['calls']
    _require(type(calls) is list and 0 < len(calls) <= MAX_ATTEMPTS * 4, 'invalid controller inventory')
    previous, wall, previous_index, completed = None, 0.0, -1, Counter()
    births = set()
    for call in calls:
        start, finish, duration = (call[k] for k in ('started_monotonic', 'finished_monotonic', 'wall_seconds'))
        _require(all(number(v) for v in (start, finish, duration)) and start <= finish
                 and math.isclose(finish-start, duration, rel_tol=0, abs_tol=1e-8)
                 and (previous is None or previous <= start), 'invalid controller clocks')
        before, after = call['before_snapshot'], call['after_snapshot']
        _require(before in snapshots and after in snapshots and previous_index < ordered[before] < ordered[after],
                 'controller snapshot order mismatch')
        supervision = call['controller_supervision'];binding = supervision['binding']
        _require(call['exit_code'] == 0, 'controller did not complete')
        _closed(supervision['closure'], binding, call['exit_code'])
        birth = (binding['worker']['pid'], binding['worker']['start_ticks'])
        _require(birth not in births, 'duplicate controller invocation')
        births.add(birth)
        prior = {digest(_row_ref(r)): r for r in snapshots[before]['database']['execution_attempts']}
        for row in snapshots[after]['database']['execution_attempts']:
            key = digest(_row_ref(row))
            if row['state'] == 'TERMINAL' and (key not in prior or prior[key]['state'] != 'TERMINAL'):
                _require(key in attempts and _same(row, attempts[key]), 'controller terminal differs from final ledger')
                completed[key] += 1
        previous, previous_index, wall = finish, ordered[after], wall + duration
    _require(dict(completed) == {key: 1 for key in attempts}, 'attempt missing from controller transitions')
    _require(number(wall), 'controller time overflow')
    return wall


def audit_scheduling(record, *, instance, protocol, expected_attempt_refs, selected_ref,
                     confirmation_ref=None, worker_command=None):
    """Verify task artifacts, native accounting and evaluator replication.

Returns ``measurement`` in the reducer's shape, plus scope and coverage.
Callers must verify campaign/arm/model/input provenance and resource ledgers
before using it as a complete comparison adapter. The current evaluator is
reapplied to bytes; saved scores are never accepted on their own.
"""
    try:
        # Capture ordinary JSON containers once. The existing in-process test
        # collector uses a Path only for its top-level root label.
        captured = dict(record)
        if isinstance(captured.get('root'), Path):
            captured['root'] = str(captured['root'])
        raw = json.dumps({'record': captured, 'instance': instance, 'protocol': protocol,
            'expected': expected_attempt_refs, 'selected': selected_ref, 'confirmation': confirmation_ref,
            'command': worker_command}, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
        _require(len(raw) <= MAX_CAPTURE_BYTES, 'capture exceeds byte limit')
        inputs = read_json(raw)
        return _audit(inputs)
    except (KeyError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ValueError('scheduling audit: malformed capture') from exc


def _audit(inputs):
    record = inputs['record'];instance = domain.validate_instance(inputs['instance'])
    protocol = inputs['protocol']
    _require(protocol in domain.PROTOCOLS, 'unsupported expected evaluator protocol')
    expected = inputs['expected']
    _require(type(expected) is list and 0 < len(expected) <= MAX_ATTEMPTS, 'invalid expected attempt inventory')
    expected = [digest(_ref(row)) for row in expected]
    _require(len(expected) == len(set(expected)), 'duplicate expected attempt')
    selected = digest(_ref(inputs['selected']))
    confirmation = digest(_ref(inputs['confirmation'])) if inputs['confirmation'] is not None else None
    _require(selected in expected and (confirmation is None or confirmation in expected and confirmation != selected),
             'selection or independent confirmation reference invalid')
    cfg = record['cfg'];scope = cfg['results_dir']
    expected_domain = {'version': 1, 'kind': 'schedule_holdout', 'config': {'instance': instance}}
    _require(_same(cfg['action_domain'], expected_domain), 'instance or domain identity mismatch')
    worker_command = inputs['command'] or [sys.executable, str(Path(domain.__file__).resolve())]
    _require(type(worker_command) is list and len(worker_command) == 2
             and all(type(v) is str and v and '\0' not in v for v in worker_command), 'invalid expected worker command')
    db = record['database']
    raw_attempts = _mapping(db['execution_attempts'], 'attempt_id', 'attempt')
    attempts = {digest(_row_ref(row)): row for row in raw_attempts.values()}
    _require(set(attempts) == set(expected), 'actual attempt universe differs from expected inventory')
    ideas = _mapping(db['ideas'], 'idea_id', 'idea')
    reservations = _mapping(db['cpu_action_reservations'], 'reservation_id', 'reservation')
    _require(len(reservations) == len(attempts), 'reservation universe mismatch')
    artifacts, observations = {}, {}
    for row in _mapping(db['research_artifacts'], 'artifact_id', 'artifact').values():
        artifact = read_json(row['record_json']);artifact_record(artifact)
        _require(row['artifact_id'] == artifact['artifact_id'] and row['logical_name'] == artifact['logical_name']
                 and all(row['producer_'+k] == artifact['producer'][k] for k in _REF), 'artifact row identity mismatch')
        artifacts[artifact['artifact_id']] = artifact
    for row in _mapping(db['research_observations'], 'observation_id', 'observation').values():
        observation = validate_observation_record(read_json(row['record_json']))
        _require(row['observation_id'] == observation['observation_id'] and row['name'] == observation['name']
                 and all(row['evaluator_'+k] == observation['evaluator'][k] for k in _REF), 'observation row identity mismatch')
        observations[observation['observation_id']] = observation
    _require(type(record['artifact_contents']) is dict and set(record['artifact_contents']) == set(artifacts),
             'artifact byte inventory mismatch')
    contents = {}
    for identity, artifact in artifacts.items():
        text = record['artifact_contents'][identity]
        _require(type(text) is str, 'artifact bytes unavailable')
        content = text.encode('utf-8')
        _require(len(content) == artifact['size_bytes'] <= domain.MAX_ARTIFACT_BYTES
                 and _sha(content) == artifact['content_sha256'], 'artifact bytes differ from publication')
        contents[identity] = content
    wall = _clocks(record, attempts)
    outcomes = Counter({'completed':0, 'failed':0, 'interrupted':0})
    evaluations, bindings = {}, {}
    used_artifacts, used_observations, used_reservations = set(), set(), set()
    reserved_ns, native_elapsed, analysis_actions = 0, 0.0, 0
    evaluator = domain.SchedulingDomain({'instance': instance})
    for key, attempt in attempts.items():
        ref = _row_ref(attempt);binding = read_json(attempt['binding_json']);terminal = read_json(attempt['terminal_json'])
        _require(attempt['state'] == 'TERMINAL' and attempt['hold_reason'] is None
                 and terminal['outcome'] in outcomes and _same(binding['attempt_ref'], ref)
                 and binding['attempt_id'] == ref['attempt_id'] and binding['resource'] == 'cpu'
                 and binding['kind'] == 'native_cpu_action', 'invalid native terminal or identity')
        _require(_same(binding['supervision']['identity']['attempt_ref'], ref), 'native supervision reference mismatch')
        _closed(terminal['process_tree'], binding['supervision'], terminal['return_code'])
        _require(number(terminal['elapsed_wall_seconds']), 'invalid native elapsed time')
        outcome = terminal['outcome'];outcomes[outcome] += 1
        native_elapsed += terminal['elapsed_wall_seconds'];bindings[key] = binding
        reservation = reservations.get(binding['reservation_id'])
        _require(reservation is not None and reservation['reservation_id'] not in used_reservations
                 and reservation['state'] == 'SETTLED' and _same(read_json(reservation['ref_json']), ref)
                 and reservation['task_id'] == ref['task_id'] and reservation['terminal_sha256'] == digest(terminal),
                 'settled reservation differs from terminal')
        permit = read_json(reservation['permit_json']);charge = permit['reserved_nanoseconds']
        _require(type(charge) is str and re.fullmatch(r'[1-9][0-9]{0,18}', charge) is not None
                 and permit['reservation_id'] == reservation['reservation_id'] and permit['task_id'] == ref['task_id']
                 and permit['wall_limit_seconds'] == 2 and int(charge) == 2000000000
                 and _same(permit['budget_scope']['declaration'], cfg['execution'])
                 and permit['budget_scope']['results_dir'] == scope, 'invalid scoped budget charge')
        reserved_ns += int(charge);used_reservations.add(reservation['reservation_id'])
        native = binding['domain_run'];idea = ideas[ref['task_id']]
        config = read_json(idea['config'])
        keys(config, ('kind','domain_request'), 'native idea config')
        _require(config['kind'] == 'native_cpu_action' and native['domain_kind'] == 'schedule_holdout'
                 and native['domain_id'] == 'acceptance.schedule.v1'
                 and native['domain_config_sha256'] == digest(expected_domain)
                 and native['request_sha256'] == _sha(idea['config'].encode()), 'native task input identity mismatch')
        request = config['domain_request'];sources = []
        for item in native['source_snapshot']['inputs']:
            source = item['artifact']
            _require(source['artifact_id'] in artifacts and _same(source, artifacts[source['artifact_id']]),
                     'native source differs from published artifact')
            _require(digest(source['producer']) in attempts, 'source producer outside declared attempt universe')
            sources.append(source)
        prepared = evaluator.prepare(request, sources)
        prepared['action']['command'] = worker_command
        fingerprint = action_fingerprint(prepared['action'])
        _require(binding['action_sha256'] == native['action_sha256'] == fingerprint
                 and binding['command_sha256'] == digest(worker_command)
                 and binding['inputs_sha256'] == digest(prepared['action']['inputs'])
                 and _same(native['observation'], prepared['observation']), 'prepared action identity mismatch')
        produced = [value for value in artifacts.values() if _same(value['producer'], ref)]
        observed = [value for value in observations.values() if _same(value['evaluator'], ref)]
        produced_ids = [value['artifact_id'] for value in produced]
        observed_ids = [value['observation_id'] for value in observed]
        _require(type(terminal['artifact_ids']) is list and type(terminal['observation_ids']) is list
                 and sorted(terminal['artifact_ids']) == sorted(produced_ids)
                 and sorted(terminal['observation_ids']) == sorted(observed_ids), 'terminal publication membership mismatch')
        used_artifacts.update(produced_ids);used_observations.update(observed_ids)
        operation = request['payload']['operation'];analysis_actions += int(operation == 'evaluate')
        if outcome != 'completed':
            _require(not produced and not observed, 'failed scheduling action published qualified output')
            continue
        _require(terminal['return_code'] == 0 and type(terminal['effect_receipt_sha256']) is str
                 and re.fullmatch(r'[0-9a-f]{64}', terminal['effect_receipt_sha256']), 'unconfirmed native effect')
        _require(len(produced) == 1 and produced[0]['scope'] == scope
                 and produced[0]['spec_fingerprint'] == fingerprint, 'scheduling output identity mismatch')
        expected_name = 'candidate' if operation == 'produce' else 'evaluation'
        _require(produced[0]['logical_name'] == expected_name, 'unexpected scheduling output')
        if operation == 'produce':
            _require(not observed, 'producer cannot assign scientific quality')
            continue
        envelope = read_json(contents[produced_ids[0]])
        verdict = domain.evaluate(instance, contents[sources[0]['artifact_id']], request['payload']['protocol'])
        _require(_same(envelope['verdict'], verdict), 'saved evaluator verdict differs from candidate bytes')
        claims = evaluator.interpret(prepared, envelope)
        declaration = prepared['observation']
        publication = cpu_observation_binding(adapter_id=declaration['adapter_id'],
            spec_fingerprint=declaration['spec_fingerprint'], protocol_fingerprint=declaration['protocol_fingerprint'],
            scope=scope, input_artifacts=sources)
        rebuilt = cpu_observation_records(AttemptRef(**ref), publication, produced_ids, claims)
        _require(_same(binding['observation_publication'], publication) and _same(list(rebuilt), observed),
                 'published observation differs from recomputed quality')
        evaluations[key] = {'ref':ref, 'protocol':request['payload']['protocol'],
            'candidate_artifact_id':sources[0]['artifact_id'], 'candidate_sha256':sources[0]['content_sha256'],
            'verdict':verdict}
    _require(used_artifacts == set(artifacts) and used_observations == set(observations)
             and used_reservations == set(reservations), 'unassigned publication or reservation')
    _require(number(reserved_ns, integer=True) and number(native_elapsed), 'aggregate overflow')
    selected_value = evaluations.get(selected)
    if selected_value is not None:
        _require(selected_value['protocol'] == protocol, 'selected evaluation protocol differs from expected')
    valid = selected_value is not None and selected_value['verdict']['status'] == 'valid'
    confirmed = False
    if confirmation is not None:
        other = evaluations.get(confirmation)
        _require(other is not None and other['protocol'] == protocol, 'confirmation evaluator unavailable or incompatible')
        confirmed = bool(valid and other['verdict']['status'] == 'valid'
            and _same(other['verdict'], selected_value['verdict'])
            and other['candidate_artifact_id'] == selected_value['candidate_artifact_id']
            and bindings[confirmation]['action_sha256'] == bindings[selected]['action_sha256'])
    metrics = {key:None for key in METRICS}
    metrics.update(native_actions=len(attempts), reserved_seconds=reserved_ns/1e9,
        analysis_actions=analysis_actions, cli_wall_seconds=wall, native_elapsed_seconds=native_elapsed)
    counts = Counter({'valid':0,'invalid':0,'unknown':0})
    for value in evaluations.values():counts[value['verdict']['status']] += 1
    measurement = {'status':'completed' if valid else 'failed', 'metrics':metrics,
        'quality':{'valid':valid,'confirmed':confirmed,'score':selected_value['verdict']['scheduled_value'] if valid else None,
            'comparison_key':{'instance_sha256':digest(instance),'protocol':protocol,'direction':'maximize'}},
        'observations':dict(counts),'native_outcomes':dict(outcomes)}
    return {'schema':1,'scope':'frozen_scheduling_artifacts_and_declared_native_attempts',
        'new_research_evidence':False,'campaign_identity_verified':False,
        'confirmation_scope':'independent_evaluator_attempt_same_candidate',
        'expected_attempt_inventory_sha256':digest(inputs['expected']), 'measurement':measurement,
        'evaluations':list(evaluations.values()),'evaluator_sha256':_sha(Path(domain.__file__).read_bytes())}


def read_capture(path, expected_sha256):
    """Read one bounded, digest-pinned capture without following its embedded paths."""
    _require(type(expected_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', expected_sha256), 'invalid capture digest')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_CAPTURE_BYTES, 'capture is not a bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(MAX_CAPTURE_BYTES+1)
        after = os.fstat(fd)
        identity = lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        _require(len(raw) <= MAX_CAPTURE_BYTES and identity(before) == identity(after)
                 and _sha(raw) == expected_sha256, 'capture changed or digest mismatch')
        return read_json(raw)
    finally:
        os.close(fd)
