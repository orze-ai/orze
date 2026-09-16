"""Independent stdlib audit of actual GC bytes/plans and reported cost data.

No product imports, process observations, signals, managers or execution.
Synthetic history-reader fixtures are identified separately and never counted
as completed GC or research experiments.
"""
import hashlib
import json
from pathlib import Path
import stat
import sys


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def retirements(project):
    count, files = 0, 0
    for intent_path in sorted(project.glob('results/idea-*/_gc_retirements/*/intent.json')):
        raw = intent_path.read_bytes()
        intent = json.loads(raw)
        assert intent['task'] == str(intent_path.parents[2])
        assert intent['token'] == intent_path.parent.name
        assert read(intent_path.with_name('completed.json')) == {'version':1, 'intent_sha256':sha(raw)}
        journal = Path(intent['journal'])
        plan_raw = (journal / 'guarded-plan.json').read_bytes()
        assert sha(plan_raw) == intent['plan_sha256']
        plan = json.loads(plan_raw)
        assert plan['version'] == 1 and plan['storage_mode'] == 'guarded'
        assert not Path(plan['source']).exists() and not Path(plan['source']).is_symlink()
        assert plan['bytes'] == sum(row[1][4] for row in plan['entries'] if stat.S_ISREG(row[1][2]))
        expected = 'archived' if plan['destination'] else 'deleted'
        assert read(journal / 'completed.json') == {'status':expected}
        if plan['destination']:
            destination = Path(plan['destination'])
            hashes = {}
            for path in (journal / 'archive-files').glob('*.json'):
                value = read(path)
                assert value['path'] not in hashes
                hashes[value['path']] = value['sha256']
            for relative, metadata in plan['entries']:
                target = destination if relative == '.' else destination / relative
                assert target.is_dir() == stat.S_ISDIR(metadata[2])
                if stat.S_ISREG(metadata[2]):
                    assert target.stat().st_size == metadata[4]
                    assert sha(target.read_bytes()) == hashes.pop(relative)
                    files += 1
            assert not hashes
        count += 1
    return count, files


def main():
    root, output = map(Path, sys.argv[1:])
    raw = root / 'gc-guarded'
    baseline = read(raw / 'baseline-product-fixture/result.json')
    final = read(raw / 'product-v2-fixture/result.json')
    assert baseline['filesystem'] == final['filesystem'] == 'ceph'
    assert all(r['stats']['errors'] == 1 and r['stats']['reasons'] == ['storage_atomic_rename_unsupported']
               for r in baseline['records'])
    expected = {'checkpoints': ('deleted',1,16), 'results': ('deleted_files',1,12), 'archive': ('archived_files',2,32)}
    for run in final['records']:
        project = raw / 'product-v2-fixture' / run['mode']
        key, count, size = expected[run['mode']]
        assert run['stats']['errors'] == 0 and run['stats'][key] == count
        assert run['stats']['moved_bytes' if key == 'archived_files' else 'freed_bytes'] == size
        current = {str(p.relative_to(project)): sha(p.read_bytes()) for p in project.rglob('*') if p.is_file()}
        assert current == run['remaining']
        assert retirements(project)[0] == count
        task = project / 'results/idea-disposable'
        assert (task / 'metrics.json').read_text() == '{"status":"FAILED"}'
        if run['mode'] == 'archive':
            target = project / 'archive/idea-disposable'
            assert (target / 'scratch.pt').read_bytes() == b'owned result'
            assert (target / 'overlays/nested/data').read_bytes() == b'owned nested archive'
    cost = read(raw / 'cost-v1-fixture/result.json')
    assert len(cost['gc']) == 12 and len(cost['reader']) == 8
    block_hash = sha(bytes(range(256)) * 2048)
    guarded, archive_files = 0, 0
    for run in cost['gc']:
        project = Path(run['root'])
        assert read(project / 'measurement.json') == run
        assert run['seconds'] > 0 and len(run['guard_seconds']) == 1
        assert 0 < run['guard_seconds'][0] < run['seconds']
        assert run['stats']['errors'] == 0
        assert run['stats']['moved_bytes' if run['operation'] == 'archive' else 'freed_bytes'] == 32 * 1024 * 1024
        if run['backend'] == 'guarded':
            count, files = retirements(project)
            assert count == 1
            guarded += count
            archive_files += files
        if run['operation'] == 'archive':
            files = list((project / 'archive/idea-cost/overlays').iterdir())
            assert len(files) == 64 and all(sha(path.read_bytes()) == block_hash for path in files)
    for run in cost['reader']:
        assert 'never execution' in run['scope']
        assert len(run['samples']) == 3
        assert run['median_seconds'] == sorted(s['seconds'] for s in run['samples'])[1]
        assert len(list(Path(run['fixture']).glob('_gc_retirements/*'))) == run['records']
    admission = Path('/tmp/gg-t4/test_fresh_claim_is_denied_whi0')
    assert read(admission / 'fresh-claim.json') == {'claimed':False, 'reason':'gc_retirement_unconfirmed', 'claim_exists':False}
    assert read(admission / 'fresh-claim-after.json') == {'claimed':False, 'reason':None, 'claim_exists':False}
    result = {'final_ceph_collect_calls':3, 'final_ceph_candidates':4,
              'cost_collect_calls':12, 'cost_guarded_retirements':guarded,
              'cost_guarded_archive_files':archive_files, 'reader_only_fixtures':8,
              'script_sha256':sha(Path(__file__).read_bytes()),
              'scope':'Static bytes and records; not new experiments or a repeat of past concurrent observations'}
    with output.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
