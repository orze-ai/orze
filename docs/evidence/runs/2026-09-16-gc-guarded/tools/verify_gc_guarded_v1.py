"""Read-only check of packaged bytes, final source binding, and private link."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REL = Path('docs/evidence/runs/2026-09-16-gc-guarded')
PUBLIC, PRIVATE = ROOT / 'orze' / REL, ROOT / 'pro' / REL


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


counts = {}
for name, directory in (('public', PUBLIC), ('private', PRIVATE)):
    index = read(directory / 'files.json')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    assert actual == set(index) | {'files.json'}
    for relative, expected in index.items():
        raw = (directory / relative).read_bytes()
        assert expected == {'bytes':len(raw), 'sha256':sha(raw)}, relative
    counts[name] = len(index)
assert read(PRIVATE / 'public-link.json') == {
    'public_relative_path':str(REL), 'summary_sha256':sha((PUBLIC / 'summary.json').read_bytes())}
final = read(PUBLIC / 'runs/targeted-v5/projection.json')['before']['core']
for name, expected in final.items():
    assert sha((ROOT / 'orze' / name).read_bytes()) == expected, name
summary = read(PUBLIC / 'summary.json')
result = {'files':counts, 'runs':len(summary['runs']), 'archives':len(summary['archives']),
          'archive_entries':sum(a['entries'] for a in summary['archives']),
          'final_core_inventory_files':len(final), 'script_sha256':sha(Path(__file__).read_bytes())}
with (ROOT / 'gc-guarded/package-verification.json').open('x') as output:
    output.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
print(json.dumps(result))
