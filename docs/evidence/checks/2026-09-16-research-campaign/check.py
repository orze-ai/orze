"""Read-only final file/source linkage check; no product or provider imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REL = Path('docs/evidence/runs/2026-09-16-research-campaign')
PUBLIC, PRIVATE = ROOT / 'orze' / REL, ROOT / 'pro' / REL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    counts = {}
    for label, destination in (('public', PUBLIC), ('private', PRIVATE)):
        files = read(destination / 'files.json')
        for name, pinned in files.items():
            path = destination / name
            assert path.stat().st_size == pinned['bytes'] and sha(path) == pinned['sha256']
        counts[label] = len(files)
    assert read(PRIVATE / 'public-link.json')['summary_sha256'] == sha(PUBLIC / 'summary.json')
    final = read(PRIVATE / 'runs/targeted-v6/run.json')
    for key, names in final['before'].items():
        root = Path(final['repositories'][key])
        for name, pinned in names.items():
            assert sha(root / name) == pinned
    for name in ('targeted-v6', 'pro-regression', 'paired'):
        original = PRIVATE / 'runs' / name / 'run.json'
        projection = read(PUBLIC / 'runs' / name / 'projection.json')
        assert projection['original_run_sha256'] == sha(original)
        assert projection['before'] == projection['after'] and projection['exit_code'] == 0
        assert all(not key.startswith('src/orze_pro/') for key in projection['before']['core'])
    expected = read(PUBLIC / 'snapshots/candidate-v7/manifest.json')
    assert all(sha(ROOT / 'orze' / name) == pinned['sha256'] for name, pinned in expected.items())
    result = {'files': counts, 'final_sources_unchanged': True, 'private_public_link_matches': True,
              'checker_sha256': sha(Path(__file__))}
    out = ROOT / 'research-campaign/final-check.json'
    with out.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
