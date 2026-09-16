"""Preserve the Core-only runtime comparison and verify every archived byte."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'research-runtime-comparison'
OUT = ROOT / 'orze/docs/evidence/runs/2026-09-16-research-runtime-comparison'
spec = importlib.util.spec_from_file_location('retained', ROOT / 'package_research_campaign_v1.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    read = lambda path: json.loads(path.read_bytes())
    summary, proof = read(RAW / 'formal-03/summary.json'), read(RAW / 'independent-v1.json')
    assert summary['all_quality_passed'] and summary['native_actions'] == proof['native_actions'] == 192
    for path in RAW.glob('*.json'):
        helper.copy(path, OUT / path.name)
    for name in ('plan.json', 'summary.json'):
        helper.copy(RAW / 'formal-03' / name, OUT / name)
    for path in RAW.glob('*.log'):
        helper.copy(path, OUT / 'logs' / path.name)
    for arm in ('A', 'B'):
        helper.copy(RAW / arm / 'source.json', OUT / 'runtime' / arm / 'source.json')
        helper.copy(RAW / arm / 'wheels/orze-4.6.2-py3-none-any.whl', OUT / 'runtime' / arm / 'orze-4.6.2-py3-none-any.whl')
        for path in (RAW / arm).glob('*.log'):
            helper.copy(path, OUT / 'logs' / arm / path.name)
    for path in (RAW / 'application').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            helper.copy(path, OUT / 'application' / path.relative_to(RAW / 'application'))
    for name in ('research_runtime_comparison_v1.py', 'research_runtime_comparison_v2.py', 'research_runtime_comparison_v3.py',
                 'research_runtime_comparison_v4.py', 'audit_research_runtime_v1.py', 'package_research_runtime_v1.py', 'package_research_campaign_v1.py'):
        helper.copy(ROOT / name, OUT / 'tools' / name)
    helper.copy(ROOT / 'orze/docs/evidence/checks/2026-09-12-efficiency-formal-review.py', OUT / 'tools/independent_oracle.py')
    (OUT / 'raw').mkdir()
    archive = helper.archive(RAW / 'formal-03', OUT / 'raw/formal-03.tar.gz')
    helper.write(OUT / 'archives.json', [archive])
    files = {str(p.relative_to(OUT)): {'bytes': p.stat().st_size, 'sha256': helper.sha(p.read_bytes())}
             for p in sorted(OUT.rglob('*')) if p.is_file()}
    helper.write(OUT / 'files.json', files)
    assert set(files) | {'files.json'} == {str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()}
    for name, pinned in files.items():
        raw = (OUT / name).read_bytes()
        assert pinned == {'bytes': len(raw), 'sha256': helper.sha(raw)}
    value = {'payloads': len(files), 'index_sha256': helper.sha((OUT / 'files.json').read_bytes()),
             'archive_entries': archive['entries'], 'archive_sha256': archive['sha256'],
             'runs': 48, 'pairs': 24, 'native_actions': 192, 'archived_bytes_equal_to_original': True,
             'raw_root': str(RAW / 'formal-03'), 'real_model_calls': 0,
             'scope': 'New actual Core CPU comparison, no product changes. Existing independent scientific oracles and metric checks reused. No Pro inventory or source in public artifacts.'}
    helper.write(RAW / 'package-verification-v1.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
