"""Archive closed route audit fixtures using the existing immutable archive implementation."""
import importlib.util
from pathlib import Path
CORE=Path(__file__).resolve().parents[3]
HELPER=CORE/'docs/evidence/checks/2026-09-16-watchdog-runtime-admission-archive-v1.py'
spec=importlib.util.spec_from_file_location('existing_archive',HELPER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.OUT=CORE/'docs/evidence/runs/2026-09-16-service-route-audit'
module.ROOTS={
    'baseline':Path('/tmp/sra-b1'),'old-behavior':Path('/tmp/sra-old1'),'targeted-v1':Path('/tmp/sra-t1'),
    'old-route-probe':Path('/tmp/orze-service-route-audit-old-v2'),
    'fixed-route-probe':Path('/tmp/orze-service-route-audit-fixed-v1'),
    'canary-v1':Path('/tmp/orze-service-route-canary-v1'),
    'product-v1':Path('/tmp/orze-service-route-product-v1'),
    'cost-v1':Path('/tmp/orze-service-route-cost-v1')}
module.RUNS=[(name,1 if name in ('old-behavior','old-route-probe') else 0) for name in module.ROOTS]
module.main()
