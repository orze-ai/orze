"""Negative cost control: 128 genuine cross-ID duplicates, retained source."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ingress_bench", REPO / "docs/evidence/checks/2026-09-15-ingress-benchmark.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "results").mkdir()
    source = root / "ideas.md"
    text = "".join(f"## idea-duplicate-{i:04d}: Duplicate\n```yaml\nseed: 1\n```\n" for i in range(128))
    source.write_text(text)
    cfg = {"ideas_file": str(source), "results_dir": str(root / "results"),
           "idea_lake_db": str(root / "lake.db"), "_orze_dir": str(root / ".orze")}
    engine = b.Orze.__new__(b.Orze)
    engine.cfg, engine.results_dir, engine.active_roles = cfg, root / "results", {}
    engine.lake = b.IdeaLake(cfg["idea_lake_db"])
    try:
        engine.lake.insert("idea-winner", "Winner", "seed: 1\n", "", status="completed")
        engine._save_config_hash("idea-winner", {"seed": 1})
        old = b.load("orze.engine._duplicate_old", REPO / "docs/evidence/runs/2026-09-15-ingress-bounds/baseline/idea_ingress.py")
        before = b.sha("\n".join(engine.lake.conn.iterdump()).encode())
        def call(module):
            engine._idea_ingress_cursor = None
            raw, inserted = module.ingest_ideas_source(engine, cfg)
            assert not inserted and len(raw) == 128
            assert source.read_text() == text
            return {"raw_sha256": b.sha(json.dumps(raw, sort_keys=True).encode()), "inserted": inserted}
        with patch.object(b.idea_ingress.logger, "warning", lambda *a, **k: None):
            report = b.measure({"old": lambda: call(old), "new": lambda: call(b.idea_ingress)}, 7, memory=True)
        after = b.sha("\n".join(engine.lake.conn.iterdump()).encode())
        assert before == after
        report.update(database_sha256_before=before, database_sha256_after=after,
                      source_sha256=b.sha(text.encode()),
                      limits=["Synthetic completed winner metadata, not real execution evidence.",
                              "Single batch of genuine cross-ID duplicates, not a full source traversal.",
                              "Per-proposal warning output suppressed equally in both measurement arms."])
        (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: a["median_seconds"] * 1000 for k, a in report["arms"].items()}))
    finally:
        engine.lake.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
