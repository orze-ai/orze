"""Alternate immutable ingress over identical metadata, including whole-file control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


helper = load("ingress_measure", ROOT / "docs/evidence/checks/2026-09-15-ingress-benchmark.py")
sha = helper.sha


def fingerprint():
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")}


def main(root, layout, repetitions):
    from orze.engine import idea_ingress, sidecar_prefix
    root.mkdir(parents=True, exist_ok=False)
    before_sources = fingerprint()
    base = ROOT / "docs/evidence/runs/2026-09-15-sidecar-prefix/baseline"
    original_ideas = load("original_ideas", base / "ideas.py")
    original = load("original_ingress", base / "idea_ingress.py")
    original._iter_sidecar_ideas = original_ideas._iter_sidecar_ideas
    modules = {"old": original, "new": idea_ingress}
    report = {"baseline": "24a7eea460fc528ac573db661afdd405c2b39b64", "source_files": before_sources,
              "baseline_files": {p.name: sha(p.read_bytes()) for p in base.glob("*.py")},
              "script_sha256": sha(Path(__file__).read_bytes()), "helper_sha256": sha(Path(helper.__file__).read_bytes()),
              "layout": layout, "repetitions": repetitions, "rows": [],
              "limits": ["Synthetic completed metadata and conflicting source titles; no scientific outcome or worker.",
                         "Real current source lock, fresh selected payload and unchanged immutable admission in both arms.",
                         "Each full cycle starts without hints; no cross-cycle cache of source contents.",
                         "A single partially traversed file still requires fresh prefix parsing.",
                         "Name windows bound names held at once, not total directory enumeration or I/O."]}
    for size in ((100, 1000, 5000) if layout == "many" else (1000,)):
        project = root / str(size)
        results = project / "results"
        results.mkdir(parents=True)
        source = project / "ideas.md"
        source.write_text("# Ideas\n")
        side = project / "ideas.d"
        side.mkdir()
        cfg = {"ideas_file": str(source), "results_dir": str(results),
               "idea_lake_db": str(project / "lake.db"), "_orze_dir": str(project / ".orze")}
        engine = helper.Orze.__new__(helper.Orze)
        engine.cfg, engine.results_dir, engine.active_roles = cfg, results, {}
        engine.lake = helper.IdeaLake(cfg["idea_lake_db"])
        try:
            blocks = []
            for index in range(size):
                key = f"idea-side-{index:05d}"
                engine.lake.conn.execute("INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,'','completed')",
                    (key, "Stored metadata", f"seed: {index}\n"))
                text = f"## {key}: Conflicting authored title\n```yaml\nseed: {index}\n```\n" + "note " * 40
                if layout == "many":
                    (side / f"{index:05d}.md").write_text(text)
                else:
                    blocks.append(text + "\n")
            if layout == "one":
                (side / "all.md").write_text("".join(blocks))
            engine.lake.conn.commit()
            database_before = sha("\n".join(engine.lake.conn.iterdump()).encode())
            source_files = {str(p): sha(p.read_bytes()) for p in [source, *side.glob("*.md")]}
            def call(module, complete):
                engine._idea_ingress_cursor = None
                count, batches, digest = 0, 0, hashlib.sha256()
                while True:
                    raw, inserted = module.ingest_ideas_source(engine, cfg)
                    assert not inserted
                    count += len(raw)
                    batches += 1
                    digest.update(json.dumps(raw, sort_keys=True).encode())
                    if not complete or engine._idea_ingress_cursor[2] == 0:
                        break
                    assert batches <= (size + 127) // 128 + 1
                return {"records": count, "batches": batches, "payload_sha256": digest.hexdigest()}
            with patch.object(idea_ingress.logger, "warning", lambda *a, **k: None):
                first = helper.measure({name: lambda m=m: call(m, False) for name, m in modules.items()}, repetitions, memory=True)
                full = helper.measure({name: lambda m=m: call(m, True) for name, m in modules.items()}, repetitions)
                reads = {}
                for name, module in modules.items():
                    fresh = module._read_source
                    stats = {"sidecar_payload_reads": 0, "sidecar_payload_bytes": 0}
                    def counted(path):
                        value = fresh(path)
                        if path.parent == side:
                            stats["sidecar_payload_reads"] += 1
                            stats["sidecar_payload_bytes"] += len(value[0].encode())
                        return value
                    with patch.object(module, "_read_source", counted):
                        assert call(module, True) == full["value"]
                    reads[name] = stats
            assert first["value"]["records"] == min(size, 128) and full["value"]["records"] == size
            assert source_files == {p: sha(Path(p).read_bytes()) for p in source_files}
            assert database_before == sha("\n".join(engine.lake.conn.iterdump()).encode())
            row = {"records": size, "source_files": source_files, "database_sha256": database_before,
                   "first_batch": first, "full_cycle": full, "separate_io_audit": reads,
                   "limits": {"files": sidecar_prefix.MAX_FILES, "ids": sidecar_prefix.MAX_IDS,
                              "id_bytes": sidecar_prefix.MAX_ID_BYTES, "name_window": sidecar_prefix.NAME_WINDOW}}
            report["rows"].append(row)
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"layout": layout, "records": size,
                "first_ms": {k: a["median_seconds"] * 1000 for k, a in first["arms"].items()},
                "full_ms": {k: a["median_seconds"] * 1000 for k, a in full["arms"].items()}, "reads": reads}), flush=True)
        finally:
            engine.lake.close()
    assert before_sources == fingerprint()
    report["passed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--layout", choices=("many", "one"), default="many")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    main(args.root.resolve(), args.layout, args.repetitions)
