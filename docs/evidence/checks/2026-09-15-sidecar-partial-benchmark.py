"""Compare consumed partial-file ID hints through complete current ingress."""
import argparse
import contextlib
import os
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


def main(root, layout, repetitions, memory=False):
    from orze.engine import idea_ingress, sidecar_prefix
    from orze.core import ideas as current_ideas
    root.mkdir(parents=True, exist_ok=False)
    before_sources = fingerprint()
    base = ROOT / "docs/evidence/runs/2026-09-15-sidecar-partial-prefix/baseline"
    original_ideas = load("original_ideas", base / "ideas.py")
    original = load("original_prefix", base / "sidecar_prefix.py")
    original._iter_sidecar_ideas = original_ideas._iter_sidecar_ideas
    original._iter_sidecar_text = original_ideas._iter_sidecar_text
    modules = {"old": original, "new": sidecar_prefix}
    report = {"baseline": "e5289587a0c4aa445cfc420d77b7e174a9c24c03", "source_files": before_sources,
              "baseline_files": {p.name: sha(p.read_bytes()) for p in base.glob("*.py")},
              "script_sha256": sha(Path(__file__).read_bytes()), "helper_sha256": sha(Path(helper.__file__).read_bytes()),
              "layout": layout, "repetitions": repetitions, "mode": "memory" if memory else "time", "rows": [],
              "limits": ["Synthetic completed metadata and conflicting source titles; no scientific outcome or worker.",
                         "Real current source lock, fresh selected payload and unchanged immutable admission in both arms.",
                         "Each full cycle starts without hints; no cross-cycle cache of source contents.",
                         "Partial files are read fresh; only verified consumed IDs skip YAML in the previously consumed prefix.",
                         "Only baseline/current prefix classes change; section parser, ingress and current writers are identical.",
                         "Separate memory mode traces complete and first calls; tracing times are not ordinary latency.",
                         "Name windows bound names held at once, not total directory enumeration or I/O."]}
    for size in ((100, 1000, 5000) if layout == "many" else (1000, 5000)):
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
            def call(module, complete, observe=None):
                engine._idea_ingress_cursor = None
                count, batches, digest = 0, 0, hashlib.sha256()
                with patch.object(sidecar_prefix, "SidecarPrefix", module.SidecarPrefix):
                    while True:
                        raw, inserted = idea_ingress.ingest_ideas_source(engine, cfg)
                        assert not inserted
                        if observe is not None:
                            observe(engine._idea_sidecar_prefix)
                        count += len(raw)
                        batches += 1
                        digest.update(json.dumps(raw, sort_keys=True).encode())
                        if not complete or engine._idea_ingress_cursor[2] == 0:
                            break
                        assert batches <= (size + 127) // 128 + 1
                return {"records": count, "batches": batches, "payload_sha256": digest.hexdigest()}
            with patch.object(idea_ingress.logger, "warning", lambda *a, **k: None):
                first = helper.measure({name: lambda m=m: call(m, False) for name, m in modules.items()}, repetitions, memory=memory)
                full = helper.measure({name: lambda m=m: call(m, True) for name, m in modules.items()}, repetitions, memory=memory)
                reads = {}
                for name, module in modules.items():
                    fresh = idea_ingress._read_source
                    stamp = module._file_stamp
                    scan = os.scandir
                    parser = original_ideas if name == "old" else current_ideas
                    compile_re, parse_yaml = parser.re.compile, parser.yaml.safe_load
                    stats = {"sidecar_payload_reads": 0, "sidecar_payload_bytes": 0,
                             "file_identity_checks": 0, "directory_enumerations": 0,
                             "directory_entries": 0, "heading_matches": 0, "yaml_loads": 0,
                             "max_name_window": 0, "max_partial_ids": 0, "max_partial_id_bytes": 0, "begins": 0, "rollbacks": 0}
                    def counted(path):
                        value = fresh(path)
                        if path.parent == side:
                            stats["sidecar_payload_reads"] += 1
                            stats["sidecar_payload_bytes"] += len(value[0].encode())
                        return value
                    def counted_stamp(path):
                        stats["file_identity_checks"] += 1
                        return stamp(path)
                    @contextlib.contextmanager
                    def counted_scan(path):
                        with scan(path) as entries:
                            if Path(path) != side:
                                yield entries
                                return
                            stats["directory_enumerations"] += 1
                            def names():
                                for entry in entries:
                                    stats["directory_entries"] += 1
                                    yield entry
                            yield names()
                    class Pattern:
                        def __init__(self, value):
                            self.value = value
                        def finditer(self, text):
                            for match in self.value.finditer(text):
                                stats["heading_matches"] += 1
                                yield match
                    def compiled(pattern, *args, **kwargs):
                        value = compile_re(pattern, *args, **kwargs)
                        return Pattern(value) if isinstance(pattern, str) and pattern.startswith("^## (") else value
                    def parsed(value, *args, **kwargs):
                        stats["yaml_loads"] += 1
                        return parse_yaml(value, *args, **kwargs)
                    def observed(prefix):
                        stats["max_name_window"] = max(stats["max_name_window"], len(getattr(prefix, "name_window", ())))
                        partial = getattr(prefix, "partial", None)
                        ids = partial[2] if partial else ()
                        amount = sum(len(key.encode()) for key in ids)
                        stats["max_partial_ids"] = max(stats["max_partial_ids"], len(ids))
                        stats["max_partial_id_bytes"] = max(stats["max_partial_id_bytes"], amount)
                        assert len(prefix.files) + bool(partial) <= sidecar_prefix.MAX_FILES
                        assert prefix.id_count + len(ids) <= sidecar_prefix.MAX_IDS
                        assert prefix.id_bytes + amount <= sidecar_prefix.MAX_ID_BYTES
                    def statement(sql):
                        stats["begins"] += int(sql == "BEGIN IMMEDIATE")
                        stats["rollbacks"] += int(sql == "ROLLBACK")
                    engine.lake.conn.set_trace_callback(statement)
                    try:
                        with patch.object(idea_ingress, "_read_source", counted), patch.object(module, "_file_stamp", counted_stamp), patch.object(os, "scandir", counted_scan), patch.object(parser.re, "compile", compiled), patch.object(parser.yaml, "safe_load", parsed):
                            assert call(module, True, observed) == full["value"]
                    finally:
                        engine.lake.conn.set_trace_callback(None)
                    assert stats["max_name_window"] <= sidecar_prefix.NAME_WINDOW
                    assert stats["begins"] == stats["rollbacks"] == size
                    reads[name] = stats
                for key in ("sidecar_payload_reads", "sidecar_payload_bytes", "heading_matches", "directory_enumerations", "directory_entries", "begins", "rollbacks"):
                    assert reads["old"][key] == reads["new"][key], (key, reads)
                if layout == "many":
                    assert reads["old"]["file_identity_checks"] == reads["new"]["file_identity_checks"]
                    assert reads["old"]["yaml_loads"] == reads["new"]["yaml_loads"]
                else:
                    assert reads["old"]["yaml_loads"] > reads["new"]["yaml_loads"]
                    assert reads["old"]["file_identity_checks"] < reads["new"]["file_identity_checks"]
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
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    main(args.root.resolve(), args.layout, args.repetitions, args.memory)
