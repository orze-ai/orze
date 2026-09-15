"""Compare full ingress and unchanged databases across edits and hint limits."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/evidence/runs/2026-09-15-sidecar-partial-prefix/baseline"
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


helper = load("partial_diff_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-benchmark.py")


def block(index, value=None, title="Current definition"):
    return f"## idea-side-{index:04d}: {title}\n```yaml\nseed: {index if value is None else value}\n```\n"


def fixtures(layout):
    chunks = []
    for index in range(360):
        text = block(index)
        if index % 23 == 0:
            text = block(index, "[") + text
        if index % 17 == 0:
            text += block(index, 999, "Later duplicate")
        if index % 31 == 0:
            text += block(index + 1000, 0, "PI_DIRECTIVE") + "## Unknown heading\nretained raw\n"
        chunks.append(text)
    groups = ({"one": [360], "many": [1] * 360, "mixed": [20, 200, 140], "small_groups": [30] * 12})[layout]
    files, start = {}, 0
    for index, count in enumerate(groups):
        files[f"{index:04d}.md"] = "".join(chunks[start:start + count]).encode()
        start += count
    files["z.md"] = "".join(block(i) for i in range(360, 376)).encode()
    return files, "0001.md" if layout == "mixed" else "0000.md"


def publish(side, files):
    for p in side.iterdir():
        p.unlink()
    for name, raw in files.items():
        (side / name).write_bytes(raw)


def main(root):
    from orze.engine import idea_ingress, sidecar_prefix
    assert Path(sidecar_prefix.__file__).resolve() == ROOT / "src/orze/engine/sidecar_prefix.py"
    root.mkdir()
    original = load("partial_diff_original", BASE / "sidecar_prefix.py")
    sources = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")}
    report = {"baseline": "e5289587a0c4aa445cfc420d77b7e174a9c24c03", "script_sha256": sha(Path(__file__).read_bytes()),
              "helper_sha256": sha(Path(helper.__file__).read_bytes()), "sources": sources,
              "baseline_sha256": sha((BASE / "sidecar_prefix.py").read_bytes()), "cases": [],
              "limits": ["64 deterministic complete-ingress pairs; imported completed metadata is not executed research.",
                         "Same current source reader, locks and normal writer in both arms; conflicting proposals remain retained.",
                         "Edits occur after the first page; transient empty-reader failures have unchanged file identities.",
                         "No elapsed-time claim from this behavioral differential."]}
    mutations = ["none", "rewrite_restore_mtime", "replace", "namespace", "delete", "invalid_utf8", "empty_reader", "truncate"]
    for case in range(64):
        layout = ("one", "many", "mixed", "small_groups")[(case // 8) % 4]
        mutation = mutations[case % 8]
        limited = case >= 32
        limits = {"MAX_FILES": 1, "MAX_IDS": 64, "MAX_ID_BYTES": 512} if limited else {"MAX_FILES": 8192, "MAX_IDS": 32768, "MAX_ID_BYTES": 1024 * 1024}
        files, target = fixtures(layout)
        project = root / f"case-{case:03d}"
        results = project / "results"
        results.mkdir(parents=True)
        side = project / "ideas.d"
        side.mkdir()
        source = project / "ideas.md"
        source.write_text("# Ideas\n")
        plan = {"layout": layout, "mutation": mutation, "target": target, "limits": limits,
                "initial_files_base64": {n: base64.b64encode(raw).decode() for n, raw in files.items()}}
        (project / "input.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        cfg = {"ideas_file": str(source), "results_dir": str(results), "idea_lake_db": str(project / "lake.db"), "_orze_dir": str(project / ".orze")}
        engine = helper.Orze.__new__(helper.Orze)
        engine.cfg, engine.results_dir, engine.active_roles = cfg, results, {}
        engine.lake = helper.IdeaLake(cfg["idea_lake_db"])
        try:
            engine.lake.conn.executemany("INSERT INTO ideas(idea_id,title,config,raw_markdown,status) VALUES (?,?,?,'','completed')",
                [(f"idea-side-{i:04d}", "Stored metadata", f"seed: {i}\n") for i in range(376)])
            engine.lake.conn.commit()
            database = sha("\n".join(engine.lake.conn.iterdump()).encode())
            arms = {}
            final_sources = {}
            for arm, module in (("old", original), ("new", sidecar_prefix)):
                publish(side, files)
                engine._idea_ingress_cursor = None
                changed = []
                real_read = idea_ingress._read_sidecar
                def read(path):
                    return "" if mutation == "empty_reader" and changed and path.name == target else real_read(path)
                outputs = []
                with patch.multiple(module, **limits), patch.object(sidecar_prefix, "SidecarPrefix", module.SidecarPrefix), patch.object(idea_ingress, "_read_sidecar", read), patch.object(idea_ingress.logger, "warning", lambda *a, **k: None):
                    while True:
                        raw, inserted = idea_ingress.ingest_ideas_source(engine, cfg)
                        assert not inserted
                        outputs.append({"ids": list(raw), "raw_sha256": sha(json.dumps(raw, sort_keys=True).encode()), "offset": engine._idea_ingress_cursor[2]})
                        if not changed:
                            path = side / target
                            old = path.stat()
                            if mutation == "rewrite_restore_mtime":
                                path.write_bytes(re.sub(rb"seed: [0-9]+", b"seed: 777", path.read_bytes(), count=1))
                                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                            elif mutation == "replace":
                                replacement = path.with_suffix(".replacement")
                                replacement.write_bytes(path.read_bytes())
                                replacement.replace(path)
                            elif mutation == "namespace":
                                (side / "-new.md").write_text(block(0, 777, "New first definition"))
                            elif mutation == "delete":
                                path.unlink()
                            elif mutation == "invalid_utf8":
                                path.write_bytes(b"\xff\xfe invalid UTF-8")
                            elif mutation == "truncate":
                                path.write_text(block(359, 777))
                            changed.append(True)
                        if engine._idea_ingress_cursor[2] == 0:
                            break
                        assert len(outputs) < 10
                arms[arm] = outputs
                final_sources[arm] = {str(p.relative_to(project)): sha(p.read_bytes()) for p in [source, *sorted(side.iterdir())]}
                assert database == sha("\n".join(engine.lake.conn.iterdump()).encode())
            assert arms["old"] == arms["new"], (case, layout, mutation, arms)
            assert final_sources["old"] == final_sources["new"]
            report["cases"].append({"case": case, "layout": layout, "mutation": mutation, "limits": limits,
                                     "input_sha256": sha((project / "input.json").read_bytes()), "outputs": arms["new"],
                                     "sources": final_sources["new"], "database_sha256": database})
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        finally:
            engine.lake.close()
    assert sources == {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")}
    report["passed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "pairs": len(report["cases"]), "report_sha256": sha((root / "report.json").read_bytes())}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
