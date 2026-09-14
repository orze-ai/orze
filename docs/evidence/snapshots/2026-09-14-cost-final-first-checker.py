"""Independent read-only final audit. No product imports, tests, workers or installs.
Only aggregate/hash/private-location results are printed; Pro source stays private.
"""
import argparse
import ast
import base64
import csv
from email.parser import BytesParser
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

C = Path(__file__).resolve().parents[3]
P = C.parent / "orze-pro"
CC = "df446ab35e745daf02cc7c4a19cadffd57518103"
PC = "879859cb383d7f113f58ba1da054c6785e4fce6a"
CB = "e606ba6688b4e1972e26a08617821a427668dd24"
V = Path("docs/evidence/runs/2026-09-14-cost-validation")
PK = P / "docs/evidence/runs/2026-09-14-cost-package"
EXTRA = "docs/evidence/runs/2026-09-14-cost-equivalence/baseline/src/orze/core/cpu_action_budget.py"
BUILD = ("pyproject.toml","setup.cfg","requirements.txt","conftest.py","pytest.ini","setup.py","MANIFEST.in","tox.ini")
sha = lambda raw: hashlib.sha256(raw).hexdigest()
js = lambda p: json.loads(p.read_bytes())
digest = lambda obj: sha(json.dumps(obj,sort_keys=True,separators=(",",":")).encode())

def git(repo, *args, data=None):
    return subprocess.check_output(["git","-C",str(repo),*args], input=data)

def gitnames(repo, commit):
    return [n.decode() for n in git(repo,"ls-tree","-rz","--name-only",commit).split(b"\0") if n]

def blobs(repo, commit, names):
    names = list(names)
    raw = git(repo,"cat-file","--batch",data=b"".join((commit+":"+n+"\n").encode() for n in names))
    stream = io.BytesIO(raw); out = {}
    for n in names:
        header = stream.readline().split()
        assert len(header)==3 and header[1]==b"blob", (n,header)
        out[n] = stream.read(int(header[2]))
        assert stream.read(1)==b"\n"
    assert stream.read()==b""
    return out

def in_scope(n):
    return (n.startswith(("src/","tests/","examples/")) and "__pycache__" not in Path(n).parts
            and Path(n).suffix not in (".pyc",".pyo")) or n in BUILD or n==EXTRA

def inventory(repo, commit, count):
    expected_names = {n for n in gitnames(repo,commit) if in_scope(n)}
    current_names = {str(p.relative_to(repo)) for folder in ("src","tests","examples")
                     for p in (repo/folder).rglob("*") if p.is_file() and in_scope(str(p.relative_to(repo)))}
    current_names |= {n for n in (*BUILD,EXTRA) if (repo/n).is_file()}
    assert current_names==expected_names and len(current_names)==count
    original = blobs(repo,commit,sorted(expected_names))
    hashes = {n:sha(raw) for n,raw in original.items()}
    assert hashes=={n:sha((repo/n).read_bytes()) for n in current_names}
    return hashes

def pin(path, item):
    raw=path.read_bytes()
    assert sha(raw)==item["sha256"] and len(raw)==item["bytes"], str(path)
    return raw

def frozen(folder, maps):
    r=js(folder/"run.json")
    assert r["exit_code"]==0 and r["frozen"] is True and r["before"]==r["after"]
    assert js(folder/"before.json")==r["before"] and js(folder/"after.json")==r["after"]
    for key, repo in r["repositories"].items(): assert r["before"][key]==maps[repo]
    for name, item in r["files"].items(): pin(folder/name,item)
    assert sha(Path(r["recorder"]["path"]).read_bytes())==r["recorder"]["sha256"]
    return r

def junit(path, passed, skipped=0):
    tree=ET.parse(path); cases=tree.findall(".//testcase")
    assert len(cases)==passed+skipped
    assert len({(c.get("classname"),c.get("name")) for c in cases})==len(cases)
    assert not tree.findall(".//failure") and not tree.findall(".//error")
    actual_skips=[{"class":c.get("classname"),"name":c.get("name"),"reason":c.find("skipped").get("message"),
                  "text":c.find("skipped").text} for c in cases if c.find("skipped") is not None]
    assert len(actual_skips)==skipped
    for suite in tree.findall(".//testsuite"):
        assert int(suite.get("failures","0"))==int(suite.get("errors","0"))==0
        assert int(suite.get("tests"))==len(suite.findall("testcase"))
        assert int(suite.get("skipped","0"))==sum(c.find("skipped") is not None for c in suite.findall("testcase"))
    return {"passed":passed,"skipped":skipped,"skip_reasons":actual_skips,"sha256":sha(path.read_bytes())}

def footer(path, passed, skipped, warnings):
    rx=re.compile(r"(?:=+\s*)?(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) warnings?)? in ([0-9]+\.[0-9]+)s(?: \([0-9:]+\))?(?:\s*=+)?")
    matches=[rx.fullmatch(line.strip()) for line in path.read_text().splitlines()]
    matches=[m for m in matches if m]
    assert len(matches)==1
    m=matches[0]
    assert tuple(int(x or 0) for x in m.groups()[:3])==(passed,skipped,warnings)
    return {"line":m.group(0),"seconds":float(m.group(4)),"sha256":sha(path.read_bytes())}

def old_tests(repo, commit, expected_count, migration=False):
    names=[n for n in gitnames(repo,commit) if n.startswith("tests/")]
    raw=blobs(repo,commit,names); assert len(raw)==expected_count
    recorded=js(repo/V/("core-original-tests" if migration else "pro-original-tests")/"stdout.log")
    assert recorded["baseline"]==commit and set(recorded["files"])==set(names)
    changes={
        b"original_get_ids = instance.lake.get_all_ids\n":b"original_get_ids = instance.lake.find_existing_ids\n",
        b"def stale_ids_then_competing_admission():\n":b"def stale_ids_then_competing_admission(idea_ids):\n",
        b"old_ids = original_get_ids()\n":b"old_ids = original_get_ids(idea_ids)\n",
        b'monkeypatch.setattr(instance.lake, "get_all_ids", stale_ids_then_competing_admission)':b'monkeypatch.setattr(instance.lake, "find_existing_ids", stale_ids_then_competing_admission)',
    }
    changed=[]
    for n,original in raw.items():
        actual=(repo/n).read_bytes(); item=recorded["files"][n]
        assert item["original_sha256"]==sha(original) and item["current_sha256"]==sha(actual)
        expected=original
        if migration and n=="tests/test_idea_ingress_contract.py":
            for a,b in changes.items():
                assert expected.count(a)==1
                expected=expected.replace(a,b)
            assertions=lambda body:[ast.dump(a,include_attributes=False) for a in ast.walk(ast.parse(body)) if isinstance(a,ast.Assert)]
            assert assertions(original)==assertions(actual) and len(assertions(original))==26
            assert item["approved_migration"] is True and item["all_original_assertion_ast_exact"] is True
            changed.append({"path":n,"old_sha256":sha(original),"new_sha256":sha(actual),"assertions":26,"exact_replacements":4})
        assert actual==expected and item["preserved"] is True
    return {"files":expected_count,"byte_exact":expected_count-len(changed),"changes":changed,"record_sha256":sha((repo/V/("core-original-tests" if migration else "pro-original-tests")/"stdout.log").read_bytes())}

def record_check(raw, loader, expected_names=None):
    rows=list(csv.reader(io.StringIO(raw.decode())))
    assert all(len(r)==3 for r in rows) and len(rows)==len({r[0] for r in rows})
    if expected_names is not None: assert {r[0] for r in rows}==set(expected_names)
    for name,encoded,size in rows:
        if not encoded:
            assert name.endswith(".dist-info/RECORD") and not size
            continue
        body=loader(name)
        assert encoded=="sha256="+base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
        assert size==str(len(body))
    return len(rows)

def package_check():
    idx=js(PK/"archive-index.json"); review=js(PK/"review.json")
    assert review["status"]==idx["status"]=="passed"
    assert idx["source_commits"]=={"core":CC,"pro":PC}
    assert len(review["commands"])==22 and review["commands"]==js(PK/"commands.json")
    raw=(PK/idx["evidence"]["archive"]["path"]).read_bytes()
    assert sha(raw)==idx["evidence"]["archive"]["sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        payload={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        assert len(payload)==121 and len(t.getmembers())==121
    assert set(payload)==set(idx["evidence"]["archive_members"])
    original=Path(idx["root"])
    for name, data in payload.items():
        item=idx["evidence"]["archive_members"][name]
        assert sha(data)==item["sha256"] and len(data)==item["bytes"]
        assert (original/name).read_bytes()==data
        if (PK/name).is_file(): assert (PK/name).read_bytes()==data
    for name in ("review","verification"):
        item=idx["evidence"][name]; assert sha((PK/item["path"]).read_bytes())==item["sha256"]
    for command in review["commands"]:
        assert command["exit_code"]==0
        for stream in ("stdout","stderr"):
            p=command[stream]; assert pin(Path(p["path"]),p)==payload[str(Path(p["path"]).relative_to(original))]
    for label in ("core-only-check","paired-check"):
        command=next(c for c in review["commands"] if c["label"]==label)
        assert Path(command["stdout"]["path"]).read_text()=="No broken requirements found.\n"
    wheels={}; summary={}
    import tomli
    for label, repo, commit, module in (("core",C,CC,"orze"),("pro",P,PC,"orze_pro")):
        package=review["packages"][label]; assert package["commit"]==commit
        archive=Path(package["source_archive"]["path"]).read_bytes()
        assert archive==git(repo,"archive","--format=tar",commit,*package["archive_paths"])
        assert sha(archive)==package["source_archive"]["sha256"]
        with tarfile.open(fileobj=io.BytesIO(archive)) as t:
            sources={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
        wheel=Path(package["wheel"]).read_bytes();assert sha(wheel)==package["sha256"]
        with zipfile.ZipFile(io.BytesIO(wheel)) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            content={n:z.read(n) for n in z.namelist() if not n.endswith("/")}
        actual={n for n in content if n.startswith(module+"/")}
        conf=tomli.loads(sources["pyproject.toml"].decode())
        patterns=conf["tool"]["setuptools"]["package-data"][module]
        assert patterns==package["package_data_patterns"]
        expected={n[4:] for n in sources if n.startswith("src/"+module+"/") and
                  (n.endswith(".py") or any(fnmatch.fnmatch(n[len("src/"+module+"/"):],p) for p in patterns))}
        assert actual==expected==set(package["package_files"])
        for n in actual:
            assert content[n]==sources["src/"+n] and sha(content[n])==package["package_files"][n]
        if label=="core":assert "orze/SKILL.md" in actual
        recs={"wheel":record_check(content[package["record"]],content.__getitem__,content)}
        for mode, install in review["installs"].items():
            if label not in install["records"]:continue
            venv=Path(install["venv"]);site,=(venv/"lib").glob("python*/site-packages")
            for n in actual: assert (site/n).read_bytes()==content[n]
            def read(n):
                p=(site/n).resolve();assert p.is_relative_to(venv);return p.read_bytes()
            rec=read(package["record"])
            assert sha(rec)==install["records"][label]["sha256"]
            recs[mode]=record_check(rec,read)
        assert sum(n.endswith(".py") for n in actual)==package["python_files"]
        assert sum(not n.endswith(".py") for n in actual)==package["resource_files"]
        summary[label]={"commit":commit,"wheel_sha256":sha(wheel),"python_files":package["python_files"],
                        "resources":package["resource_files"],"record_rows":recs}
        wheels[label]=content
    lock=js(PK/"dependency-lock.json")
    old=js(Path("/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP/dependency-lock.json"))
    def normalize(name):return re.sub(r"[-_.]+","-",name).lower()
    deps={d["name"]:d for d in lock if d["name"] not in ("orze","orze-pro")}
    orig={d["name"]:d for d in old if d["name"] not in ("orze","orze-pro")}
    assert set(deps)==set(orig) and len(deps)==23
    for name,dep in deps.items():
        assert all(dep[k]==orig[name][k] for k in ("version","sha256","source_url"))
        assert sha(Path(dep["offline_wheel"]).read_bytes())==dep["sha256"]
    for mode, install in review["installs"].items():
        site,=(Path(install["venv"])/"lib").glob("python*/site-packages")
        distributions={}
        for path in site.glob("*.dist-info/METADATA"):
            meta=BytesParser().parsebytes(path.read_bytes())
            distributions[normalize(meta["Name"])]=meta["Version"]
        assert all(distributions[normalize(n)]==d["version"] for n,d in deps.items())
        assert distributions["orze"]=="4.6.2"
        assert ("orze-pro" in distributions)==(mode=="paired")
        if mode=="paired": assert distributions["orze-pro"]=="0.13.1"
        # RECORD verification above covers Core and Pro. Dependency wheel hashes
        # and installed versions are checked; no claim of byte-testing every dependency module.
    for mode, name in (("paired","paired.lock"),("core-only","core-only.lock")):
        data=(PK/name).read_text()
        expected=[d for d in lock if mode=="paired" or d["name"]!="orze-pro"]
        lines=[line for line in data.splitlines() if line.strip() and not line.startswith("#")]
        assert set(lines)=={d["name"]+"=="+d["version"]+" --hash=sha256:"+d["sha256"] for d in expected}
    assert review["workers_started"]==0 and review["license_read_or_patched"] is False
    assert review["pro_runtime_validated"] is False and review["production_switched"] is False
    return {"index_sha256":sha((PK/"archive-index.json").read_bytes()),"archive_members":121,"archive_sha256":sha(raw),
            "commands":22,"unchanged_dependency_wheels":23,"packages":summary,"bootstrap_copy_correction_retained":idx["archive_copy_correction"]},wheels

def installed_check(maps, wheels):
    folder=C/V/"core-installed-payload"; idx=js(folder/"index.json");r=js(folder/"report.json")
    assert r["core_commit"]==CC and r["exit_code"]==0 and r["source_test_before"]==r["source_test_after"]==maps[str(C)]
    assert r["wheel_sha256"]==sha(Path(r["wheel"]).read_bytes())
    for name,item in idx["files"].items():assert pin(folder/name,item)==Path(item["original"]).read_bytes()
    assert len(r["loaded_orze_modules"])==111
    for name,pin_data in r["loaded_orze_modules"].items():
        raw=Path(pin_data["file"]).read_bytes()
        assert Path(pin_data["file"]).is_relative_to(Path(r["installed_core_only_site"])/"orze")
        assert raw==wheels["core"][pin_data["wheel_member"]]
        assert sha(raw)==pin_data["sha256"]==maps[str(C)]["src/"+pin_data["wheel_member"]]
    assert not (Path(r["installed_core_only_site"])/"orze_pro").exists()
    for observation in r["sys_path_observations"]:
        assert str(C/"src") not in observation["sys_path"]
        assert observation["sys_path"][0]==r["installed_core_only_site"]
    assert len(r["events"])==35*3 and all(e["outcome"]=="passed" for e in r["events"])
    args=r["pytest_arguments"]
    assert args[args.index("--confcutdir")+1]==str(C/"tests")
    assert sha((C/"docs/evidence/runs/2026-09-14-cost-validation/run_installed.py").read_bytes())==r["launcher_sha256"]
    junit_data=junit(folder/"junit.xml",35)
    tarpin=idx["archive"]; data=(folder/tarpin["path"]).read_bytes();assert sha(data)==tarpin["sha256"]
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        contents={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
        assert len(contents)==len(tar.getmembers())==155 and set(contents)==set(tarpin["members"])
    for n,raw in contents.items():
        pin_data=tarpin["members"][n];assert sha(raw)==pin_data["sha256"] and len(raw)==pin_data["bytes"]
        assert (Path(idx["original"])/"pytest"/n).read_bytes()==raw
    products=[];sumworkers=0
    for path in sorted(folder.glob("*-report.json")):
        p=js(path); assert p["actual_cli_invocation_in_same_pytest_interpreter"] is True
        assert len(p["admissions"])==35 and len(p["after"]["cpu_proposal_requests"])==35
        sumworkers+=p["actual_native_workers"]
        if p["actual_native_workers"]:
            assert p["actual_native_workers"]==1 and p["outcomes"]==[0]
            row,=p["after"]["execution_attempts"];term=json.loads(row["terminal_json"]);binding=json.loads(row["binding_json"])
            tree,=p["actual_closure_receipts"]
            assert row["state"]=="TERMINAL" and term["outcome"]=="completed"
            assert tree==term["process_tree"] and tree["event"]=="TREE_CLOSED" and tree["wait_proof"]=="ECHILD_WALL"
            assert tree["binding"]==binding["supervision"]
            ref={k:row[k] for k in ("task_id","phase","attempt_id","generation")}
            assert tree["binding"]["identity"]["attempt_ref"]==ref
            reservation,=p["after"]["cpu_action_reservations"]
            assert reservation["state"]=="SETTLED" and json.loads(reservation["ref_json"])==ref
            permit=json.loads(reservation["permit_json"]);assert permit["reserved_nanoseconds"]=="2000000000"
            record=json.loads(p["after"]["research_artifacts"][0]["record_json"])
            artifact_member=next(n for n in contents if n.endswith("/"+record["artifact_id"]+"/content"))
            assert sha(contents[artifact_member])==record["content_sha256"] and record["producer"]==ref
            suffix="/"+ref["task_id"]+"/_execution_effects/"+ref["attempt_id"]+"/"
            prepared_key=next(n for n in contents if n.endswith(suffix+"prepared.json"))
            committed_key=next(n for n in contents if n.endswith(suffix+"committed.json"))
            prepared=contents[prepared_key];committed=json.loads(contents[committed_key])
            assert sha(prepared)==term["effect_receipt_sha256"]==committed["prepared_sha256"]
            assert {k:committed[k] for k in ref}==ref
        else:
            assert not p["after"]["cpu_action_reservations"]
        products.append({"report":path.name,"sha256":sha(path.read_bytes()),"native_workers":p["actual_native_workers"],
                         "metadata_prefixes":p["metadata_evidence_prefixes"],"cli_exits":p["outcomes"],
                         "proposal_receipts":35})
    assert len(products)==4 and sumworkers==1
    return {"index_sha256":sha((folder/"index.json").read_bytes()),"report_sha256":sha((folder/"report.json").read_bytes()),
            "loaded_modules":111,"every_observed_module_wheel_git_installed_exact":True,
            "junit":junit_data,"products":products,"archive_members":155,"archive_sha256":sha(data),
            "root_conftest_excluded_sha256":maps[str(C)]["conftest.py"],
            "unchanged_tests_conftest_sha256":maps[str(C)]["tests/conftest.py"],
            "limits":["External global pytest harness, not pure venv interpreter or fresh CLI process per test.",
                      "Only 1 real native worker among 4 product cases; 140 real proposal receipts and 33 metadata prefixes are not workers.",
                      "Source/terminal/effect historical raw checks do not grant replay or adoption authority."]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--core-passed",type=int);parser.add_argument("--core-skipped",type=int,default=7);parser.add_argument("--core-warnings",type=int,default=2);args=parser.parse_args()
    assert __debug__, "no -O"
    maps={str(C):inventory(C,CC,808),str(P):inventory(P,PC,231)}
    result={"schema":1,"source_commits":{"core":CC,"pro":PC},
            "input_inventory":{"core":{"files":808,"sha256":digest(maps[str(C)])},"pro":{"files":231,"sha256":digest(maps[str(P)])}},
            "original_tests":{"core":old_tests(C,CB,505,True),"pro":old_tests(P,PC,140)}}
    for repo,label in ((C,"core-original-tests"),(P,"pro-original-tests"),(C,"core-installed"),(P,"pro-full"),(P,"core-pro-optional")):
        frozen(repo/V/label,maps)
    result["completed_suites"]={}
    for repo,label,n in ((P,"pro-full",1037),(P,"core-pro-optional",31)):
        result["completed_suites"][label]={"junit":junit(repo/V/label/"junit.xml",n),
             "footer":footer(repo/V/label/"stdout.log",n,0,0),"run_sha256":sha((repo/V/label/"run.json").read_bytes())}
    result["installed_footer"]=footer(C/V/"core-installed/stdout.log",35,0,0)
    result["package"],wheels=package_check()
    result["installed"]=installed_check(maps,wheels)
    if args.core_passed is None:
        result["core_full"]={"status":"pending; full completion not claimed","before_sha256":sha((C/V/"core-full/before.json").read_bytes())}
        assert js(C/V/"core-full/before.json")["primary"]==maps[str(C)]
        result["status"]="partial_completed_items_verified"
    else:
        folder=C/V/"core-full";frozen(folder,maps)
        result["core_full"]={"status":"passed","junit":junit(folder/"junit.xml",args.core_passed,args.core_skipped),
                            "footer":footer(folder/"stdout.log",args.core_passed,args.core_skipped,args.core_warnings),
                            "run_sha256":sha((folder/"run.json").read_bytes())}
        result["status"]="all_requested_items_verified"
    assert inventory(C,CC,808)==maps[str(C)] and inventory(P,PC,231)==maps[str(P)]
    result["private_evidence_root"]="Pro: docs/evidence/runs/2026-09-14-cost-package and cost-validation; no private payload copied to Core."
    result["limits"]=["Read-only audit of original archived logs/JUnit/files; no rerun of test/worker/install commands.",
                      "Pro test runs use explicit offline license fixture, not actual paid license validation.",
                      "Package green or CPU canary does not establish production/Pro/role-release/GC filesystem compatibility.",
                      "No scientific research speedup or end-to-end throughput claim.",
                      "Dependency hashes and installed versions verified; individual dependency implementations not executed."]
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
