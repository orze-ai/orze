"""Interactive init — analyze codebase, generate project, start orze.

Calling spec:
    from orze.cli_interactive import do_interactive_init
    do_interactive_init(project_dir, venv_python)

Called after do_init() scaffolds template files. Detects the codebase,
calls an LLM to generate GOAL.md / train.py / ideas.md, asks the user
to confirm, installs deps, and starts orze.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _print(msg: str):
    print(f"  {msg}")


def _step(msg: str):
    print(f"\n{BOLD}{CYAN}==> {msg}{RESET}")


def _ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str):
    print(f"  {YELLOW}!{RESET} {msg}")


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def _detect_ai_cli() -> str | None:
    """Detect if running inside an AI CLI. Returns CLI name or None."""
    if os.environ.get("CLAUDE_CODE"):
        return "claude"
    if os.environ.get("GEMINI_CLI"):
        return "gemini"
    if os.environ.get("CODEX_CLI"):
        return "codex"
    # Heuristic: check parent process names
    try:
        ppid = os.getppid()
        cmdline = Path(f"/proc/{ppid}/cmdline").read_bytes().decode(errors="ignore")
        if "claude" in cmdline.lower():
            return "claude"
        if "gemini" in cmdline.lower():
            return "gemini"
    except Exception:
        pass
    return None


def _detect_gpus() -> list[dict]:
    """Detect GPUs via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10,
        )
        gpus = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_mib": int(parts[2].replace(" MiB", "")),
                })
        return gpus
    except Exception:
        return []


def _detect_framework(project_dir: Path) -> str | None:
    """Detect ML framework from imports in Python files."""
    for pattern in ["*.py", "**/*.py"]:
        for f in list(project_dir.glob(pattern))[:50]:
            try:
                text = f.read_text(errors="ignore")[:5000]
                if "import torch" in text or "from torch" in text:
                    return "pytorch"
                if "import tensorflow" in text or "from tensorflow" in text:
                    return "tensorflow"
                if "import jax" in text or "from jax" in text:
                    return "jax"
            except Exception:
                continue
    return None


def _find_all_api_keys() -> list[tuple[str, str]]:
    """Find all available LLM API keys. Returns list of (key, backend) pairs."""
    found = []
    seen_backends = set()

    for var, backend in [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("GEMINI_API_KEY", "gemini"),
        ("OPENAI_API_KEY", "openai"),
    ]:
        val = os.environ.get(var)
        if val:
            found.append((val, backend))
            seen_backends.add(backend)

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            for env_var, backend in [
                ("ANTHROPIC_API_KEY", "anthropic"),
                ("GEMINI_API_KEY", "gemini"),
                ("OPENAI_API_KEY", "openai"),
            ]:
                if key == env_var and val and backend not in seen_backends:
                    found.append((val, backend))
                    seen_backends.add(backend)

    return found


def _find_api_key() -> tuple[str | None, str | None]:
    """Find the first available LLM API key."""
    keys = _find_all_api_keys()
    return keys[0] if keys else (None, None)


def _gather_codebase_context(project_dir: Path) -> str:
    """Gather codebase context for LLM analysis."""
    ctx_parts = []

    # README
    for name in ["README.md", "README.rst", "README.txt", "README"]:
        p = project_dir / name
        if p.exists():
            ctx_parts.append(f"=== {name} ===\n{p.read_text()[:5000]}")
            break

    # GOAL.md if exists
    goal = project_dir / "GOAL.md"
    if goal.exists():
        text = goal.read_text()
        if len(text) > 50:  # not just a stub
            ctx_parts.append(f"=== GOAL.md ===\n{text[:3000]}")

    # Existing Python files (first 20)
    py_files = sorted(project_dir.glob("**/*.py"))[:20]
    for f in py_files:
        if "venv" in str(f) or ".orze" in str(f) or "orze/" in str(f):
            continue
        try:
            text = f.read_text()[:2000]
            ctx_parts.append(f"=== {f.relative_to(project_dir)} ===\n{text}")
        except Exception:
            continue

    # Data directories
    data_dirs = []
    for d in project_dir.iterdir():
        if d.is_dir() and d.name.lower() in ("data", "datasets", "dataset"):
            contents = list(d.iterdir())[:20]
            data_dirs.append(f"{d.name}/: {[c.name for c in contents]}")
    if data_dirs:
        ctx_parts.append(f"=== Data Directories ===\n" + "\n".join(data_dirs))

    # requirements.txt / pyproject.toml
    for name in ["requirements.txt", "pyproject.toml", "setup.py"]:
        p = project_dir / name
        if p.exists():
            ctx_parts.append(f"=== {name} ===\n{p.read_text()[:2000]}")
            break

    return "\n\n".join(ctx_parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(api_key: str, backend: str, prompt: str, system: str = "") -> str | None:
    """Call an LLM API. Falls back to other available keys on failure."""
    all_keys = _find_all_api_keys()
    # Put the requested key first, then any others
    ordered = [(api_key, backend)] + [(k, b) for k, b in all_keys if b != backend]

    for key, be in ordered:
        try:
            if be == "anthropic":
                return _call_anthropic(key, prompt, system)
            elif be == "gemini":
                return _call_gemini(key, prompt, system)
            elif be == "openai":
                return _call_openai(key, prompt, system)
        except Exception as e:
            _warn(f"LLM call failed ({be}): {e}")
    return None


def _call_anthropic(api_key: str, prompt: str, system: str) -> str:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(f"Anthropic API {e.code}: {body}") from e


def _call_gemini(api_key: str, prompt: str, system: str) -> str:
    import urllib.request
    import urllib.error
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}],
        "generationConfig": {"maxOutputTokens": 8192},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(f"Gemini API {e.code}: {body_text}") from e


def _call_openai(api_key: str, prompt: str, system: str) -> str:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": "gpt-4o-mini",
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(f"OpenAI API {e.code}: {body_text}") from e


# ---------------------------------------------------------------------------
# File generation via LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an ML research engineer setting up an automated experiment system.
You analyze codebases and generate configuration files for orze, a GPU experiment orchestrator.

When generating files, output them in this exact format:
===FILE: path/to/file.ext===
<file content>
===END===

Generate complete, runnable files. No placeholders, no TODOs. Real code that works."""

ANALYSIS_PROMPT_TEMPLATE = """\
Analyze this codebase and generate the files needed to run automated ML experiments with orze.

## Environment
- GPUs: {gpu_info}
- Framework: {framework}
- Python: {python_path}

## Codebase
{codebase_context}

## What to generate

Generate these files:

1. **GOAL.md** — Research objective, dataset, metrics, current state, directions to explore.

2. **train.py** — Training script that follows this contract:
   - Receives: `python train.py --idea-id <id> --results-dir <dir> --ideas-md <file> --config <yaml>`
   - Reads `<results-dir>/<idea_id>/idea_config.yaml` for merged config (orze writes this before launch)
   - Outputs: `<results-dir>/<idea_id>/metrics.json` with `{{"status": "COMPLETED", "<metric>": <value>, "training_time": <seconds>}}`
   - IMPORTANT: Use the --results-dir argument, do NOT hardcode a path
   - Must import all needed modules, handle GPU assignment via CUDA_VISIBLE_DEVICES (set by orze)
   - Must use the actual dataset and model architecture appropriate for the task

3. **configs/base.yaml** — Default hyperparameters (nested YAML is fine).

4. **ideas.md** — 5-8 seed experiments in this format:
   ```
   ## idea-XXXX: Title
   - **Priority**: high|medium|low
   ```yaml
   key: value
   ```
   ```

5. **RESEARCH_RULES.md** — Research agent instructions: goal, metrics, strategy, config keys, idea format.

6. **orze.yaml** — Orze config. Must include:
   - train_script, ideas_file, base_config, results_dir
   - python: {python_path}
   - nested_config_whitelist for any nested config keys used in ideas
   - report section with primary_metric, sort order, columns matching metrics.json keys
   - timeout appropriate for the task

7. **requirements.txt** — Python dependencies needed by train.py.

8. Any **src/*.py** helper modules needed by train.py (dataset loaders, models, etc.)

Output each file using the ===FILE: path=== / ===END=== format. Make them complete and correct."""


def _parse_files(llm_output: str) -> dict[str, str]:
    """Parse ===FILE: path=== ... ===END=== blocks from LLM output."""
    files = {}
    parts = llm_output.split("===FILE:")
    for part in parts[1:]:
        # Try newline-terminated header first, fall back to bare ===
        if "===\n" in part:
            header, _, body = part.partition("===\n")
        else:
            header, _, body = part.partition("===")
        path = header.strip().rstrip("=").strip()
        content = body.split("===END===")[0]
        if path and content.strip():
            files[path] = content.rstrip() + "\n"
    return files


def _write_generated_files(project_dir: Path, files: dict[str, str]):
    """Write generated files, overwriting template files but not user files."""
    safe_overwrite = {
        "GOAL.md", "train.py", "configs/base.yaml", "ideas.md",
        "RESEARCH_RULES.md", "orze.yaml", "requirements.txt",
    }
    resolved_root = project_dir.resolve()
    for rel_path, content in files.items():
        full_path = (project_dir / rel_path).resolve()
        if not full_path.is_relative_to(resolved_root):
            _warn(f"Skipping {rel_path} (path traversal)")
            continue
        is_template = rel_path in safe_overwrite
        if full_path.exists() and not is_template:
            # Don't overwrite user files — only new src/ files
            existing = full_path.read_text()
            if len(existing) > 100:  # not a stub
                _warn(f"Skipping {rel_path} (already exists with content)")
                continue
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        _ok(f"Generated {rel_path}")


# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------

def _install_deps(venv_python: Path, project_dir: Path):
    """Install requirements.txt into the project venv."""
    req_path = project_dir / "requirements.txt"
    if not req_path.exists():
        return

    _step("Installing dependencies")
    pip_cmd = [str(venv_python), "-m", "pip", "install", "-q", "-r", str(req_path)]
    try:
        result = subprocess.run(
            pip_cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            _ok("Dependencies installed")
        else:
            _warn(f"pip install failed:\n{result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        _warn("pip install timed out (10 min)")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def do_interactive_init(project_dir: Path, venv_python: Path) -> bool:
    """Analyze codebase, generate project files via LLM, and start orze.

    Called after do_init() scaffolds template files. Returns True if it
    ran to completion (so the caller can skip redundant "Next steps").
    """
    print(f"\n{BOLD}Orze — Interactive Setup{RESET}")
    print("=" * 40)

    # --- Step 1: Detect environment ---
    _step("Detecting environment")

    ai_cli = _detect_ai_cli()
    if ai_cli:
        _ok(f"Running inside {ai_cli} CLI — interactive init continues")

    gpus = _detect_gpus()
    if gpus:
        gpu_summary = f"{len(gpus)}× {gpus[0]['name']} ({gpus[0]['memory_mib']}MiB)"
        _ok(f"GPUs: {gpu_summary}")
    else:
        gpu_summary = "none detected"
        _warn("No GPUs detected — experiments will run on CPU")

    framework = _detect_framework(project_dir)
    _ok(f"Framework: {framework or 'not detected'}")

    api_key, backend = _find_api_key()
    if api_key:
        _ok(f"API key: {backend}")
    else:
        # Ask for API key
        print()
        print(f"  {DIM}An API key lets orze analyze your codebase and generate")
        print(f"  real training scripts, ideas, and configs automatically.{RESET}")
        print()
        api_key_input = ""
        if sys.stdin.isatty():
            api_key_input = input("  Enter API key (Gemini/Anthropic/OpenAI, or Enter to skip): ").strip()
        if api_key_input:
            # Auto-detect backend from key prefix
            if api_key_input.startswith("sk-ant"):
                backend = "anthropic"
                os.environ["ANTHROPIC_API_KEY"] = api_key_input
            elif api_key_input.startswith("AIza"):
                backend = "gemini"
                os.environ["GEMINI_API_KEY"] = api_key_input
            elif api_key_input.startswith("sk-"):
                backend = "openai"
                os.environ["OPENAI_API_KEY"] = api_key_input
            else:
                backend = "gemini"
                os.environ["GEMINI_API_KEY"] = api_key_input
            api_key = api_key_input

            # Save to .env
            env_path = project_dir / ".env"
            env_var = {
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
            }[backend]
            saved = False
            if env_path.exists():
                content = env_path.read_text()
                lines = content.splitlines()
                new_lines = [
                    l for l in lines
                    if not l.strip().startswith(f"# {env_var}")
                    and not l.strip().startswith(f"{env_var}=")
                ]
                new_lines.append(f"{env_var}={api_key}")
                env_path.write_text("\n".join(new_lines) + "\n")
                saved = True
            else:
                env_path.write_text(f"{env_var}={api_key}\n")
                saved = True
            if saved:
                _ok(f"API key saved to .env ({backend})")

    if not api_key:
        _warn("No API key — generating basic template files only")
        print(f"\n  {BOLD}To enable smart setup later:{RESET}")
        print(f"  1. Add your API key to .env")
        print(f"  2. Run: orze init")
        return False

    # --- Step 2: Analyze codebase ---
    _step("Analyzing codebase")

    context = _gather_codebase_context(project_dir)
    if not context.strip():
        _warn("No codebase context found (empty project)")
        context = "Empty project — generate a demo ML experiment setup."

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        gpu_info=gpu_summary,
        framework=framework or "pytorch (default)",
        python_path=str(venv_python),
        codebase_context=context,
    )

    _print(f"Calling {backend} to generate project files...")
    llm_output = _call_llm(api_key, backend, prompt, SYSTEM_PROMPT)

    if not llm_output:
        _warn("LLM analysis failed — keeping template files")
        return False

    # --- Step 3: Parse and write files ---
    _step("Generating project files")

    files = _parse_files(llm_output)
    if not files:
        _warn("Could not parse LLM output into files")
        return False

    _write_generated_files(project_dir, files)

    # --- Step 4: Show GOAL.md and ask for confirmation ---
    goal_path = project_dir / "GOAL.md"
    if goal_path.exists():
        print(f"\n{BOLD}Generated GOAL.md:{RESET}")
        print(f"{DIM}{'─' * 50}{RESET}")
        goal_text = goal_path.read_text()
        for line in goal_text.splitlines()[:30]:
            print(f"  {line}")
        if len(goal_text.splitlines()) > 30:
            print(f"  {DIM}... ({len(goal_text.splitlines()) - 30} more lines){RESET}")
        print(f"{DIM}{'─' * 50}{RESET}")

    if sys.stdin.isatty():
        print()
        response = input(f"  {BOLD}Looks right? [Y/n]{RESET} ").strip().lower()
        if response == "n":
            print(f"\n  Edit the generated files, then run: {CYAN}orze start -c orze.yaml{RESET}")
            return True

    # --- Step 5: Install dependencies ---
    _install_deps(venv_python, project_dir)

    # --- Step 6: Validate and start ---
    _step("Starting orze")

    cfg_path = project_dir / "orze.yaml"
    if not cfg_path.exists():
        _warn("No orze.yaml found — cannot start")
        return False

    # Find the orze binary — it's installed system-wide or in the current Python env,
    # NOT in the project venv (which is for training dependencies)
    orze_bin = shutil.which("orze")
    if orze_bin:
        check_cmd = [orze_bin, "--check", "-c", str(cfg_path)]
        start_cmd = [orze_bin, "start", "-c", str(cfg_path)]
    else:
        check_cmd = [sys.executable, "-m", "orze.cli", "--check", "-c", str(cfg_path)]
        start_cmd = [sys.executable, "-m", "orze.cli", "start", "-c", str(cfg_path)]

    try:
        check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
        if check_result.returncode != 0:
            _warn("Config check had issues — starting anyway")
    except subprocess.TimeoutExpired:
        _warn("Config check timed out — starting anyway")

    try:
        start_result = subprocess.run(start_cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        _warn("orze start timed out — it may still be starting in the background")
        return True

    combined_output = (start_result.stdout or "") + (start_result.stderr or "")
    if start_result.returncode == 0 or "already running" in combined_output.lower():
        print()
        n_gpus = len(gpus) if gpus else 0
        _ok(f"Orze is running on {n_gpus} GPU{'s' if n_gpus != 1 else ''}")
        _print(f"Admin UI: {CYAN}http://localhost:8787{RESET}")
        _print(f"Edit {CYAN}GOAL.md{RESET} anytime to change direction.")
    else:
        _warn("Failed to start orze:")
        print(start_result.stderr[-500:] if start_result.stderr else "unknown error")
        print(f"\n  Try manually: {CYAN}orze start -c orze.yaml{RESET}")

    return True
