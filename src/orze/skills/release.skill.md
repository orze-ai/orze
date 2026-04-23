---
name: release
---

## Cutting a release

This skill covers cutting a new release for **orze** (basic, public PyPI) and **orze-pro** (pro, private PyPI at `pypi.orze.ai`). The two packages share the same release shape — only the publish step differs.

### Identity

All git operations use the project's configured committer (do not override). Releases are cut from `main`.

### What "release" means here

- **A new version on PyPI** that `pip install --upgrade <pkg>` will pull
- **A pushed git tag `vX.Y.Z`** matching the version in `pyproject.toml`
- **A CHANGELOG.md entry** under that version, summarizing user-facing changes
- **Working sdist + wheel** that include all package data (skills, UI assets, rule files)

If any of those is missing, the release isn't done.

---

## Pre-flight (do this for both basic and pro)

```bash
# 1. Clean tree on main
git status -s            # must be empty
git rev-parse --abbrev-ref HEAD   # must be `main`
git pull --ff-only origin main

# 2. Read the current version (canonical source: pyproject.toml)
grep '^version' pyproject.toml

# 3. Review what's actually shipping
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# 4. Sanity-check package-data so .md / asset files end up in the wheel
grep -A1 '\[tool.setuptools.package-data\]' pyproject.toml
```

**Hidden gotcha** (caused a real bug — fixed in 4.0.4): `[tool.setuptools.package-data]` only ships files explicitly listed. New `*.skill.md`, `*.md`, or other non-Python data files **must be added there** or they won't appear in the wheel. Editable installs (`pip install -e .`) hide this — they resolve to the source tree, so the missing entries don't bite until a real PyPI install. After adding new data files, always inspect the wheel:

```bash
python -m build
python -c "import zipfile; print('\n'.join(zipfile.ZipFile('dist/orze-X.Y.Z-py3-none-any.whl').namelist()))" | grep <new-file>
```

---

## Version bump + CHANGELOG

Single source of truth: `pyproject.toml` `[project] version`. `orze.__version__` reads from installed package metadata, so no manual edit needed there.

```bash
# Bump version in pyproject.toml: e.g. 4.0.3 -> 4.0.4
# Add a CHANGELOG.md section directly under "# Changelog" with Fixed / Added / Security subsections.
git add pyproject.toml CHANGELOG.md <other changed files>
git commit -m "chore: bump version to vX.Y.Z"
```

Use real semver: bug fixes → patch, additive features → minor, breaking changes → major.

---

## Tag and push (git side)

```bash
git tag vX.Y.Z                        # lightweight tag matches existing convention
git push origin main
git push origin vX.Y.Z
```

If `gh` is installed, also create a GitHub release:

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-from-tag
```

`gh` may not be installed on every machine. The PyPI publish step is what users actually consume — the GitHub release is supplemental.

---

## Build artifacts

```bash
rm -rf dist build *.egg-info
python -m build
ls dist/                 # expect <pkg>-X.Y.Z.tar.gz and <pkg>-X.Y.Z-py3-none-any.whl
```

Both files must exist. If only one appears, the build failed silently — read the full output.

**Verify the wheel contents** before publishing (especially after package-data changes):

```bash
python -c "
import zipfile, sys
names = zipfile.ZipFile(sys.argv[1]).namelist()
expected = ['orze/SKILL.md', 'orze/skills/core.skill.md', 'orze/skills/release.skill.md']
missing = [e for e in expected if e not in names]
print('OK' if not missing else f'MISSING: {missing}')
" dist/orze-X.Y.Z-py3-none-any.whl
```

---

## Publish — basic (orze → public PyPI)

```bash
# PyPI API token: lives in the repo's release secrets / 1Password / env (PYPI_TOKEN).
# Token starts with `pypi-`. Never commit it.
twine upload dist/* --username __token__ --password "$PYPI_TOKEN"
```

Defaults to `https://upload.pypi.org/legacy/` — the public PyPI upload endpoint.

**Verify in a clean venv:**

```bash
python -m venv /tmp/orze-verify && /tmp/orze-verify/bin/pip install --upgrade orze
/tmp/orze-verify/bin/python -c "import orze; print(orze.__version__)"  # must print X.Y.Z
/tmp/orze-verify/bin/orze --version
```

PyPI propagation takes 30–60s. If `pip install` reports an older version, wait and retry.

---

## Publish — pro (orze-pro → private PyPI at `pypi.orze.ai`)

orze-pro is distributed via the license-gated private index `https://pypi.orze.ai/simple/`. Consumer installs are documented in this repo's README and used by `orze upgrade` / `extensions._auto_install_pro` / `engine/upgrade.py`. The publisher endpoint and credential are **orze.ai-internal** — they are not in this repo and should not be committed here.

### Consumer side (what users do — already wired into orze)

```bash
pip install --upgrade orze-pro \
    --extra-index-url "https://__token__:${ORZE_PRO_KEY}@pypi.orze.ai/simple/"
```

`ORZE_PRO_KEY` resolution order (see `orze.extensions._find_pro_key`):
1. `ORZE_PRO_KEY` environment variable
2. `.env` file in cwd with `ORZE_PRO_KEY=...`
3. `~/.orze-pro.key`

### Publisher side (release engineer)

Same `python -m build` step as basic. Then upload with `twine`:

```bash
twine upload \
    --repository-url "$ORZE_PRO_PYPI_UPLOAD_URL" \
    --username __token__ \
    --password "$ORZE_PRO_PUBLISHER_TOKEN" \
    dist/*
```

`ORZE_PRO_PYPI_UPLOAD_URL` and `ORZE_PRO_PUBLISHER_TOKEN` are kept in the orze.ai operator secrets store. **Do not** reuse a consumer license key (`ORZE-PRO-...`) for upload — those grant read-only `simple/` index access.

### Verify pro release end-to-end

After publishing, run `orze upgrade` from any pro-licensed install (with `ORZE_PRO_KEY` set). It should pick up the new version automatically:

```bash
orze upgrade --no-restart           # exits 0; redacts the URL in printed cmd
pip show orze-pro | grep Version    # must report X.Y.Z
```

Or do a clean-venv install matching the consumer flow:

```bash
python -m venv /tmp/pro-verify
/tmp/pro-verify/bin/pip install --upgrade orze-pro \
    --extra-index-url "https://__token__:${ORZE_PRO_KEY}@pypi.orze.ai/simple/"
/tmp/pro-verify/bin/python -c "import orze_pro; print(orze_pro.__version__)"
```

---

## Security — license keys never leak to logs

orze ships `orze.extensions.redact_basic_auth(text)`, which scrubs `user:password@` from any URL embedded in a string. Use it whenever you print, log, or surface a pip command or pip stderr that may contain `--extra-index-url`. Both `cli.py:upgrade` (stdout) and `engine/upgrade.py` (logger.warning of stderr) already use it. Any new code path that handles the private-PyPI URL must do the same.

---

## Common gotchas

- **`pip3` ≠ `sys.executable -m pip`.** Bare `pip3` resolves outside the active venv ("Defaulting to user installation"). Always use `sys.executable -m pip` in code; in scripts, source the venv first.
- **Lightweight vs annotated tags.** Existing tags (`v3.x`, `v4.0.x`) are lightweight (`git tag vX.Y.Z`). Match that — switching to annotated mid-stream confuses changelog tooling.
- **Detached HEAD after `git checkout vX.Y.Z`.** Don't commit on detached HEAD by accident. If you find yourself there, `git checkout main` first, then merge or fast-forward.
- **`--force-reinstall -e` on a non-editable install** silently does nothing useful for the `<pkg>` half but still reinstalls deps. Detect editable vs non-editable and branch (see `cli.py:upgrade._editable_project_root`).
- **Wheel without skills**: if `pip show orze` works but `orze` crashes with `Built-in skill file not found`, you shipped a wheel missing `package-data` entries. Re-check `pyproject.toml`'s `[tool.setuptools.package-data]`.

---

## Quick reference — release checklist

```
[ ] Clean tree on main, pulled from origin
[ ] Bumped version in pyproject.toml
[ ] CHANGELOG.md entry under new version
[ ] git commit -m "chore: bump version to vX.Y.Z"
[ ] git tag vX.Y.Z && git push origin main && git push origin vX.Y.Z
[ ] rm -rf dist build && python -m build
[ ] Inspected wheel namelist — no missing data files
[ ] twine upload dist/*  (basic: public PyPI; pro: private PyPI)
[ ] Verified in clean venv: pip install --upgrade <pkg>; check __version__
[ ] (basic) gh release create vX.Y.Z, if gh available
[ ] Cleared dist/ and build/ on success
```
