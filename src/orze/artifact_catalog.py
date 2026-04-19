"""Artifact catalog — SQLite index of on-disk ML artifacts.

Feature F9. Tracks checkpoints, per-clip prediction NPZs, feature caches and
TTA view NPZs so that post-hoc search (F10/F11/F15) can enumerate everything
derived from a single model without walking the filesystem on every cycle.

The DB lives alongside the idea_lake DB (same directory) and uses the same
``PRAGMA journal_mode=DELETE`` because some deployments run on Lustre/NFS
where WAL's shared-memory segment is not supported.

Schema:

    artifacts(
        path TEXT PRIMARY KEY,
        kind TEXT NOT NULL,                  -- ckpt | preds_npz | features | tta_preds
        ckpt_sha TEXT,                       -- SHA-256 of the underlying weights
        idea_id TEXT,
        inference_config_hash TEXT,          -- hash({tta_ops, frame_stride, crop, ...})
        inference_config TEXT,               -- JSON of that config
        metric_val REAL,                     -- val-split pgmAP (or None)
        metric_test REAL,                    -- test pgmAP (or None)
        size_bytes INTEGER,
        created_at TEXT NOT NULL
    )

CLI:

    orze catalog scan --results-dir <path>

Programmatic:

    cat = ArtifactCatalog("idea_lake.db".replace(".db", "_artifacts.db"))
    cat.upsert("results/foo/best_model.pt", "ckpt", ckpt_sha=sha, metric_val=0.89)
    cat.by_ckpt_sha(sha)            # list[dict]
    cat.bundle(sha)                 # list[dict] — preds_npz rows sharing sha
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("artifact_catalog")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    path TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    ckpt_sha TEXT,
    idea_id TEXT,
    inference_config_hash TEXT,
    inference_config TEXT,
    metric_val REAL,
    metric_test REAL,
    size_bytes INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_ckpt_sha ON artifacts(ckpt_sha);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
CREATE INDEX IF NOT EXISTS idx_artifacts_idea_id ON artifacts(idea_id);
"""


# Anything outside this set is rejected in upsert() — a hard contract so the
# rest of the engine (bundle_combiner, posthoc_runner, search role) can rely
# on a closed vocabulary.
ALLOWED_KINDS = {"ckpt", "preds_npz", "features", "tta_preds"}


def _retry_on_busy(func, max_retries: int = 5, base_delay: float = 0.2):
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


def hash_ckpt(path: str | os.PathLike, chunk_mib: int = 4) -> str:
    """Cheap content hash for big checkpoint files.

    Hashes the first and last ``chunk_mib`` MiB of the file plus the size.
    This is ~1000x faster than hashing a 4 GB checkpoint and collision-free
    in practice for ML weight files (which have very high entropy).
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(p, "rb") as f:
        chunk = f.read(chunk_mib * 1024 * 1024)
        h.update(chunk)
        if size > chunk_mib * 1024 * 1024 * 2:
            f.seek(max(0, size - chunk_mib * 1024 * 1024))
            h.update(f.read(chunk_mib * 1024 * 1024))
    return h.hexdigest()


def hash_inference_config(cfg: Dict[str, Any]) -> str:
    """Stable hash of a dict, invariant to key order."""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


class ArtifactCatalog:
    """SQLite-backed catalog of ML artifacts."""

    def __init__(self, db_path: str | os.PathLike):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA busy_timeout=15000")
        self.conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------ #
    # writes                                                             #
    # ------------------------------------------------------------------ #

    def upsert(
        self,
        path: str | os.PathLike,
        kind: str,
        *,
        ckpt_sha: Optional[str] = None,
        idea_id: Optional[str] = None,
        inference_config: Optional[Dict[str, Any]] = None,
        metric_val: Optional[float] = None,
        metric_test: Optional[float] = None,
        size_bytes: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> None:
        if kind not in ALLOWED_KINDS:
            raise ValueError(
                f"artifact kind={kind!r} not in {sorted(ALLOWED_KINDS)}"
            )

        path_str = str(path)
        cfg_json = json.dumps(inference_config) if inference_config else None
        cfg_hash = (
            hash_inference_config(inference_config) if inference_config else None
        )
        if size_bytes is None:
            try:
                size_bytes = Path(path_str).stat().st_size
            except OSError:
                size_bytes = None
        ts = created_at or datetime.datetime.utcnow().isoformat()

        def _do():
            self.conn.execute(
                """INSERT INTO artifacts (
                       path, kind, ckpt_sha, idea_id,
                       inference_config_hash, inference_config,
                       metric_val, metric_test, size_bytes, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       kind=excluded.kind,
                       ckpt_sha=COALESCE(excluded.ckpt_sha, artifacts.ckpt_sha),
                       idea_id=COALESCE(excluded.idea_id, artifacts.idea_id),
                       inference_config_hash=COALESCE(excluded.inference_config_hash,
                                                     artifacts.inference_config_hash),
                       inference_config=COALESCE(excluded.inference_config,
                                                 artifacts.inference_config),
                       metric_val=COALESCE(excluded.metric_val, artifacts.metric_val),
                       metric_test=COALESCE(excluded.metric_test, artifacts.metric_test),
                       size_bytes=COALESCE(excluded.size_bytes, artifacts.size_bytes)
                """,
                (path_str, kind, ckpt_sha, idea_id,
                 cfg_hash, cfg_json, metric_val, metric_test, size_bytes, ts),
            )
            self.conn.commit()

        _retry_on_busy(_do)

    def delete(self, path: str | os.PathLike) -> None:
        def _do():
            self.conn.execute("DELETE FROM artifacts WHERE path = ?", (str(path),))
            self.conn.commit()

        _retry_on_busy(_do)

    # ------------------------------------------------------------------ #
    # reads                                                              #
    # ------------------------------------------------------------------ #

    def get(self, path: str | os.PathLike) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM artifacts WHERE path = ?", (str(path),)
        ).fetchone()
        return _row_to_dict(row)

    def by_ckpt_sha(self, sha: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE ckpt_sha = ? ORDER BY kind, path",
            (sha,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def bundle(self, sha: str) -> List[Dict[str, Any]]:
        """All preds_npz / tta_preds rows sharing a checkpoint SHA.

        This is the primitive F10 uses to construct an InferenceBundle —
        multiple TTA views of ONE model. The ckpt itself (kind='ckpt') is
        intentionally not part of the bundle.
        """
        rows = self.conn.execute(
            """SELECT * FROM artifacts
               WHERE ckpt_sha = ? AND kind IN ('preds_npz', 'tta_preds')
               ORDER BY path""",
            (sha,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_by_kind(self, kind: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE kind = ? ORDER BY path", (kind,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # bulk scan                                                          #
    # ------------------------------------------------------------------ #

    def scan(
        self,
        results_dir: str | os.PathLike,
        *,
        hash_ckpts: bool = True,
        max_files: Optional[int] = None,
    ) -> Dict[str, int]:
        """Walk ``results_dir`` and upsert every ckpt / preds_npz we find.

        Returns a counter dict {'ckpt': N, 'preds_npz': M, 'tta_preds': K}.
        Idempotent: re-runs only rehash checkpoints whose size changed.
        """
        results_dir = Path(results_dir)
        if not results_dir.exists():
            return {}

        counts = {k: 0 for k in ALLOWED_KINDS}
        seen = 0

        # Cache existing rows by path so we can skip re-hashing unchanged ckpts.
        existing = {
            r["path"]: (r["size_bytes"], r["ckpt_sha"])
            for r in self.conn.execute(
                "SELECT path, size_bytes, ckpt_sha FROM artifacts"
            )
        }

        # 1) checkpoints
        for ckpt_path in _iter_ckpts(results_dir):
            if max_files is not None and seen >= max_files:
                break
            seen += 1
            size = ckpt_path.stat().st_size
            prev = existing.get(str(ckpt_path))
            if prev and prev[0] == size and prev[1]:
                sha = prev[1]
            else:
                sha = hash_ckpt(ckpt_path) if hash_ckpts else None
            idea_id = ckpt_path.parent.name if ckpt_path.parent != results_dir else None
            self.upsert(
                ckpt_path, "ckpt",
                ckpt_sha=sha, idea_id=idea_id, size_bytes=size,
            )
            counts["ckpt"] += 1

        # 2) prediction NPZs
        for npz_path in _iter_npzs(results_dir):
            if max_files is not None and seen >= max_files:
                break
            seen += 1
            kind = "tta_preds" if "tta" in npz_path.name.lower() else "preds_npz"
            # Best-effort ckpt_sha association: if the npz sits in a dir that
            # contains a best_model.pt, reuse its SHA.
            ckpt_sha = _find_sibling_ckpt_sha(npz_path, existing, self)
            idea_id = npz_path.parent.name if npz_path.parent != results_dir else None
            self.upsert(
                npz_path, kind,
                ckpt_sha=ckpt_sha, idea_id=idea_id,
                size_bytes=npz_path.stat().st_size,
            )
            counts[kind] += 1

        return counts


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    if d.get("inference_config"):
        try:
            d["inference_config"] = json.loads(d["inference_config"])
        except (ValueError, TypeError):
            pass
    return d


_CKPT_NAMES = ("best_model.pt", "best_ema_model.pt", "ema_best.pt")


def _iter_ckpts(root: Path) -> Iterable[Path]:
    for name in _CKPT_NAMES:
        yield from root.rglob(name)


def _iter_npzs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.npz"):
        # skip obvious non-predictions (caches, features)
        low = p.name.lower()
        if "cache" in low or "feature" in low or "riskprop" in low:
            continue
        yield p


def _find_sibling_ckpt_sha(
    npz_path: Path,
    existing: Dict[str, tuple],
    catalog: "ArtifactCatalog",
) -> Optional[str]:
    """Look for a best_model.pt in the same directory; inherit its SHA."""
    for name in _CKPT_NAMES:
        sibling = npz_path.parent / name
        prev = existing.get(str(sibling))
        if prev and prev[1]:
            return prev[1]
        if sibling.exists():
            try:
                row = catalog.get(sibling)
                if row and row.get("ckpt_sha"):
                    return row["ckpt_sha"]
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------- #
# CLI entry                                                              #
# ---------------------------------------------------------------------- #


def cli_scan(args) -> int:
    """Implementation of ``orze catalog scan --results-dir …``."""
    results_dir = Path(args.results_dir)
    db_path = args.db or (results_dir / "idea_lake_artifacts.db")
    cat = ArtifactCatalog(db_path)
    counts = cat.scan(results_dir, hash_ckpts=not args.no_hash, max_files=args.limit)
    total = sum(counts.values())
    print(f"scanned {results_dir}: {total} artifacts indexed → {db_path}")
    for k, v in sorted(counts.items()):
        if v:
            print(f"  {k:12s} {v}")
    cat.close()
    return 0
