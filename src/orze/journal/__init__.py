"""orze.Journal — append-only structured iteration log that survives compaction.

A Journal is the agent's persistent memory for a long-running workflow. State
lives on disk, not in process memory, so resuming after a crash, restart, or
agent context compaction is `Journal(path)` and you have all prior iterations
back.

Two on-disk shapes are supported:

* ``.md`` — human-readable Markdown sections per iteration. Use this when the
  journal is the primary status doc the operator reads. The `Journal` class
  parses existing ``.md`` content on open so resume is lossless for the
  structured fields it controls.
* ``.jsonl`` — one JSON object per line. Use this when the journal is meant
  for programmatic aggregation. Faster to read, easier to query, but not
  human-pretty.

Typical use::

    from orze import Journal
    j = Journal("results/STATUS.md")
    with j.iter("iter-9", hypothesis="iter-4 LoRA on chk-38000 base") as it:
        it.recipe(lora="iter-4", base="chk-38000", decode="default")
        # ... long-running work ...
        it.result(macro_wer=5.793, per_dataset={"ami": 11.98, ...})
        it.decide("rejected", reason="worse than baseline 5.449")

Resume after restart::

    j = Journal("results/STATUS.md")
    print(j.iters[-1].name)            # "iter-9"
    print(j.iters[-1].result)          # {"macro_wer": 5.793, ...}
    print(j.iters[-1].decision)        # ("rejected", "worse than baseline 5.449")
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Iteration:
    """One entry in the Journal."""
    name: str
    started_at: float
    hypothesis: str | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    notes: list[tuple[float, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    decision: tuple[str, str] | None = None  # (verdict, reason)
    closed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "hypothesis": self.hypothesis,
            "recipe": self.recipe,
            "notes": self.notes,
            "result": self.result,
            "decision": list(self.decision) if self.decision else None,
            "closed_at": self.closed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Iteration":
        decision = d.get("decision")
        return cls(
            name=d["name"],
            started_at=d["started_at"],
            hypothesis=d.get("hypothesis"),
            recipe=d.get("recipe", {}),
            notes=[tuple(n) for n in d.get("notes", [])],
            result=d.get("result"),
            decision=tuple(decision) if decision else None,
            closed_at=d.get("closed_at"),
        )


class _IterContext:
    """Mutating handle returned by Journal.iter() — calls flush after each edit."""

    def __init__(self, journal: "Journal", it: Iteration):
        self._j = journal
        self._it = it

    def hypothesis(self, text: str) -> "_IterContext":
        self._it.hypothesis = text
        self._j._flush()
        return self

    def recipe(self, **kwargs: Any) -> "_IterContext":
        self._it.recipe.update(kwargs)
        self._j._flush()
        return self

    def note(self, text: str) -> "_IterContext":
        self._it.notes.append((time.time(), text))
        self._j._flush()
        return self

    def result(self, **kwargs: Any) -> "_IterContext":
        self._it.result = kwargs
        self._j._flush()
        return self

    def decide(self, verdict: str, reason: str = "") -> "_IterContext":
        self._it.decision = (verdict, reason)
        self._j._flush()
        return self

    def close(self) -> None:
        if self._it.closed_at is None:
            self._it.closed_at = time.time()
            self._j._flush()

    def __enter__(self) -> "_IterContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Mark closed even on exception. Caller can still set decision in handler.
        if self._it.closed_at is None:
            self._it.closed_at = time.time()
            if exc is not None and self._it.decision is None:
                self._it.decision = ("errored", f"{exc_type.__name__}: {exc}")
            self._j._flush()


class Journal:
    """Append-only iteration log persisted on disk.

    Args:
        path: Output path. ``.md`` or ``.jsonl`` extension chooses the format.
            If the file exists, prior iterations are loaded.
        meta: Optional dict written to the file header. Useful for recording
            invariants like the experiment goal, target metric, etc.
    """

    def __init__(self, path: str | os.PathLike[str], meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self._fmt = "jsonl" if self.path.suffix == ".jsonl" else "md"
        self.meta: dict[str, Any] = meta or {}
        self.iters: list[Iteration] = []
        if self.path.exists():
            self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    # -- public API --

    def iter(self, name: str, hypothesis: str | None = None) -> _IterContext:
        """Start a new iteration. Returns a context manager that can be used
        with ``with`` to ensure ``closed_at`` is set on exit."""
        it = Iteration(name=name, started_at=time.time(), hypothesis=hypothesis)
        self.iters.append(it)
        self._flush()
        return _IterContext(self, it)

    def get(self, name: str) -> Iteration | None:
        """Look up an iteration by name. Returns the LAST one if names collide."""
        for it in reversed(self.iters):
            if it.name == name:
                return it
        return None

    @contextmanager
    def session(self) -> Iterator["Journal"]:
        """Optional context: flushes on exit. Mostly for explicit cleanliness."""
        try:
            yield self
        finally:
            self._flush()

    # -- persistence --

    def _flush(self) -> None:
        if self._fmt == "jsonl":
            self._write_jsonl()
        else:
            self._write_md()

    def _write_jsonl(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            f.write(json.dumps({"_meta": self.meta}) + "\n")
            for it in self.iters:
                f.write(json.dumps(it.to_dict()) + "\n")
        tmp.replace(self.path)

    def _write_md(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            f.write("# Journal\n\n")
            if self.meta:
                f.write("```json\n")
                f.write(json.dumps(self.meta, indent=2))
                f.write("\n```\n\n")
            f.write("<!-- orze.Journal: do not hand-edit lines between BEGIN/END markers -->\n\n")
            for it in self.iters:
                f.write(f"## {it.name}\n\n")
                f.write(f"- started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(it.started_at))}\n")
                if it.hypothesis:
                    f.write(f"- hypothesis: {it.hypothesis}\n")
                if it.recipe:
                    f.write(f"- recipe: `{json.dumps(it.recipe, sort_keys=True)}`\n")
                if it.result is not None:
                    f.write(f"- result: `{json.dumps(it.result, sort_keys=True)}`\n")
                if it.decision is not None:
                    verdict, reason = it.decision
                    suffix = f" — {reason}" if reason else ""
                    f.write(f"- decision: **{verdict}**{suffix}\n")
                if it.closed_at:
                    dur = it.closed_at - it.started_at
                    f.write(f"- closed: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(it.closed_at))} ({dur:.0f}s)\n")
                if it.notes:
                    f.write("\n<details><summary>notes</summary>\n\n")
                    for ts, text in it.notes:
                        f.write(f"- `{time.strftime('%H:%M:%S', time.gmtime(ts))}` {text}\n")
                    f.write("\n</details>\n")
                # Machine-readable block for round-trip resume
                f.write(f"\n<!--BEGIN orze-iter {it.name}-->\n")
                f.write("```json orze-iter\n")
                f.write(json.dumps(it.to_dict()))
                f.write("\n```\n")
                f.write(f"<!--END orze-iter {it.name}-->\n\n")
        tmp.replace(self.path)

    def _load(self) -> None:
        text = self.path.read_text()
        if self._fmt == "jsonl":
            for i, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                d = json.loads(line)
                if i == 0 and "_meta" in d:
                    self.meta = d["_meta"]
                    continue
                self.iters.append(Iteration.from_dict(d))
        else:
            # Parse the meta header block (first ```json fenced block before
            # any iter blocks).
            head = text.split("<!--BEGIN orze-iter", 1)[0]
            m_meta = re.search(r"```json\s*(\{.*?\})\s*```", head, flags=re.DOTALL)
            if m_meta:
                try:
                    self.meta = json.loads(m_meta.group(1))
                except json.JSONDecodeError:
                    pass
            # Pull the embedded JSON blocks. Anything outside the markers is
            # human-edited prose we leave alone (next flush will rebuild
            # the controlled regions verbatim).
            for m in re.finditer(
                r"<!--BEGIN orze-iter [^>]+-->\s*```json orze-iter\s*(\{.*?\})\s*```\s*<!--END orze-iter [^>]+-->",
                text,
                flags=re.DOTALL,
            ):
                self.iters.append(Iteration.from_dict(json.loads(m.group(1))))


__all__ = ["Journal", "Iteration"]
