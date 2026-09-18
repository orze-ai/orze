"""Resolve optional exact edits against a measured research program.

The application supplies current, source-qualified history and reviews the
resolved program before execution. A small textual edit is not evidence of a
single scientific factor: its consequences still require inspection.
"""
import copy

from .open_experiment import identity


def parent_program(history, action_id):
    """Return a copy of the valid method bound to this complete identity."""
    if type(action_id) is not str or len(action_id) != 64:
        raise ValueError("provide the full parent action identity")
    matches = [row for row in history if row.get("action_id") == action_id]
    if not matches:
        raise ValueError("parent is absent from measured history")
    for row in matches:
        action = row.get("action")
        if (row.get("valid") is not True or type(action) is not dict
                or set(action) != {"kind", "source"}
                or action["kind"] != "method" or type(action["source"]) is not str
                or identity(action) != action_id):
            raise ValueError("parent must be an intact valid measured method")
    return copy.deepcopy(matches[0]["action"])


def resolve(action, history):
    """Resolve sequential, uniquely matching replacements without mutating history.

Full new programs remain supported. Every replacement must match once in the
current source; missing/ambiguous matches are errors, never guessed edits.
"""
    if type(action) is not dict:
        raise ValueError("action must be an object")
    if set(action) == {"kind", "source"} and action["kind"] in {"method", "analyze"}:
        return copy.deepcopy(action)
    if set(action) != {"kind", "parent_id", "edits"} or action["kind"] != "method":
        raise ValueError("provide a complete program or a method with parent_id and edits")
    edits = action["edits"]
    if type(edits) is not list or not 1 <= len(edits) <= 32:
        raise ValueError("provide 1..32 sequential exact edits")
    result = parent_program(history, action["parent_id"])
    for edit in edits:
        if (type(edit) is not dict or set(edit) != {"old", "new"}
                or type(edit["old"]) is not str or not edit["old"]
                or type(edit["new"]) is not str):
            raise ValueError("each edit requires nonempty old and string new")
        if result["source"].count(edit["old"]) != 1:
            raise ValueError("edit old must match exactly once")
        result["source"] = result["source"].replace(edit["old"], edit["new"], 1)
        if not 1 <= len(result["source"].encode()) <= 32768:
            raise ValueError("resolved source must contain 1..32768 UTF-8 bytes")
    return result
