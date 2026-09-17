"""Resolve an explicit choice in already qualified, same-scope history.

The application supplies canonicalization and identity rules. A full candidate
is an alternative to its opaque ID, not a new fit or a nearest-match query.
Callers still verify the selected measurements and perform final confirmation.
"""


def select_existing(selection, history, *, canonicalize, identity):
    if (type(selection) is not dict or selection.get("kind") != "select"
            or set(selection) not in ({"kind", "candidate_id"}, {"kind", "candidate"})):
        raise ValueError("select requires exactly one candidate ID or complete candidate")
    candidate = None
    if "candidate" in selection:
        candidate = canonicalize(selection["candidate"])
        candidate_id = identity(candidate)
    else:
        candidate_id = selection["candidate_id"]
        if type(candidate_id) is not str or not candidate_id:
            raise ValueError("candidate ID must be a nonempty string")
    found = next((row for row in history if row["id"] == candidate_id), None)
    if found is None:
        raise ValueError("candidate is absent from verified history")
    stored = canonicalize(found["candidate"])
    if identity(stored) != found["id"] or (candidate is not None and stored != candidate):
        raise ValueError("stored candidate identity differs from the explicit choice")
    return found
