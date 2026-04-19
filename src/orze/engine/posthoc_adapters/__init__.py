"""Post-hoc adapters — pluggable inference drivers for each consumer project.

Each adapter is a callable ``(idea_id, cfg, idea_dir) -> metrics_dict``
registered via ``@register_adapter(name)``. Importing this package
auto-registers every shipped adapter.
"""

from orze.engine.posthoc_runner import register_adapter  # noqa: F401
