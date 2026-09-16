"""Application registration followed by the unchanged installed Orze CLI."""
from orze.core.research_interfaces import register_domain, register_policy
from examples.acceptance.compression import CompressionDomain
from examples.acceptance.sorting import SortingDomain
from examples.acceptance.policy import CommonPolicy
from .policy import DominancePruningPolicy


def main():
    import orze.cli
    register_domain("acceptance_sorting", "acceptance.sorting.v1", SortingDomain)
    register_domain("acceptance_compression", "acceptance.compression.v1", CompressionDomain)
    register_policy("acceptance", "acceptance.policy.v1", CommonPolicy)
    register_policy("dominance_pruning", "research.dominance_pruning.v1", DominancePruningPolicy)
    return orze.cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
