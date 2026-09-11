"""Register the example adapters and use the unchanged real Orze CLI."""
from orze.core.research_interfaces import register_domain, register_policy

from .compression import CompressionDomain
from .policy import CommonPolicy
from .sorting import SortingDomain


def main():
    import orze.cli
    register_domain("acceptance_sorting", "acceptance.sorting.v1", SortingDomain)
    register_domain("acceptance_compression", "acceptance.compression.v1", CompressionDomain)
    register_policy("acceptance", "acceptance.policy.v1", CommonPolicy)
    return orze.cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
