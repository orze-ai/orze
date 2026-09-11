"""Register only the holdout Domain, then use the existing real Orze CLI."""
from orze.core.research_interfaces import register_domain

from .scheduling import SchedulingDomain


def main():
    import orze.cli
    register_domain("schedule_holdout", "acceptance.schedule.v1", SchedulingDomain)
    return orze.cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
