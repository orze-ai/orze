"""Register the campaign's task before entering the ordinary Pro research CLI."""
from orze.core.research_interfaces import register_domain
from examples.holdout.scheduling import SchedulingDomain


def main():
    register_domain('schedule_holdout', 'acceptance.schedule.v1', SchedulingDomain)
    from orze_pro.agents.research import main as research_main
    return research_main()


if __name__ == '__main__':
    raise SystemExit(main())
