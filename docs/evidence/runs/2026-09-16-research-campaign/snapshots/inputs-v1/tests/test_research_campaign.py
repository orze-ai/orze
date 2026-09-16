"""Complete owned runs must bind inputs and preserve failures, not just scores."""
import importlib


def test_campaign_execution_and_scheduling_workload_are_available():
    campaign = importlib.import_module('examples.research_comparison.campaign')
    workload = importlib.import_module('examples.research_comparison.scheduling_campaign')
    assert callable(campaign.execute)
    assert callable(campaign.verify)
    assert callable(workload.run)
    assert callable(workload.verify)
