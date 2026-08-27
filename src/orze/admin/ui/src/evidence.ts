import type { ResearchEfficiency } from './types';

export function canPresentEfficiencyScore(eff: ResearchEfficiency): boolean {
  const presentation = eff.presentation;
  const qualification = eff.evidence_qualification;
  if (
    presentation?.qualification_applied !== true
    || presentation.claim_scope !== 'internal_research_efficiency'
    || presentation.leaderboard_rank_comparable !== false
    || typeof presentation.evidence_label !== 'string'
    || presentation.evidence_label.length === 0
    || qualification == null
    || !['verified_local_artifact', 'benchmark_contract'].includes(qualification.mode)
    || typeof qualification.primary_metric !== 'string'
    || qualification.primary_metric.length === 0
    || qualification.fallback_metrics_allowed !== false
    || !Number.isInteger(qualification.accepted)
    || qualification.accepted < 0
    || qualification.rejected == null
    || Array.isArray(qualification.rejected)
    || !Object.values(qualification.rejected).every(
      (count) => Number.isInteger(count) && count >= 0,
    )
    || typeof eff.score !== 'number'
    || !Number.isFinite(eff.score)
  ) return false;

  if (qualification.mode === 'benchmark_contract') {
    return [
      qualification.benchmark_id,
      qualification.benchmark_view,
      qualification.evidence_scope,
      qualification.selection_mode,
    ].every((value) => typeof value === 'string' && value.length > 0);
  }
  return true;
}

export function rejectedEvidenceCount(eff: ResearchEfficiency): number {
  if (!canPresentEfficiencyScore(eff)) return 0;
  return Object.values(eff.evidence_qualification!.rejected)
    .reduce((sum, count) => sum + count, 0);
}
