/* ── API response types ── */

export interface ActiveRun {
  idea_id: string;
  gpu: number;
  elapsed_min: number;
  host: string;
  title?: string;
}

export interface StatusResponse {
  timestamp: string;
  host: string;
  iteration: number;
  active: ActiveRun[];
  free_gpus: number[];
  free_gpus_by_host: Record<string, number[]>;
  queue_depth: number;
  completed: number;
  failed: number;
  disk_free_gb: number;
  top_results: TopResult[];
}

export interface TopResult {
  idea_id: string;
  title: string;
  [metric: string]: any;
}

export interface GpuInfo {
  index: number;
  name: string;
  memory_used_mib: number;
  memory_total_mib: number;
  utilization_pct: number;
  temperature_c: number;
}

export interface Heartbeat {
  host: string;
  pid: number;
  timestamp: string;
  epoch: number;
  active: ActiveRun[];
  free_gpus: number[];
  gpu_info?: GpuInfo[];
  disk_free_gb?: number;
  os?: string;
  status: 'online' | 'degraded' | 'offline';
  heartbeat_age_sec: number;
}

export interface NodesResponse {
  heartbeats: Heartbeat[];
  local_gpus: GpuInfo[];
}

export interface Run {
  idea_id: string;
  status: 'RUNNING' | 'QUEUED' | 'COMPLETED' | 'FAILED' | 'ERROR' | string;
  host?: string;
  gpu?: number;
  metric?: number;
  metric_name?: string;
  started_at?: string;
  elapsed_min?: number;
  title?: string;
  tags?: string[];
  claimed_by?: string;
  claimed_gpu?: number;
  error?: string;
}

export interface RunsResponse {
  active: ActiveRun[];
  top_results?: TopResult[];
  recent: Run[];
}

export interface RunDetail {
  idea_id: string;
  metrics: Record<string, any> | null;
  claim: { claimed_by: string; claimed_at: string; gpu: number } | null;
  config?: Record<string, any>;
  title?: string;
}

export interface RunLog {
  idea_id: string;
  log: string;
}

export interface LeaderboardEntry {
  idea_id: string;
  title: string;
  metric_value: number;
  metric_name: string;
  training_time?: number;
}

export interface LeaderboardResponse {
  top: LeaderboardEntry[];
  metric: string;
  view?: string;
  title?: string;
}

export interface LeaderboardViewsResponse {
  views: string[];
}

export interface Alert {
  type: string;
  idea_id?: string;
  host?: string;
  error?: string;
  minutes_ago?: number;
  disk_free_gb?: number;
}

export interface AlertsResponse {
  alerts: Alert[];
  count: number;
}

export interface Idea {
  id: string;
  title: string;
  priority?: string;
  status: string;
  config?: any;
}

export interface IdeasResponse {
  ideas: Record<string, any>;
  count: number;
}

export interface QueueItem {
  idea_id: string;
  title: string;
  priority: string;
  status: string;
  config: Record<string, any>;
  category: string;
  parent: string;
  hypothesis: string;
  sweep_parent?: string;
  huggingface?: {
    model_id: string;
    url: string;
    feature_dim: number;
    img_size: number;
    source: string;
  };
}

export interface QueueResponse {
  queue: QueueItem[];
  total: number;
  total_all: number;
  page: number;
  per_page: number;
  total_pages: number;
  counts: Record<string, number>;
}

export interface IdeaDetail {
  idea_id: string;
  found: boolean;
  title?: string;
  hypothesis?: string;
  category?: string;
  parent?: string;
  research_cycle?: string;
  priority?: string;
  config?: Record<string, any>;
  origin?: string;
  metrics?: Record<string, any>;
  lake_metrics?: Record<string, any>;
  claim?: { claimed_by: string; claimed_at: string; gpu: number };
}

export type Tab = 'overview' | 'nodes' | 'runs' | 'queue' | 'leaderboard' | 'research-tree' | 'alerts' | 'settings';

export interface SearchPathDelta {
  key: string;
  parent: string | number | boolean | null;
  child: string | number | boolean | null;
}

export interface SearchPathNode {
  id: string;
  title: string;
  parent: string | null;
  category: string;
  approach_family: string;
  status: string;
  priority?: string;
  metric: number | null;
  score_pct: number | null;
  depth: number;
  n_children: number;
  subtree_size: number;
  subtree_depth?: number;
  delta_vs_parent: number | null;
  improved: boolean;
  evolution_type?: string;
  parent_delta?: SearchPathDelta[];
  delta_size?: number;
  rationale?: string | null;
  contract_ok?: boolean | null;
  contract_violations?: string[];
  training_time?: number | null;
  x: number;
  y: number;
  problems: string[];
}

export interface SearchPathProblem {
  kind: 'under_researched' | 'over_researched' | 'failed_cluster' | 'missing_coverage'
      | 'flat_hub' | 'pseudo_evolution' | 'unjustified_branch' | string;
  severity: 'high' | 'medium' | 'low' | string;
  reason: string;
  suggestion: string;
  node_id: string | null;
  region: string | null;
}

export interface ResearchEfficiencyComponent {
  value: number;
  score: number;
  weight: number;
}

export interface DepthYieldRow {
  depth: number;
  label: string;
  n: number;
  scored: number;
  scored_frac: number;
  best_metric: number | null;
}

export interface ResearchEfficiency {
  score: number | null;
  grade: string;
  components: Record<string, ResearchEfficiencyComponent>;
  weights_sum?: number;
  exploration_exploitation: { explore: number; exploit: number; exploit_share: number };
  concentration: { top1_share: number; top5_share: number; max_fanout: number; gini: number };
  failure_rate: number;
  yield_rate: number;
  depth_yield: DepthYieldRow[];
}

export interface ResearchEfficiencyResponse extends ResearchEfficiency {
  metric?: { name: string; lower_is_better: boolean };
  n_total?: number;
  n_scored?: number;
  genuine_evolution_rate?: number | null;
  error?: string;
  _loading?: boolean;
}

export interface SearchPathResponse {
  metric: { name: string; lower_is_better: boolean };
  nodes: SearchPathNode[];
  edges: { source: string; target: string }[];
  problems: SearchPathProblem[];
  coverage: Record<string, Record<string, number>>;
  stats: {
    n_total: number;
    n_in_tree?: number;
    n_rendered: number;
    n_scored?: number;
    n_roots?: number;
    max_depth?: number;
    mean_depth?: number;
    status_counts?: Record<string, number>;
    refinement_success_rate?: number | null;
    refinement_pairs?: number;
    evolution_rate?: number | null;
    intermediate_nodes?: number;
    n_edges?: number;
    judged_edges?: number;
    undiffable_edges?: number;
    genuine_evolution_rate?: number | null;
    contract_ok_edges?: number;
    zero_delta_edges?: number;
    no_rationale_edges?: number;
    evolution_types?: Record<string, number>;
    truncated?: boolean;
    problem_counts?: Record<string, number>;
    error?: string;
  };
  research_efficiency?: ResearchEfficiency;
}
