import { useMemo, useState, useCallback, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  type Node,
  type Edge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  AlertTriangle,
  GitBranch,
  Sparkles,
  Repeat,
  XCircle,
  Compass,
  Network,
  Copy,
  HelpCircle,
  Gauge,
} from 'lucide-react';

import { useSearchPath } from './hooks';
import { useIdeaDetail } from './IdeaDetailContext';
import { Card, LoadingState, Pill } from './components';
import type { SearchPathNode, SearchPathProblem, ResearchEfficiency } from './types';

const X_SCALE = 26;
const Y_SCALE = 96;
const VIEW_NODE_CAP = 1400; // keep one view responsive

const PROBLEM_META: Record<string, { label: string; color: string; icon: any }> = {
  flat_hub: { label: 'Flat hub (no evolution)', color: '#38bdf8', icon: Network },
  under_researched: { label: 'Under-researched', color: '#fbbf24', icon: Sparkles },
  over_researched: { label: 'Over-researched', color: '#fb923c', icon: Repeat },
  failed_cluster: { label: 'Failed cluster', color: '#ef4444', icon: XCircle },
  missing_coverage: { label: 'Missing coverage', color: '#a78bfa', icon: Compass },
  pseudo_evolution: { label: 'Pseudo-evolution (re-run/eval)', color: '#f472b6', icon: Copy },
  unjustified_branch: { label: 'Unjustified branch (no rationale)', color: '#facc15', icon: HelpCircle },
};

// Generic evolution-operator styling for edges. Colour encodes *how* a child
// was derived from its parent; a dashed line marks an edge that fails the
// evolution contract (identical config, or no recorded rationale).
const EVO_META: Record<string, { label: string; color: string }> = {
  seed: { label: 'Seed', color: '#64748b' },
  mutate_param: { label: 'Single-param mutation', color: '#22c55e' },
  multi_param: { label: 'Multi-param change', color: '#3b82f6' },
  combine: { label: 'Combine / merge', color: '#a855f7' },
  cross_eval: { label: 'Cross-eval (same config)', color: '#f472b6' },
  audit: { label: 'Audit', color: '#fb923c' },
  replicate: { label: 'Replicate (identical)', color: '#f43f5e' },
  unknown: { label: 'Unknown (config missing)', color: '#52525b' },
};

function evoColor(t?: string): string {
  return EVO_META[t || 'unknown']?.color || EVO_META.unknown.color;
}

const SEV_COLOR: Record<string, string> = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#8b5cf6',
};

// Map a node's "goodness percentile" (1 = best) to a red→green heat colour.
function heatColor(pct: number | null): string {
  if (pct === null) return '#3f3f46';
  const hue = 120 * pct; // 0=red, 120=green
  return `hsl(${hue}, 70%, 45%)`;
}

function nodeFill(n: SearchPathNode): string {
  const s = (n.status || '').toLowerCase();
  if (s === 'failed' || s === 'error') return '#7f1d1d';
  if (n.metric === null) return '#3f3f46';
  return heatColor(n.score_pct);
}

function problemRing(n: SearchPathNode): string | null {
  if (n.problems.includes('failed_cluster')) return PROBLEM_META.failed_cluster.color;
  if (n.problems.includes('pseudo_evolution')) return PROBLEM_META.pseudo_evolution.color;
  if (n.problems.includes('flat_hub')) return PROBLEM_META.flat_hub.color;
  if (n.problems.includes('under_researched')) return PROBLEM_META.under_researched.color;
  if (n.problems.includes('over_researched')) return PROBLEM_META.over_researched.color;
  if (n.problems.includes('unjustified_branch')) return PROBLEM_META.unjustified_branch.color;
  return null;
}

function ResearchTreeInner() {
  const data = useSearchPath();
  const { openIdea } = useIdeaDetail();
  const rf = useReactFlow();

  const loading = (data as any)._loading && data.nodes.length === 0;

  // group nodes by their root (tree id) ----------------------------------
  const { byId, rootOf, roots } = useMemo(() => {
    const byId = new Map<string, SearchPathNode>();
    data.nodes.forEach((n) => byId.set(n.id, n));
    const rootOf = new Map<string, string>();
    const resolveRoot = (id: string): string => {
      const seen = new Set<string>();
      let cur = id;
      while (true) {
        if (rootOf.has(cur)) return rootOf.get(cur)!;
        const node = byId.get(cur);
        if (!node || !node.parent || !byId.has(node.parent) || seen.has(cur)) return cur;
        seen.add(cur);
        cur = node.parent;
      }
    };
    data.nodes.forEach((n) => rootOf.set(n.id, resolveRoot(n.id)));
    const sizeByRoot = new Map<string, number>();
    const depthByRoot = new Map<string, number>();
    data.nodes.forEach((n) => {
      const r = rootOf.get(n.id)!;
      sizeByRoot.set(r, (sizeByRoot.get(r) || 0) + 1);
      depthByRoot.set(r, Math.max(depthByRoot.get(r) || 0, n.depth));
    });
    const roots = [...sizeByRoot.entries()]
      .map(([id, size]) => ({
        id,
        size,
        depth: depthByRoot.get(id) || 0,
        title: byId.get(id)?.title || id,
      }));
    return { byId, rootOf, roots };
  }, [data.nodes]);

  const [selectedRoot, setSelectedRoot] = useState<string>('');
  const [filter, setFilter] = useState<'all' | 'problems' | 'scored'>('all');
  const [sortBy, setSortBy] = useState<'depth' | 'size'>('depth');
  const [hover, setHover] = useState<SearchPathNode | null>(null);

  const sortedRoots = useMemo(() => {
    const rs = [...roots];
    rs.sort((a, b) => (sortBy === 'depth' ? b.depth - a.depth || b.size - a.size : b.size - a.size));
    return rs;
  }, [roots, sortBy]);

  // default to the most-evolved (deepest) tree once data arrives, so branching
  // lineages are visible rather than the giant flat hub.
  useEffect(() => {
    if (sortedRoots.length && (!selectedRoot || !sortedRoots.find((r) => r.id === selectedRoot))) {
      setSelectedRoot(sortedRoots[0].id);
    }
  }, [sortedRoots, selectedRoot]);

  const { rfNodes, rfEdges, viewCount, capped } = useMemo(() => {
    if (!selectedRoot) return { rfNodes: [], rfEdges: [], viewCount: 0, capped: false };
    let pool = data.nodes.filter((n) => rootOf.get(n.id) === selectedRoot);
    pool.sort((a, b) => a.depth - b.depth);
    const capped = pool.length > VIEW_NODE_CAP;
    if (capped) pool = pool.slice(0, VIEW_NODE_CAP);
    const visible = new Set(pool.map((n) => n.id));

    const passes = (n: SearchPathNode) =>
      filter === 'all' ||
      (filter === 'problems' && n.problems.length > 0) ||
      (filter === 'scored' && n.metric !== null);

    const rfNodes: Node[] = pool.map((n) => {
      const dim = Math.max(14, Math.min(40, Math.sqrt(n.subtree_size) * 6));
      const ring = problemRing(n);
      const dimmed = !passes(n);
      const deltaBorder =
        n.delta_vs_parent === null ? null : n.delta_vs_parent > 0 ? '#22c55e' : '#ef4444';
      return {
        id: n.id,
        position: { x: n.x * X_SCALE, y: n.y * Y_SCALE },
        data: { label: '' },
        draggable: false,
        connectable: false,
        selectable: true,
        style: {
          width: dim,
          height: dim,
          borderRadius: '50%',
          background: nodeFill(n),
          border: ring
            ? `3px solid ${ring}`
            : deltaBorder
            ? `2px solid ${deltaBorder}`
            : '1px solid rgba(255,255,255,0.25)',
          boxShadow: ring ? `0 0 10px ${ring}` : 'none',
          opacity: dimmed ? 0.12 : 1,
          padding: 0,
          cursor: 'pointer',
        },
      } as Node;
    });

    const rfEdges: Edge[] = data.edges
      .filter((e) => visible.has(e.source) && visible.has(e.target))
      .map((e) => {
        const child = byId.get(e.target);
        const evoType = child?.evolution_type;
        const failed = child?.contract_ok === false; // zero-delta or no rationale
        const color = evoColor(evoType);
        return {
          id: `${e.source}->${e.target}`,
          source: e.source,
          target: e.target,
          type: 'straight',
          animated: false,
          style: {
            stroke: color,
            strokeOpacity: failed ? 0.9 : 0.45,
            strokeWidth: failed ? 1.6 : 1,
            strokeDasharray: failed ? '4 3' : undefined,
          },
        };
      });

    return { rfNodes, rfEdges, viewCount: pool.length, capped };
  }, [data.nodes, data.edges, selectedRoot, rootOf, filter, byId]);

  const focusNode = useCallback(
    (nodeId: string) => {
      const n = byId.get(nodeId);
      if (!n) return;
      const root = rootOf.get(nodeId);
      if (root && root !== selectedRoot) setSelectedRoot(root);
      setTimeout(() => {
        rf.setCenter(n.x * X_SCALE, n.y * Y_SCALE, { zoom: 1.4, duration: 600 });
      }, 60);
    },
    [byId, rootOf, selectedRoot, rf],
  );

  if (loading) return <LoadingState label="Building research tree…" />;

  if (data.stats?.error || data.nodes.length === 0) {
    return (
      <Card>
        <div className="text-sm text-gray-400">
          No genealogy data available
          {data.stats?.error ? ` — ${data.stats.error}` : ''}.
        </div>
      </Card>
    );
  }

  const s = data.stats;
  const pc = s.problem_counts || {};

  return (
    <div className="space-y-4">
      {/* stat header */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-8">
        <Pill label="Ideas" value={s.n_total} sub={`${s.n_in_tree ?? s.n_rendered} in tree`} />
        <Pill label="Branches" value={s.n_roots ?? roots.length} sub="root lineages" />
        <Pill label="Max depth" value={s.max_depth ?? 0} sub={`mean ${s.mean_depth ?? 0}`} />
        <Pill
          label="Evolution"
          value={
            s.evolution_rate === null || s.evolution_rate === undefined
              ? '—'
              : `${(s.evolution_rate * 100).toFixed(1)}%`
          }
          sub={`${s.intermediate_nodes ?? 0} evolved`}
        />
        <Pill
          label="Genuine evo"
          value={
            s.genuine_evolution_rate === null || s.genuine_evolution_rate === undefined
              ? '—'
              : `${(s.genuine_evolution_rate * 100).toFixed(1)}%`
          }
          sub={`${s.judged_edges ?? 0} judged · ${s.undiffable_edges ?? 0} n/a`}
        />
        <Pill label="Scored" value={s.n_scored ?? 0} sub={data.metric.name} />
        <Pill
          label="Refine win%"
          value={
            s.refinement_success_rate === null || s.refinement_success_rate === undefined
              ? '—'
              : `${Math.round(s.refinement_success_rate * 100)}%`
          }
          sub={`${s.refinement_pairs ?? 0} pairs`}
        />
        <Pill label="Problems" value={data.problems.length} sub="see attention list" />
      </div>

      {data.research_efficiency && (
        <EfficiencyPanel eff={data.research_efficiency} metricName={data.metric.name} />
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_360px]">
        {/* graph */}
        <Card className="!p-0 overflow-hidden">
          <div className="flex flex-wrap items-center gap-2 border-b border-white/5 p-3">
            <GitBranch size={15} className="text-purple-400" />
            <span className="text-sm font-semibold">Research Tree</span>
            <select
              value={selectedRoot}
              onChange={(e) => setSelectedRoot(e.target.value)}
              className="ml-2 rounded-md bg-white/5 px-2 py-1 text-xs text-gray-200 border border-white/10 max-w-[260px]"
            >
              {sortedRoots.slice(0, 120).map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} · depth {r.depth} · {r.size} nodes
                </option>
              ))}
            </select>
            <button
              onClick={() => setSortBy((s) => (s === 'depth' ? 'size' : 'depth'))}
              className="rounded-md bg-white/5 px-2 py-1 text-[11px] text-gray-300 border border-white/10 hover:text-white"
              title="Sort branch list"
            >
              sort: {sortBy}
            </button>
            <div className="ml-auto flex gap-1">
              {(['all', 'problems', 'scored'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`rounded-md px-2 py-1 text-[11px] capitalize transition-colors ${
                    filter === f
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                      : 'bg-white/5 text-gray-400 border border-transparent hover:text-white'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
          <div style={{ height: 560 }} className="relative">
            <ReactFlow
              nodes={rfNodes}
              edges={rfEdges}
              onNodeClick={(_e, n) => openIdea(n.id)}
              onNodeMouseEnter={(_e, n) => setHover(byId.get(n.id) || null)}
              onNodeMouseLeave={() => setHover(null)}
              onlyRenderVisibleElements
              minZoom={0.05}
              maxZoom={3}
              fitView
              proOptions={{ hideAttribution: true }}
              nodesDraggable={false}
              nodesConnectable={false}
            >
              <Background color="#27272a" gap={24} />
              <Controls showInteractive={false} />
              <MiniMap
                pannable
                zoomable
                nodeColor={(n) => (n.style?.background as string) || '#52525b'}
                maskColor="rgba(9,9,11,0.7)"
              />
            </ReactFlow>
            {hover && <EvolutionInspector node={hover} byId={byId} metricName={data.metric.name} />}
            <div className="pointer-events-none absolute bottom-2 left-2 rounded-md bg-black/50 px-2 py-1 text-[10px] text-gray-400">
              {viewCount} nodes shown{capped ? ' (capped — pick a smaller branch)' : ''} · color =
              {' '}{data.metric.name} rank · size = subtree · ring = problem · edge = evolution type (dashed = fails contract)
            </div>
          </div>
          <Legend metricName={data.metric.name} />
        </Card>

        {/* attention panel */}
        <Card className="!p-0 flex flex-col" >
          <div className="flex items-center gap-2 border-b border-white/5 p-3">
            <AlertTriangle size={15} className="text-amber-400" />
            <span className="text-sm font-semibold">Attention list</span>
            <div className="ml-auto flex gap-2 text-[10px] text-gray-400">
              {Object.entries(pc).map(([k, v]) => (
                <span key={k}>
                  {PROBLEM_META[k]?.label || k}: <b className="text-gray-200">{v}</b>
                </span>
              ))}
            </div>
          </div>
          <div className="max-h-[620px] overflow-y-auto divide-y divide-white/5">
            {data.problems.length === 0 && (
              <div className="p-4 text-xs text-gray-500">No problems detected. 🎉</div>
            )}
            {data.problems.map((p, i) => (
              <ProblemRow key={i} p={p} onFocus={focusNode} />
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function ProblemRow({
  p,
  onFocus,
}: {
  p: SearchPathProblem;
  onFocus: (id: string) => void;
}) {
  const meta = PROBLEM_META[p.kind] || { label: p.kind, color: '#94a3b8', icon: AlertTriangle };
  const Icon = meta.icon;
  return (
    <button
      onClick={() => p.node_id && onFocus(p.node_id)}
      className="block w-full px-3 py-2.5 text-left hover:bg-white/5 transition-colors"
    >
      <div className="flex items-center gap-2">
        <Icon size={13} style={{ color: meta.color }} />
        <span className="text-[11px] font-semibold" style={{ color: meta.color }}>
          {meta.label}
        </span>
        <span
          className="ml-auto rounded px-1.5 py-0.5 text-[9px] font-bold uppercase"
          style={{ background: `${SEV_COLOR[p.severity] || '#52525b'}22`, color: SEV_COLOR[p.severity] || '#94a3b8' }}
        >
          {p.severity}
        </span>
      </div>
      {p.node_id && (
        <div className="mt-1 font-mono text-[11px] text-purple-400">{p.node_id}</div>
      )}
      {p.region && (
        <div className="mt-1 font-mono text-[11px] text-purple-300">{p.region}</div>
      )}
      <div className="mt-0.5 text-[11px] text-gray-300">{p.reason}</div>
      <div className="mt-0.5 text-[10px] text-gray-500">→ {p.suggestion}</div>
    </button>
  );
}

function Legend({ metricName }: { metricName: string }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-white/5 p-2.5 text-[10px] text-gray-400">
      <span className="flex items-center gap-1">
        <i className="h-2.5 w-2.5 rounded-full" style={{ background: 'hsl(120,70%,45%)' }} /> best {metricName}
      </span>
      <span className="flex items-center gap-1">
        <i className="h-2.5 w-2.5 rounded-full" style={{ background: 'hsl(0,70%,45%)' }} /> worst
      </span>
      <span className="flex items-center gap-1">
        <i className="h-2.5 w-2.5 rounded-full bg-zinc-700" /> no eval
      </span>
      <span className="flex items-center gap-1">
        <i className="h-2.5 w-2.5 rounded-full" style={{ background: '#7f1d1d' }} /> failed
      </span>
      {Object.values(PROBLEM_META).map((m) => (
        <span key={m.label} className="flex items-center gap-1">
          <i className="h-2.5 w-2.5 rounded-full" style={{ boxShadow: `0 0 0 2px ${m.color}` }} />
          {m.label}
        </span>
      ))}
      <span className="w-full" />
      <span className="text-gray-500">edges:</span>
      {(['mutate_param', 'multi_param', 'combine', 'cross_eval', 'unknown'] as const).map((t) => (
        <span key={t} className="flex items-center gap-1">
          <i className="h-0.5 w-4" style={{ background: EVO_META[t].color }} /> {EVO_META[t].label}
        </span>
      ))}
      <span className="flex items-center gap-1">
        <i className="h-0.5 w-4" style={{ background: '#f43f5e', borderTop: '1px dashed #f43f5e' }} />
        dashed = fails contract
      </span>
    </div>
  );
}

// Floating inspector: shows *why* a node is (or isn't) a genuine evolution of
// its parent — the structural config delta, the recorded rationale, and the
// contract verdict. Task-agnostic: keys/values are whatever the project uses.
function EvolutionInspector({
  node,
  byId,
  metricName,
}: {
  node: SearchPathNode;
  byId: Map<string, SearchPathNode>;
  metricName: string;
}) {
  const evo = EVO_META[node.evolution_type || 'unknown'] || EVO_META.unknown;
  const parent = node.parent ? byId.get(node.parent) : null;
  const delta = node.parent_delta || [];
  const ok = node.contract_ok;
  const verdict =
    ok === true ? { t: 'genuine evolution', c: '#22c55e' }
    : ok === false ? { t: 'fails contract', c: '#f43f5e' }
    : { t: 'unjudgeable (config missing)', c: '#a1a1aa' };
  return (
    <div className="absolute right-2 top-2 z-10 max-h-[540px] w-[320px] overflow-y-auto rounded-lg border border-white/10 bg-black/85 p-3 text-[11px] shadow-xl backdrop-blur">
      <div className="font-mono text-purple-300">{node.id}</div>
      <div className="mt-0.5 text-gray-300">{node.title}</div>
      <div className="mt-2 flex items-center gap-2">
        <span className="rounded px-1.5 py-0.5 font-semibold" style={{ background: `${evo.color}22`, color: evo.color }}>
          {evo.label}
        </span>
        <span className="rounded px-1.5 py-0.5 font-semibold" style={{ background: `${verdict.c}22`, color: verdict.c }}>
          {verdict.t}
        </span>
      </div>
      <div className="mt-2 text-gray-400">
        parent:{' '}
        {parent ? <span className="font-mono text-purple-400">{parent.id}</span> : <span className="italic">none (seed)</span>}
        {node.delta_vs_parent !== null && (
          <span className={node.delta_vs_parent > 0 ? 'text-green-400' : 'text-red-400'}>
            {' '}· Δ{metricName} {node.delta_vs_parent > 0 ? '+' : ''}{node.delta_vs_parent}
          </span>
        )}
      </div>
      {node.contract_violations && node.contract_violations.length > 0 && (
        <div className="mt-1 text-[10px] text-rose-300">
          ✗ {node.contract_violations.map((v) => v.replace('_', ' ')).join(', ')}
        </div>
      )}
      <div className="mt-2 text-gray-500">
        rationale {node.rationale ? '' : '(missing)'}
      </div>
      <div className="mt-0.5 text-gray-300">
        {node.rationale || <span className="italic text-rose-300">No rationale recorded for this change.</span>}
      </div>
      {parent && (
        <>
          <div className="mt-2 text-gray-500">
            config delta vs parent{' '}
            {node.delta_size !== undefined && node.delta_size >= 0
              ? `(${node.delta_size} key${node.delta_size === 1 ? '' : 's'})`
              : '(unavailable)'}
          </div>
          {delta.length === 0 ? (
            <div className="mt-0.5 italic text-gray-500">
              {node.delta_size === 0 ? 'identical config — not a real change' : 'no diff available'}
            </div>
          ) : (
            <table className="mt-1 w-full table-fixed border-collapse">
              <tbody>
                {delta.map((d) => (
                  <tr key={d.key} className="align-top">
                    <td className="truncate pr-1 font-mono text-[10px] text-sky-300" title={d.key}>{d.key}</td>
                    <td className="truncate pr-1 text-right text-[10px] text-gray-500" title={String(d.parent)}>{String(d.parent)}</td>
                    <td className="px-1 text-center text-gray-600">→</td>
                    <td className="truncate text-[10px] text-gray-200" title={String(d.child)}>{String(d.child)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
      <div className="mt-2 text-[10px] text-gray-600">click node for full idea detail</div>
    </div>
  );
}

const GRADE_COLOR: Record<string, string> = {
  A: '#22c55e', B: '#84cc16', C: '#eab308', D: '#f97316', F: '#ef4444',
};

const COMPONENT_LABEL: Record<string, string> = {
  yield: 'Yield (scored / idea)',
  success: 'Refine success',
  depth_utilization: 'Depth utilization',
  diversity: 'Hub diversity',
  reliability: 'Reliability (1 − fail)',
};

// Top-level research-efficiency panel: a single 0-100 score + grade for how well
// the engine *searches*, broken into weighted components, an explore-vs-exploit
// split, hub concentration, and a depth-yield curve (does deeper search pay off?).
function EfficiencyPanel({ eff, metricName }: { eff: ResearchEfficiency; metricName: string }) {
  const grade = eff.grade || '—';
  const gc = GRADE_COLOR[grade] || '#94a3b8';
  const ee = eff.exploration_exploitation;
  const exploitPct = Math.round((ee?.exploit_share || 0) * 100);
  const maxYield = Math.max(0.0001, ...eff.depth_yield.map((d) => d.scored_frac));
  return (
    <Card className="!p-0 overflow-hidden">
      <div className="flex items-center gap-2 border-b border-white/5 p-3">
        <Gauge size={15} className="text-emerald-400" />
        <span className="text-sm font-semibold">Evo Score</span>
        <span className="text-[11px] text-gray-500">orze's top-level metric — research efficiency (how well it searches)</span>
      </div>
      <div className="grid gap-4 p-4 lg:grid-cols-[160px_1fr_1fr]">
        {/* score + grade */}
        <div className="flex flex-col items-center justify-center rounded-lg bg-white/[0.02] p-3">
          <div className="text-4xl font-extrabold" style={{ color: gc }}>
            {eff.score == null ? '—' : Math.round(eff.score)}
          </div>
          <div className="mt-1 rounded px-2 py-0.5 text-xs font-bold" style={{ background: `${gc}22`, color: gc }}>
            grade {grade}
          </div>
          <div className="mt-2 text-center text-[10px] text-gray-500">
            yield {(eff.yield_rate * 100).toFixed(1)}% · fail {(eff.failure_rate * 100).toFixed(0)}%
          </div>
        </div>

        {/* component bars */}
        <div className="flex flex-col justify-center gap-1.5">
          {Object.entries(eff.components).map(([k, c]) => (
            <div key={k}>
              <div className="flex justify-between text-[10px] text-gray-400">
                <span>{COMPONENT_LABEL[k] || k} <span className="text-gray-600">×{c.weight}</span></span>
                <span className="text-gray-300">{Math.round(c.score * 100)}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-white/5">
                <div className="h-1.5 rounded-full" style={{ width: `${Math.min(100, c.score * 100)}%`, background: gc }} />
              </div>
            </div>
          ))}
          {/* explore vs exploit */}
          <div className="mt-1">
            <div className="flex justify-between text-[10px] text-gray-400">
              <span>Explore (breadth) vs Exploit (depth)</span>
              <span className="text-gray-300">{exploitPct}% deepened</span>
            </div>
            <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-white/5">
              <div className="h-1.5 bg-sky-500" style={{ width: `${100 - exploitPct}%` }} title={`explore ${ee?.explore}`} />
              <div className="h-1.5 bg-purple-500" style={{ width: `${exploitPct}%` }} title={`exploit ${ee?.exploit}`} />
            </div>
          </div>
        </div>

        {/* depth-yield curve */}
        <div className="flex flex-col">
          <div className="mb-1 flex justify-between text-[10px] text-gray-400">
            <span>Depth-yield — does deeper search produce scored results?</span>
            <span className="text-gray-600">top hub {Math.round((eff.concentration?.top1_share || 0) * 100)}% · gini {eff.concentration?.gini}</span>
          </div>
          <div className="flex flex-1 items-end gap-1" style={{ minHeight: 90 }}>
            {eff.depth_yield.map((d) => {
              const h = Math.max(2, (d.scored_frac / maxYield) * 78);
              const hot = d.scored > 0;
              return (
                <div key={d.label} className="flex flex-1 flex-col items-center justify-end" title={`depth ${d.label}: ${d.scored}/${d.n} scored (${(d.scored_frac * 100).toFixed(1)}%)${d.best_metric != null ? `, best ${metricName} ${d.best_metric}` : ''}`}>
                  <div className="w-full rounded-t" style={{ height: h, background: hot ? '#22c55e' : '#3f3f46' }} />
                  <span className="mt-0.5 text-[8px] text-gray-500">{d.label}</span>
                </div>
              );
            })}
          </div>
          <div className="mt-1 text-[9px] text-gray-600">bar = % of ideas at that depth that produced a score · grey = none</div>
        </div>
      </div>
    </Card>
  );
}

export default function ResearchTreeTab() {
  return (
    <ReactFlowProvider>
      <ResearchTreeInner />
    </ReactFlowProvider>
  );
}
