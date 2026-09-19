class Policy:
    """Anchor-and-momentum portfolio explorer for bounded, batched discovery.

    Decision signals (all computed from the revealed prefix only, normalised by
    score_scale and measured relative to the root score):

    * anchor: a branch's best successful score so far.  It is never erased by a
      later failure or by a worse evaluation, so an implementation failure does
      not destroy the evidence that a direction works.
    * momentum: the normalised change between the two most recent valid scores.
      Positive momentum strongly favours deepening the same line.
    * branch stall: number of valid evaluations since that branch last improved
      on its own anchor.  Each stalled evaluation costs a small, bounded amount
      of priority, so a plateau demotes a line gradually instead of closing it;
      a stalled line can be resumed when the current leader also flattens.
    * maturation: a line whose steps returned analysis-only results (null score)
      has produced no evidence yet, not bad evidence.  Such lines get a
      maturation allowance of a few steps at near-leader priority before their
      priority decays.
    * bounded repair: a `repairable` tail keeps the branch's anchor and is
      retried while consecutive failures stay within repair_patience; beyond
      that the episode is abandoned.  `blocked` tails are closed immediately.
    * breadth pressure: unopened roots are valued optimistically from the
      current leader plus an exploration bonus that grows when the global best
      has stalled and shrinks as more roots are already open or when too few
      calls remain for a new start to mature.  At least two independent starts
      are always opened so trajectories can be compared.

    Candidates from every role (deepen, mature, repair, open) are scored on one
    scale and the top ones are taken, one per branch, so a parallel batch always
    spends its workers on distinct hypotheses rather than duplicating a guess.
    """

    def __init__(self, maturation=3, patience=2, repair_patience=2):
        self.maturation = max(1, int(maturation))
        self.patience = max(1, int(patience))
        self.repair_patience = max(1, int(repair_patience))

    @staticmethod
    def _num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def decide(self, view):
        legal = list(view.get("legal") or [])
        slots = min(int(view.get("workers") or 1), int(view.get("remaining_calls") or 0))
        if not legal or slots <= 0:
            return {"actions": [], "reason": "No legal action or no remaining call budget; stopping."}

        scale = view.get("score_scale")
        scale = float(scale) if self._num(scale) and float(scale) > 0 else 1.0
        rootnode = view.get("root") or {}
        base = float(rootnode.get("score")) if self._num(rootnode.get("score")) else 0.0
        remaining = int(view.get("remaining_calls") or 0)
        obs = list(view.get("observed") or [])

        paths = {}
        for i, n in enumerate(obs):
            paths.setdefault(n.get("branch"), []).append((i, n))

        info, best_norm = {}, 0.0
        for b, items in paths.items():
            items.sort(key=lambda p: (p[1].get("step", 0), p[0]))
            nodes = [n for _, n in items]
            valid = [n for n in nodes if n.get("status") == "ok" and self._num(n.get("score"))]
            anchor = max((n["score"] for n in valid), default=None)
            norm = (anchor - base) / scale if anchor is not None else None
            cur, stall = None, 0
            for n in valid:
                if cur is None or n["score"] > cur:
                    cur, stall = n["score"], 0
                else:
                    stall += 1
            gain = (valid[-1]["score"] - valid[-2]["score"]) / scale if len(valid) >= 2 else 0.0
            fails = 0
            for n in reversed(nodes):
                if n.get("status") == "ok":
                    break
                fails += 1
            info[b] = {"norm": norm, "stall": stall, "gain": gain, "fails": fails,
                       "tail": nodes[-1], "depth": len(nodes), "valid": len(valid)}
            if norm is not None and norm > best_norm:
                best_norm = norm

        gstall, seen = 0, None
        for n in obs:
            if n.get("status") == "ok" and self._num(n.get("score")):
                if seen is None or n["score"] > seen:
                    seen, gstall = n["score"], 0
                else:
                    gstall += 1

        opened = len(paths)
        cands = []
        for a in legal:
            b = a.get("branch")
            d = info.get(b)
            if d is None:
                v = 0.30 * max(best_norm, 0.0) + 0.12 - 0.05 * max(0, opened - 2)
                if gstall >= self.patience:
                    v += 0.12
                if opened < 2:
                    v += 1.0
                if opened >= 1 and remaining < self.maturation:
                    v -= 1.0
                kind = "open"
            else:
                status = d["tail"].get("status")
                if status == "blocked":
                    v, kind = -1.0, "blocked"
                elif status == "repairable":
                    if d["fails"] <= self.repair_patience:
                        anchor = d["norm"] if d["norm"] is not None else 0.25 * max(best_norm, 0.0)
                        v, kind = anchor + 0.10 - 0.10 * (d["fails"] - 1), "repair"
                    else:
                        v, kind = -0.8, "abandon"
                elif d["valid"] == 0:
                    if d["depth"] < self.maturation:
                        v, kind = 0.45 * max(best_norm, 0.0) + 0.18, "mature"
                    else:
                        v, kind = 0.20 * max(best_norm, 0.0) - 0.10, "slow"
                else:
                    v = (d["norm"] + 1.2 * max(0.0, d["gain"]) - 0.13 * d["stall"]
                         + 0.05 * max(0, self.maturation - d["depth"]))
                    kind = "deepen" if d["stall"] == 0 else "revisit"
            cands.append((v, kind, b, a))

        cands.sort(key=lambda t: (-t[0], t[3].get("step", 0), str(t[3].get("id"))))

        chosen, used, roles = [], set(), {}
        for v, kind, b, a in cands:
            if len(chosen) >= slots:
                break
            if kind in ("blocked", "abandon") or b in used:
                continue
            used.add(b)
            chosen.append(a)
            roles[kind] = roles.get(kind, 0) + 1
        if not chosen:
            v, kind, b, a = cands[0]
            if v > -0.9:
                chosen = [a]
                roles[kind] = 1

        if not chosen:
            return {"actions": [], "reason": "All revealed paths are blocked or past repair budget; stopping."}

        summary = ", ".join("%s=%d" % (k, c) for k, c in sorted(roles.items()))
        reason = ("lead=%.3f global-stall=%d | %d opened paths, %d unopened roots, %d calls left | roles: %s "
                  "| deepen live momentum, keep anchors through failures, let analysis-only starts mature, "
                  "bounded repair, widen when the leader flattens." %
                  (best_norm, gstall, opened,
                   sum(1 for a in legal if a.get("branch") not in info),
                   remaining, summary or "none"))
        return {"actions": [a["id"] for a in chosen], "reason": reason}