"""
Binary handover decision problem (STAY vs HO-to-top-challenger) for the
interpretable-policy experiment (G2). Same real-orbit scenario; the per-slot
decision is reduced to a clean binary choice so the learned rule is readable.

  - top(s,n): highest-SINR visible sat != serving n (the challenger).
  - action a ∈ {STAY (serve n if visible), HO (serve top)}.
  - DP over state (s, n, p) [p = serving two slots ago, for W_pp=2 ping-pong]
    gives V and an EXPERT policy expert(s,n,p) -> optimal action (for DAgger).
  - features(s, n, p): 8-dim causal/ephemeris-predictable vector.
  - rollout(policy_fn): run a policy to get a serving sequence + J.

Comments in English by house rule.
"""
from __future__ import annotations
import numpy as np
from handover import evaluate, Scenario

NONE = -1

def top_challenger(sc: Scenario, n, s):
    """Highest-SINR visible sat != n at slot s, or NONE."""
    vis = sc.visible[s]
    cand = vis[vis != n] if n != NONE else vis
    if len(cand) == 0:
        return NONE
    return int(cand[np.argmax(sc.sinr[cand, s])])

def vis_remain(sc: Scenario, n, s):
    """#future consecutive slots n stays visible (ephemeris-predictable, geometric)."""
    if n == NONE:
        return 0
    r = 0
    for t in range(s, sc.S):
        if sc.vis[n, t] == 1:
            r += 1
        else:
            break
    return r

def features(sc: Scenario, n, p, s):
    """8-dim decision features at slot s given serving n (prev2 = p). Causal."""
    top = top_challenger(sc, n, s)
    sinr_n = sc.sinr[n, s] if n != NONE else 0.0
    sinr_t = sc.sinr[top, s] if top != NONE else 0.0
    el_n = sc.el[n, s] if n != NONE else -90.0
    return np.array([
        sinr_n,                                   # serving SINR
        sinr_t,                                   # challenger SINR
        sinr_t - sinr_n,                          # Delta SINR (key hysteresis var)
        el_n,                                     # serving elevation
        vis_remain(sc, n, s),                     # serving visibility remaining (ephemeris)
        vis_remain(sc, top, s),                   # challenger visibility remaining
        1.0 if (n != NONE and sc.vis[n, s] == 1) else 0.0,   # serving visible
        1.0 if (top != NONE and top == p) else 0.0,          # HO would ping-pong (top==prev2)
    ], dtype=np.float32), top

FEATURE_NAMES = ["sinr_serv", "sinr_chal", "dsinr", "el_serv",
                 "visrem_serv", "visrem_chal", "serv_visible", "ho_is_pingpong"]

# ----------------------------- binary DP + expert --------------------------

def solve_dp(sc: Scenario):
    """DP over (s, n, p) on the binary action space. Returns V dict and expert fn."""
    import sys
    sys.setrecursionlimit(20000)
    p_ = sc.p
    lam1, lam2, lam3, sinr_th = p_["lam1"], p_["lam2"], p_["lam3"], p_["sinr_th"]
    S = sc.S
    memo = {}

    def slot_cost(m, s):
        c = 0.0
        if m == NONE or sc.sinr[m, s] < sinr_th:
            c += lam2
        if m != NONE:
            c += -sc.sinr[m, s]
        return c

    def actions(s, n):
        """Valid (label, resulting m) at slot s given serving n."""
        acts = []
        if n != NONE and sc.vis[n, s] == 1:
            acts.append((0, n))                   # STAY
        top = top_challenger(sc, n, s)
        if top != NONE:
            acts.append((1, top))                 # HO -> top
        if not acts:
            acts.append((0, NONE))                # forced outage
        return acts

    def rec(s, n, p):
        if s == S:
            return 0.0, None
        key = (s, n, p)
        if key in memo:
            return memo[key]
        best, best_a = float("inf"), None
        for (lab, m) in actions(s, n):
            c = slot_cost(m, s)
            if m != n and m != NONE and n != NONE:    # a real handover
                c += lam3
                if m == p:                            # ping-pong (returned to prev2)
                    c += lam1
            elif m != NONE and n == NONE:             # (re)acquire
                c += lam3
            rest, _ = rec(s + 1, m, n)
            tot = c + rest
            if tot < best:
                best, best_a = tot, lab
        memo[key] = (best, best_a)
        return best, best_a

    V = lambda s, n, p: rec(s, n, p)[0]
    expert = lambda s, n, p: rec(s, n, p)[1]
    # optimal serving sequence
    serv = np.full(S, NONE, dtype=int)
    n, p = NONE, NONE
    for s in range(S):
        lab = rec(s, n, p)[1]
        acts = dict(actions(s, n))
        m = acts.get(lab, NONE)
        serv[s] = m
        n, p = m, n
    return V, expert, serv

# ----------------------------- rollout -------------------------------------

def rollout(sc: Scenario, policy_fn):
    """Run a binary policy (features -> {0 STAY,1 HO}) -> serving sequence.
    Returns serv and the list of (features, expert-free) visited states."""
    S = sc.S
    serv = np.full(S, NONE, dtype=int)
    visited = []                                  # (s, n, p, feat, top)
    n, p = NONE, NONE
    for s in range(S):
        feat, top = features(sc, n, p, s)
        visited.append((s, n, p, feat, top))
        serv_visible = (n != NONE and sc.vis[n, s] == 1)
        if not serv_visible:                      # forced: must acquire top (or outage)
            m = top
        else:
            a = policy_fn(feat)                   # 0 STAY, 1 HO
            m = top if (a == 1 and top != NONE) else n
        serv[s] = m
        n, p = m, n
    return serv, visited

def expert_dataset(sc: Scenario, states, expert):
    """Label a set of visited states with the DP expert action."""
    X, y = [], []
    for (s, n, p, feat, top) in states:
        if n != NONE and sc.vis[n, s] == 1 and top != NONE:   # a real decision point
            X.append(feat)
            y.append(expert(s, n, p))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    from handover import build_scenario, evaluate
    sc = build_scenario(seed=0)
    V, expert, serv_star = solve_dp(sc)
    m = evaluate(serv_star, sc)
    print(f"binary DP optimum: J={m['J']:.1f} HO={m['handovers']} "
          f"ping-pong={m['pingpong']} outage={m['outage']}")
    # expert-trajectory dataset
    _, visited = rollout(sc, lambda f: expert_from_feat(f) if False else 0)
    X, y = expert_dataset(sc, visited, expert)
    print(f"decision points: {len(y)}, HO fraction: {y.mean():.2f}")
    # baselines in binary space
    from handover import greedy_max_sinr, greedy_max_elev
    for nm, fn in [("max-SINR", greedy_max_sinr), ("max-elev", greedy_max_elev)]:
        mm = evaluate(fn(sc), sc)
        print(f"  {nm}: J={mm['J']:.1f} ping-pong={mm['pingpong']} HO={mm['handovers']}")
