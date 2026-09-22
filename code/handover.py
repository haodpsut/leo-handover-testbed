"""
LEO handover scenario + offline DP optimum + independent feasibility/consistency
checker (G1 phase) for the interpretable-KAN-handover paper.

Implements formulation P1 v2:
  - Scenario from real orbital mechanics (Walker constellation via sgp4, one user).
  - evaluate(serv): multi-criteria cost J (throughput, ping-pong, outage, HO-freq),
    computed INDEPENDENTLY from a serving sequence.
  - dp_optimum(): offline optimum via DP with EXPANDED state (serv + W_pp window)
    so ping-pong (non-Markovian on serv alone) is handled exactly -> pi* IS the
    true optimum of J (fixes G0 hard-defect #1).
  - check_feasible(): independent hard-constraint check (c1/c2/c3) + the KEY
    machine check that DP's optimal value equals evaluate(serv*).J (DP<->J
    consistency, the load-bearing G0 fix), with a positive-control self-test.
  - baselines: max-SINR, max-elevation heuristics.

sinr is DETERMINISTIC (elevation-based) here so the offline DP is a valid
ground truth. Comments in English by house rule.
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache

# ----------------------------- constellation -------------------------------

def walker_sats(n_planes=6, sats_per_plane=6, inc_deg=53.0, alt_km=550.0,
                epoch=(2026, 7, 28, 12, 0)):
    """Reproducible Walker-delta constellation via sgp4init (no network)."""
    from sgp4.api import Satrec, WGS72
    from skyfield.api import load, EarthSatellite
    import datetime as dt
    ts = load.timescale()
    mu, Re = 398600.4418, 6378.137
    no_kozai = np.sqrt(mu / (Re + alt_km) ** 3) * 60.0
    y, mo, d, h, mi = epoch
    ep = (dt.datetime(y, mo, d, h, mi) - dt.datetime(1949, 12, 31)).total_seconds() / 86400.0
    T = n_planes * sats_per_plane
    sats, k = [], 1
    for p in range(n_planes):
        raan = 360.0 * p / n_planes
        for q in range(sats_per_plane):
            ma = (360.0 * q / sats_per_plane + 360.0 * p / T) % 360
            s = Satrec()
            s.sgp4init(WGS72, 'i', k, ep, 0.0, 0.0, 0.0, 1e-6,
                       np.radians(0.0), np.radians(inc_deg),
                       np.radians(ma), no_kozai, np.radians(raan))
            es = EarthSatellite.from_satrec(s, ts)
            es.name = f"S{p:02d}-{q:02d}"
            sats.append(es)
            k += 1
    return sats


class Scenario:
    def __init__(self, el, vis, sinr, params):
        self.el = el              # [N,S] elevation deg
        self.vis = vis            # [N,S] {0,1} pure geometric
        self.sinr = sinr          # [N,S] deterministic
        self.N, self.S = el.shape
        self.p = params           # dict: lam1,lam2,lam3,W_pp,sinr_th,el_min
        # precompute per-slot visible satellite lists
        self.visible = [np.where(vis[:, s] == 1)[0] for s in range(self.S)]


def build_scenario(seed=0, n_planes=18, sats_per_plane=18, S=120, slot_seconds=15,
                   el_min=5.0, station=(45.0, 0.0), sinr_th=8.0, fading_db=2.5,
                   lam1=2.0, lam2=5.0, lam3=1.0, W_pp=2):
    """One ground user, dense Walker constellation. Elevation-based SINR PLUS
    seeded Rician-like fading -> max-SINR jitters and ping-pongs, motivating a
    learned hysteresis policy. Fading is deterministic-per-seed, so the DP optimum
    is a valid ground truth for that realization."""
    from skyfield.api import load, wgs84
    ts = load.timescale()
    rng = np.random.default_rng(seed)
    sats = walker_sats(n_planes, sats_per_plane)
    N = len(sats)
    gs = wgs84.latlon(station[0], station[1])
    secs = np.arange(S) * slot_seconds + slot_seconds / 2.0
    times = ts.utc(2026, 7, 28, 12, 0, secs)
    el = np.zeros((N, S))
    for n, sat in enumerate(sats):
        el[n, :] = (sat - gs).at(times).altaz()[0].degrees
    vis = (el >= el_min).astype(np.int8)
    base = 5.0 + 20.0 * np.sin(np.radians(np.clip(el, 0, 90)))   # elevation-driven mean
    fade = fading_db * rng.standard_normal((N, S))               # seeded fading
    sinr = np.where(vis == 1, np.maximum(0.0, base + fade), 0.0)
    params = dict(lam1=lam1, lam2=lam2, lam3=lam3, W_pp=W_pp, sinr_th=sinr_th, el_min=el_min)
    return Scenario(el, vis, sinr, params)


# ----------------------------- evaluation ----------------------------------

def evaluate(serv, sc: Scenario):
    """Independent cost of a serving sequence serv (len S, sat idx or -1 for none).
    J = sum_s [ -sinr + lam2*out + lam1*pp + lam3*1{HO} ]. Returns dict."""
    p = sc.p
    S, W = sc.S, p["W_pp"]
    thr = 0.0; out = 0; pp = 0; ho = 0
    for s in range(S):
        n = serv[s]
        if n == -1 or sc.sinr[n, s] < p["sinr_th"]:
            out += 1
        if n != -1:
            thr += sc.sinr[n, s]
        if s > 0 and serv[s] != serv[s - 1]:          # handover happened
            ho += 1
            k = serv[s]
            # ping-pong: HO into k that was served within the last W_pp slots
            window = serv[max(0, s - W):s]            # serv[s-W .. s-1]
            if k != -1 and k in window:
                pp += 1
    J = -thr + p["lam2"] * out + p["lam1"] * pp + p["lam3"] * ho
    return dict(J=J, throughput=thr, outage=out, pingpong=pp, handovers=ho)


# ----------------------------- offline DP optimum --------------------------

def dp_optimum(sc: Scenario):
    """Offline optimum of J via DP over EXPANDED state (window of last W_pp serving
    sats). Returns (J_opt, serv_star). This is the ground-truth upper bound."""
    p = sc.p
    S, W = sc.S, p["W_pp"]
    lam1, lam2, lam3, sinr_th = p["lam1"], p["lam2"], p["lam3"], p["sinr_th"]
    sinr, visible = sc.sinr, sc.visible
    NONE = -1

    def slot_cost(n, s):
        c = 0.0
        if n == NONE or sinr[n, s] < sinr_th:
            c += lam2
        if n != NONE:
            c += -sinr[n, s]
        return c

    import sys
    sys.setrecursionlimit(10000)
    memo = {}

    def rec(s, win):
        # win: tuple length W of serving sats at slots s-1..s-W (NONE padded)
        if s == S:
            return 0.0, []
        key = (s, win)
        if key in memo:
            return memo[key]
        best, best_seq = float("inf"), None
        choices = list(visible[s])
        choices.append(NONE)                          # allow outage (no serving)
        for m in choices:
            c = slot_cost(m, s)
            if s > 0:
                prev = win[0]
                if m != prev:                         # handover (or acquire from NONE)
                    if prev != NONE:                  # a real HO costs lam3 + maybe pp
                        c += lam3
                        if m != NONE and m in win:    # ping-pong
                            c += lam1
                    else:                             # re-acquire after outage: HO cost
                        if m != NONE:
                            c += lam3
            new_win = (m,) + win[:W - 1]
            rest, seq = rec(s + 1, new_win)
            tot = c + rest
            if tot < best:
                best, best_seq = tot, [m] + seq
        memo[key] = (best, best_seq)
        return best, best_seq

    win0 = (NONE,) * W
    J_opt, serv_star = rec(0, win0)
    return J_opt, np.array(serv_star, dtype=int)


# ----------------------------- baselines -----------------------------------

def greedy_max_sinr(sc: Scenario):
    """Readable-but-suboptimal heuristic: each slot serve the visible sat with
    highest SINR (causes ping-pong)."""
    serv = np.full(sc.S, -1, dtype=int)
    for s in range(sc.S):
        vis = sc.visible[s]
        if len(vis):
            serv[s] = vis[np.argmax(sc.sinr[vis, s])]
    return serv


def greedy_max_elev(sc: Scenario):
    serv = np.full(sc.S, -1, dtype=int)
    for s in range(sc.S):
        vis = sc.visible[s]
        if len(vis):
            serv[s] = vis[np.argmax(sc.el[vis, s])]
    return serv


# ----------------------------- feasibility ---------------------------------

def check_feasible(serv, sc: Scenario, verbose=True):
    """Independent hard-constraint check (c1/c2/c3). Returns (feasible, rows).
    c1: every served sat is visible. c2/c3 reduce to c1 given serv is a scalar
    sequence (STAY keeps serv, HO target must be visible)."""
    rows = []
    bad_c1 = 0
    for s in range(sc.S):
        n = serv[s]
        if n != -1 and sc.vis[n, s] != 1:
            bad_c1 += 1
    rows.append(("c1_serve_visible", bad_c1 == 0, bad_c1, 0, "#slots serving a non-visible sat"))
    feasible = all(ok for _, ok, *_ in rows)
    if verbose:
        print(f"{'id':20} {'ok':4} {'value':>6} {'thr':>4}  desc")
        print("-" * 60)
        for cid, ok, val, thr, desc in rows:
            print(f"{cid:20} {'YES' if ok else 'NO':4} {val:>6} {thr:>4}  {desc}"
                  f"{'' if ok else '  <-- VIOLATED'}")
        print(f"VERDICT: {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    return feasible, rows


def check_dp_consistency(sc: Scenario):
    """KEY machine check (G0 hard-defect #1 fix): DP optimal value must equal the
    independently-evaluated J of the returned sequence."""
    J_opt, serv_star = dp_optimum(sc)
    J_eval = evaluate(serv_star, sc)["J"]
    match = abs(J_opt - J_eval) < 1e-6
    return match, J_opt, J_eval, serv_star


# ----------------------------- self-test -----------------------------------

def _self_test():
    """Positive control: (1) an infeasible sequence must be caught; (2) DP<->J
    consistency must hold on a tiny scenario."""
    # tiny synthetic scenario
    N, S = 4, 8
    rng = np.random.default_rng(0)
    vis = (rng.random((N, S)) > 0.4).astype(np.int8)
    vis[0, :] = 1                                   # ensure sat 0 always visible
    el = vis * rng.uniform(10, 80, size=(N, S))
    sinr = np.where(vis == 1, 5 + 20 * np.sin(np.radians(np.clip(el, 0, 90))), 0.0)
    sc = Scenario(el, vis, sinr,
                  dict(lam1=2.0, lam2=5.0, lam3=1.0, W_pp=2, sinr_th=3.0, el_min=10.0))

    # (1) feasibility positive control: force serving a non-visible sat
    bad = np.zeros(S, dtype=int)
    for s in range(S):
        nonvis = np.where(sc.vis[:, s] == 0)[0]
        bad[s] = nonvis[0] if len(nonvis) else 0
    feas, rows = check_feasible(bad, sc, verbose=True)
    assert not feas, "positive control FAILED: infeasible sequence not caught"
    print("[self-test] infeasible sequence caught -> checker alive.\n")

    # (2) DP<->J consistency
    match, J_opt, J_eval, serv_star = check_dp_consistency(sc)
    print(f"[self-test] DP J_opt={J_opt:.4f}  evaluate J={J_eval:.4f}  match={match}")
    assert match, "DP<->J INCONSISTENT: DP does not optimize J (hard-defect #1 not fixed)"
    feas_star, _ = check_feasible(serv_star, sc, verbose=False)
    assert feas_star, "DP optimum is infeasible!"
    print("[self-test] PASS: DP optimizes J AND optimum is feasible.")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
