"""
G1 smoke test: real-orbit handover scenario. Verifies (i) DP optimum is feasible
and DP<->J consistent (the load-bearing G0 fix, machine-checked), (ii) scenario is
non-degenerate (handovers genuinely needed: sats rise/set, multiple visible), (iii)
readable heuristics are feasible but suboptimal (ping-pong), leaving room for a
learned interpretable policy.
"""
import numpy as np
from handover import (build_scenario, dp_optimum, evaluate, check_feasible,
                      check_dp_consistency, greedy_max_sinr, greedy_max_elev)

def main():
    print("=== real-orbit handover scenario (dense Walker 18x18=324, 1 user, 120 slots x 15s) ===")
    sc = build_scenario(seed=0)
    vis_per_slot = sc.vis.sum(axis=0)
    print(f"N={sc.N} sats, S={sc.S} slots (1h horizon)")
    print(f"slots with >=1 visible sat: {int((vis_per_slot>=1).sum())}/{sc.S}")
    print(f"slots with >=2 visible (choice exists): {int((vis_per_slot>=2).sum())}/{sc.S}")
    print(f"avg visible sats/slot: {vis_per_slot.mean():.1f}")

    if (vis_per_slot >= 2).sum() == 0:
        print("BLOCK: never >1 visible -> no handover decision to make (degenerate).")
        return

    print("\n=== DP<->J consistency (machine-check of G0 hard-defect #1 fix) ===")
    match, J_opt, J_eval, serv_star = check_dp_consistency(sc)
    print(f"DP J_opt={J_opt:.3f}  independent evaluate J={J_eval:.3f}  MATCH={match}")

    print("\n=== feasibility of DP optimum ===")
    feas, _ = check_feasible(serv_star, sc, verbose=True)
    m_star = evaluate(serv_star, sc)
    print(f"optimum: J={m_star['J']:.1f}  HO={m_star['handovers']}  "
          f"ping-pong={m_star['pingpong']}  outage={m_star['outage']}")

    print("\n=== baselines (readable heuristics) ===")
    for name, fn in [("max-SINR", greedy_max_sinr), ("max-elev", greedy_max_elev)]:
        serv = fn(sc)
        fe, _ = check_feasible(serv, sc, verbose=False)
        m = evaluate(serv, sc)
        gap = 100 * (m['J'] - J_opt) / abs(J_opt)
        print(f"  {name:9} J={m['J']:8.1f} (gap +{gap:.1f}% vs opt)  HO={m['handovers']:3d}  "
              f"ping-pong={m['pingpong']:2d}  outage={m['outage']:2d}  feasible={fe}")

    # handover-count of optimum should be >0 (policy must act) and heuristics should
    # ping-pong more than the optimum (room for a smarter readable policy)
    pp_heur = evaluate(greedy_max_sinr(sc), sc)['pingpong']
    print("\n=== G1 SMOKE VERDICT ===")
    checks = {
        "DP<->J consistent (ground-truth valid)": match,
        "DP optimum feasible": feas,
        "scenario non-degenerate (choice slots>0)": (vis_per_slot >= 2).sum() > 0,
        "optimum actually hands over (HO>0)": m_star['handovers'] > 0,
        "heuristic ping-pongs more than optimum (room to learn)": pp_heur > m_star['pingpong'],
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if all(checks.values()):
        print("\nPASS: valid deterministic ground truth on real orbits, non-degenerate "
              "handover problem, readable heuristics leave a ping-pong gap. Ready for G2 "
              "(train interpretable KAN policy vs MLP, DAgger from DP optimum).")
    else:
        print("\nBLOCK: a required condition failed.")

if __name__ == "__main__":
    main()
