"""
Robustness sweep (G2): how does each policy behave as channel fading grows?
A KAN trained at the default fading is evaluated, together with the heuristics,
across fading levels on fresh test scenarios. Expectation: max-SINR ping-pong
explodes with fading (it chases every fluctuation) while the learned policy stays
stable. Saves results/robustness.csv + figures/fig_robust.pdf (house style).
"""
import csv, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from handover import build_scenario, evaluate, greedy_max_sinr, greedy_max_elev
from handover_binary import solve_dp, features, NONE
from policy_train import SplineKAN, Scaler, train_policy, eval_policy

R = lambda n: os.path.join(os.path.dirname(__file__), "..", "results", n)
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
# House plotting style is vendored in style/ (see make_figures_v2.py).
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "style"))
sys.path.insert(0, ASSETS); import agentra_plot as ap

FADES = [1.0, 2.0, 3.0, 4.0, 5.0]

def main():
    # train one KAN at default fading (2.5 dB)
    tr = [build_scenario(seed=s) for s in [0, 1, 2]]
    trdp = [solve_dp(sc) for sc in tr]
    allX = []
    for sc, (V, ex, ss) in zip(tr, trdp):
        n, p = NONE, NONE
        for s in range(sc.S):
            f, top = features(sc, n, p, s)
            if n != NONE and sc.vis[n, s] == 1 and top != NONE: allX.append(f)
            n, p = ss[s], n
    scaler = Scaler(np.array(allX, np.float32))
    torch.manual_seed(0); kan = SplineKAN()
    kpol = train_policy(kan, scaler, tr[0], trdp[0][1], trdp[0][2])

    rows = []
    for fd in FADES:
        te = [build_scenario(seed=10 + i, fading_db=fd) for i in range(3)]
        tedp = [solve_dp(sc) for sc in te]
        for name, fn in [("KAN", None), ("max-SINR", greedy_max_sinr), ("max-elev", greedy_max_elev)]:
            pps, gaps = [], []
            for ti in range(len(te)):
                Jopt = evaluate(tedp[ti][2], te[ti])["J"]
                if name == "KAN":
                    r = eval_policy(te[ti], kpol, tedp[ti][1], Jopt)
                    pps.append(r["pingpong"]); gaps.append(r["gap"])
                else:
                    m = evaluate(fn(te[ti]), te[ti])
                    pps.append(m["pingpong"]); gaps.append(100 * (m["J"] - Jopt) / abs(Jopt))
            rows.append(dict(fading=fd, method=name, pingpong=round(np.mean(pps), 2), gap=round(np.mean(gaps), 3)))
        print(f"fading {fd}dB: " + "  ".join(f"{r['method']} pp={r['pingpong']}" for r in rows[-3:]))
    with open(R("robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fading", "method", "pingpong", "gap"]); w.writeheader(); w.writerows(rows)

    # figure (house style)
    ap.use_style(); ap.register(["max-SINR", "max-elev", "KAN"], headline="KAN", rival="max-SINR",
                                store=R(".methods_rob.json"))
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    for m in ["max-SINR", "max-elev", "KAN"]:
        xs = FADES; ys = [next(r["pingpong"] for r in rows if r["fading"] == fd and r["method"] == m) for fd in FADES]
        s = ap.sty(m); ax.plot(xs, ys, label=m, **s)
    ax.set_xlabel("fading level [dB]"); ax.set_ylabel("ping-pong count")
    ax.legend(frameon=False, fontsize=8)
    ap.save(fig, os.path.join(FIG, "fig_robust"), audit=False); plt.close(fig)
    print("saved results/robustness.csv + figures/fig_robust")

if __name__ == "__main__":
    main()
