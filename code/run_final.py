"""
Final consolidated experiment for the paper. Fixed seeds -> fully reproducible.
Trains KAN + MLP once per seed, evaluates on held-out scenarios, extracts
interpretability, and SAVES all numbers to CSV so figures and the paper cite
exact traceable values (G4 data-producer discipline: numbers come from real
model rollouts, not hand-written).

Writes:
  results/perf.csv         per (method, train_seed, test_scenario) J/gap/pp/HO/F1
  results/perf_summary.csv  aggregated mean/std per method
  results/interp.csv        per train_seed: KAN/MLP thresholds + M1 fidelity
  results/kan_splines.csv   KAN phi_i(x_i) curves for the interpretability figure
"""
import csv, os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from handover import build_scenario, evaluate, greedy_max_sinr, greedy_max_elev
from handover_binary import solve_dp, rollout, expert_dataset, features, FEATURE_NAMES, NONE
from policy_train import (SplineKAN, MLP, Scaler, train_policy, eval_policy,
                          kan_extract_threshold, n_params, GRID)

R = lambda n: os.path.join(os.path.dirname(__file__), "..", "results", n)
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS = [5, 6, 7, 8, 9]

def collect_decisions(pol, sc):
    _, visited = rollout(sc, pol)
    X, A = [], []
    for (s, n, p, feat, top) in visited:
        if n != NONE and sc.vis[n, s] == 1 and top != NONE:
            X.append(feat); A.append(pol(feat))
    return np.array(X, np.float32), np.array(A)

def main():
    print("Building scenarios + DP ground truth...")
    train_scs = [build_scenario(seed=s) for s in TRAIN_SEEDS]
    test_scs = [build_scenario(seed=s) for s in TEST_SEEDS]
    train_dp = [solve_dp(sc) for sc in train_scs]
    test_dp = [solve_dp(sc) for sc in test_scs]
    test_Jopt = [evaluate(test_dp[i][2], test_scs[i])["J"] for i in range(len(test_scs))]

    # scaler on train decision features
    allX = []
    for sc, (V, expert, serv_star) in zip(train_scs, train_dp):
        n, p = NONE, NONE
        for s in range(sc.S):
            feat, top = features(sc, n, p, s)
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                allX.append(feat)
            m = serv_star[s]; n, p = m, n
    scaler = Scaler(np.array(allX, np.float32))
    kan_np, mlp_np = n_params(SplineKAN()), n_params(MLP(h=10))

    perf_rows, interp_rows, spline_rows = [], [], []
    for si, seed in enumerate(TRAIN_SEEDS):
        sc = train_scs[si]; V, expert, serv_star = train_dp[si]
        torch.manual_seed(seed); kan = SplineKAN(); kpol = train_policy(kan, scaler, sc, expert, serv_star)
        torch.manual_seed(seed); mlp = MLP(h=10); mpol = train_policy(mlp, scaler, sc, expert, serv_star)
        # performance on each held-out test scenario
        for ti in range(len(test_scs)):
            for name, pol in [("KAN", kpol), ("MLP", mpol)]:
                r = eval_policy(test_scs[ti], pol, test_dp[ti][1], test_Jopt[ti])
                perf_rows.append(dict(method=name, train_seed=seed, test_scen=TEST_SEEDS[ti],
                                      J=round(r["J"], 2), gap=round(r["gap"], 3),
                                      pingpong=r["pingpong"], HO=r["HO"], macroF1=round(r["macroF1"], 4)))
        # interpretability: KAN direct thresholds + MLP logistic-surrogate thresholds
        d_kan = kan_extract_threshold(kan, scaler, 2); t_kan = kan_extract_threshold(kan, scaler, 4)
        Xm, Am = collect_decisions(mpol, sc); d_mlp = np.nan
        if len(np.unique(Am)) == 2:
            lr = LogisticRegression(max_iter=500).fit(scaler(Xm), Am)
            w = lr.coef_[0]; b = lr.intercept_[0]; xbar = scaler(Xm).mean(0)
            if abs(w[2]) > 1e-6:
                z = (-b - (w @ xbar - w[2] * xbar[2])) / w[2]; d_mlp = scaler.lo[2] + z * scaler.rng[2]
        # M1 rule-fidelity (2-var rule vs model's own decisions)
        def fid(pol, d, t):
            X, A = collect_decisions(pol, sc)
            d = 0.0 if np.isnan(d) else d; t = 0.0 if np.isnan(t) else t
            rule = ((X[:, 2] > d) | (X[:, 4] < t)).astype(int)
            return round(float((rule == A).mean()), 4)
        interp_rows.append(dict(train_seed=seed, kan_dsinr_thr=round(d_kan, 3),
                                mlp_dsinr_thr=round(float(d_mlp), 3) if not np.isnan(d_mlp) else "nan",
                                kan_M1=fid(kpol, d_kan, t_kan), mlp_M1=fid(mpol, d_mlp, t_kan)))
        # save KAN spline curves for seed 0 (interpretability figure)
        if seed == 0:
            with torch.no_grad():
                for fi in range(8):
                    th = kan.theta[fi].numpy()
                    xs = scaler.lo[fi] + np.linspace(0, 1, GRID) * scaler.rng[fi]
                    for gi in range(GRID):
                        spline_rows.append(dict(feature=FEATURE_NAMES[fi], x=round(float(xs[gi]), 4),
                                                phi=round(float(th[gi]), 5)))

    # heuristics on test
    for nm, fn in [("max-SINR", greedy_max_sinr), ("max-elev", greedy_max_elev)]:
        for ti in range(len(test_scs)):
            mm = evaluate(fn(test_scs[ti]), test_scs[ti])
            gap = 100 * (mm["J"] - test_Jopt[ti]) / abs(test_Jopt[ti])
            perf_rows.append(dict(method=nm, train_seed=-1, test_scen=TEST_SEEDS[ti],
                                  J=round(mm["J"], 2), gap=round(gap, 3), pingpong=mm["pingpong"],
                                  HO=mm["handovers"], macroF1=""))

    # write CSVs
    def wr(path, rows, fields):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    wr(R("perf.csv"), perf_rows, ["method", "train_seed", "test_scen", "J", "gap", "pingpong", "HO", "macroF1"])
    wr(R("interp.csv"), interp_rows, ["train_seed", "kan_dsinr_thr", "mlp_dsinr_thr", "kan_M1", "mlp_M1"])
    wr(R("kan_splines.csv"), spline_rows, ["feature", "x", "phi"])

    # aggregated summary
    methods = ["KAN", "MLP", "max-SINR", "max-elev"]
    summ = []
    for m in methods:
        rs = [r for r in perf_rows if r["method"] == m]
        gap = [r["gap"] for r in rs]; pp = [r["pingpong"] for r in rs]
        f1 = [r["macroF1"] for r in rs if r["macroF1"] != ""]
        summ.append(dict(method=m, gap_mean=round(np.mean(gap), 3), gap_std=round(np.std(gap), 3),
                         pingpong_mean=round(np.mean(pp), 2), macroF1_mean=round(np.mean(f1), 4) if f1 else "",
                         params={"KAN": kan_np, "MLP": mlp_np}.get(m, "")))
    wr(R("perf_summary.csv"), summ, ["method", "gap_mean", "gap_std", "pingpong_mean", "macroF1_mean", "params"])

    print("\n=== perf_summary.csv ===")
    for s in summ:
        print(f"  {s['method']:9} gap +{s['gap_mean']}% (std {s['gap_std']})  "
              f"ping-pong {s['pingpong_mean']}  F1 {s['macroF1_mean']}  params {s['params']}")
    print("\n=== interp.csv (per seed) ===")
    kanM1 = np.mean([r["kan_M1"] for r in interp_rows]); mlpM1 = np.mean([r["mlp_M1"] for r in interp_rows])
    kanT = [r["kan_dsinr_thr"] for r in interp_rows]
    mlpT = [float(r["mlp_dsinr_thr"]) for r in interp_rows if r["mlp_dsinr_thr"] != "nan"]
    print(f"  M1 fidelity: KAN {kanM1:.3f}  MLP {mlpM1:.3f}")
    print(f"  dsinr threshold std: KAN {np.std(kanT):.2f}  MLP {np.std(mlpT):.2f}")
    print("\nCSVs written to results/. All numbers traceable to model rollouts.")

if __name__ == "__main__":
    main()
