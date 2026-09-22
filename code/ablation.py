"""
Ablation for the paper (G2): isolate the two design choices.
  (a) DAgger vs plain behavioural cloning (BC, no dataset aggregation).
  (b) KAN with vs without the ping-pong feature.
Reuses the same scenarios / DP expert / KAN policy. Saves results/ablation.csv.
"""
import csv, os
import numpy as np
import torch
from handover import build_scenario, evaluate
from handover_binary import solve_dp, features, NONE
from policy_train import SplineKAN, Scaler, train_policy, eval_policy

R = lambda n: os.path.join(os.path.dirname(__file__), "..", "results", n)
TRAIN = [0, 1, 2]; TEST = [5, 6, 7]

class MaskScaler(Scaler):
    def __init__(self, base, mask_idx):
        self.lo = base.lo; self.hi = base.hi; self.rng = base.rng; self.mask = mask_idx
    def __call__(self, X):
        Y = ((X - self.lo) / self.rng).astype(np.float32); Y[..., self.mask] = 0.0
        return Y

def main():
    tr = [build_scenario(seed=s) for s in TRAIN]
    te = [build_scenario(seed=s) for s in TEST]
    trdp = [solve_dp(sc) for sc in tr]; tedp = [solve_dp(sc) for sc in te]
    teJ = [evaluate(tedp[i][2], te[i])["J"] for i in range(len(te))]
    allX = []
    for sc, (V, ex, ss) in zip(tr, trdp):
        n, p = NONE, NONE
        for s in range(sc.S):
            f, top = features(sc, n, p, s)
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                allX.append(f)
            n, p = ss[s], n
    scaler = Scaler(np.array(allX, np.float32))
    mscaler = MaskScaler(scaler, 7)              # zero the ho_is_pingpong feature

    variants = {
        "KAN + DAgger (full)": dict(scaler=scaler, rounds=4),
        "KAN, BC only (no DAgger)": dict(scaler=scaler, rounds=1),
        "KAN, no ping-pong feat.": dict(scaler=mscaler, rounds=4),
    }
    rows = []
    for name, cfg in variants.items():
        agg = {"gap": [], "pp": [], "f1": []}
        for si, seed in enumerate(TRAIN):
            sc = tr[si]; V, ex, ss = trdp[si]
            torch.manual_seed(seed)
            kan = SplineKAN()
            pol = train_policy(kan, cfg["scaler"], sc, ex, ss, rounds=cfg["rounds"])
            for ti in range(len(te)):
                r = eval_policy(te[ti], pol, tedp[ti][1], teJ[ti])
                agg["gap"].append(r["gap"]); agg["pp"].append(r["pingpong"]); agg["f1"].append(r["macroF1"])
        rows.append(dict(variant=name, gap=round(np.mean(agg["gap"]), 2),
                         pingpong=round(np.mean(agg["pp"]), 1), macroF1=round(np.mean(agg["f1"]), 3)))
        print(f"  {name:28} gap +{np.mean(agg['gap']):.2f}%  ping-pong {np.mean(agg['pp']):.1f}  F1 {np.mean(agg['f1']):.3f}")
    with open(R("ablation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "gap", "pingpong", "macroF1"]); w.writeheader(); w.writerows(rows)
    print("saved results/ablation.csv")

if __name__ == "__main__":
    main()
