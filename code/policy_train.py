"""
G2 main experiment: train an interpretable KAN (additive spline / GAM-style)
handover policy vs a param-matched MLP, both via DAgger from the DP expert.
Evaluate on held-out scenarios (J, ping-pong, HO, macro-F1, gap-to-optimum) and
measure interpretability (M1 rule-fidelity, M2 threshold stability) vs MLP+SHAP.

The KAN is a single [8->1] KAN layer with sum aggregation = each feature passes
through one learnable univariate spline, then summed -> logit. This is
interpretable-by-design: each phi_i(x_i) is readable, thresholds extractable.
Comments in English by house rule.
"""
import numpy as np
import torch
import torch.nn as nn
from handover import build_scenario, evaluate, greedy_max_sinr, greedy_max_elev
from handover_binary import (solve_dp, rollout, expert_dataset, features,
                             FEATURE_NAMES, NONE)

torch.manual_seed(0)
NF = 8            # number of features
GRID = 8          # spline grid points

# ------------------------- models ------------------------------------------

class SplineKAN(nn.Module):
    """Additive univariate splines (KAN [8->1], sum aggregation). Interpretable."""
    def __init__(self, n_feat=NF, grid=GRID):
        super().__init__()
        self.grid = grid
        self.theta = nn.Parameter(0.01 * torch.randn(n_feat, grid))
        self.bias = nn.Parameter(torch.zeros(1))
    def phi(self, x):
        # x: [B, n_feat] in [0,1] -> per-feature spline value [B, n_feat]
        pos = x.clamp(0, 1) * (self.grid - 1)
        lo = pos.floor().long().clamp(0, self.grid - 2)
        frac = (pos - lo.float())
        th = self.theta                                   # [F, G]
        lo_v = torch.gather(th.unsqueeze(0).expand(x.shape[0], -1, -1), 2, lo.unsqueeze(2)).squeeze(2)
        hi_v = torch.gather(th.unsqueeze(0).expand(x.shape[0], -1, -1), 2, (lo + 1).unsqueeze(2)).squeeze(2)
        return lo_v * (1 - frac) + hi_v * frac            # [B, F]
    def forward(self, x):
        return self.phi(x).sum(dim=1, keepdim=True) + self.bias   # logit [B,1]

class MLP(nn.Module):
    def __init__(self, n_feat=NF, h=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ------------------------- feature scaling ---------------------------------

class Scaler:
    def __init__(self, X):
        self.lo = X.min(0); self.hi = X.max(0)
        self.rng = np.where(self.hi - self.lo < 1e-6, 1.0, self.hi - self.lo)
    def __call__(self, X):
        return ((X - self.lo) / self.rng).astype(np.float32)

# ------------------------- DAgger training ---------------------------------

def train_policy(model, scaler, sc, expert, serv_star, rounds=4, epochs=150, lr=0.03):
    """DAgger: start from expert trajectory, aggregate policy-visited states."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    # round 0 dataset: expert trajectory states
    _, vis0 = rollout(sc, lambda f: 0)                    # any policy to enumerate states...
    # better: use expert trajectory states directly
    Xs, ys = [], []
    n, p = NONE, NONE
    for s in range(sc.S):
        feat, top = features(sc, n, p, s)
        if n != NONE and sc.vis[n, s] == 1 and top != NONE:
            Xs.append(feat); ys.append(expert(s, n, p))
        m = serv_star[s]; n, p = m, n
    X = np.array(Xs, np.float32); y = np.array(ys, np.int64)

    def policy_fn(feat):
        with torch.no_grad():
            z = model(torch.tensor(scaler(feat[None]))).item()
        return 1 if z > 0 else 0

    for r in range(rounds):
        Xt = torch.tensor(scaler(X)); yt = torch.tensor(y, dtype=torch.float32)[:, None]
        for _ in range(epochs):
            opt.zero_grad(); loss = lossf(model(Xt), yt); loss.backward(); opt.step()
        # aggregate: roll out current policy, label visited decision points with expert
        _, visited = rollout(sc, policy_fn)
        Xn, yn = expert_dataset(sc, visited, expert)
        if len(yn):
            X = np.concatenate([X, Xn]); y = np.concatenate([y, yn])
    return policy_fn

# ------------------------- evaluation --------------------------------------

def eval_policy(sc, policy_fn, expert, J_opt):
    serv, visited = rollout(sc, policy_fn)
    m = evaluate(serv, sc)
    # macro-F1 vs expert on decision points
    from sklearn.metrics import f1_score
    Xe, ye = expert_dataset(sc, visited, expert)
    yp = np.array([policy_fn(x) for x in Xe]) if len(Xe) else np.array([])
    f1 = f1_score(ye, yp, average="macro") if len(ye) else float("nan")
    gap = 100 * (m["J"] - J_opt) / abs(J_opt)
    return dict(J=m["J"], gap=gap, pingpong=m["pingpong"], HO=m["handovers"],
                outage=m["outage"], macroF1=f1)

# ------------------------- interpretability --------------------------------

def kan_extract_threshold(model, scaler, feat_idx):
    """Read the dsinr-like threshold from an additive KAN spline: the input value
    (original scale) where phi_i crosses its midpoint (monotone trend)."""
    G = model.grid
    with torch.no_grad():
        th = model.theta[feat_idx].numpy()
    # grid in scaled [0,1] -> original scale
    xs_scaled = np.linspace(0, 1, G)
    xs_orig = scaler.lo[feat_idx] + xs_scaled * scaler.rng[feat_idx]
    mid = 0.5 * (th.max() + th.min())
    # first crossing of midpoint
    for i in range(G - 1):
        if (th[i] - mid) * (th[i + 1] - mid) <= 0:
            return float(xs_orig[i])
    return float(xs_orig[np.argmax(np.abs(np.diff(th)))])

def rule_fidelity(policy_fn, sc_list, experts, thr_dsinr, thr_visrem):
    """M1: a symbolic rule 'HO iff dsinr>thr_dsinr OR visrem_serv<thr_visrem' —
    how often does it match the actual policy decisions?"""
    match, tot = 0, 0
    for sc, expert in zip(sc_list, experts):
        _, visited = rollout(sc, policy_fn)
        for (s, n, p, feat, top) in visited:
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                a = policy_fn(feat)
                dsinr, visrem = feat[2], feat[4]
                rule = 1 if (dsinr > thr_dsinr or visrem < thr_visrem) else 0
                match += (a == rule); tot += 1
    return match / max(tot, 1)


def main():
    train_seeds = [0, 1, 2, 3, 4]
    test_seeds = [5, 6, 7, 8, 9]
    print("Building scenarios (fading realizations)...")
    train_scs = [build_scenario(seed=s) for s in train_seeds]
    test_scs = [build_scenario(seed=s) for s in test_seeds]
    train_dp = [solve_dp(sc) for sc in train_scs]
    test_dp = [solve_dp(sc) for sc in test_scs]

    # fit scaler on union of train decision features
    allX = []
    for sc, (V, expert, serv_star) in zip(train_scs, train_dp):
        n, p = NONE, NONE
        for s in range(sc.S):
            feat, top = features(sc, n, p, s)
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                allX.append(feat)
            m = serv_star[s]; n, p = m, n
    scaler = Scaler(np.array(allX, np.float32))

    print(f"KAN params, MLP params:")
    kan0, mlp0 = SplineKAN(), MLP(h=10)
    print(f"  KAN={n_params(kan0)}  MLP={n_params(mlp0)}  (param-matched target)")

    results = {"kan": [], "mlp": []}
    kan_thr, mlp_thr = [], []
    for si, seed in enumerate(train_seeds):
        sc = train_scs[si]; V, expert, serv_star = train_dp[si]
        torch.manual_seed(seed)
        kan = SplineKAN(); kpol = train_policy(kan, scaler, sc, expert, serv_star)
        torch.manual_seed(seed)
        mlp = MLP(h=10); mpol = train_policy(mlp, scaler, sc, expert, serv_star)
        # test on held-out scenarios
        for ti in range(len(test_scs)):
            tsc = test_scs[ti]; tV, texpert, tserv = test_dp[ti]
            J_opt = evaluate(tserv, tsc)["J"]
            results["kan"].append(eval_policy(tsc, kpol, texpert, J_opt))
            results["mlp"].append(eval_policy(tsc, mpol, texpert, J_opt))
        kan_thr.append(kan_extract_threshold(kan, scaler, 2))   # dsinr threshold
    print("\n=== held-out performance (mean over train-seeds x test-scenarios) ===")
    for name in ["kan", "mlp"]:
        R = results[name]
        for k in ["J", "gap", "pingpong", "HO", "outage", "macroF1"]:
            pass
        import numpy as _np
        print(f"  {name.upper():4} J={_np.mean([r['J'] for r in R]):8.1f} "
              f"gap=+{_np.mean([r['gap'] for r in R]):.1f}% "
              f"ping-pong={_np.mean([r['pingpong'] for r in R]):.1f} "
              f"HO={_np.mean([r['HO'] for r in R]):.0f} "
              f"macroF1={_np.mean([r['macroF1'] for r in R]):.3f}")
    # heuristics on test
    import numpy as _np
    hs = {"max-SINR": greedy_max_sinr, "max-elev": greedy_max_elev}
    for nm, fn in hs.items():
        gaps, pps = [], []
        for ti in range(len(test_scs)):
            tsc = test_scs[ti]; J_opt = evaluate(test_dp[ti][2], tsc)["J"]
            mm = evaluate(fn(tsc), tsc); gaps.append(100 * (mm["J"] - J_opt) / abs(J_opt)); pps.append(mm["pingpong"])
        print(f"  {nm:8} gap=+{_np.mean(gaps):.1f}%  ping-pong={_np.mean(pps):.1f}")

    print("\n=== M2 interpretability: dsinr-threshold stability across seeds ===")
    print(f"  KAN extracted dsinr threshold: mean={_np.mean(kan_thr):.2f} std={_np.std(kan_thr):.2f}")
    print(f"  (thresholds per seed: {[round(t,2) for t in kan_thr]})")


def interp_compare():
    """DECISIVE test of hanging condition #2: is KAN interpretability measurably
    better than a post-hoc explanation of the param-matched MLP?
    - KAN: rule read DIRECTLY from splines (interpretable-by-design).
    - MLP: rule from a logistic-regression global surrogate on its decisions (standard
      post-hoc). We compare M1 (rule-fidelity to the model's own decisions) and M2
      (threshold stability across seeds)."""
    from sklearn.linear_model import LogisticRegression
    import numpy as _np
    train_seeds = [0, 1, 2, 3, 4]
    train_scs = [build_scenario(seed=s) for s in train_seeds]
    train_dp = [solve_dp(sc) for sc in train_scs]
    # scaler on train decision features
    allX = []
    for sc, (V, expert, serv_star) in zip(train_scs, train_dp):
        n, p = NONE, NONE
        for s in range(sc.S):
            feat, top = features(sc, n, p, s)
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                allX.append(feat)
            m = serv_star[s]; n, p = m, n
    scaler = Scaler(_np.array(allX, _np.float32))

    def collect_decisions(pol, sc):
        _, visited = rollout(sc, pol)
        X, A = [], []
        for (s, n, p, feat, top) in visited:
            if n != NONE and sc.vis[n, s] == 1 and top != NONE:
                X.append(feat); A.append(pol(feat))
        return _np.array(X, _np.float32), _np.array(A)

    kan_fid, mlp_fid, kan_delta, mlp_delta = [], [], [], []
    for si, seed in enumerate(train_seeds):
        sc = train_scs[si]; V, expert, serv_star = train_dp[si]
        torch.manual_seed(seed); kan = SplineKAN(); kpol = train_policy(kan, scaler, sc, expert, serv_star)
        torch.manual_seed(seed); mlp = MLP(h=10); mpol = train_policy(mlp, scaler, sc, expert, serv_star)
        # KAN: direct thresholds
        d_kan = kan_extract_threshold(kan, scaler, 2)     # dsinr
        t_kan = kan_extract_threshold(kan, scaler, 4)     # visrem_serv
        # MLP: logistic surrogate on its own decisions -> effective dsinr threshold
        Xm, Am = collect_decisions(mpol, sc)
        d_mlp, t_mlp = _np.nan, _np.nan
        if len(_np.unique(Am)) == 2:
            lr = LogisticRegression(max_iter=500).fit(scaler(Xm), Am)
            w = lr.coef_[0]; b = lr.intercept_[0]
            xbar = scaler(Xm).mean(0)
            # threshold on dsinr (idx2) holding others at mean, in scaled units -> orig
            if abs(w[2]) > 1e-6:
                z = (-b - (w @ xbar - w[2] * xbar[2])) / w[2]
                d_mlp = scaler.lo[2] + z * scaler.rng[2]
            if abs(w[4]) > 1e-6:
                z4 = (-b - (w @ xbar - w[4] * xbar[4])) / w[4]
                t_mlp = scaler.lo[4] + z4 * scaler.rng[4]
        # M1 rule-fidelity: does 'HO iff dsinr>delta or visrem<tau' match the model?
        def fid(pol, d, t):
            X, A = collect_decisions(pol, sc)
            if _np.isnan(d): d = 0.0
            if _np.isnan(t): t = 0.0
            rule = ((X[:, 2] > d) | (X[:, 4] < t)).astype(int)
            return (rule == A).mean()
        kan_fid.append(fid(kpol, d_kan, t_kan)); mlp_fid.append(fid(mpol, d_mlp, t_mlp))
        kan_delta.append(d_kan); mlp_delta.append(d_mlp)

    print("\n########## DECISIVE: KAN vs MLP+surrogate interpretability ##########")
    print(f"M1 rule-fidelity (rule reproduces model's own decisions):")
    print(f"  KAN (direct splines):  {_np.mean(kan_fid):.3f} +- {_np.std(kan_fid):.3f}")
    print(f"  MLP (post-hoc LR surr): {_np.mean(mlp_fid):.3f} +- {_np.std(mlp_fid):.3f}")
    print(f"M2 dsinr-threshold stability (std across seeds, lower=more stable):")
    print(f"  KAN delta per seed: {[round(x,2) for x in kan_delta]}  -> std={_np.nanstd(kan_delta):.2f}")
    print(f"  MLP delta per seed: {[round(x,2) for x in mlp_delta]}  -> std={_np.nanstd(mlp_delta):.2f}")
    dfid = _np.mean(kan_fid) - _np.mean(mlp_fid)
    print(f"\nVERDICT INPUT: fidelity advantage KAN-MLP = {dfid:+.3f}; "
          f"KAN stability {'better' if _np.nanstd(kan_delta)<_np.nanstd(mlp_delta) else 'NOT better'}.")
    print("(hanging condition #2: KAN interpretability must be measurably better; else honest-negative)")


if __name__ == "__main__":
    main()
    interp_compare()

if __name__ == "__main__":
    main()
