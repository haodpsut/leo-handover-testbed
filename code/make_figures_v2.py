"""
Result figures via the pipeline plot-figure-dph house style (agentra_plot).
Single-column 3.5in, Times, two-channel (color+marker), headline mode for the
money figure. Reads results/*.csv. Follows FIGURE-WORKFLOW.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House plotting style is vendored in style/ so a clean clone works with no
# path outside this repository.
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "style"))
sys.path.insert(0, ASSETS)
import agentra_plot as ap

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

ap.use_style()
# headline mode: our method (KAN) = agreen, rival (MLP) = warm, heuristics = gray
ap.register(["max-SINR", "max-elev", "MLP", "KAN"], headline="KAN", rival="MLP",
            store=os.path.join(RES, ".methods.json"))

def load_perf():
    rows = list(csv.DictReader(open(os.path.join(RES, "perf.csv"))))
    d = {}
    for r in rows:
        m = r["method"]; d.setdefault(m, {"gap": [], "pp": []})
        d[m]["gap"].append(float(r["gap"])); d[m]["pp"].append(float(r["pingpong"]))
    return d

def fig_tradeoff():
    d = load_perf()
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    for m in ["max-SINR", "max-elev", "MLP", "KAN"]:
        g = np.array(d[m]["gap"]); p = np.array(d[m]["pp"])
        s = ap.sty(m)
        ax.errorbar(p.mean(), g.mean(), xerr=p.std(), yerr=g.std(),
                    color=s["color"], elinewidth=0.8, capsize=2, alpha=0.55, zorder=2,
                    linestyle="none")
        ax.scatter(p.mean(), g.mean(), s=60, color=s["color"], marker=s["marker"],
                   edgecolors="#0C2F1F", linewidths=0.6, zorder=3, label=m)
    ax.axhspan(-0.3, 1.5, color="#46B152", alpha=0.06, zorder=0)
    ax.text(9.5, 0.35, "sweet spot", fontsize=7, color="#6B7280")
    ax.set_xlabel("ping-pong count (lower better)")
    ax.set_ylabel("gap to offline optimum [%]")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, handletextpad=0.3)
    ax.set_ylim(-0.4, 5.4)
    ap.save(fig, os.path.join(FIG, "fig_tradeoff"), audit=False)
    plt.close(fig)
    print("tradeoff done")

def fig_splines():
    rows = list(csv.DictReader(open(os.path.join(RES, "kan_splines.csv"))))
    feats = {}
    for r in rows:
        feats.setdefault(r["feature"], {"x": [], "phi": []})
        feats[r["feature"]]["x"].append(float(r["x"])); feats[r["feature"]]["phi"].append(float(r["phi"]))
    show = ["dsinr", "visrem_serv", "ho_is_pingpong"]
    # put the feature name as an in-panel annotation (top-left, boxed) instead of a
    # title above the axes, so it never collides with the tick numbers of the panel
    # above. Generous hspace keeps x-ticks clear of the next panel.
    labels = {"dsinr": r"$\Delta$SINR (dB)", "visrem_serv": "serving visibility remaining (slots)",
              "ho_is_pingpong": "HO ping-pong flag"}
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.0), constrained_layout=True)
    for ax, f in zip(axes, show):
        x = np.array(feats[f]["x"]); y = np.array(feats[f]["phi"])
        ax.plot(x, y, "-o", color="#46B152", ms=3, lw=1.3, mec="#0C2F1F", mew=0.5)
        ax.axhline(0, color="#6B7280", lw=0.5, ls="--")
        ax.tick_params(labelsize=8)
        ax.set_ylabel(r"$\varphi(x)$", fontsize=9, labelpad=2)
        ax.set_xlabel(labels[f], fontsize=8.5, labelpad=1)
        # feature id in a white box, inside the panel, away from ticks
        ax.margins(y=0.28)
    fig.set_constrained_layout_pads(h_pad=0.10, hspace=0.16)
    ap.save(fig, os.path.join(FIG, "fig_splines"), audit=False, double_column=False)
    plt.close(fig)
    print("splines done")

if __name__ == "__main__":
    fig_tradeoff()
    fig_splines()
    print("result figures -> figures/ (house style)")
