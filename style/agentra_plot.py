"""agentra_plot — house style for IEEE-Transactions result figures (single column).

Two rules this module exists to enforce, because both are easy to violate by hand:

1. **One identity per method, held across every figure in the paper.** Register the
   methods once; every figure then asks for the same method by name and gets the same
   colour, marker and dash. A reader learns the legend once.
2. **Every series is encoded twice: hue AND (marker + dash).** Hue alone dies in
   grayscale print — the same failure that once erased the role coding of our flow
   figures. Never style a series by colour only.

Usage
-----
    from agentra_plot import use_style, register, sty, save, headline
    use_style()
    register(["MOEA/D", "NSGA-II", "AGEA", "Ours"])       # once, order = draw order
    ax.plot(x, y, **sty("Ours"), label="Ours")
    save(fig, "out/fig-convergence")   # pdf+png+gray, then audits; RAISES if the gate fails
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, "agentra.mplstyle")

# ---------------------------------------------------------------- palette ----
# Option B: five identities, all drawn from the Agentra brand. Ordered so the most
# distinct pairs come first — a 2-series plot gets forest vs agreen, not the two greens.
INK = "#0C2F1F"      # forest
AGREEN = "#46B152"
WARM = "#D97742"
GRAY = "#6B7280"     # g500
LIME = "#B3D335"
HAIRLINE = "#DADEE2"  # g300

_CYCLE = [
    dict(color=INK,    marker="o", linestyle="-"),
    dict(color=AGREEN, marker="s", linestyle="--"),
    dict(color=WARM,   marker="^", linestyle="-."),
    dict(color=GRAY,   marker="D", linestyle=(0, (3, 1, 1, 1))),
    dict(color=LIME,   marker="v", linestyle=(0, (1, 1))),
]

# Option C ("headline" mode): every baseline is neutral gray and only the proposed
# method — plus, optionally, the strongest rival — carries brand colour. Use it for the
# one comparison figure that has to land; using it everywhere reads as visual bias.
_HEADLINE_OURS = dict(color=AGREEN, marker="o", linestyle="-")
_HEADLINE_RIVAL = dict(color=WARM, marker="^", linestyle="-.")
_HEADLINE_BASE = [
    dict(color=GRAY, marker="s", linestyle="--"),
    dict(color=GRAY, marker="D", linestyle=":"),
    dict(color=GRAY, marker="v", linestyle=(0, (3, 1, 1, 1))),
    dict(color=GRAY, marker="x", linestyle=(0, (1, 1))),
    # deeper pool: a paper may compare different baseline sets in different figures,
    # and reusing an identity across figures is what confuses a reader
    dict(color=INK, marker="P", linestyle="--"),
    dict(color=INK, marker="*", linestyle=":"),
    dict(color=INK, marker="<", linestyle=(0, (3, 1, 1, 1))),
    dict(color=GRAY, marker=">", linestyle=(0, (5, 1))),
]

# Bar hatch is the FOURTH channel, and in a grouped bar chart it is the one that
# actually survives grayscale (white-filled bars differ by <5 grey levels). It is
# therefore assigned by name and persisted like the other three. Plain "" first: the
# first-registered method should not look textured without reason.
_HATCHES = ["", "///", "\\\\\\", "...", "xxx", "++", "ooo", "***"]

_registry: dict[str, dict] = {}
STORE = os.environ.get("AGENTRA_METHODS", ".agentra-methods.json")


def use_style() -> None:
    """Apply the house style. Call before creating any figure."""
    plt.style.use(STYLE)


def reset_registry(store: str | None = None) -> None:
    """Start a new paper. Deletes the persisted identity map."""
    _registry.clear()
    path = store or STORE
    if os.path.exists(path):
        os.remove(path)


def _fix_linestyle(style: dict) -> dict:
    """JSON has no tuples, so a dash pattern round-trips as [0, [3,1,1,1]] and
    matplotlib rejects it. Restore the nested tuple form on load."""
    ls = style.get("linestyle")
    if isinstance(ls, list):
        offset, dashes = ls
        style["linestyle"] = (offset, tuple(dashes))
    return style


def _load(path):
    if os.path.exists(path):
        with open(path, encoding="utf8") as fh:
            raw = json.load(fh)
        methods = {m: _fix_linestyle(v) for m, v in raw.get("methods", {}).items()}
        return methods, raw.get("headline"), raw.get("rival")
    return {}, None, None


def _save(path, methods, headline, rival):
    with open(path, "w", encoding="utf8") as fh:
        json.dump({"methods": methods, "headline": headline, "rival": rival},
                  fh, indent=1, ensure_ascii=False)


def register(methods, *, headline=None, rival=None, store: str | None = None) -> None:
    """Bind method names to fixed visual identities **for the whole paper**.

    The map is keyed by NAME and persisted to `store` (default
    `.agentra-methods.json` in the working directory), because each figure is a
    separate process: an in-memory, position-keyed map silently gives two different
    methods in two different figures the same colour+marker+dash, and a reader who
    learns the legend in Fig. 2 then mis-reads Fig. 5. Names already in the store keep
    their identity; new names take the next free slot.

    methods : names appearing in THIS figure. headline : your method (enables Option C).
    rival   : the strongest baseline, kept warm. Pass it BY NAME, never by position.
    """
    path = store or STORE
    known, st_head, st_rival = _load(path)
    # precedence: explicit argument > what the paper already decided > environment.
    # The env hooks (AGENTRA_HEADLINE / AGENTRA_RIVAL) let one paper pick its emphasis
    # without every template hard-coding a method name.
    headline = headline or st_head or os.environ.get("AGENTRA_HEADLINE")
    rival = rival or st_rival or os.environ.get("AGENTRA_RIVAL")

    used = {(v["color"], v["marker"], str(v["linestyle"])) for v in known.values()}

    def take(pool):
        for cand in pool:
            key = (cand["color"], cand["marker"], str(cand["linestyle"]))
            if key not in used:
                used.add(key)
                return dict(cand)
        raise RuntimeError(
            f"out of distinct identities for {len(known) + 1} methods; a single-column "
            "figure cannot carry this many series legibly — split it into small multiples"
        )

    for m in methods:
        if m in known:
            continue
        if headline is not None and m == headline:
            known[m] = dict(_HEADLINE_OURS)
            used.add((_HEADLINE_OURS["color"], _HEADLINE_OURS["marker"],
                      str(_HEADLINE_OURS["linestyle"])))
        elif rival is not None and m == rival:
            known[m] = dict(_HEADLINE_RIVAL)
            used.add((_HEADLINE_RIVAL["color"], _HEADLINE_RIVAL["marker"],
                      str(_HEADLINE_RIVAL["linestyle"])))
        else:
            known[m] = take(_HEADLINE_BASE if headline is not None else _CYCLE)

    # Hatches, in the same pass and by the same rule. Entries loaded from an older
    # store have no "hatch" key; they get one here on first use and it is persisted,
    # so the assignment is decided once per paper and not once per process.
    used_h = {v["hatch"] for v in known.values() if "hatch" in v}
    for m, v in known.items():
        if "hatch" in v:
            continue
        free = next((c for c in _HATCHES if c not in used_h), None)
        if free is None:
            raise RuntimeError(
                f"out of distinct hatches for {len(known)} methods; a single-column "
                "grouped bar chart cannot carry this many series legibly")
        v["hatch"] = free
        used_h.add(free)

    _save(path, known, headline, rival)
    _registry.clear()
    _registry.update({m: dict(v) for m, v in known.items()})


def sty(method: str, **overrides) -> dict:
    """Style kwargs for a registered method. Unknown names fail loudly on purpose:
    a silent fallback is how two methods end up sharing one identity."""
    if method not in _registry:
        raise KeyError(
            f"method {method!r} not registered; call register([...]) first "
            f"(registered: {sorted(_registry)})"
        )
    out = dict(_registry[method])
    # Line2D has no `hatch` property; it belongs to patches only, so it is not part of
    # the line-plot kwargs. Ask for it explicitly with hatch_for().
    out.pop("hatch", None)
    out.update(overrides)
    return out


def hatch_for(method: str) -> str:
    """Bar hatch for a method, keyed by NAME like every other channel.

    In a grouped bar chart the hatch is what actually carries grayscale separation
    (white-filled bars differ by <5 grey levels), so assigning it by list position —
    as an earlier version did — silently hands a different method the same texture in
    the next figure. That is the exact failure `register()` exists to prevent.
    """
    if method not in _registry:
        raise KeyError(
            f"method {method!r} not registered; call register([...]) first "
            f"(registered: {sorted(_registry)})"
        )
    return _registry[method]["hatch"]


def box_colors(method: str) -> dict:
    """Matching kwargs for boxplots: outline carries the identity, fill stays white.

    Strokes are deliberately thin. A boxplot with a small IQR relative to the axis
    range can end up ~2pt tall, and 0.9+1.2+0.9pt of stroke then fuses the median into
    a quartile edge — the one mark a boxplot exists to show. If the box is still thinner
    than ~4pt, the axis is wrong for the data: plot gap-to-best instead of raw values.
    """
    s = sty(method)
    ls = s["linestyle"]
    return dict(
        boxprops=dict(color=s["color"], linewidth=0.7, linestyle=ls),
        whiskerprops=dict(color=s["color"], linewidth=0.6),
        capprops=dict(color=s["color"], linewidth=0.6),
        medianprops=dict(color=s["color"], linewidth=1.0),
        flierprops=dict(marker=s["marker"], markersize=2.2,
                        markerfacecolor="white", markeredgecolor=s["color"],
                        markeredgewidth=0.5),
    )


def headline(ax, text, *, loc="upper left") -> None:
    """Put the number the figure exists to communicate INSIDE the panel.
    A reader should not have to hunt the caption for the punchline."""
    xy = {"upper left": (0.03, 0.97), "upper right": (0.97, 0.97),
          "lower left": (0.03, 0.03), "lower right": (0.97, 0.03)}[loc]
    ax.annotate(text, xy=xy, xycoords="axes fraction", fontsize=7, color=INK,
                ha="right" if "right" in loc else "left",
                va="top" if "upper" in loc else "bottom")


def panel_tag(ax, tag) -> None:
    """(a)/(b)/(c) tag for small multiples, set flush left above the axes."""
    ax.set_title(f"({tag})", loc="left", fontsize=8, color=INK)


def save(fig, stem: str, *, audit: bool = True, double_column: bool = False,
         strict: bool = True):
    """Save pdf + png + grayscale png, then audit print size and text size.

    Grayscale is written every time because that is the check people skip.

    A failing gate raises (exit 1). It used only to *return* the gate's exit code, and
    since the documented call is a bare `save(fig, stem)`, a `PRINT-READY: FAIL`
    scrolled past and the script still exited 0 — a gate nobody obeys. Pass
    strict=False if you genuinely want the code back instead of an exception.
    """
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png")
    plt.close(fig)
    # The grayscale proof is the check people skip, so a missing one must be loud:
    # a silent skip would let an unreadable-in-print figure pass unnoticed.
    from PIL import Image
    Image.open(stem + ".png").convert("L").save(stem + "-gray.png")
    if audit:
        script = os.path.join(HERE, "measure_plot.py")
        cmd = [sys.executable, script, stem + ".pdf"]
        if double_column:
            cmd.append("--double-column")
        rc = subprocess.call(cmd)
        if rc and strict:
            raise SystemExit(
                f"\n{'='*66}\nPRINT-READY GATE FAILED for {stem}.pdf — see the report "
                f"above.\nThe figure was written but is NOT usable in the paper. Fix the "
                f"figsize\nor the font sizes and re-run; do not paste this PDF into "
                f"LaTeX.\n{'='*66}")
        return rc
    return 0
