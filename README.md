# leo-handover-testbed

A reproducible testbed for **interpretable LEO satellite handover control**, in which a deterministic
offline optimum, computed by dynamic programming over SGP4-propagated orbital geometry, serves both
as ground truth for measurement and as the imitation signal for training. A compact
interpretable-by-design Kolmogorov-Arnold Network (KAN) policy is trained by dataset aggregation from
that optimum and benchmarked, under matched supervision, against a standard multilayer perceptron and
against readable heuristics.

This is the reproducibility artifact for the paper *"Interpretable LEO Handover Control with
Offline-Optimal Supervision"* (FAIR 2026, paper #1571332169). Every figure and table in the paper
regenerates from the code below with fixed seeds and no network access.

## What the paper finds, including the part that did not work

1. **The offline optimum is computable and verified.** Dynamic programming over an expanded state,
   carrying the window of the last `W_pp` serving satellites, gives the exact optimum per fading
   realisation. Its value is checked against an independent cost evaluator to machine precision.
2. **Learned policies reach an operating point no heuristic attains.** The KAN sits `+1.16%` from the
   optimum with `6.2` ping-pongs per 30-minute scenario, against `+1.79%` / `19.8` for strongest
   signal and `+4.36%` / `0.0` for highest elevation. It matches the MLP on objective quality and on
   decision agreement with the expert, at `3.2x` fewer parameters (65 against 211).
3. **The interpretability claim is NOT supported.** We pre-registered a two-metric test before
   running it. Rule-fidelity is indistinguishable (`0.681` KAN against `0.682` for a post-hoc
   surrogate of the MLP over 5 seeds), and the extracted threshold is in fact *less* stable for the
   KAN (`1.48` against `0.99`). We therefore do not claim superior interpretability. On this
   essentially threshold-shaped decision, a compact post-hoc explanation of a black box is already
   as faithful as the built-in rule.

Point 3 is a negative result and is reported as one. It is the reason this testbed exists: a
computable ground truth is what makes it possible to say which claim the data refuses.

## Layout

```
code/      simulation, dynamic-programming optimum, policies, training, measurement scripts
results/   every CSV the paper's tables and figures are built from
figures/   the figures as they appear in the paper, plus the TikZ source of the pipeline diagram
style/     house plotting style, vendored so a clean clone needs no path outside this repository
```

## Install

```
python3 -m pip install -r requirements.txt
```

Tested with Python 3.9, numpy 2.0, torch 2.8, scikit-learn 1.6, matplotlib 3.9, skyfield 1.54,
sgp4 2.25. CPU only; no GPU and no network are required at any point.

## Reproduce

```
python3 code/smoke_g1.py             # self-test: DP value == independent evaluator, feasibility,
                                     # and every slot has at least two visible candidates
python3 code/run_final.py            # main comparison  -> results/perf.csv, perf_summary.csv,
                                     #                     interp.csv, kan_splines.csv
python3 code/ablation.py             # ablation         -> results/ablation.csv
python3 code/robustness.py           # fading sweep     -> results/robustness.csv, figures/fig_robust
python3 code/make_figures_v2.py      # figures          -> figures/fig_tradeoff, fig_splines
cd figures && pdflatex fig_flow.tex  # pipeline diagram -> figures/fig_flow.pdf
```

The whole study runs on a laptop CPU in minutes: only a handful of satellites are visible per slot,
so the dynamic program solves one hour-scale scenario in a fraction of a second, and each policy
trains in seconds.

## Setup reproduced by the scripts

Walker constellation of 324 satellites at 53 degrees inclination and 550 km altitude, propagated by
SGP4; one mid-latitude ground user; 120 slots of 15 s, i.e. a 30-minute horizon; 5 degree elevation
mask; 2.5 dB fading; cost weights `lambda_1 = 2` (ping-pong), `lambda_2 = 5` (outage),
`lambda_3 = 1` (handover), with a ping-pong window of 2 slots. Five training fading realisations and
five held-out realisations give 25 train-by-test runs, and every reported number is the mean over
all of them.

Note on units: ping-pong counts are **per 30-minute scenario** (120 slots of 15 s), not per hour.

## Scope

The action is binary, stay or switch to the strongest visible challenger. The optimum is exact per
fading realisation rather than in expectation. A single ground user is modelled, so there is no
multi-user contention, load or beam constraint. The fading sweep shows the learned advantage
narrowing and effectively vanishing above the training fading level, which is distribution shift in
the channel rather than in the geometry.

## Citation

```bibtex
@inproceedings{do2026interpretable,
  title     = {Interpretable {LEO} Handover Control with Offline-Optimal Supervision},
  author    = {Do, Phuc Hao},
  booktitle = {Proc. Conf. Fundamental and Applied Information Technology (FAIR)},
  year      = {2026},
  note      = {Paper \#1571332169}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Camera-ready additions (September 2026)

`code/camera_ready_extra.py` reproduces the two measurements added for the camera-ready version
in response to the reviewers, and writes `results/camera-ready-extra.json`.

**Cost of the binary action space.** The paper reduces the per-slot decision to stay-or-switch
toward the strongest challenger. The script re-solves the dynamic program over the *full*
candidate set (every visible satellite plus an outage option) and reports the difference. With
an average of 12.6 visible satellites per slot, the reduction costs 0.17% of the objective on
average and 0.28% at worst.

**3GPP A3 baseline.** A hysteresis plus time-to-trigger mechanism, tuned on the training
realisations and evaluated on the held-out ones, against the same offline optimum the paper
uses. The result does not favour the paper and is reported as found: the best configuration
(h = 2 dB, TTT = 1) reaches a smaller gap than either learned policy at comparable stability.
Its performance is highly parameter-sensitive, spanning 23.7 percentage points of gap across
the 15-point grid.

A positive control is built in: A3 with h = 0 and TTT = 1 must reproduce the strongest-signal
baseline exactly, and the script asserts this before reporting anything.

```bash
python3 code/camera_ready_extra.py
```
