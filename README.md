# leo-handover-testbed

A reproducible testbed for **interpretable LEO satellite handover control**, in which a deterministic
offline optimum, computed by dynamic programming over SGP4-propagated orbital geometry, serves both
as ground truth for measurement and as the imitation signal for training. A compact
interpretable-by-design Kolmogorov-Arnold Network (KAN) policy is trained by dataset aggregation from
that optimum and benchmarked, under matched supervision, against a standard multilayer perceptron and
against readable heuristics.

### Status, read this first

This repository began as the artifact for the conference paper *"Interpretable LEO Handover Control
with Offline-Optimal Supervision"* (FAIR 2026, paper #1571332169). **That paper was withdrawn before
publication and was never published**, so it should not be cited; the code and results of that
version remain here because they are correct and because the withdrawal was not about them.

The repository is now also the artifact for a **journal manuscript in preparation**, which asks a
different question of the same testbed: *how far is the 3GPP A3 handover mechanism from optimal in
LEO, and what moves that distance?* The manuscript is **not submitted, not accepted and not
published**, so nothing here should be cited as a published result. See
[the journal study](#the-journal-study-how-far-is-a3-from-optimal) below for what was added.

Every figure and table in either study regenerates from the code below with fixed seeds and no
network access.

## What the conference version found, including the part that did not work

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

## The journal study: how far is A3 from optimal

The conference version used the offline optimum as a *training signal*. The journal study uses it as
a *measuring instrument*, and the object measured is the standardised mechanism rather than a
learned policy. Four things are new, and they are the reason this is a separate study rather than an
extension:

1. **Two action spaces, and two optima.** The 3GPP A3 event compares the serving satellite against
   one challenger, so it acts in `A_bin = {serv[s-1], top[s]}`, while an unrestricted controller may
   choose any visible satellite. `code/handover_binary.py` solves the restricted problem exactly and
   `handover.py:dp_optimum` the unrestricted one, giving `J_bin` and `J_opt`.
2. **The action-space floor, `F`.** Every binary-action policy's optimality gap splits exactly into
   `F = 100 (J_bin - J_opt)/|J_opt|`, which no such policy can cross, plus a residual that belongs to
   its decision rule. All three quantities use the same denominator `|J_opt|`; using the natural but
   different `|J_bin|` on the residual breaks the identity, which `cong_ket_qua.py` check K8 is there
   to catch. Measured over 27 regime points, `F` is a median 13.4% of the tuned A3 gap, and in one
   regime more than half of it.
3. **A regime map.** `quet_che_do.py` defines five axes (constellation density, fading severity,
   handover cost, slot length, ground-station latitude) plus a two-dimensional latitude-by-cost
   slice of 80 cells. `chay_vps.py` sweeps A3 and the heuristics over them; `quet_chinh_sach.py`
   sweeps the learned reference policies.
4. **Per-regime tuning on disjoint instances.** The A3 parameters `(h, T)` are selected on training
   seeds and the gap is reported on test seeds, separately in every regime, so a reported gap is that
   of a mechanism tuned *for* the regime and not *on* the measurement.

### Reproducing the journal study

```
python3 code/quet_che_do.py   --tu-kiem    # self-test of the regime definitions (7 checks)
python3 code/quet_chinh_sach.py --tu-kiem  # self-test of the policy sweep (6 checks)
python3 code/chay_vps.py      --viec ofat  --nhan 24    # -> results/vps-ofat.csv
python3 code/chay_vps.py      --viec mat2d --nhan 24    # -> results/vps-mat2d.csv
python3 code/quet_chinh_sach.py --viec ofat  --nhan 16  # -> results/cs-ofat.csv
python3 code/quet_chinh_sach.py --viec mat2d --nhan 16  # -> results/cs-mat2d.csv
python3 code/cong_ket_qua.py               # 10 invariant checks over every result row
python3 code/sinh_hinh_bang.py             # 11 figures + 5 tables + the number macros
```

The four sweep files are shipped **gzipped** (`results/*.csv.gz`, 12 MB instead of 63 MB) and every
loader reads `.csv.gz` transparently, so a clean clone reproduces all figures and tables without
unpacking anything by hand.

### One machine, and why that is not pedantry

Each result file carries `*.csv.xuat-xu.json`, recording the host, the Python and package versions,
and a SHA-256 over the source files that produced it. `cong_ket_qua.py` checks K6a, K6b and K6c
refuse to pass if the result files disagree on any of the three, and `sinh_hinh_bang.py` refuses to
draw anything if the gate fails.

The constraint is empirical, not ceremonial. On this problem the dynamic-programming optimum is
bit-identical across machines, but imitation training is not: the same seed and the same code gave
multilayer-perceptron weight sums differing by 0.64% between two PyTorch versions, which is training
divergence rather than floating-point noise. Mixing machines would have put a visible but spurious
difference into the learned columns. All published rows come from one host, one version set and one
code hash.

### Gates, and the fact that they were injection-tested

Three gates guard the journal study, and each was tested by deliberately breaking the artifact to
confirm it catches what it claims to catch:

| gate | what it checks | injection cases caught |
|---|---|---|
| `code/cong_ket_qua.py` | 10 invariants over every result row: positive control, optimum ordering, exact decomposition, provenance uniqueness | checks are themselves asserted on synthetic violations |
| `code/cong_van_lieu.py` | the countable claims about the literature: every cited key resolves, no survey is counted as a proposal | 5 of 5 |
| `code/cong_bai.py` | the built manuscript: clean rebuild, no hand-typed numbers in result prose, printed numbers reconciled against the raw CSVs, no orphaned equation, figure and table | 8 of 8 |

The reconciliation check is worth one sentence of warning to anyone reusing it. Its first version
asked only whether a given number appeared *somewhere* in the rendered PDF, and an injection proved
that vacuous: swapping the nominal gap macro (0.76) for the worst-case macro (24.3) left the gate
green, because 0.76 happened to also be the value of another macro elsewhere in the paper. It now
reconciles in three tiers, and matches macro names on a word boundary, because the substring test
also passed the swap.

## Scope

A single ground user is modelled, so there is no multi-user contention, load or beam constraint. The
optimum is exact per fading realisation rather than in expectation, and it is clairvoyant: it sees
the whole horizon including the fading, so it is an upper bound on achievable performance and not a
causal controller. Reported gaps are therefore conservative as statements about how much a causal
mechanism could recover.

The *policies* act in the binary space, stay or switch to the strongest visible challenger, because
that is the shape of the 3GPP A3 event. The *unrestricted optimum* does not, and keeping the two
apart is the point of the floor `F` described above. Code that assumes a single binary optimum, as
the conference version did, is the thing the journal study had to separate.

Orbital kinematics come from SGP4 propagation of a synthesised Walker-delta element set, so the
geometry is physical for that shell but is not an operator's ephemeris. The link model is a
deliberately simple elevation-driven mean with a per-seed Gaussian perturbation in dB: no rain
attenuation, no scintillation spectrum, and no correlation in time or across satellites. Fading
severity is swept widely precisely because the model is a stand-in.

## Citation

**There is nothing here to cite as a publication yet, and that is deliberate.** The conference paper
this repository started as was withdrawn before publication, so citing it would point readers at a
record that does not exist. The journal manuscript is in preparation and is neither submitted nor
accepted at the time of writing.

If you use the code or the result files, cite the repository itself:

```bibtex
@misc{do2026leohandovertestbed,
  title  = {leo-handover-testbed: an offline-optimal measuring instrument for {LEO} handover},
  author = {Do, Phuc Hao},
  year   = {2026},
  note   = {Software and result artifact, MIT licence},
  url    = {https://github.com/haodpsut/leo-handover-testbed}
}
```

This file will be updated with a publication reference if and when one exists.

## AI assistance

An AI coding assistant was used for software engineering tasks in this repository, including writing
and reviewing the gate and self-test code, and commits carry its co-authorship trailer. The
experimental design, the formulation and the interpretation of results are the author's. No
generative model produced any number in `results/`: every row is emitted by the code here, and the
gates in `code/cong_*.py` exist so that claim can be checked rather than believed.

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
