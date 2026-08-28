# PROJECT

## Goal

Given trajectory data `q_i(t)` (in the hardest case, only noisy positions),
recover a Lagrangian `L` that reproduces the observed dynamics; decide whether
two recovered Lagrangians are the same physical theory; and, for
higher-derivative Lagrangians, decide whether the theory carries an Ostrogradski
ghost.

## Current status

### 2nd-order discovery — working within a noise band

`experiments/noise_robustness_sweep.py` drives
`finding_L/main_streaming.runDiscoveryStreaming` over the two benchmark systems
in `experiments/systems.py`.

- **`isotropic_quartic_calibration`** (reference system): exact recovery of all
  quadratic + quartic coefficients at 0 and 1 % position noise; fails from ~2 %.
- **`anharmonic_chain_blind`** (blind holdout): the 5 interior chain sites
  recover exactly at 0 noise; the boundary site fails even there; the whole
  system collapses from ~1 %.

All tolerances are the **frozen `finding_L` library defaults** — no tuning
search was run — held in `experiments/discovery.FROZEN_TOLERANCES` and identical
for every system (`PhysicalSystem` has no tolerance fields). `runSystemDiscovery`
returns the exact tolerance set it used and the sweep asserts it equals
`FROZEN_TOLERANCES`, so the holdout provably runs on the same knobs as the
reference system. `startingMaxDegree` and `maxRounds` are per-system *search
budgets*, not tolerances; `maxRounds` is large enough (150) that a stopping
condition, never the round cap, decides success or failure.

### Equivalence-class classification — wired into the pipeline

`finding_L/equivalence_class.py` decides whether `discovered − expected` is a
genuine null Lagrangian (Euler–Lagrange operator identically zero) rather than
merely numerically close. It is order-aware (ordinary EL at order 1, full
Ostrogradski operator at order ≥ 2), so it judges both tracks. It is called from
`experiments/discovery.compareToExpected` (verdict on `RecoveryComparison`,
`equiv?` column and per-level verdict in the sweep artifacts) and directly from
the higher-derivative validation studies.

### Higher-derivative (Ostrogradski) track — working at order 2

- `generation/ostrogradski.py` / `ostrogradski_hamiltonian.py`: EL operator,
  top-derivative solve, RK4 state derivative, canonical momenta and Hamiltonian,
  for arbitrary order and arbitrary `coords` length. The Hamiltonian build
  raises `NonUniqueTopDerivativeError` when `L` is nonlinear in its highest
  derivative.
- `finding_L/higher_order_discovery.py`: single-coordinate recovery. Recovers
  the Pais–Uhlenbeck Lagrangian up to a total derivative, robust to ~3 % noise.
- `generation/ghost_detection.detectGhost`: ghost verdict from `H` indefiniteness
  + oscillatory dynamics. Stable to ≥ 35 % noise on a data-recovered PU
  Lagrangian — conditional on correct order identification.

### Higher-derivative track — multi-field, order-inferring, end-to-end

- **Multi-coordinate recovery** (`recoverMultiFieldHigherOrderLagrangian`):
  recovers a position-coupled Pais–Uhlenbeck chain (2–3 fields) up to a total
  derivative on clean data, cross-field coupling coefficient exact.
- **Order inference** (`inferLagrangianOrder`): infers the Lagrangian order from
  data (PU → 2, anharmonic oscillator → 1) by testing successive orders against
  an Euler–Lagrange feasibility residual.
- **End-to-end pipeline** (`finding_L/pipeline.endToEndPipeline`): noisy
  positions only → differentiation-method grid-search → order inference →
  recovery → ghost verdict, with separate order / ghost / coefficient
  confidences.

### Constrained-Hamiltonian analysis

`generation/constraints.py` + `ostrogradskiHamiltonian` detect a degenerate
(non-invertible) Lagrangian, extract the primary constraints, and classify them
first- / second-class via the canonical Poisson bracket. `detectGhost` reports
degeneracy instead of crashing. Stops before Dirac-bracket reduction.

### Tests

`tests/` (`uv run pytest`, 43): EL / Ostrogradski Hamiltonian vs closed form,
PU EOM + Hamiltonian conservation, the equivalence-class classifier, forward
selection and STLSQ/LASSO on synthetic Gram matrices, the frozen-tolerance
discipline, degenerate-constraint classification, two-field mixing, multi-field
higher-order recovery, order inference, the end-to-end pipeline, and the ghost
ROC battery.

## Open problems

### A. The ~1–2 % measurement-noise ceiling is the *greedy selector*, not OLS recovery

At ≥ 2 % position noise the **greedy forward-selection** path fills with spurious
velocity-dependent degree-4 terms (`q_i² q̇_i²`, `q_i q_j q̇_i q̇_j`): these are
genuinely well-correlated with the noise-corrupted residual, and greedy commits
to them before the true cubics. This was thought to be structural.

**Item 14 shows it is not.** A regularisation-path selector (`finding_L/regularized_select.py`),
run on the *same* Gram matrix, does much better:

| noise | greedy | STLSQ | debiased LASSO |
|---|---|---|---|
| 1 % | exact | exact | exact |
| 2 % | fails (7/19 spurious) | exact | exact |
| 5 % | fails | ~10 spurious | **exact, both benchmark systems** |

Starting from the full least-squares fit and thresholding down (rather than
adding terms greedily) recovers the true sparse Lagrangian through ~5 % noise. An
errors-in-variables formulation would still be the principled endpoint (the EL
columns are all built from noisy derivatives), but the debiased LASSO path is a
strong, cheap improvement that is available now. The production path is
unchanged pending a wider validation (`experiments/model_selection_comparison.py`).

### B. Jerk/snap libraries fail via on-shell EOM degeneracy

`jerk_snap_distractor_study` recovers `s2² − 7/30 s3²` (jerk-squared) instead of
the order-2 PU Lagrangian. On the solution manifold the higher derivatives
satisfy `q⁽⁶⁾ = −5 q⁽⁴⁾ − 4 q̈` exactly, so the EL column of `q'''²` is a linear
combination of lower-order columns *for every trajectory*. Stacking trajectories
does not lift this — they are all on-shell. `dropKineticAliasColumns` removes the
one exact alias it can see; the rest need either off-manifold data or a hard
prior that the Lagrangian order equals the kinetic term's order. **Order-2
libraries work; order-≥3 libraries do not.** The equivalence-class check flags
every failed recovery, so the failure is at least detectable.

### C. Blind-chain boundary site

Even at 0 noise the last coordinate `q5` of the open chain is mis-recovered (a
`q5² q̇5²` term shadows `q5⁴`). Same mechanism as A — a velocity term genuinely
correlates with the residual better than the true quartic once the other `q5`
terms carry small errors. The interior 5 sites recover exactly.

### D. Noisy multi-field higher-order differentiation

Multi-field higher-derivative recovery is exact on clean data but the
noisy-positions → spline-derivatives step collapses well before the
single-field case does — each Euler–Lagrange column mixes several fields'
derivative levels. Better differentiation, not a better recovery, is the gap.

### E. `detectGhost` only handles quadratic Hamiltonians

A recovered Lagrangian with a spurious `q² q̇²` term (or a genuinely nonlinear
system) gives a non-quadratic `H`; `detectGhost` returns `ghost=None`
("needs nonlinear boundedness analysis") rather than a verdict. In the ROC
battery this shows up as a rising "undetermined" rate under noise — safe (no
false alarms) but incomplete.

### F. Linear order-1 systems are degenerate for order inference

For a purely linear order-1 system (a harmonic oscillator), `EL(q̇²) ∝ EL(q²)`
exactly on-shell, so `inferLagrangianOrder`'s feasibility residual for order 1 is
trivially zero and the method still returns 1 — correct, but by a degenerate
route. Nonlinear order-1 systems (with a large enough library) infer cleanly.

## Roadmap

Items 8–14 are implemented (see `src/REPORT.md`, 2026-08-29). Next candidates:

- Errors-in-variables / total-least-squares Lagrangian recovery (the principled
  endpoint of problem A).
- Wider validation of the debiased LASSO path, then a possible switch of the
  production default.
- Off-manifold data generation to lift the on-shell degeneracy (problems B, D).
- A boundedness test for non-quadratic Hamiltonians (problem E).
- Multi-field extension of the end-to-end pipeline once multi-field
  differentiation improves.
