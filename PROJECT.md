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

### Tests

`tests/` (`uv run pytest`, 19): EL / Ostrogradski Hamiltonian vs closed form,
PU EOM + Hamiltonian conservation, the equivalence-class classifier both ways,
forward selection on a synthetic Gram matrix, and the frozen-tolerance
discipline.

## Open problems

### A. The ~1–2 % measurement-noise ceiling is structural, not a tolerance bug

At ≥ 2–3 % position noise the pipeline fills with spurious velocity-dependent
degree-4 terms (`q_i² q̇_i²`, `q_i q_j q̇_i q̇_j`) and the quadratic coefficients
are biased ~30–50 % high. These terms are **genuinely well-correlated** with the
noise-corrupted residual — their EL columns are not linearly dependent on the
true terms' columns, so no dependency filter removes them, and their correlation
score (~0.15–0.25) sits above any cutoff that still admits the real cubic terms
(~0.13–0.20). Greedy forward selection against on-shell data cannot separate
them.

**Direction of a real fix:** replace greedy forward selection + sequential
thresholding with an **errors-in-variables** sparse regression — the EL columns
are all built from noisy `q, q̇, q̈`, so ordinary least squares is biased. A
total-least-squares or SINDy-with-measurement-error formulation, with a
regularisation path chosen by cross-validation on held-out *trajectories*, is the
right tool. This is a design change to the estimator (tracked as item 14).

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

### D. Model order and state-variable count are still hand-specified

The higher-derivative discovery scripts hardcode `LAGRANGIAN_ORDER` and
`NO_STATE_VARS`. Inferring the order from data is item 11.

## Roadmap (items 8–14, not yet started)

8. Degenerate-Lagrangian (Dirac constraint) detection + first/second-class
   classification in `ostrogradski_hamiltonian`.
9. Two-coordinate mixing testbed with an explicit mass matrix.
10. Multi-coordinate higher-order discovery.
11. Automatic Lagrangian-order inference from data.
12. End-to-end pipeline on noisy position-only data (differentiation choice →
    discovery → ghost verdict → uncertainty).
13. ROC-style ghost-detection validation over a battery of systems and noise.
14. Regularisation-path / SINDy model selection as an additive comparison to the
    current greedy forward selection.
