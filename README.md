# Trajectory-to-Lagrangian Inference

Given trajectory data `q_i(t)` — in the hardest case only noisy positions —
recover a Lagrangian `L` that reproduces the observed dynamics, decide whether
two recovered Lagrangians are the same physical theory, and, for
higher-derivative Lagrangians, decide whether the theory carries an Ostrogradski
ghost.

This is the single design doc. The dated, append-only work log is
[`REPORT.md`](REPORT.md).

---

## Layout

`src/` holds three packages, each with an empty `__init__.py`; there is no
installable distribution (`[tool.uv] package = false`), so everything runs from
`src/`. The tree carries no comments and no docstrings by design — the
explanation lives here.

### `generation/` — the forward direction: a known `L` → data / analysis

| module | contents |
|---|---|
| `eqnofmotion` | the single `TIME` symbol, `defineCoordinates`, the ordinary Euler–Lagrange operator |
| `integrator`, `higher_order_integrator` | RK4 for 2nd-order and higher-order state |
| `generate_data`, `noise` | streamed noisy-trajectory CSV generation |
| `ostrogradski` | Ostrogradski EL operator `Σ (−d/dt)^k ∂L/∂q^(k)`, top-derivative solve, RK4 state derivative — arbitrary order, arbitrary `coords` length |
| `ostrogradski_hamiltonian` | canonical momenta and Hamiltonian; `NonUniqueTopDerivativeError`; `analyzeDegenerateLagrangian` |
| `constraints` | canonical Poisson bracket, weak (on-shell) vanishing, first/second-class constraint classification |
| `ghost_detection` | `detectGhost` — `H` indefiniteness + EOM characteristic roots |
| `numerical_diff` | finite-difference / Savitzky–Golay / quintic-smoothing-spline derivative estimates |

### `finding_L/` — the inverse direction: data → `L`

| module | contents |
|---|---|
| `candidates`, `higher_order_candidates` | monomial libraries; `buildMultiFieldElMatrix` for coupled fields |
| `build_matrix` | the streaming Gram matrix `G = Θᵀ Θ` (chunked, cell-budgeted) |
| `gram_forward_select` | fit / score / prune primitives on `G` |
| `stopping_conditions` | the three stopping conditions |
| `regularized_select` | STLSQ and debiased-LASSO selectors, Gram-only (additive alternative to greedy) |
| `main_streaming` | `runDiscoveryStreaming` — the 2nd-order greedy driver |
| `higher_order_discovery` | `recoverHigherOrderLagrangian`, `recoverMultiFieldHigherOrderLagrangian`, `forwardSelectFromGram`, `inferLagrangianOrder` |
| `pipeline` | `endToEndPipeline` — noisy positions → `L` + ghost verdict + confidences |
| `report` | `assembleDiscoveredLagrangian`, coefficient snapping, the readable text |
| `equivalence_class` | `classifyLagrangianPair` / `isNullLagrangian` — null-Lagrangian test, order-aware |

### `experiments/` — benchmarks and runnable studies

`systems` (2nd-order benchmarks), `pu_system` (Pais–Uhlenbeck helpers),
`generate_dataset`, `discovery` (the frozen-tolerance policy + `compareToExpected`),
and the studies listed under **Install and run**.

---

## Install and run

Dependencies: numpy, scipy, sympy, pandas, matplotlib (`pyproject.toml`; the
checked-in `.venv` already has them). Tests also need `pytest` (the `dev` group).

```
cd src
PY=../.venv/bin/python

# 1. build a dataset from a known symbolic Lagrangian
$PY -m experiments.generate_dataset anharmonic_chain_blind --noise 0.0 0.05 0.10 0.25
$PY -m generation.main                            # the isotropic calibration system's CSVs

# 2. recover a Lagrangian from a dataset (readable report to stdout)
$PY -m finding_L.main_streaming                    # __main__ points at a generated CSV

# 3. 2nd-order discovery vs measurement noise
$PY -m experiments.noise_robustness_sweep isotropic_quartic_calibration   # reference system
$PY -m experiments.noise_robustness_sweep anharmonic_chain_blind          # blind holdout
#   optional: --noise 0.0 0.01 0.02 0.05 0.10   --chunk-rows N

# 4. higher-derivative (Ostrogradski) track
$PY -m finding_L.equivalence_class                 # constructed null-Lagrangian checks
$PY -m experiments.pu_oscillator_validation
$PY -m experiments.differentiation_method_study
$PY -m experiments.higher_order_discovery_validation
$PY -m experiments.jerk_snap_distractor_study
$PY -m experiments.two_field_mixing                # mass-matrix spectrum + 2-field recovery
$PY -m experiments.multi_field_discovery_validation   # coupled Pais-Uhlenbeck chain
$PY -m experiments.order_inference_validation      # infer the Lagrangian order from data

# 5. ghost detection (symbolic checks, noise boundary, ROC battery)
$PY -m experiments.ghost_detection_validation

# 6. end-to-end: noisy positions -> L + ghost verdict + confidence
$PY -m experiments.end_to_end_pipeline_validation

# 7. model selection: greedy vs STLSQ vs LASSO (additive comparison)
$PY -m experiments.model_selection_comparison
```

Each study writes `.txt` / `.json` (and some `.png`) into
`src/experiments/results/`. Datasets under `src/experiments/data/` and
`src/generation/data/` are git-ignored and regenerable (the generators are
seeded).

## Tests

```
uv sync --group dev
uv run pytest
```

43 tests: EL / Ostrogradski Hamiltonian vs closed form; the Pais–Uhlenbeck EOM
and Hamiltonian conservation; the equivalence-class classifier both ways; forward
selection, STLSQ and LASSO on synthetic Gram matrices; the frozen-tolerance
discipline; degenerate-constraint classification; two-field mixing; multi-field
higher-order recovery; order inference; the end-to-end pipeline; the ghost ROC
battery.

---

## How the pipeline works

### Symbol conventions

`generation/eqnofmotion.py` owns the single time symbol `TIME = Symbol("t")` and
`defineCoordinates(n)` → `q_i(t)` functions plus their first derivatives. Two
symbol spaces, mapped in `finding_L/report.py`:

- **functional** (`q0(t)`, `Derivative(q0(t), t)`) — used for all calculus
  (Euler–Lagrange derivatives, total time derivatives); differentiation w.r.t.
  time and w.r.t. the coordinate only makes sense when the coordinate is a
  function of `t`.
- **state** (`q0, v0, …`; `s0, s1, s2, …` for the higher-derivative track) — used
  for the numeric regression and the readable output; plain symbols lambdify
  cleanly and read well.

### 1. Dataset generation — `experiments/generate_dataset.py`, `generation/`

`experiments/systems.py` holds `PhysicalSystem` records (a symbolic Lagrangian
builder, the expected scaled Lagrangian, integration parameters).
`generation/integrator.GetAccelFunctions` solves the Euler–Lagrange system for
the accelerations `q̈_i = f(q, q̇)` and lambdifies them;
`generation/generate_data.generateDatasetStreaming` integrates many seeded random
initial conditions with RK4, adds Gaussian noise scaled per column by
`noisePercentage · std`, and streams rows to CSV. The ground truth is a known
`L`, so recovery can be scored exactly and the noise model is explicit.

### 2. Candidate library — `finding_L/candidates.py`

All monomials in `(q_i, q̇_i)` of total degree `1..maxDegree`, de-duplicated by
exponent tuple; pure-velocity monomials are dropped (they add nothing to the EL
residual structure). The kinetic term `Σ q̇_i²` is appended and is the regression
target. This turns "find a function" into "find sparse coefficients over a fixed
basis" — a linear problem once the EL operator is applied.

### 3. Streaming Gram matrix — `finding_L/build_matrix.py`

For each candidate `θ`, its Euler–Lagrange column `EL(θ) = d/dt ∂θ/∂q̇ − ∂θ/∂q`
is formed (with `q̈_i` substituted as an independent data symbol), lambdified once
(cached across degree expansions), evaluated on the CSV in row chunks, and
sub-chunked to a cell budget (`GRAM_DENSE_CELL_BUDGET`, peak RSS ~1 GB). The
accumulators are the row count `n`, the column sums, and `G = Θᵀ Θ`.

The model being fit: the discovered `L` must satisfy the observed equations of
motion,

```
EL(kinetic) + Σ_j c_j · EL(θ_j) ≈ 0     over all data rows
```

The design matrix `Θ` (one row per data point × coordinate, one column per
candidate) is never materialised — for a degree-4 library on 6 DOF it is ~23 GB.
`G` is `n_candidates × n_candidates`; everything downstream needs only `G` and
`b = −G[:, kinetic]`.

### 4. Forward selection — `finding_L/main_streaming.py`, `gram_forward_select.py`, `stopping_conditions.py`

Greedy, orthogonal-matching-pursuit style. Per round, with active set `S`:

1. **Refit** — `c_S = solve(G[S,S], b[S])`
   (`fitActiveCoefficientsFromGram`). A singular block falls back to `lstsq` and
   emits a `RuntimeWarning` — that only happens when two selected EL columns are
   collinear, and their coefficient split is then not identifiable.
2. **Residual** — `residualNormSq = targetNormSq − c_S · b[S]`;
   `scaledResidual = sqrt(residualNormSq / targetNormSq)`.
3. **Score reserves** — for each reserve `j`,
   `score_j = (b_j − c_S · G[j, S]) / (‖θ_j‖ · ‖r‖)`, the cosine between
   candidate `j`'s EL column and the current residual. Take `argmax |score|`.
4. **Select or expand** — add the best candidate if
   `|score| ≥ correlationCutoff`; otherwise attempt a degree expansion.
5. After the loop, `pruneNearZeroCoefficients` iteratively drops any active term
   with `|c| < pruneRelativeThreshold · max|c|` and refits.

The full best-subset problem is combinatorial; this greedy score is cheap and, on
clean data, exact. Its noise failure mode — and the selectors that do better —
are in **Open problem A** and section 12.

**Stopping conditions**

| id | fires when | meaning |
|---|---|---|
| **A — converged** | `scaledResidual < residualRmsTolerance` | the active set explains the dynamics |
| **B — stalled** | best reserve `|score| < correlationCutoff` and the library is at `degreeCap` | no remaining candidate meaningfully helps |
| **C — stagnated** | `scaledResidual` improved by `< stagnationTolerance` for `stagnationPatience` consecutive rounds | flattened above the noise floor — graceful exit instead of grinding to `maxRounds` |

Between B and continuing: if the score stalled but the degree is below
`degreeCap`, the library is widened by one degree, `G` is re-streamed for the new
columns (reusing the lambdify cache), the residual history is reset, and the loop
continues.

**Frozen tolerances** — `experiments/discovery.FROZEN_TOLERANCES`

These are the `finding_L` library default values, frozen verbatim — **not** the
output of any calibration or tuning search. They are identical for every system;
`PhysicalSystem` carries no tolerance fields, only search budgets
(`startingMaxDegree`, `maxRounds`). `maxRounds` is 150 — large enough that A / B /
C, never the round cap, ends the search (asserted in `tests/`).

| key | value | consumed by |
|---|---|---|
| `correlationCutoff` | `0.1` | `checkCorrelationCutoff` |
| `residualRmsTolerance` | `0.01` | `checkResidualToleranceFromGram` |
| `pruneRelativeThreshold` | `1e-2` | `pruneNearZeroCoefficients` |
| `stagnationTolerance` | `0.01` | `checkResidualStagnation` |
| `stagnationPatience` | `3` | `checkResidualStagnation` |
| `degreeCap` | `4` | `checkDegreeExpansionNeeded` (both benchmark Lagrangians are quartic) |

Every value is threaded explicitly into `runDiscoveryStreaming` — none falls back
to a function default at runtime. `runSystemDiscovery` returns
`(discovered, logFrame, tolerancesUsed)` and `noise_robustness_sweep.py` asserts
`tolerancesUsed == FROZEN_TOLERANCES` on every run, so a blind-holdout system
provably runs on the same knobs as the reference system.

### 5. Readable report — `finding_L/report.py`

`assembleDiscoveredLagrangian` → `DiscoveredLagrangian(expression, rawExpression,
…, text)`: `rawExpression` keeps the fitted floats; `expression` snaps each
coefficient to the nearest small-denominator rational within 1 % relative; `text`
groups terms by degree and shared coefficient. Physical Lagrangians have simple
rational coefficients, so snapping turns `-0.24999` into `-1/4` and makes the
equivalence check below exact rather than approximate.

### 6. Equivalence-class check — `finding_L/equivalence_class.py`

`classifyLagrangianPair(A, B, coords, vels, order=None)`:

1. `ΔL = expand(A − B)`. `ΔL == 0` → identical.
2. `isNullLagrangian(ΔL)` applies the Euler–Lagrange operator and checks every
   component reduces to `0` (`expand`, then `simplify`). Order-aware: order 1
   uses the ordinary EL operator (`eqnofmotion.EulerLagrangeEqn`); order ≥ 2 uses
   the full Ostrogradski operator (`ostrogradski.eulerLagrangeExpression`); order
   defaults to the highest derivative present.
3. If null and first-order, `reconstructBoundaryPotential` recovers `F` with
   `ΔL = dF/dt` (curl-free check, then integrate the velocity coefficients).

Verdict: **equivalent** if `ΔL == 0` or `ΔL` is a nonzero null Lagrangian (same
physics, differ by a total time derivative); **not equivalent** if `ΔL` has a
nonzero EL residual (different equations of motion). Two Lagrangians can differ
by `d/dt F` and be physically identical (e.g. `q q̈` vs `−q̇²`), and two with the
same monomials but slightly-off coefficients are *not* the same theory —
coefficient proximity answers neither question, the EL operator answers both.

`experiments/discovery.compareToExpected` runs this on every
discovered-vs-expected pair; the noise sweep surfaces it per level as `exact`
(`ΔL == 0`) / `null-L` (nonzero null Lagrangian) / `no` (nonzero EL residual —
the recovery failed even if the coefficients looked close). The higher-derivative
studies call `isNullLagrangian` directly.

### 7. Higher-derivative track — `finding_L/higher_order_*.py`, `generation/ostrogradski*.py`

Same shape as steps 2–5, single-coordinate, with an explicit Lagrangian order.
`higher_order_candidates.py` builds monomials in `(q, q', q'', …)`; each
candidate's EL column uses the full Ostrogradski operator, and
`dropKineticAliasColumns` removes columns collinear with the kinetic column (the
exact `q''² ↔ q' q'''` alias). `recoverHigherOrderLagrangian` builds a dense Gram
matrix, runs `forwardSelectFromGram`, and snaps coefficients. Noisy derivatives
come from `generation/numerical_diff.py`; the quintic smoothing spline is the
only method that survives to 3rd/4th order. On Pais–Uhlenbeck it recovers `L` up
to a total derivative, robust to ~3 % noise.

### 8. Ghost detection — `generation/ghost_detection.py`

1. Build the Ostrogradski Hamiltonian `H`
   (`ostrogradski_hamiltonian.ostrogradskiHamiltonian`). Raises
   `NonUniqueTopDerivativeError` if `L` is nonlinear in its highest derivative
   (the Legendre transform is then multi-valued and the physical branch is a
   caller decision).
2. If `H` is quadratic in the phase-space variables, take its Hessian and count
   negative eigenvalues; otherwise report "needs nonlinear boundedness analysis".
3. Compute the EOM characteristic roots; classify the dynamics (oscillatory /
   runaway / damped).
4. **Ghost** iff `H` is indefinite *and* the dynamics are oscillatory — a
   bounded-motion negative-energy mode, the Ostrogradski signature. Indefinite
   `H` with runaway dynamics is a tachyon, not a ghost.

The Ostrogradski theorem guarantees a linear-in-momentum term in `H` for a
non-degenerate higher-derivative theory, so `H` is unbounded below; the
oscillatory-dynamics condition rules out the trivial unbounded-but-harmless
runaway. The verdict is invariant under the total-derivative freedom from step 6
— PU and PU + a total derivative both return `ghost = True`. On a data-recovered
PU Lagrangian the verdict is stable to ≥ 35 % noise, conditional on correct order
identification.

### 9. Degenerate Lagrangians — `generation/constraints.py`

When `ostrogradskiHamiltonian` cannot invert `p_n = ∂L/∂q^(n)` for `q^(n)` it
returns a `DegenerateLagrangianResult` instead of a Hamiltonian dict, and
`detectGhost` returns `ghost=None, degenerate=True` with the analysis attached
rather than crashing:

1. Primary constraints come from the null space of the Hessian
   `W_ab = ∂²L/∂q_a^(n) ∂q_b^(n)`: each null vector `v` gives a phase-space
   relation `Σ_a v_a (P_a − ∂L/∂q_a^(n)) = 0` (the `q^(n)` parts cancel because
   `W v = 0`).
2. `classifyConstraints` builds the canonical Poisson-bracket matrix
   `C_ab = {φ_a, φ_b}`. A constraint whose whole row of `C` vanishes *weakly*
   (modulo the constraint ideal, checked with a Gröbner reduction) is
   first-class; otherwise second-class.
3. For a first-class candidate, `{φ_a, H}` is also tested weakly; if it does not
   vanish, consistency requires a secondary constraint — flagged, not computed.

First/second-class classification already gives the physical phase-space
dimension and whether there is gauge freedom. Dirac-bracket construction and the
full Dirac–Bergmann iteration are out of scope.

### 10. Multi-field recovery and order inference — `finding_L/higher_order_*.py`

- `buildMultiFieldElMatrix` stacks one row-block per field (rows = timepoints ×
  fields), exactly as the 2nd-order streaming path stacks per-coordinate blocks.
  `recoverMultiFieldHigherOrderLagrangian` fixes the isotropic top-derivative
  kinetic `Σ_i (q_i^(n))²` and forward-selects the rest, cross-field coupling
  monomials included. On a position-coupled Pais–Uhlenbeck chain (2–3 fields) it
  recovers `L` up to a total derivative with the coupling coefficient exact on
  clean data.
- `inferLagrangianOrder` tries orders `1..maxOrder`; for each it measures the
  least-squares residual of projecting the `q^(n)²` kinetic EL column onto the
  span of the other EL columns (a feasibility test: does the data satisfy an
  order-`n` Euler–Lagrange equation?). It returns the smallest order below
  tolerance (Condition A), else the order after which the residual stops
  improving (Condition C). PU → 2, anharmonic oscillator → 1.

### 11. End-to-end pipeline — `finding_L/pipeline.py`

`endToEndPipeline(noisyPositions, dt)` chains the pieces with no ground-truth
input:

1. For each differentiation method (Savitzky–Golay, SG poly-8, quintic spline):
   estimate derivatives → `inferLagrangianOrder` → `recoverHigherOrderLagrangian`
   → `detectGhost`, and record the recovered Lagrangian's *own* Euler–Lagrange
   residual on that method's derivatives.
2. Consensus order = majority vote. Method selection = among recoveries with
   plausible (not absurdly large) coefficients, the lowest own-EL residual.
3. Three confidences: **order** (cross-method agreement), **ghost** (agreement on
   the verdict), **coefficient** (spread across methods + plausibility). Overall
   confidence is the order/ghost minimum — the robust part; coefficients are
   reported separately because the differentiation step limits them. On PU: order
   2 and ghost True with full cross-method agreement through ≥ 1 % noise; the
   coefficients drift with noise.

### 12. Alternative selectors — `finding_L/regularized_select.py`

Two regularisation-path selectors solve the same Gram-only problem as the greedy
path, as an additive comparison (never a silent replacement):

- **`sequentialThresholdedLeastSquares`** (SINDy STLSQ): start from the full
  least-squares fit `G c = b`, zero the coefficients below
  `relativeThreshold · max|c|`, refit on the survivors, iterate to a fixed point.
- **`lassoSelect`**: a coordinate-descent LASSO path from `G` and `b` (minimising
  `½ cᵀG c − bᵀc + λ‖c‖₁` over a geometric λ sequence), then the sparsest solution
  whose refit residual is within 5 % of the densest, then hard-threshold and
  refit (debiased LASSO).

`experiments/model_selection_comparison.py` runs all three on one degree-4
streaming Gram per benchmark system and noise level:

| noise | greedy | STLSQ | debiased LASSO |
|---|---|---|---|
| 1 % | exact | exact | exact |
| 2 % | fails (7/19 spurious) | exact | exact |
| 5 % | fails | ~10 spurious | **exact, both benchmark systems** |

So the ~1–2 % ceiling is a property of the greedy selector, not of least-squares
Lagrangian recovery. The production path is unchanged pending wider validation.

---

## Current status

- **2nd-order discovery.** `isotropic_quartic_calibration` (reference system):
  exact recovery of all quadratic + quartic coefficients at 0 and 1 % position
  noise; fails from ~2 % (greedy) — see problem A. `anharmonic_chain_blind`
  (blind holdout, same frozen tolerances): 5 interior chain sites exact at 0
  noise, boundary site fails even there, whole system collapses from ~1 %.
- **Equivalence-class classification** is wired into `compareToExpected` and the
  higher-derivative studies.
- **Higher-derivative track** works at order 2: single-coordinate PU recovery
  robust to ~3 % noise; multi-coordinate coupled-chain recovery exact on clean
  data; `inferLagrangianOrder` correct on PU and the anharmonic oscillator;
  `endToEndPipeline` correct on order + ghost from noisy positions only.
- **Constrained-Hamiltonian analysis** detects and classifies degenerate
  Lagrangians (stops before Dirac brackets).

## Open problems

### A. The ~1–2 % noise ceiling is the greedy selector, not OLS recovery

At ≥ 2 % position noise the greedy forward-selection path fills with spurious
velocity-dependent degree-4 terms (`q_i² q̇_i²`, `q_i q_j q̇_i q̇_j`): genuinely
well-correlated with the noise-corrupted residual, and greedy commits to them
before the true cubics. This was thought to be structural, but the STLSQ / LASSO
comparison (section 12) shows it is not — starting from the full least-squares
fit and thresholding down recovers the true sparse Lagrangian through ~5 % noise.
An errors-in-variables formulation is still the principled endpoint (the EL
columns are all built from noisy derivatives); the debiased LASSO path is a cheap
improvement available now.

### B. Jerk/snap libraries fail via on-shell EOM degeneracy

`jerk_snap_distractor_study` recovers `s2² − 7/30 s3²` (jerk-squared) instead of
the order-2 PU Lagrangian. On the solution manifold `q⁽⁶⁾ = −5 q⁽⁴⁾ − 4 q̈`
exactly, so the EL column of `q'''²` is a linear combination of lower-order
columns for every trajectory; stacking trajectories does not lift this — they are
all on-shell. Order-2 libraries work; order-≥3 libraries do not. The
equivalence-class check flags every failed recovery, so the failure is
detectable. A fix needs off-manifold data or a hard order prior.

### C. Blind-chain boundary site

Even at 0 noise the last coordinate `q5` of the open chain is mis-recovered
(a `q5² q̇5²` term shadows `q5⁴`) — same mechanism as A. The interior 5 sites
recover exactly.

### D. Noisy multi-field higher-order differentiation

Multi-field higher-derivative recovery is exact on clean data but the
noisy-positions → spline-derivatives step collapses well before the single-field
case does — each Euler–Lagrange column mixes several fields' derivative levels.
Better differentiation, not a better recovery, is the gap.

### E. `detectGhost` only handles quadratic Hamiltonians

A recovered Lagrangian with a spurious `q² q̇²` term (or a genuinely nonlinear
system) gives a non-quadratic `H`; `detectGhost` returns `ghost=None` rather than
a verdict. In the ROC battery this is a rising "undetermined" rate under noise —
safe (no false alarms) but incomplete.

### F. Linear order-1 systems are degenerate for order inference

For a purely linear order-1 system (a harmonic oscillator), `EL(q̇²) ∝ EL(q²)`
exactly on-shell, so `inferLagrangianOrder`'s feasibility residual for order 1 is
trivially zero — it still returns 1, correctly, but by a degenerate route.
Nonlinear order-1 systems (with a large enough library) infer cleanly.

## Roadmap

- Errors-in-variables / total-least-squares recovery (the principled endpoint of
  problem A).
- Wider validation of the debiased LASSO path, then a possible switch of the
  production default.
- Off-manifold data generation to lift the on-shell degeneracy (B, D).
- A boundedness test for non-quadratic Hamiltonians (E).
- Multi-field extension of the end-to-end pipeline once multi-field
  differentiation improves.
