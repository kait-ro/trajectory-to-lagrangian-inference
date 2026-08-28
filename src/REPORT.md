# finding_L — hardening, higher-derivative extension, cleanup

Work log covering the readable-Lagrangian fix, Phases 1–3, a cleanup pass, and the
follow-up fixes. Everything here is reproducible with the commands in
[How to run](#how-to-run). Artifacts land in `src/experiments/results/`.

---

## How to run

The project is a `src/`-layout set of packages (`generation`, `finding_L`,
`experiments`) with no single entry point. Dependencies: numpy, scipy, sympy, pandas,
matplotlib (declared in `pyproject.toml`; `.venv` already has them).

Run everything from `src/` (which puts the packages on `sys.path`):

```
cd src
PY=../.venv/bin/python

# --- generation: build a trajectory dataset from a symbolic Lagrangian ---
$PY -m generation.main                                    # writes the calibration system's noisy CSVs
$PY -m experiments.generate_dataset anharmonic_chain_blind --noise 0.0 0.05 0.10 0.25

# --- discovery: recover a Lagrangian from a dataset, with the readable report ---
$PY -m finding_L.main_streaming                           # __main__ points at a generated dataset

# --- Phase 1 ---
$PY -m experiments.equivalence_class                      # 1.2 constructed null-Lagrangian checks
$PY -m experiments.phase1_noise_curve isotropic_quartic_calibration   # 1.3 (also does 1.1's blind system)
$PY -m experiments.phase1_noise_curve anharmonic_chain_blind

# --- Phase 2 ---
$PY -m experiments.phase2_pu_oscillator                   # 2.1 + 2.2 + 3.1 smoke test
$PY -m experiments.phase2_differentiation_study           # 2.3 noisy higher-order differentiation
$PY -m experiments.phase2_higher_order_discovery          # 2.4 (correct-order library)
$PY -m experiments.phase2_jerk_distractor                 # 2.4 (jerk/snap distractors)

# --- Phase 3 ---
$PY -m experiments.phase3_ghost_detection                 # 3.2 + 3.3 ghost verdict vs noise
```

Programmatic use:

```python
from finding_L.main_streaming import runDiscoveryStreaming
discovered, log = runDiscoveryStreaming("path/to/trajectories.csv", noCoords=6, degreeCap=4)
print(discovered.text)          # readable, grouped, coefficients snapped
discovered.expression           # sympy expression in clean q0, v0, ... symbols
```

Datasets (`src/experiments/data/`, `src/generation/data/`) are git-ignored and
regenerable — the generators are seeded.

---

## Cleanup pass

**Deleted (dead / superseded):**

| file | why |
|---|---|
| `generation/lagrangian.py` | superseded by `eqnofmotion.py`; nothing imported it |
| `finding_L/main.py` (dense) | superseded by the Gram-streaming path in `main_streaming.py` |
| `finding_L/forward_select.py` | only used by the deleted dense path + benchmark |
| `finding_L/correlation_scoring.py` | only used by the deleted dense path |
| `finding_L/benchmark_forward_selection.py` | benchmarked the deleted dense path on synthetic data |
| `finding_L/compare_methods.py` | orphan (`compareSelectionResults` never imported) |
| `finding_L/trajectory_split.py` | orphan (`splitTrajectoryGroups` never imported) |
| `src/__init__.py` | vestigial `uv init` stub (`def main(): print("Hello…")`); `src/` is a path entry, not a package |
| `candidates.computeResidual`, `build_matrix.{buildCandidateColumnMatrix, splitKineticTarget, gramToCovariance}` | only used by the deleted dense path |
| `stopping_conditions.checkResidualTolerance` | non-Gram variant, only used by the deleted dense path |

**Consolidated:**

- One time symbol. `TIME = sympy.Symbol("t")` is defined once in `generation/eqnofmotion.py`;
  every other module imports it. Removed the four scattered redefinitions
  (`eqnofmotion` had two local `sp.symbols("t")`, `ostrogradski` and
  `equivalence_class` each had their own).
- `experiments/pu_system.py` — all Pais–Uhlenbeck helpers (`OMEGA1/OMEGA2`,
  `paisUhlenbeckLagrangian`, `paisUhlenbeckStateLagrangian`, `groundTruthColumns`)
  in one place. They were spread across three phase-2 files.
- `finding_L/higher_order_discovery.py` — the shared higher-order recovery
  (`recoverHigherOrderLagrangian`, `forwardSelectFromGram`, `stateToCoordinate`).
  `phase2_higher_order_discovery` and `phase2_jerk_distractor` had ~80 duplicated lines each.

**`pyproject.toml`:** declared the real dependencies (were `[]`), removed the broken
`something:main` script entry and the build backend, marked it `[tool.uv] package = false`
(it is a workspace of scripts, not an installable library).

Net: 8 modules deleted, 3 added. `finding_L/` went from 12 files to 9; the two
`phase2_*` files lost ~80 duplicated lines each. Whole tree ≈ 2.5k lines, no comments.

---

## Fixes applied

### Kept — safe and beneficial

1. **Gram-matrix OOM** (`build_matrix.buildGramMatrixChunked`). The dense `Θ` block for a
   degree-4 expansion on 6 DOF was ~23 GB and got OOM-killed. Now sub-chunked to a cell
   budget; peak RSS ~1 GB.

2. **Euler–Lagrange column cache** (`build_matrix.lambdifiedColumnsForTerms`). Each degree
   expansion used to re-derive and re-lambdify *every* candidate's EL columns from sympy.
   `runDiscoveryStreaming` now threads a `lambdifiedCache` dict, so an expansion only pays
   the sympy cost for the genuinely new terms. Roughly halves discovery wall-time on
   systems that expand twice.

3. **Degree-expansion cap** (`runDiscoveryStreaming(degreeCap=…)`, threaded into
   `checkDegreeExpansionNeeded`). Without it, moderate noise makes the RMS-convergence test
   unreachable and the search expands to degree 6 (~12k candidates), each expansion
   exponentially slower. Default stays 6; the experiments pin it to the true max degree + 0.

4. **Residual-stagnation stop — Condition C** (`stopping_conditions.checkResidualStagnation`).
   Stops when the scaled residual improves by less than `stagnationTolerance` (default 1 %)
   for `stagnationPatience` (default 3) consecutive rounds. Verified it does **not** perturb
   the clean recoveries (isotropic 0-noise still recovers exactly in 53 rounds). It gives
   the search a graceful exit when the noise floor is above the RMS tolerance, instead of
   grinding to `maxRounds`.

### Attempted and removed — did not work

5. **Linear-dependency filter for the streaming path.** I built a greedy
   conditional-variance filter to drop candidate EL columns that are linear combinations of
   already-kept columns (the velocity aliases that shadow `q_i^3`). It removed the aliases
   *and* legitimate `q_i^2 q_j^2` cross terms, breaking the isotropic recovery — because the
   EL columns are genuinely highly correlated on-shell, the projection step goes singular
   once the kept set is large. Removed. The narrow cosine-similarity alias filter for the
   higher-order path (`higher_order_candidates.dropKineticAliasColumns`) stays — it only
   compares each column against the kinetic column, no matrix inversion, and it removes the
   exact `q''^2 ↔ q' q'''` alias reliably.

---

## Still open (documented, not cleanly fixable in scope)

### A. The ~1–2 % measurement-noise ceiling is structural, not a tolerance bug

At ≥ 2–3 % position noise the pipeline fills with spurious velocity-dependent degree-4
terms (`q_i^2 v_i^2`, `q_i q_j v_i v_j`) and the quadratic coefficients are biased ~30–50 %
high. These terms are **genuinely well-correlated** with the noise-corrupted residual —
their EL columns are not linearly dependent on the true terms' columns, so no dependency
filter removes them, and their correlation score (~0.15–0.25) is above any cutoff that
still admits the real cubic terms (~0.13–0.20). Greedy forward selection against on-shell
data cannot separate them.

**Recommended fix (a real change, not a knob):** replace greedy forward selection +
sequential thresholding with an **errors-in-variables** sparse regression — the EL columns
are all built from noisy `q, q̇, q̈`, so ordinary least squares is biased. A total-least-squares
or SINDy-with-measurement-error formulation, plus a regularisation path chosen by
cross-validation on held-out *trajectories* (`splitTrajectoryGroups` was the start of this
before it was removed), is the right tool. This is a design change to the estimator, out
of scope for a cleanup pass.

### B. Jerk/snap libraries fail via on-shell EOM degeneracy

`phase2_jerk_distractor` still recovers `s2^2 - 7/30 s3^2` (picking jerk-squared) instead
of the order-2 PU Lagrangian. On the solution manifold the higher derivatives satisfy
`q^(6) = -5 q^(4) - 4 q̈` exactly, so the EL column of `q'''^2` is a linear combination of
lower-order columns *for every trajectory*. Stacking trajectories does not lift this — they
are all on-shell. The `dropKineticAliasColumns` filter removes the one exact alias it can
see; the rest need either off-manifold data (perturb trajectories away from the EOM, which
requires a different data source) or a hard prior that the Lagrangian order equals the
kinetic term's order. **Order-2 libraries work; order-≥3 libraries do not.** The equivalence-class
check (Phase 1.2) correctly flags every failed recovery as "not equivalent to the true
Lagrangian", so at least the failure is detectable.

### C. Blind-chain boundary site (Phase 1.1)

Even at 0 noise the last coordinate `q5` of the open chain is mis-recovered (a
`q5^2 v5^2` term shadows `q5^4`). Same root cause as A — a velocity term genuinely
correlates with the residual better than the true quartic once the other `q5` terms carry
small errors. The interior 5 sites recover exactly.

---

## Results (unchanged by the cleanup — re-verified)

### Immediate fix — readable Lagrangian
`finding_L/report.assembleDiscoveredLagrangian` → `DiscoveredLagrangian(expression,
rawExpression, kineticTerm, contributions, text)`. `expression` is a real sympy object in
clean `q0, v0, …` symbols with coefficients snapped to smallest-denominator rationals
within 1 % relative; `text` groups terms by degree and shared coefficient with a
`raw -> clean` table. Wired into `runDiscoveryStreaming`.

### Phase 1.1 blind holdout
Unseen system (anisotropic anharmonic chain: per-site stiffness, sparse nearest-neighbour
coupling, cubic + quartic). At 0 noise, all 6 stiffnesses, all 5 couplings, and cubics +
quartics on the 5 interior sites are **recovered exactly**; the boundary site `q5` fails
(see Still-open C).

### Phase 1.2 equivalence-class check
`experiments/equivalence_class.py`. `ΔL` → existing Euler–Lagrange derivation → identically
zero ⇒ genuine total-derivative degeneracy (also reconstructs `F` with `ΔL = dF/dt`);
nonzero ⇒ numerical slop. Constructed both-way tests pass. Used throughout Phase 2 to
interpret higher-derivative recoveries.

### Phase 1.3 noise curve
`results/noise_curve_*.{txt,png,json}`. **Recovers cleanly only for noise ≲ 1–2 %**
(see Still-open A for the mechanism).

Isotropic quartic calibration system, 6 DOF (position noise as % of signal std):

| noise | outcome | missing | spurious | max &#124;Δcoef&#124; |
|---|---|---|---|---|
| 0 % | recovered | 0 | 0 | 0.000 |
| 1 % | recovered | 0 | 0 | 0.001 |
| 2 % | failed | 7 | 19 | 0.26 |
| 5 % | failed | 20 | 20 | 0.51 |
| 10 % | failed | 21 | 11 | 0.57 |

Blind anharmonic chain, 6 DOF: fails at 0 % on the boundary site only (1 missing / 3
spurious, max Δcoef 0.016) and collapses fully from 1 % — it has no clean-recovery band at
all, i.e. the anisotropic/odd-parity system is strictly harder than the calibration system.

### Phase 2.1 / 2.2 — Ostrogradski
`generation/ostrogradski.py` — full operator `Σ (−d/dt)^k ∂L/∂q^(k)`, `pipelineSign` flag
to match the existing 2nd-order sign convention (regression-tested identical).
`generation/higher_order_integrator.py` — RK4 on `(q, q̇, q̈, q⃛)`. Pais–Uhlenbeck: EOM
`q'''' + (ω₁²+ω₂²)q'' + ω₁²ω₂²q` exact; recovered frequencies [0.995, 1.990] vs [1, 2];
Ostrogradski `H` conserved to 1e-10.

### Phase 2.3 — noisy higher-order differentiation
`results/differentiation_study.{txt,png}`. Finite differences unusable at order ≥ 2 under
any noise (`error ∝ noise/dt^k`). Quintic **smoothing spline with GCV-style λ is the only
method that survives to 3rd/4th order** — jerk / snap relative error ~2 % / ~18 % at 1 %
position noise. Savitzky–Golay usable to ~3 %.

### Phase 2.4 — jerk/snap library
Correct-order library: recovers the PU Lagrangian *up to a total derivative*
(`q q''` in place of `−q'^2`), confirmed by the Phase 1.2 check, robust to ~3 % noise.
Jerk-extended library: fails (Still-open B).

### Phase 3 — ghost detection
`generation/ghost_detection.detectGhost` — Ostrogradski `H`, Hessian definiteness, and
EOM characteristic roots. Healthy oscillator → no ghost; PU → ghost; PU + boundary term →
still ghost (verdict invariant under equivalence-class freedom). Run on the Lagrangian
*recovered from data*, the ghost verdict is **stable to ≥ 35 % measurement noise** — the
signature (indefinite `H` + oscillatory dynamics) is qualitative and survives large
coefficient errors. Conditional on correct model-order identification, which (Still-open A)
is itself the fragile step.

---

## 2026-08-28 — Items 1–7 (hardening + docs)

Append-only entry. Nothing above this line was rewritten. Companion docs added:
[`../README.md`](../README.md), [`../PROJECT.md`](../PROJECT.md),
[`../LOGIC.md`](../LOGIC.md), [`../ForwardSelection.md`](../ForwardSelection.md).

### 1. Blind-holdout discipline made real
- `experiments/discovery.py`: `FROZEN_TOLERANCES` — the **finding_L library
  default** tolerances (`correlationCutoff` 0.1, `residualRmsTolerance` 0.01,
  `pruneRelativeThreshold` 1e-2, `stagnationTolerance` 0.01, `stagnationPatience`
  3, `degreeCap` 4), written out verbatim. **No calibration / tuning search was
  run** — the wording everywhere says "frozen defaults", not "calibrated".
- Provenance bug fixed: the old `LOCKED_TOLERANCES["degreeCap"]` read the
  `checkDegreeExpansionNeeded` signature default (6) while runs used 4. Now a
  single value (4, the degree of both quartic benchmarks) that is both reported
  and used.
- `correlationCutoff` and `pruneRelativeThreshold` are now real parameters of
  `runDiscoveryStreaming` and are threaded from `FROZEN_TOLERANCES`; previously
  only the function defaults were reachable, so "locking" them was unenforceable.
- `PhysicalSystem` lost its `degreeCap` and `residualRmsTolerance` fields — there
  are no per-system tolerances. `startingMaxDegree` / `maxRounds` remain but are
  documented as search budgets; `maxRounds` raised to 150 for every system so a
  stopping condition, never the round cap, decides success/failure.
- `runSystemDiscovery` returns `(discovered, logFrame, tolerancesUsed)` and the
  sweep asserts `tolerancesUsed == FROZEN_TOLERANCES` for every run — the holdout
  provably uses the identical set. `tests/test_frozen_tolerances.py` locks this
  in.
- `noise_robustness_sweep.py` prints `frozenTolerancesReport()` and a
  `ROLE: REFERENCE SYSTEM` / `ROLE: BLIND HOLDOUT` banner, plus an `equiv?` column
  and the full per-noise-level `EquivalenceVerdict`.

### 2. Equivalence-class check consolidated + order-aware
- `experiments/equivalence_class.py` → `finding_L/equivalence_class.py`.
- `eulerLagrangeResidual` now dispatches on order: order 1 uses the ordinary
  `EulerLagrangeEqn`; order ≥ 2 uses the full Ostrogradski operator
  `eulerLagrangeExpression`. So it can judge Phase 2 (higher-derivative)
  recoveries, not just 2nd-order.
- `experiments/discovery.compareToExpected` runs `classifyLagrangianPair` on
  every discovered-vs-expected pair and returns the `EquivalenceVerdict` on
  `RecoveryComparison`; `formatComparison` and the sweep artifacts surface it.
- The inline null-Lagrangian checks in `higher_order_discovery_validation.py` and
  `jerk_snap_distractor_study.py` now call `isNullLagrangian` — no parallel
  copies. (`ghost_detection_validation.py` has no such inline check.)

### 3. Renamed experiment scripts (cosmetic)
`phase1_noise_curve → noise_robustness_sweep`,
`phase2_differentiation_study → differentiation_method_study`,
`phase2_pu_oscillator → pu_oscillator_validation`,
`phase2_jerk_distractor → jerk_snap_distractor_study`,
`phase2_higher_order_discovery → higher_order_discovery_validation`,
`phase3_ghost_detection → ghost_detection_validation`. No Python module imported
any of them. `[2.1]` / `[3.1]` / `Phase 2.4` labels in report text replaced with
descriptive ones. (The "How to run" block above still lists the old names — see
`README.md` for the current commands; this log is not rewritten.)

### 4. `ostrogradskiHamiltonian` branch safety
Raises `NonUniqueTopDerivativeError` (carrying `.branches`) when `sp.solve`
returns >1 branch, i.e. `L` is nonlinear in its highest derivative and the
Legendre transform is multi-valued. All existing systems (quadratic in the top
derivative) are unaffected.

### 5. `fitActiveCoefficientsFromGram` lstsq fallback
Now emits a `RuntimeWarning` ("singular active Gram block ... coefficients on
collinear terms are not individually identifiable") when it falls back from
`np.linalg.solve` to `lstsq`. Behaviour otherwise unchanged.

### 6. Test suite — `tests/` (`uv run pytest`, 19 tests)
`test_ostrogradski.py`, `test_pu_oscillator.py`, `test_equivalence_class.py`,
`test_gram_forward_select.py`, `test_frozen_tolerances.py`. Added `pytest` to the
`dev` dependency group and `[tool.pytest.ini_options] pythonpath = ["src"]`.

### 7. Docs
`README.md` (was empty), `PROJECT.md`, `LOGIC.md`, `ForwardSelection.md` created
with fixed, non-overlapping scope. The "Still-open A/B/C" material is carried
forward in `PROJECT.md`; the copy above is the earlier snapshot and stays as
history.

### Numerical results — unchanged
Items 1–7 are plumbing / reporting / tests. Threaded tolerances equal the old
effective defaults; `degreeCap` stays 4; `maxRounds` only grew (every run still
terminates via Condition A/B/C, never the cap). Re-ran the full Phase 1 noise
sweep for both systems and every Phase 2/3 study; results are identical to the
"Results" section above:

| run | baseline (miss / spur / max\|Δcoef\|) | re-run | 
|---|---|---|
| isotropic 0 % | recovered 0 / 0 / 0.000 | recovered 0 / 0 / 0.0000 |
| isotropic 1 % | recovered 0 / 0 / 0.001 | recovered 0 / 0 / 0.0006 |
| isotropic 2 % | failed 7 / 19 / 0.26 | failed 7 / 19 / 0.2607 |
| isotropic 5 % | failed 20 / 20 / 0.51 | failed 20 / 20 / 0.5087 |
| isotropic 10 % | failed 21 / 11 / 0.57 | failed 21 / 11 / 0.5651 |
| anharmonic 0 % | 1 / 3 / 0.016 | failed 1 / 3 / 0.0159 |

PU EOM exact; Ostrogradski H drift 1.04e-10; higher-order recovery still PU-up-to-
total-derivative; jerk library still fails; ghost verdicts and eigenvalues
identical; differentiation breakdown table identical.

---

## 2026-08-29 — Items 8–14 (research features)

Append-only. Nothing above this line was changed. New modules: `generation/constraints.py`,
`finding_L/pipeline.py`, `finding_L/regularized_select.py`. Package `__init__.py`
added to `generation/`, `finding_L/`, `experiments/`.

### 8. Degenerate Lagrangians / Dirac constraints
`generation/constraints.py` — canonical `poissonBracket`, `weaklyVanishes` (reduce
modulo the constraint ideal via a Groebner basis), `classifyConstraints`.
`ostrogradski_hamiltonian.analyzeDegenerateLagrangian` — when `sp.solve` cannot
invert the top-momentum relation, it returns a `DegenerateLagrangianResult`
(primary constraints from the Hessian null space, the `{φ_a, φ_b}` bracket
matrix, first/second-class classification, `{φ, H}` consistency check flagging a
secondary constraint) **instead of raising**. `detectGhost` returns
`ghost=None, degenerate=True` with the constraint analysis attached.
Verified: `L = q1' q2 − ½q2² − ½q1²` → two second-class constraints, bracket
`[[0,−1],[1,0]]`; `L = ½q1'² + q1 q2' − …` → one constraint, `{φ,H} ≠ 0` → secondary
flagged. Dirac-bracket construction is out of scope. `tests/test_constraints.py` (4).

### 9. Two-field mixing testbed
`experiments/two_field_mixing.py` — `eig(M⁻¹K)` decides the normal-mode spectrum
(real → oscillatory, negative → runaway); an indefinite mass matrix (`|μ| ≥ 1`,
eigenvalues `1 ± μ`) forces a negative `ω²`. `buildEulerLagrangeMatrix` generalised
to `buildMultiFieldElMatrix` (one row-block per field). Recovers the coupled
**potential** including the off-diagonal `q0 q1` term exactly. The velocity
(mass-matrix) sector is not recoverable from on-shell data. `tests/test_two_field_mixing.py` (2).

### 10. Multi-coordinate higher-derivative discovery
`finding_L/higher_order_candidates.py` — `stateGridSymbols`, `multiFieldLibrary`,
`buildMultiFieldElMatrix`. `higher_order_discovery.recoverMultiFieldHigherOrderLagrangian`
— isotropic top-derivative kinetic fixed, forward selection recovers the rest.
On a position-coupled Pais–Uhlenbeck chain (2 and 3 fields) it recovers the
Lagrangian up to a total derivative **with the cross-field coupling coefficient
exact** on clean data and under direct column perturbation; the
noisy-position → spline path fails (multi-field higher-order differentiation).
`experiments/multi_field_discovery_validation.py`, `tests/test_multi_field_discovery.py` (4).

### 11. Automatic Lagrangian-order inference
`higher_order_discovery.inferLagrangianOrder` — for orders 1..maxOrder, measures
the least-squares residual of projecting the `q^(n)²` kinetic EL column onto the
other EL columns (a feasibility test for an order-`n` Euler-Lagrange equation).
Smallest order below tolerance (Condition A), else where the residual stopped
improving (Condition C). Infers PU → order 2, anharmonic oscillator → order 1.
A purely linear order-1 system (SHO) is a degenerate case (`EL(q'²) ∝ EL(q²)`),
documented. `experiments/order_inference_validation.py`, `tests/test_order_inference.py` (3).

### 12. End-to-end pipeline on noisy positions only
`finding_L/pipeline.endToEndPipeline(noisyPositions, dt)` — grid-searches the
differentiation methods (Savitzky–Golay, SG poly-8, quintic spline), infers the
order, recovers the Lagrangian, runs the ghost verdict, and returns **separate**
order / ghost / coefficient confidences. Method selection is unsupervised: among
plausible recoveries (no absurd coefficients) pick the lowest own-EL residual.
No step is given ground-truth derivatives. On PU: order 2 and ghost True with full
cross-method agreement through ≥ 1% noise; coefficients drift (reported as low
coefficient confidence). `recoverHigherOrderLagrangian` gained a `kineticLevel`
param so order 1 works. `experiments/end_to_end_pipeline_validation.py`, `tests/test_pipeline.py` (3).

### 13. ROC-style ghost-detection validation
`ghost_detection_validation.rocReport` — a labelled battery (3 healthy + 4 ghost
single-coordinate systems, plus a degenerate borderline case), run through the
end-to-end pipeline over a noise sweep with multiple seeds. Reports false-positive,
false-negative and undetermined rates. **FP = FN = 0** across 0–1% noise; the
undetermined rate rises 0.38 → 0.57 (nonlinear recovered `H`, or noise-degraded
structure). The degenerate system is correctly flagged, not scored.
`tests/test_ghost_roc.py` (4).

### 14. Regularisation-path model selection (additive)
`finding_L/regularized_select.py` — `sequentialThresholdedLeastSquares` (SINDy
STLSQ) and `lassoSelect` (coordinate-descent LASSO path + debiased refit), **both
from the Gram matrix** (no `Θ`, no new dependency). `experiments/model_selection_comparison.py`
runs all three selectors on one degree-4 streaming Gram per system/noise.

**Finding — the ~1–2% ceiling is the greedy selector, not OLS recovery.** On both
benchmark systems (one seed per level):

| noise | greedy | STLSQ | debiased LASSO |
|---|---|---|---|
| 1% | exact | exact | exact |
| 2% | failed (7/19, 6/6) | **exact** | **exact** |
| 5% | failed | 0 miss / ~10 spurious | **exact** |

The production path (`main_streaming` → `gram_forward_select`) is unchanged.
LASSO is a strong replacement candidate; a switch should first be validated over
more seeds / systems / the streaming degree-expanding path. `tests/test_regularized_select.py` (4).

### Effect on earlier results
Items 8–14 add modules and one parameter (`recoverHigherOrderLagrangian(kineticLevel=…)`,
default `min(2, order)` = the previous hard-coded 2 for orders ≥ 2). The Phase 1
noise sweep, `pu_oscillator_validation`, `differentiation_method_study`,
`higher_order_discovery_validation`, `jerk_snap_distractor_study` and the
`ghost_detection_validation` reference/noise-boundary sections are unchanged.
Test count 19 → 43.
