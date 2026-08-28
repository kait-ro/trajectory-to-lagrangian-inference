# finding_L — hardening, higher-derivative extension, cleanup

Work log covering the readable-Lagrangian fix, the ordinary and higher-derivative
discovery tracks, ghost detection, a cleanup pass, and the follow-up hardening
(blind-holdout discipline, equivalence-class wiring, a pytest suite, file
renames). Everything here is reproducible with the commands in
[How to run](#how-to-run). Artifacts land in `src/experiments/results/`.

> **Doc map.** Project overview → [`PROJECT.md`](../PROJECT.md); end-to-end
> pipeline → [`LOGIC.md`](../LOGIC.md); the sparse-regression method and its
> locked tolerances → [`ForwardSelection.md`](../ForwardSelection.md). This file
> is the chronological log.

## Hardening pass

- **Blind-holdout discipline** (`experiments/discovery.py`,
  `experiments/noise_robustness_sweep.py`). All discovery tolerances are
  calibrated on `isotropic_quartic_calibration` and frozen in
  `LOCKED_TOLERANCES`. `runSystemDiscovery(..., enforceLocked=True)` threads the
  frozen values into `runDiscoveryStreaming` (which now takes `correlationCutoff`
  explicitly) and raises if a non-calibration system overrides a locked
  tolerance. The sweep prints the locked values and a `ROLE: CALIBRATION` /
  `ROLE: BLIND HOLDOUT` banner into every artifact, plus an `equiv?` column.
- **Equivalence-class check wired in.** `equivalence_class.py` moved
  `experiments/ → finding_L/` and generalised to any Lagrangian order (full
  Ostrogradski EL operator, order auto-resolved).
  `experiments/discovery.compareToExpected` now calls `classifyLagrangianPair`
  on every discovered-vs-expected comparison and returns the
  `EquivalenceVerdict` on `RecoveryComparison`. The higher-derivative validation
  studies call `isNullLagrangian` instead of ad-hoc inline residual checks.
- **`ostrogradskiHamiltonian` branch safety.** Raises
  `NonUniqueTopDerivativeError` when `L` is nonlinear in its highest derivative
  (multi-valued Legendre transform) instead of silently taking `solutions[0]`.
- **`fitActiveCoefficientsFromGram`** emits a `RuntimeWarning` when it falls back
  to `lstsq` (singular active Gram block).
- **Test suite** — `tests/` (`uv run pytest`): SHO Euler–Lagrange + Ostrogradski
  Hamiltonian, Pais–Uhlenbeck EOM + Hamiltonian conservation, the
  equivalence-class classifier both ways, forward selection on a synthetic Gram
  matrix.
- **File renames** — the `phaseN_*` experiment scripts are renamed to
  descriptive names (`phase1_noise_curve → noise_robustness_sweep`,
  `phase2_pu_oscillator → pu_oscillator_validation`, etc.); phase-number labels
  removed from report text.

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

# --- equivalence-class machinery ---
$PY -m finding_L.equivalence_class                        # constructed null-Lagrangian checks

# --- ordinary (2nd-order) discovery: calibration + blind holdout ---
$PY -m experiments.noise_robustness_sweep isotropic_quartic_calibration   # calibration system
$PY -m experiments.noise_robustness_sweep anharmonic_chain_blind          # blind holdout, locked tolerances

# --- higher-derivative (Ostrogradski) track ---
$PY -m experiments.pu_oscillator_validation               # Ostrogradski EL / Hamiltonian smoke test
$PY -m experiments.differentiation_method_study           # noisy higher-order differentiation
$PY -m experiments.higher_order_discovery_validation      # correct-order library recovery
$PY -m experiments.jerk_snap_distractor_study             # jerk/snap distractor library

# --- ghost detection ---
$PY -m experiments.ghost_detection_validation             # ghost verdict vs measurement noise
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
  `equivalence_class` each had their own). `equivalence_class.py` later moved to
  `finding_L/`.
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

`jerk_snap_distractor_study` still recovers `s2^2 - 7/30 s3^2` (picking jerk-squared) instead
of the order-2 PU Lagrangian. On the solution manifold the higher derivatives satisfy
`q^(6) = -5 q^(4) - 4 q̈` exactly, so the EL column of `q'''^2` is a linear combination of
lower-order columns *for every trajectory*. Stacking trajectories does not lift this — they
are all on-shell. The `dropKineticAliasColumns` filter removes the one exact alias it can
see; the rest need either off-manifold data (perturb trajectories away from the EOM, which
requires a different data source) or a hard prior that the Lagrangian order equals the
kinetic term's order. **Order-2 libraries work; order-≥3 libraries do not.** The
equivalence-class check (`finding_L/equivalence_class.isNullLagrangian`, called directly by
`jerk_snap_distractor_study.py`) correctly flags every failed recovery as "not equivalent to
the true Lagrangian", so at least the failure is detectable.

### C. Blind-chain boundary site (holdout: interior vs. boundary)

Even at 0 noise the last coordinate `q5` of the open chain is mis-recovered (a
`q5^2 v5^2` term shadows `q5^4`). Same root cause as A — a velocity term genuinely
correlates with the residual better than the true quartic once the other `q5` terms carry
small errors. The interior 5 sites recover exactly.

---

## Results (unchanged by the cleanup / hardening — re-verified)

### Immediate fix — readable Lagrangian
`finding_L/report.assembleDiscoveredLagrangian` → `DiscoveredLagrangian(expression,
rawExpression, kineticTerm, contributions, text)`. `expression` is a real sympy object in
clean `q0, v0, …` symbols with coefficients snapped to smallest-denominator rationals
within 1 % relative; `text` groups terms by degree and shared coefficient with a
`raw -> clean` table. Wired into `runDiscoveryStreaming`.

### Blind holdout
Unseen system (anisotropic anharmonic chain: per-site stiffness, sparse nearest-neighbour
coupling, cubic + quartic), scored with the tolerances locked on the calibration system.
At 0 noise, all 6 stiffnesses, all 5 couplings, and cubics + quartics on the 5 interior
sites are **recovered exactly**; the boundary site `q5` fails (see Still-open C).
`noise_robustness_sweep.py` marks this run `ROLE: BLIND HOLDOUT` and would raise if the
system tried to retune a locked tolerance.

### Equivalence-class check
`finding_L/equivalence_class.py` (moved from `experiments/`, generalised to any Lagrangian
order). `ΔL` → full Ostrogradski Euler–Lagrange operator → identically zero ⇒ genuine
total-derivative degeneracy (also reconstructs `F` with `ΔL = dF/dt` for first-order `ΔL`);
nonzero ⇒ physically distinct. Constructed both-way tests pass (`tests/test_equivalence_class.py`).
Wired into `experiments/discovery.compareToExpected` (verdict on `RecoveryComparison`,
`equiv?` column in the sweep) and called directly by the higher-derivative validation studies.

### Noise curve
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

### Ostrogradski EL / integrator (`pu_oscillator_validation.py`)
`generation/ostrogradski.py` — full operator `Σ (−d/dt)^k ∂L/∂q^(k)`, `pipelineSign` flag
to match the existing 2nd-order sign convention (regression-tested identical).
`generation/higher_order_integrator.py` — RK4 on `(q, q̇, q̈, q⃛)`. Pais–Uhlenbeck: EOM
`q'''' + (ω₁²+ω₂²)q'' + ω₁²ω₂²q` exact; recovered frequencies [0.995, 1.990] vs [1, 2];
Ostrogradski `H` conserved to 1e-10. `ostrogradskiHamiltonian` raises
`NonUniqueTopDerivativeError` if `L` is nonlinear in its highest derivative. Asserted in
`tests/test_ostrogradski.py`, `tests/test_pu_oscillator.py`.

### Noisy higher-order differentiation (`differentiation_method_study.py`)
`results/differentiation_study.{txt,png}`. Finite differences unusable at order ≥ 2 under
any noise (`error ∝ noise/dt^k`). Quintic **smoothing spline with GCV-style λ is the only
method that survives to 3rd/4th order** — jerk / snap relative error ~2 % / ~18 % at 1 %
position noise. Savitzky–Golay usable to ~3 %.

### Jerk/snap library (`jerk_snap_distractor_study.py`)
Correct-order library: recovers the PU Lagrangian *up to a total derivative*
(`q q''` in place of `−q'^2`), confirmed by the equivalence-class check, robust to ~3 % noise.
Jerk-extended library: fails (Still-open B).

### Ghost detection (`ghost_detection_validation.py`)
`generation/ghost_detection.detectGhost` — Ostrogradski `H`, Hessian definiteness, and
EOM characteristic roots. Healthy oscillator → no ghost; PU → ghost; PU + boundary term →
still ghost (verdict invariant under equivalence-class freedom). Run on the Lagrangian
*recovered from data*, the ghost verdict is **stable to ≥ 35 % measurement noise** — the
signature (indefinite `H` + oscillatory dynamics) is qualitative and survives large
coefficient errors. Conditional on correct model-order identification, which (Still-open A)
is itself the fragile step.
