# LOGIC

End-to-end walkthrough of how a trajectory CSV becomes a Lagrangian (and a
ghost verdict). File references are `src/`-relative.

## 0. Symbols and conventions

`generation/eqnofmotion.py` defines the single global time symbol
`TIME = Symbol("t")` and `defineCoordinates(n)` → `q_i(t)` functions and their
first derivatives. Every other module imports `TIME` from here.

Two symbol spaces are used and mapped back and forth in `finding_L/report.py`:

- **functional** form — `q0(t)`, `Derivative(q0(t), t)` — used for all calculus
  (Euler–Lagrange derivatives, total time derivatives).
- **state** form — plain symbols `q0, v0, q1, v1, …` (and `s0, s1, s2, …` for
  the higher-derivative track) — used for the numeric regression and the
  readable output.

## 1. Dataset generation

`experiments/systems.py` holds `PhysicalSystem` records: a symbolic Lagrangian
builder, the expected scaled Lagrangian (kinetic coefficient normalised to 1),
and integration parameters. `experiments/generate_dataset.py`:

1. `generation/integrator.GetAccelFunctions` derives `q̈_i = f(q, q̇)` by solving
   the Euler–Lagrange system for the accelerations and lambdifying.
2. `generation/generate_data.generateDatasetStreaming` integrates many seeded
   random initial conditions with RK4, adds Gaussian noise scaled per-column by
   `noisePercentage · std`, and streams rows to CSV
   (`trajectory_id, t, q0, q0dot, q0ddot, …`).

## 2. Candidate library

`finding_L/candidates.buildCandidateLibrary(coords, vels, maxDegree)` — all
monomials in `(q_i, q̇_i)` of total degree `1..maxDegree`, de-duplicated by
exponent tuple. `filterPureVelocityTerms` drops monomials with no positional
dependence (they contribute nothing to the EL residual structure being fit).
The kinetic term `Σ q̇_i²` is appended explicitly and is the regression target.

## 3. Streaming Gram matrix

`finding_L/build_matrix.py`. For each candidate monomial `θ`, its Euler–Lagrange
column is `EL(θ) = d/dt ∂θ/∂q̇ − ∂θ/∂q`, with `q̈_i` substituted as an
independent data symbol. These columns are lambdified once (cached across degree
expansions via `lambdifiedCache`), then evaluated on the CSV in row chunks and
sub-chunked to a cell budget so the dense `Θ` block never exceeds ~1 GB. The
accumulators are the row count `n`, the column sums, and `G = Θᵀ Θ`.

The regression is: find sparse `c` such that
`EL(kinetic) + Σ_j c_j EL(θ_j) ≈ 0` on the data — i.e. the discovered `L`
satisfies the observed equations of motion. Everything downstream needs only
`G` and `b = −G[:, kinetic]`.

## 4. Forward selection

`finding_L/main_streaming.runDiscoveryStreaming`. Each round:

1. Refit the active coefficients from the Gram sub-block
   (`fitActiveCoefficientsFromGram`; `lstsq` fallback with a `RuntimeWarning`
   if the block is singular).
2. Compute the scaled residual `‖r‖ / ‖EL(kinetic)‖`.
3. **Stop — Condition A** if the scaled residual is below
   `residualRmsTolerance`.
4. **Stop — Condition C** if the residual has improved by less than
   `stagnationTolerance` for `stagnationPatience` consecutive rounds.
5. Score every reserve candidate by its correlation with the current residual
   (`scoreReserveCandidatesFromGram`) and take the best.
6. If the best score is below `correlationCutoff`: try a degree expansion
   (`checkDegreeExpansionNeeded`, capped at `degreeCap`); re-stream the Gram
   matrix for the new columns and continue, or **stop — Condition B** if the
   cap is reached.
7. Otherwise add the best candidate to the active set.

After the loop, `pruneNearZeroCoefficients` iteratively drops active terms whose
coefficient is below `pruneRelativeThreshold · max|c|` and refits.

See `ForwardSelection.md` for the scoring formula and the tolerance lock.

## 5. Readable report

`finding_L/report.assembleDiscoveredLagrangian` produces `DiscoveredLagrangian`:

- `rawExpression` — state-symbol Lagrangian with the raw fitted floats.
- `expression` — coefficients snapped to the nearest small-denominator rational
  within 1 % relative (`snapCoefficient`), zeros dropped.
- `text` — terms grouped by degree and shared coefficient, plus a
  `raw -> clean` snapping table.

## 6. Equivalence-class check

`finding_L/equivalence_class.classifyLagrangianPair(A, B, coords, vels,
order=None)`:

1. `ΔL = expand(A − B)`. If `ΔL == 0` → identical.
2. `isNullLagrangian(ΔL, …)` applies the full Ostrogradski Euler–Lagrange
   operator (order resolved from the highest derivative present, so this works
   for higher-derivative `L` too) and checks every component reduces to `0`
   (`expand`, then `simplify`).
3. If null and first-order, `reconstructBoundaryPotential` recovers `F` with
   `ΔL = dF/dt` by integrating the velocity coefficients (curl-free check first).

The verdict (`EquivalenceVerdict`: `equivalent`, `difference`,
`eulerLagrangeResidual`, `boundaryPotential`, `detail`) is:

- **equivalent** — `ΔL == 0` (exact structural match) or `ΔL` is a nonzero null
  Lagrangian (differ only by a total time derivative — the same physics).
- **not equivalent** — `ΔL` has a nonzero Euler–Lagrange residual: the two
  Lagrangians produce different equations of motion, so accepting both as "the
  same" would mean the acceptance tolerances are too loose.

`experiments/discovery.compareToExpected` runs this on every
discovered-vs-expected comparison and returns it on `RecoveryComparison`
(`.equivalenceVerdict`, `.structurallyEquivalent`);
`experiments/noise_robustness_sweep.py` surfaces it as the `equiv?` column
(`exact` / `null-L` / `no`). The higher-derivative validation studies
(`higher_order_discovery_validation.py`, `jerk_snap_distractor_study.py`) call
`isNullLagrangian` directly instead of an ad-hoc inline check.

## 7. Higher-derivative track

Same shape as steps 2–5 but single-coordinate and with an explicit Lagrangian
order:

- `finding_L/higher_order_candidates.py` — library of monomials in
  `(q, q', q'', …)` state symbols; each candidate's EL column is built with the
  full Ostrogradski operator and evaluated on derivative arrays. A narrow
  cosine-similarity filter (`dropKineticAliasColumns`) removes columns collinear
  with the kinetic column (the exact `q''² ↔ q' q'''` alias).
- `finding_L/higher_order_discovery.recoverHigherOrderLagrangian` — dense Gram
  matrix, `forwardSelectFromGram`, coefficient snapping.

Noisy derivatives come from `generation/numerical_diff.py` — the smoothing
spline is the only method that survives to 3rd/4th order.

## 8. Ghost detection

`generation/ghost_detection.detectGhost`:

1. Build the Ostrogradski Hamiltonian `H` (`ostrogradski_hamiltonian.py`).
2. If `H` is quadratic in the phase-space variables, take the Hessian and count
   negative eigenvalues; otherwise report "needs nonlinear boundedness
   analysis".
3. Compute the EOM characteristic roots and classify the dynamics
   (oscillatory / runaway / damped).
4. **Ghost** iff `H` is indefinite *and* the dynamics are oscillatory — a
   bounded-motion negative-energy mode, the Ostrogradski signature. An
   indefinite `H` with runaway dynamics is a tachyon/instability, not a ghost.

The verdict is invariant under the equivalence-class freedom: PU and
PU + a total derivative both return `ghost = True`.
