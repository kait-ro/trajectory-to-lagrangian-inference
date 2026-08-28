# LOGIC

How a trajectory CSV becomes a Lagrangian and a ghost verdict, and why each step
is there. Paths are `src/`-relative. The forward-selection step has its own
document, [`ForwardSelection.md`](ForwardSelection.md).

## Symbol conventions

`generation/eqnofmotion.py` owns the single time symbol `TIME = Symbol("t")` and
`defineCoordinates(n)` → `q_i(t)` functions plus their first derivatives.

Two symbol spaces, mapped in `finding_L/report.py`:

- **functional** (`q0(t)`, `Derivative(q0(t), t)`) — used for all calculus
  (Euler–Lagrange derivatives, total time derivatives). *Why:* differentiation
  w.r.t. time and w.r.t. the coordinate only makes sense when the coordinate is a
  function of `t`.
- **state** (`q0, v0, …`; `s0, s1, s2, …` for the higher-derivative track) —
  used for the numeric regression and the readable output. *Why:* the Gram
  matrix is built by evaluating expressions on columns of data; plain symbols
  lambdify cleanly and read well.

## 1. Dataset generation — `experiments/generate_dataset.py`, `generation/`

`experiments/systems.py` holds `PhysicalSystem` records (a symbolic Lagrangian
builder, the expected scaled Lagrangian, integration parameters).

1. `generation/integrator.GetAccelFunctions` solves the Euler–Lagrange system for
   the accelerations `q̈_i = f(q, q̇)` and lambdifies them.
2. `generation/generate_data.generateDatasetStreaming` integrates many seeded
   random initial conditions with RK4, adds Gaussian noise scaled per column by
   `noisePercentage · std`, and streams rows to CSV.

*Why symbolic-then-integrate:* the ground truth is a known `L`, so recovery can
be scored exactly; the noise model is explicit and reproducible.

## 2. Candidate library — `finding_L/candidates.py`

All monomials in `(q_i, q̇_i)` of total degree `1..maxDegree`, de-duplicated by
exponent tuple; pure-velocity monomials are dropped (they add nothing to the EL
residual structure). The kinetic term `Σ q̇_i²` is appended and is the regression
target.

*Why a monomial library:* it turns "find a function" into "find sparse
coefficients over a fixed basis" — a linear problem once the EL operator is
applied.

## 3. Streaming Gram matrix — `finding_L/build_matrix.py`

For each candidate `θ`, its Euler–Lagrange column `EL(θ) = d/dt ∂θ/∂q̇ − ∂θ/∂q`
is formed (with `q̈_i` substituted as an independent data symbol), lambdified
once (cached across degree expansions), evaluated on the CSV in row chunks, and
sub-chunked to a cell budget. The accumulators are the row count, the column
sums, and `G = Θᵀ Θ`.

The model being fit: the discovered `L` must satisfy the observed equations of
motion, i.e.

```
EL(kinetic) + Σ_j c_j · EL(θ_j) ≈ 0     over all data rows
```

*Why streaming:* the dense `Θ` for a degree-4 expansion on 6 DOF is ~23 GB.
`G` is `n_candidates × n_candidates` and everything downstream needs only `G` and
`b = −G[:, kinetic]`.

## 4. Forward selection — `finding_L/main_streaming.py` + `gram_forward_select.py`

Greedy, OMP-style: each round add the reserve candidate most correlated with the
current residual, refit the active coefficients from the Gram sub-block, and stop
on one of three conditions (A converged / B stalled at the degree cap / C
residual stagnated). The library degree is expanded on demand up to `degreeCap`.
Afterwards `pruneNearZeroCoefficients` drops active terms below a relative
threshold and refits.

*Why greedy:* the full best-subset problem is combinatorial; greedy forward
selection with a correlation score is cheap and, in the clean-data regime,
exact. Its failure mode under noise — and the regularisation-path selectors that
do better — are in [`PROJECT.md`](PROJECT.md) problem A.

Full algorithm, scoring formula, stopping conditions and the tolerance table:
[`ForwardSelection.md`](ForwardSelection.md).

## 5. Readable report — `finding_L/report.py`

`assembleDiscoveredLagrangian` → `DiscoveredLagrangian(expression, rawExpression,
…, text)`: `rawExpression` keeps the fitted floats; `expression` snaps each
coefficient to the nearest small-denominator rational within 1 % relative;
`text` groups terms by degree and shared coefficient.

*Why snapping:* physical Lagrangians have simple rational coefficients; snapping
turns `-0.24999` into `-1/4` and makes the equivalence check below exact rather
than approximate.

## 6. Equivalence-class check — `finding_L/equivalence_class.py`

`classifyLagrangianPair(A, B, coords, vels, order=None)`:

1. `ΔL = expand(A − B)`. `ΔL == 0` → identical.
2. `isNullLagrangian(ΔL)` applies the Euler–Lagrange operator and checks every
   component reduces to `0` (`expand`, then `simplify`). **Order-aware:** order 1
   uses the ordinary EL operator (`generation.eqnofmotion.EulerLagrangeEqn`);
   order ≥ 2 uses the full Ostrogradski operator
   (`generation.ostrogradski.eulerLagrangeExpression`). Order defaults to the
   highest derivative present.
3. If null and first-order, `reconstructBoundaryPotential` recovers `F` with
   `ΔL = dF/dt` (curl-free check, then integrate the velocity coefficients).

Verdict: **equivalent** if `ΔL == 0` or `ΔL` is a nonzero null Lagrangian (same
physics, differ by a total time derivative); **not equivalent** if `ΔL` has a
nonzero EL residual (different equations of motion).

*Why not just compare coefficients:* two Lagrangians can differ by `d/dt F` and
be physically identical (e.g. `q q̈` vs `−q̇²`), and two Lagrangians with the same
monomials but slightly-off coefficients are *not* the same theory. Coefficient
proximity answers neither question; the EL operator answers both.

Wiring: `experiments/discovery.compareToExpected` runs it on every
discovered-vs-expected pair; `noise_robustness_sweep.py` shows the verdict per
noise level; `higher_order_discovery_validation.py` and
`jerk_snap_distractor_study.py` call `isNullLagrangian` directly.

## 7. Higher-derivative track — `finding_L/higher_order_*.py`, `generation/ostrogradski*.py`

Same shape as steps 2–5, single-coordinate, with an explicit Lagrangian order:

- `higher_order_candidates.py` — monomials in `(q, q', q'', …)`; each candidate's
  EL column uses the full Ostrogradski operator. `dropKineticAliasColumns`
  removes columns collinear with the kinetic column (the exact
  `q''² ↔ q' q'''` alias).
- `higher_order_discovery.recoverHigherOrderLagrangian` — dense Gram matrix,
  `forwardSelectFromGram`, coefficient snapping.

Noisy derivatives come from `generation/numerical_diff.py`; the quintic smoothing
spline is the only method that survives to 3rd/4th order.

The multi-field generalisation (section 10) and automatic order inference
(section 10) build on this same machinery.

## 8. Ghost detection — `generation/ghost_detection.py`

1. Build the Ostrogradski Hamiltonian `H` (`ostrogradski_hamiltonian.py`).
   Raises `NonUniqueTopDerivativeError` if `L` is nonlinear in its highest
   derivative (the Legendre transform is then multi-valued and the physical
   branch is a caller decision).
2. If `H` is quadratic in the phase-space variables, take its Hessian and count
   negative eigenvalues; otherwise report "needs nonlinear boundedness
   analysis".
3. Compute the EOM characteristic roots; classify the dynamics (oscillatory /
   runaway / damped).
4. **Ghost** iff `H` is indefinite *and* the dynamics are oscillatory — a
   bounded-motion negative-energy mode, the Ostrogradski signature. Indefinite
   `H` with runaway dynamics is a tachyon, not a ghost.

*Why the two-part test:* the Ostrogradski theorem guarantees a linear-in-momentum
term in `H` for a non-degenerate higher-derivative theory, which makes `H`
unbounded below; the oscillatory-dynamics condition rules out the trivial
unbounded-but-harmless runaway case. The verdict is invariant under the
total-derivative freedom from step 6.

## 9. Degenerate Lagrangians — `generation/constraints.py`

When `ostrogradskiHamiltonian` cannot invert `p_n = ∂L/∂q^(n)` for `q^(n)` (the
Lagrangian is degenerate), it returns a `DegenerateLagrangianResult` instead of a
Hamiltonian dict:

1. Primary constraints come from the null space of the Hessian
   `W_ab = ∂²L/∂q_a^(n) ∂q_b^(n)`: each null vector `v` gives a phase-space
   relation `Σ_a v_a (P_a − ∂L/∂q_a^(n)) = 0` (the `q^(n)` parts cancel because
   `W v = 0`).
2. `classifyConstraints` builds the canonical Poisson-bracket matrix
   `C_ab = {φ_a, φ_b}`. A constraint whose whole row of `C` vanishes *weakly*
   (modulo the constraint ideal — checked with a Gröbner reduction) is
   first-class; otherwise second-class.
3. For a first-class candidate, `{φ_a, H}` is also tested weakly. If it does not
   vanish, consistency requires a secondary constraint — flagged, not computed.

*Why stop here:* first/second-class classification already tells you the physical
phase-space dimension and whether there is gauge freedom. Dirac-bracket
construction and the full Dirac–Bergmann iteration are a separate, larger job.

## 10. Multi-field and order-inferring recovery — `finding_L/higher_order_*.py`

- `buildMultiFieldElMatrix` stacks one row-block per field (rows = timepoints ×
  fields), exactly as the 2nd-order streaming path stacks per-coordinate blocks.
  `recoverMultiFieldHigherOrderLagrangian` fixes the isotropic top-derivative
  kinetic `Σ_i (q_i^(n))²` and forward-selects the rest, including cross-field
  coupling monomials.
- `inferLagrangianOrder` tries orders `1..maxOrder`; for each it measures the
  least-squares residual of projecting the `q^(n)²` kinetic EL column onto the
  span of the other EL columns (a feasibility test: does the data satisfy an
  order-`n` Euler–Lagrange equation?). Smallest order below tolerance
  (Condition A), else the order after which the residual stops improving
  (Condition C).

## 11. End-to-end pipeline — `finding_L/pipeline.py`

`endToEndPipeline(noisyPositions, dt)` chains the pieces with no ground-truth
input:

1. For each differentiation method (Savitzky–Golay, SG poly-8, quintic spline):
   estimate derivatives → `inferLagrangianOrder` → `recoverHigherOrderLagrangian`
   → `detectGhost`, and record the recovered Lagrangian's *own* Euler–Lagrange
   residual on that method's derivatives.
2. Consensus order = majority vote. Selection = among recoveries with plausible
   (not absurdly large) coefficients, the lowest own-EL residual.
3. Three confidences: **order** (cross-method agreement), **ghost** (agreement on
   the verdict), **coefficient** (spread across methods + plausibility). Overall
   confidence is the order/ghost minimum — the robust part; coefficients are
   reported separately because the differentiation step limits them.
