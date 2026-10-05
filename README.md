# Trajectory-to-Lagrangian Inference

Recover the Lagrangian behind observed motion — from noisy data, honestly, and
far enough into higher-derivative territory to meet a real physical pathology.

---

## What it is

A research codebase for the **inverse problem of analytical mechanics**: given
trajectory data `q_i(t)` — in the hardest case only noisy positions — find a
Lagrangian `L` whose Euler–Lagrange equations reproduce the motion.

It is data-driven model discovery in the SINDy family, but the target is the
**Lagrangian**, not the equations of motion. `L` is where the physics is
actually stated — the symmetries, the Noether charges, the action that goes into
a path integral — so recovering it says more than fitting an ODE. It also makes
"did we get it right?" a subtler question: two Lagrangians that differ by a
total time derivative are the *same* theory, while the same monomials with
slightly-off coefficients are a *different* one. Coefficient proximity answers
neither; recovery is judged by an **equivalence-class check** —
is `EL(L_recovered − L_true) ≡ 0`?

Three things it does:

- **recover** `L` from trajectory data;
- **compare** two Lagrangians — same physical theory, or not;
- for higher-derivative `L`, **decide** whether the theory carries an
  Ostrogradski ghost.

Two questions motivate it. *Can Lagrangian recovery be done honestly under
measurement noise* — with a blind holdout, one frozen tolerance set for every
system, and an equivalence check instead of "the coefficients look close"? And
*what happens past 2nd order*, where recovery collides with the ghost below and
where getting the model order right becomes the fragile step?

---

## What's going on physically

The project runs in two directions, and a physical pathology sits at the end of
the second.

### Forward — `generation/`: a known `L` → data

Start from a symbolic Lagrangian. Its Euler–Lagrange equations give the
accelerations `q̈_i = f(q, q̇)`; integrate many randomised initial conditions with
RK4; add Gaussian measurement noise scaled to each signal. The output is a CSV of
noisy trajectories whose *true* Lagrangian is known — so any recovery can be
scored exactly, and the noise is a controlled, explicit quantity rather than an
unknown.

### Inverse — `finding_L/`: data → `L`

Write `L` as an unknown sparse combination of monomials in the coordinates and
velocities. Applying the Euler–Lagrange operator to each monomial turns "find a
function" into "find which basis terms are present and with what coefficients" —
a linear problem, because on any true trajectory the Euler–Lagrange combination
of the real terms vanishes. Recover that sparse set, snap the coefficients to
the simple rationals physical Lagrangians actually have, and check the result is
the same *theory* as the target, not just a lookalike.

### The higher-derivative problem

Higher-derivative (HD) Lagrangians — those depending on `q̈`, `q⃛`, … — turn up
in effective field theory, higher-curvature and modified gravity, and
Pais–Uhlenbeck-type regulators. **Ostrogradski's theorem** says a non-degenerate
HD Lagrangian has a Hamiltonian **unbounded below**: a negative-energy mode, the
"ghost", that destabilises the theory.

The project generates such systems, recovers their Lagrangian from noisy data,
and runs the **ghost verdict on the recovered model** — asking whether the
pathology is still visible after it has been through differentiation noise and
sparse regression. The benchmark is the **Pais–Uhlenbeck oscillator**,

```
L = ½ ( q̈²  −  (ω₁² + ω₂²) q̇²  +  ω₁² ω₂² q² )
```

whose fourth-order equation of motion has two normal modes, one of them a ghost.
It also handles **degenerate** HD Lagrangians, where the mass matrix is singular
and the ghost question needs Dirac constraint analysis — primary constraints,
first/second-class classification — rather than a plain Hessian.

---

## How it works

### Symbol conventions

`generation/eqnofmotion.py` owns the single time symbol `TIME = Symbol("t")` and
`defineCoordinates(n)` → `q_i(t)` functions plus their derivatives.
**Functional** symbols (`q0(t)`, `Derivative(q0(t), t)`) are used for all
calculus (Euler–Lagrange derivatives, total time derivatives) — differentiation
w.r.t. time and w.r.t. the coordinate only makes sense when the coordinate is a
function of `t`.

For numeric regression and readable output, plain **state** symbols are used
instead — they lambdify cleanly and read well. There are deliberately three
state-symbol conventions, not one, each the most readable choice for its case:
`q_i, v_i` for order-1 systems (the familiar physics notation, the most common
case); bare `s0, s1, s2, …` for single-field higher-derivative systems (no field
index, because there's only one field — adding one would be pure noise); and
`s{field}_{level}` for multi-field higher-derivative systems (the field index is
load-bearing there). Forcing all three onto one naming scheme was considered and
rejected — it would make the two most common cases strictly less readable for no
benefit, since nothing downstream actually needs the names unified. What *is*
unified is the formatting machinery that turns a discovered Lagrangian into
readable, grouped, coefficient-snapped text (`finding_L/report.py`, §5): it was
always naming-agnostic internally, just gated behind a `q_i/v_i`-specific entry
point, so all three conventions now get the same quality of output.

### 1. Dataset generation — `experiments/generate_dataset.py`, `generation/`

`experiments/systems.py` holds `PhysicalSystem` records (a symbolic Lagrangian
builder, the expected scaled Lagrangian, integration parameters).
`generation/integrator.GetAccelFunctions` solves the Euler–Lagrange system for
the accelerations `q̈_i = f(q, q̇)` and lambdifies them;
`generation/generate_data.generateDatasetStreaming` integrates many seeded random
initial conditions with RK4, adds Gaussian noise scaled per column by
`noisePercentage · std`, and streams rows to CSV. The ground truth is a known
`L`, so recovery can be scored exactly and the noise model is explicit.

### 2. Candidate library — `finding_L/higher_order_candidates.py`

One general library builder over `(order, noFields)`: `multiFieldLibrary` forms
all monomials of total degree `1..maxDegree` in the state-symbol grid
`s{field}_{level}` (`level = 0..order`), de-duplicated by exponent tuple, then
`multiFieldMonomialToCoordinates` maps each into functional space. Order 1 is
this construction's exact special case, not a separate code path:
`buildCandidateLibrary(coords, vels, maxDegree)` — the 2nd-order library, in
familiar `(q_i, q̇_i)` functional terms — is a thin wrapper around
`multiFieldLibrary(noFields, 1, maxDegree)`. Pure-velocity (more generally,
pure-top-derivative) monomials are dropped by `filterPureVelocityTerms`, and
this pre-filter matters more than it looks: the kinetic term `Σ q̇_i²` is *itself*
exactly the sum of the individual `q̇_i²` monomials, so without the filter the
regression can trivially "recover" `−Σ q̇_i²` from its own additive pieces and
stop there — a real linear degeneracy in the candidate space, not a redundant
safety check. The kinetic term is appended and is the regression target. This
turns "find a function" into "find sparse coefficients over a fixed basis" — a
linear problem once the EL operator is applied.

### 3. Gram-matrix construction — `finding_L/build_matrix.py`

For each candidate `θ`, its Euler–Lagrange column
`EL(θ) = Σ_k (−d/dt)^k ∂θ/∂q^(k)` (`generation/ostrogradski.py`'s general
operator, §6 — order 1 reduces exactly to `d/dt ∂θ/∂q̇ − ∂θ/∂q`) is formed with
the needed derivative levels substituted as independent data symbols, lambdified
once (cached across degree expansions), and evaluated in chunks. Accumulation —
row count `n`, column sums, `G = Θᵀ Θ` — is one shared, naming-agnostic core,
`accumulateGramFromChunks`: it only consumes `theta` chunks, however they were
produced, so it doesn't care whether the underlying candidate representation is
`(q_i, q̇_i)` or a state-symbol grid. Two producers feed it:

- **CSV-streamed** (`csvThetaChunks`) — for the 2nd-order path, where the design
  matrix `Θ` (one row per data point × coordinate, one column per candidate)
  would never fit in memory: a degree-4 library on 6 DOF is ~23 GB unmaterialised.
  Sub-chunked to a cell budget (`GRAM_DENSE_CELL_BUDGET`, peak RSS ~1 GB).
- **Dense, single-chunk** — for higher-derivative recovery
  (`recoverHigherOrderLagrangian`, `recoverMultiFieldHigherOrderLagrangian`),
  where the candidate matrix already fits in memory (it comes from a handful of
  differentiated trajectories, not a streamed CSV) — the same accumulator called
  once on the whole matrix, replacing what used to be a bare `Θᵀ Θ`.

The model being fit is the same either way: the discovered `L` must satisfy the
observed equations of motion,

```
EL(kinetic) + Σ_j c_j · EL(θ_j) ≈ 0     over all data rows
```

`G` is `n_candidates × n_candidates`; everything downstream needs only `G` and
`b = −G[:, kinetic]`.

### 4. Model selection — `finding_L/main_streaming.py`, `regularized_select.py`, `gram_forward_select.py`, `stopping_conditions.py`

`runDiscoveryStreaming(selector=…)` chooses the selector. The default is
`"lasso"` — the debiased-LASSO path of §12, which keeps the correct sparse term
set far further into measurement noise than greedy does (see **Results** and
**Open problems A**); it builds the Gram once at `degreeCap` and selects in one
shot. `selector="greedy"` is the earlier path, kept reachable and tested, and is
what the round-by-round diagnostics (`roundCallback`, the visualisation scripts)
exercise. Both consume only `G` and `b = −G[:, kinetic]`. The higher-derivative
recovery functions (`higher_order_discovery.forwardSelectFromGram`) run the same
greedy shape, simplified (no degree expansion, since their library is already at
full degree before selection starts).

**Greedy** — orthogonal-matching-pursuit style. Per round, with active set `S`,
the fit/score/stall step is one shared primitive,
`gram_forward_select.greedyRoundStep`, used by both `runDiscoveryStreaming`'s
loop (below) and `forwardSelectFromGram` — each caller keeps its own outer
orchestration, only the inner step is shared code:

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
   `|score| ≥ correlationCutoff`; otherwise (`runDiscoveryStreaming` only)
   attempt a degree expansion.
5. After the loop, `pruneNearZeroCoefficients` iteratively drops any active term
   with `|c| < pruneRelativeThreshold · max|c|` and refits.

The full best-subset problem is combinatorial; this greedy score is cheap and, on
clean data, exact. Its noise failure mode — and the selectors that do better —
are in §12 below and **Open problems A**.

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
| `degreeCap` | `4` | `checkDegreeExpansionNeeded` / LASSO library degree (both benchmark Lagrangians are quartic) |
| `selector` | `"lasso"` | `runDiscoveryStreaming` — the debiased-LASSO path; `pruneRelativeThreshold` doubles as its threshold, and `lassoPathFromGram`'s own defaults (40 λ's, `1e-3` ratio, 5 % residual slack) are single library constants, identical for every system |

Every value is threaded explicitly into `runDiscoveryStreaming` — none falls back
to a function default at runtime. `runSystemDiscovery` returns
`(discovered, logFrame, tolerancesUsed)` and `noise_robustness_sweep.py` asserts
`tolerancesUsed == FROZEN_TOLERANCES` on every run, so a blind-holdout system
provably runs on the same knobs — selector included — as the reference system.

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
   component reduces to `0` (`expand`, then `simplify`) — the general Ostrogradski
   operator (`ostrogradski.eulerLagrangeExpression(..., pipelineSign=True)`) for
   any order, order 1 included: it's that operator's exact algebraic special
   case, not a separate code path (verified independently by two of the three
   agents who worked on this codebase's generation/finding_L unification, down
   to the sign convention). Order defaults to the highest derivative present.
3. If null and first-order, `reconstructBoundaryPotential` recovers `F` with
   `ΔL = dF/dt` (curl-free check, then integrate the velocity coefficients).
4. Otherwise `verifyEquivalenceClass(B, A, coords, vels)` tests the full action
   symmetry `A = c·B + dF/dt` for a *constant* `c`: solve `A − c·B − D_t F ≡ 0`
   for `c` and the coefficients of `F` jointly (it is linear in both) over an
   ansatz `F = Σ aⱼ mⱼ` in the coordinates and their time derivatives up to one
   order below the highest derivative in `A, B`, degree ≤ `deg(A, B)`; a
   bare-position term pins `c` because an
   autonomous `F` cannot produce one. The EL residual of `A − c·B` is re-checked
   before the verdict is trusted. Verdicts: `equivalent-by-scale`,
   `equivalent-by-total-derivative`, `equivalent-by-scale-and-total-derivative`,
   `not-equivalent`.

Verdict: **equivalent** if `ΔL == 0`, if `ΔL` is a nonzero null Lagrangian (same
physics, differ by a total time derivative), or if `A` and `B` differ by a
constant rescaling plus a total derivative; **not equivalent** if no constant `c`
makes `A − c·B` a total derivative (different equations of motion). The recovered
`c` is on `EquivalenceVerdict.scale`. Two Lagrangians can differ by `d/dt F` and
be physically identical (e.g. `q q̈` vs `−q̇²`), `L` and `3L` are the same theory,
and two with the same monomials but slightly-off coefficients are *not* —
coefficient proximity answers none of this, the EL operator answers all of it.

`experiments/discovery.compareToExpected` runs this on every
discovered-vs-expected pair; the noise sweep surfaces it per level as `exact`
(`ΔL == 0`) / `null-L` (nonzero null Lagrangian) / `no` (nonzero EL residual —
the recovery failed even if the coefficients looked close). The higher-derivative
studies call `isNullLagrangian` directly.

### 7. Single-field higher-derivative recovery — `finding_L/higher_order_discovery.py`

`recoverHigherOrderLagrangian` is §2–5's general machinery specialised to one
field with an explicit Lagrangian order above 1: build the library
(`buildHigherOrderLibrary`, the `noFields=1` case of §2's grid, in the plainer
bare-`s{level}` notation since a field index would be redundant — §"Symbol
conventions"), build the Gram (§3), `dropKineticAliasColumns` removes columns
collinear with the kinetic column (the exact `q''² ↔ q' q'''` alias), then
select (§4) and snap. `assembleDiscoveredHigherOrderLagrangian` wraps this with
the same readable/grouped `.text` output §5 gives the 2nd-order path. Noisy
derivatives come from `generation/numerical_diff.py`; the quintic smoothing
spline is the only method that survives to 3rd/4th order. On Pais–Uhlenbeck it
recovers `L` up to a total derivative, robust to ~3 % noise.

### 8. Ghost detection — `generation/ghost_detection.py`

1. Build the Ostrogradski Hamiltonian `H`
   (`ostrogradski_hamiltonian.ostrogradskiHamiltonian`). Raises
   `NonUniqueTopDerivativeError` if `L` is nonlinear in its highest derivative
   (the Legendre transform is then multi-valued and the physical branch is a
   caller decision).
2. If `H` is quadratic in the phase-space variables, take its Hessian and count
   negative eigenvalues. Otherwise hand `H` to
   `generation/boundedness.polynomialBoundedBelow`, which returns
   `bounded_below` / `unbounded_below` / `inconclusive` from: (a) an
   Ostrogradski linear-momentum test — `H` exactly linear in some momentum
   `P_i` with a nonzero coefficient is unbounded below (the canonical
   `Σ P_{i,s} Q_{i,s+1}` term the Legendre transform always produces for a
   non-degenerate HD theory); (b) the leading homogeneous form — odd total
   degree, or negative somewhere on the sphere ⇒ unbounded below; positive
   definite ⇒ coercive, bounded below; (c) if that form is only positive
   *semidefinite* but positive definite in the variables it involves, recurse
   on `H` restricted to the variables it omits (coercive transverse to that
   subspace, so the infimum lives on it). Anything else is `inconclusive`.
3. Compute the EOM characteristic roots; classify the dynamics (oscillatory /
   runaway / damped).
4. **Ghost** iff `H` is unbounded below *and* the dynamics are oscillatory — a
   bounded-motion negative-energy mode, the Ostrogradski signature. `H` bounded
   below is never a ghost. For quadratic `H`, "unbounded below" is Hessian
   indefiniteness and indefinite + runaway is a tachyon (`ghost=False`). For
   non-quadratic `H`, "unbounded below" is the `polynomialBoundedBelow` verdict;
   unbounded + non-oscillatory, and `inconclusive`, both yield `ghost=None` —
   the linear dynamics classification a nonlinear EOM produces is not trustworthy
   enough to call the tachyon/ghost split.

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
`detectGhost` returns `degenerate=True` with the constraint analysis attached
rather than crashing:

1. Primary constraints come from the null space of the Hessian
   `W_ab = ∂²L/∂q_a^(n) ∂q_b^(n)`: each null vector `v` gives a phase-space
   relation `Σ_a v_a (P_a − ∂L/∂q_a^(n)) = 0` (the `q^(n)` parts cancel because
   `W v = 0`).
2. `diracBergmannIteration` propagates the consistency conditions to a fixed
   point: for each constraint `φ_a` not already fixing a Lagrange multiplier
   (`{φ_a, φ_primary}` weakly zero for every primary), `{φ_a, H}` is reduced
   modulo the current constraint ideal (Gröbner, `QQ`-domain fallback for
   rational coefficients); an independent nonzero remainder is appended as a
   next-generation constraint. Iterates until no new constraint appears or the
   `maxRounds` budget is hit — `chainClosed` records which.
3. `classifyConstraints` runs on the **full** set: the canonical Poisson-bracket
   matrix `C_ab = {φ_a, φ_b}` (weak vanishing checked modulo the constraint
   ideal) splits it into first-class (whole `C` row vanishes weakly) and
   second-class. `constraintGenerations` tags each with its iteration round.
4. `diracBracketMatrix` / `diracBracket` build the second-class bracket
   `{f, g}* = {f, g} − {f, φ_a} (C⁻¹)_ab {φ_b, g}` (raises if the second-class
   matrix is singular). When the chain closed,
   `physicalPhaseSpaceDimension = 2·nDOF − 2·firstClass − secondClass`.

For the ghost verdict on a degenerate Lagrangian, `detectGhost` solves the
second-class constraints, substitutes into the canonical `H`, and tests the
**reduced** `H` — Hessian eigenvalues if quadratic, `polynomialBoundedBelow`
otherwise. The verdict dict carries `ghost` (`True` / `False` / `None`),
`chainClosed`, `physicalPhaseSpaceDimension` and `reducedHamiltonian`.
`ghost=None` when the chain did not close, when a first-class constraint is
present (residual gauge freedom — needs gauge fixing), or when the second-class
constraints cannot be solved in closed form to reduce `H`.

Still out of scope: Dirac-bracket-based quantisation, and the full
Dirac–Bergmann treatment of cases where the constraint surface is not a smooth
solvable variety.

### 10. Multi-field recovery and order inference — `finding_L/higher_order_*.py`

- `buildMultiFieldElMatrix` stacks one row-block per field (rows = timepoints ×
  fields), exactly as the 2nd-order streaming path stacks per-coordinate blocks.
  `recoverMultiFieldHigherOrderLagrangian` fixes the isotropic top-derivative
  kinetic `Σ_i (q_i^(n))²` and forward-selects the rest, cross-field coupling
  monomials included. On a position-coupled Pais–Uhlenbeck chain (2–3 fields) it
  recovers `L` up to a total derivative with the coupling coefficient exact on
  clean data, and — with `numerical_diff.segmentedDerivatives` (a spline per
  trajectory, unstable edges trimmed) — from ~0.03 % position noise
  (Open problem D).
- `inferLagrangianOrder(..., noFields=1)` tries orders `1..maxOrder`; for each
  it measures the least-squares residual of projecting the summed
  `Σ_i q_i^(n)²` kinetic EL column onto the span of the other EL columns (a
  feasibility test: does the data satisfy an order-`n` Euler–Lagrange
  equation?) — general over field count via §2's library, `noFields=1` its
  exact single-field case (verified: identical output to the pre-generalisation
  single-field-only implementation). It returns the smallest order below
  tolerance (Condition A), else the order after which the residual stops
  improving (Condition C). PU → 2, anharmonic oscillator → 1, a 2-field coupled
  PU chain → 2. Each `perOrder` record also carries a `degenerate` flag
  (kept-EL-matrix numerical rank `≤ 1`): a purely linear order-1 system tests
  as feasible at every order, so its zero residual is labelled rather than
  trusted (Open problem F).
- `recoverHigherOrderLagrangian(..., orderPrior=True)` uses that verdict as a
  hard prior: an over-specified order request is reduced to the inferred order
  (columns truncated, library shrunk) before selection, so an on-shell
  higher-derivative distractor is never built. Off by default; see Open
  problem B.

### 11. End-to-end pipeline — `finding_L/pipeline.py`

`endToEndPipeline(noisyPositions, dt, noFields=1)` chains the pieces with no
ground-truth input:

1. For each differentiation method (Savitzky–Golay, SG poly-8, quintic spline):
   estimate derivatives → `inferLagrangianOrder` → recover (single-field:
   `recoverHigherOrderLagrangian`; multi-field, `noFields>1`:
   `recoverMultiFieldHigherOrderLagrangian`) → `detectGhost`, and record the
   recovered Lagrangian's *own* Euler–Lagrange residual on that method's
   derivatives.
2. Consensus order = majority vote. Method selection = among recoveries with
   plausible (not absurdly large) coefficients, the lowest own-EL residual.
3. Three confidences: **order** (cross-method agreement), **ghost** (agreement on
   the verdict), **coefficient** (spread across methods + plausibility). Overall
   confidence is the order/ghost minimum — the robust part; coefficients are
   reported separately because the differentiation step limits them. On PU: order
   2 and ghost True with full cross-method agreement through ≥ 1 % noise; the
   coefficients drift with noise.

Multi-field order inference through this pipeline is robust (verified: full
cross-method agreement on a coupled PU chain at zero noise). Ghost-verdict
reliability for the multi-field case is a live area, not yet closed — see
**Open problems** for the current, specific findings rather than a summary
here that would go stale as that work lands.

### 12. Regularisation-path selectors — `finding_L/regularized_select.py`

Two selectors solve the same Gram-only problem as greedy:

- **`sequentialThresholdedLeastSquares`** (SINDy STLSQ): start from the full
  least-squares fit `G c = b`, zero the coefficients below
  `relativeThreshold · max|c|`, refit on the survivors, iterate to a fixed point.
- **`lassoSelect`**: a coordinate-descent LASSO path from `G` and `b` (minimising
  `½ cᵀG c − bᵀc + λ‖c‖₁` over a geometric λ sequence), then the sparsest solution
  whose refit residual is within 5 % of the densest, then hard-threshold and
  refit (debiased LASSO).

`lassoSelect` is the **production 2nd-order selector** (§4,
`runDiscoveryStreaming(selector="lasso")`, the default): it holds the correct
sparse term set where the greedy score commits to noise-correlated velocity
quartics (**Open problems A**). `selector="greedy"` stays available.

`experiments/model_selection_comparison.py` runs all three side by side over
multiple seeds and every system in `SYSTEMS`, on one degree-`degreeCap` streaming
Gram (via `build_matrix.buildAdmissibleGram`) per system and noise level; see
**Results**.

### 13. One general entry point — `finding_L/main.py`

`discoverLagrangian(trajectory, dt=None, noFields=None, maxOrder=3,
libraryMaxDegree=None, degreeCap=4, chunkRows=200_000, selector="lasso")`
dispatches on the type of `trajectory`, replacing both
`finding_L/main_streaming.py`'s role and the ad hoc `__main__` blocks the
higher-derivative studies used to need:

- **A CSV path** (`str`/`Path`) — the known-derivatives, large-`N` case — routes
  through `runDiscoveryStreaming` (§3's streamed front door, §4's selectors).
  `noFields` is inferred from the header's bare `q{i}` columns if not given.
  Returns a `DiscoveredLagrangian` (§5).
- **An array of noisy positions** (`dt` required; 1-D for one field, `(steps,
  noFields)` for several) — the no-ground-truth case — routes through
  `endToEndPipeline` (§11). `noFields` is inferred from the array's shape if not
  given. Returns a `PipelineResult` (§11).

The two return types are deliberately not forced into one shape: the streamed
path has no order-inference or ghost step, the pipeline path does — unifying
them would either drop real information or fabricate fields that don't apply.
**Honest scope limit:** the CSV path still only supports order 1 — a
general-order CSV front door needs `build_matrix.py`'s column *reading*
generalised too (currently hardcoded to the legacy `q{i}`/`q{i}dot`/`q{i}ddot`
names), which only its Gram *accumulation* (§3) actually became general over;
nothing currently needs it, so it was left undone rather than half-built.

---

## Repo structure

`src/` holds three packages; there is no installable distribution
(`[tool.uv] package = false`), so everything runs from `src/`. The tree carries
no comments or docstrings by design — the explanation lives in this file.

### `generation/` — the forward direction: a known `L` → data / analysis

| module | contents |
|---|---|
| `eqnofmotion` | the single `TIME` symbol, `defineCoordinates` — the shared coordinate/symbol factory every other module builds on. `EulerLagrangeEqn` is a retained legacy order=1 operator, kept only for `pu_oscillator_validation.py`'s deliberate legacy-vs-Ostrogradski cross-check; nothing else calls it — `integrator.GetAccelFunctions` and `finding_L`'s EL-column evaluators route through the general `ostrogradski` operator instead |
| `integrator`, `higher_order_integrator` | RK4 for 2nd-order and higher-order state. `integrator.GetAccelFunctions` keeps its original signature but now solves via `ostrogradski.solveTopDerivatives(order=1)` internally; `higher_order_integrator.rk4Step` is the one RK4 core both paths share |
| `generate_data`, `noise` | streamed noisy-trajectory CSV generation — `generateDatasetStreaming` (2nd-order, `q`/`q{c}dot`/`q{c}ddot` columns) plus the general `generateHigherOrderDatasetStreaming` (arbitrary `(order, noCoords)`, `q{c}_d{level}` columns including the top derivative as an independent data column, same role `qddot` plays for order 1) |
| `ostrogradski` | Ostrogradski EL operator `Σ (−d/dt)^k ∂L/∂q^(k)`, top-derivative solve, RK4 state derivative — arbitrary order, arbitrary `coords` length |
| `ostrogradski_hamiltonian` | canonical momenta and Hamiltonian; `NonUniqueTopDerivativeError`; `analyzeDegenerateLagrangian` |
| `constraints` | canonical Poisson bracket, weak (on-shell) vanishing, first/second-class classification, the Dirac–Bergmann secondary-constraint iteration, the second-class Dirac bracket |
| `boundedness` | `polynomialBoundedBelow` — is a non-quadratic polynomial `H` bounded below? (Ostrogradski linear-momentum test + leading-form coercivity, recursive on the flat subspace) |
| `ghost_detection` | `detectGhost` — `H` boundedness (Hessian, or `boundedness` for non-quadratic `H`) + EOM characteristic roots |
| `numerical_diff` | finite-difference / Savitzky–Golay / quintic-smoothing-spline derivative estimates; `segmentedDerivatives` differentiates each trajectory segment on its own and trims the spline-unstable edges |

### `finding_L/` — the inverse direction: data → `L`

| module | contents |
|---|---|
| `main` | `discoverLagrangian` — the one general entry point (§13), CSV path or noisy-position array in |
| `higher_order_candidates` | the one candidate-library builder, general over `(order, noFields)` (§2); `monomialLibrary`, `multiFieldLibrary`/`stateGridSymbols`, `buildMultiFieldElMatrix`, and the 2nd-order-shaped `buildCandidateLibrary`/`filterPureVelocityTerms` (thin wrappers around the general form) |
| `build_matrix` | the shared, naming-agnostic Gram-accumulation core `accumulateGramFromChunks` (§3), fed by either `csvThetaChunks` (2nd-order, chunked, cell-budgeted) or a single dense chunk (higher-derivative); `buildAdmissibleGram` streams `G` and drops zero-variance columns |
| `gram_forward_select` | fit / score / prune primitives on `G`, plus the shared per-round step `greedyRoundStep` (§4) both selector paths build on |
| `stopping_conditions` | the three stopping conditions (`runDiscoveryStreaming`'s greedy path) |
| `regularized_select` | STLSQ and debiased-LASSO selectors, Gram-only; `lassoSelect` is the production 2nd-order selector |
| `main_streaming` | `runDiscoveryStreaming(selector=…)` — the 2nd-order discovery driver (LASSO default, greedy optional); superseded as the recommended entry point by `main.discoverLagrangian`, kept as the underlying driver |
| `higher_order_discovery` | `recoverHigherOrderLagrangian`, `recoverMultiFieldHigherOrderLagrangian`, `forwardSelectFromGram`, `inferLagrangianOrder(..., noFields=…)`, and the readable-report wrappers `assembleDiscoveredHigherOrderLagrangian`/`assembleDiscoveredMultiFieldLagrangian` |
| `pipeline` | `endToEndPipeline(..., noFields=…)` — noisy positions → `L` + ghost verdict + confidences, single- or multi-field |
| `report` | the naming-agnostic formatting core `assembleDiscoveredLagrangianFromState` (§ Symbol conventions, §5) plus `assembleDiscoveredLagrangian`, its thin `q_i/v_i`-converting wrapper; coefficient snapping, the readable text |
| `equivalence_class` | `classifyLagrangianPair` / `verifyEquivalenceClass` / `isNullLagrangian` — null-Lagrangian + scale-factor test, via the one general Ostrogradski EL operator for any order |

### `experiments/` — benchmarks and runnable studies

`systems` and `pu_system` share one `PhysicalSystem` record shape:
`buildLagrangian(coords) -> (L, constants)` — no `vels` argument, a builder
derives whatever velocity/higher-derivative terms it needs internally via
`sp.diff`, and a system's order is never stored, only recovered on demand via
`ostrogradski.lagrangianOrder`. `systems.SYSTEMS` holds the 2nd-order
benchmarks; `pu_system.HD_SYSTEMS` holds the higher-derivative ones (currently
`pais_uhlenbeck`) in the same record shape, kept alongside `SYSTEMS` rather
than merged into it, since `discovery.py`'s driver is still 2nd-order-only.
`generate_dataset`, `discovery` (the frozen-tolerance policy +
`compareToExpected`), and the studies listed under **How to use** are
unchanged.

---

## Setup

Dependencies: numpy, scipy, sympy, pandas, matplotlib (declared in
`pyproject.toml`; the checked-in `.venv` already has them). Tests also need
`pytest`, in the `dev` group:

```
uv sync --group dev
```

There is nothing to build or install — every entry point is a `python -m`
module run from `src/`.

---

## How to use

```
cd src
PY=../.venv/bin/python

# 1. build a dataset from a known symbolic Lagrangian
$PY -m experiments.generate_dataset anharmonic_chain_blind --noise 0.0 0.05 0.10 0.25
$PY -m generation.main                            # the isotropic calibration system's CSVs

# 2. recover a Lagrangian from a dataset (readable report to stdout)
$PY -m finding_L.main                              # discoverLagrangian; __main__ points at a generated CSV
$PY -m finding_L.main assets/some_dataset.csv      # or any other trajectory CSV

# 3. 2nd-order discovery vs measurement noise
$PY -m experiments.noise_robustness_sweep isotropic_quartic_calibration   # reference system
$PY -m experiments.noise_robustness_sweep anharmonic_chain_blind          # blind holdout
#   optional: --noise 0.0 0.01 0.02 0.05 0.10   --chunk-rows N

# 4. higher-derivative (Ostrogradski) track
$PY -m finding_L.equivalence_class                 # constructed null-Lagrangian + scale-factor checks
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

# 7. model selection: production LASSO vs greedy vs STLSQ (multi-seed, all systems)
$PY -m experiments.model_selection_comparison
```

Each study writes `.txt` / `.json` (and some `.png`) into
`src/experiments/results/`. Datasets under `assets/` are git-ignored and
regenerable (the generators are seeded).

Programmatic use — one entry point regardless of order or field count (§13):

```python
from finding_L.main import discoverLagrangian

# a CSV of known trajectories (2nd-order today)
discovered = discoverLagrangian("trajectories.csv", noFields=6)
print(discovered.text)         # readable, grouped, coefficients snapped to rationals
discovered.expression          # sympy expression in clean q0, v0, … symbols

# noisy positions only, order/field count inferred — 1-D or (steps, noFields)
result = discoverLagrangian(noisy_positions, dt=0.004)
print(result.summary())        # order, ghost verdict, confidences, per-method breakdown
```

The underlying drivers (`finding_L.main_streaming.runDiscoveryStreaming`,
`finding_L.pipeline.endToEndPipeline`) are still there and still the right
choice when a caller needs their extra knobs (`roundCallback`, `orderPrior`,
explicit selector/tolerance overrides) — `discoverLagrangian` wraps them for
the common case, it doesn't replace their APIs.

---

## Tests

```
uv run pytest
```

102 tests. Core: EL / Ostrogradski Hamiltonian vs closed form; the Pais–Uhlenbeck
EOM and Hamiltonian conservation; the equivalence-class classifier both ways
(including that the order-1 and general-Ostrogradski EL paths now agree
exactly, not just approximately); forward selection, STLSQ and LASSO on
synthetic Gram matrices; the frozen-tolerance discipline (selector included);
degenerate-constraint classification, the Dirac–Bergmann secondary-constraint
chain and the second-class Dirac bracket; two-field mixing; multi-field
higher-order recovery and order inference; the single- and multi-field
end-to-end pipeline; the ghost ROC battery; `canonicalizeLagrangian`'s
same-field cross-derivative reduction (sign tracking, odd-gap vanishing,
no-ops on already-diagonal and cross-field terms) and its
`isNullLagrangianLocal` safety net; `characteristicRoots`' coupled-system
determinant path against single-field and decoupled-multi-field legacy
behavior, a genuinely coupled system, and full `detectGhost` invariance under
a null-Lagrangian representative swap; the one general entry point
(`finding_L.main`) against both a CSV and a noisy-position array; the readable
report's naming-agnostic core against all three symbol conventions. Open-problem
regressions: the debiased-LASSO streaming path recovers a clean quartic where
greedy needs it (A); the order prior collapses an over-specified order request
on-shell (B); the blind chain recovers all six sites including the boundary
under the LASSO
default (C); segment-wise differentiation recovers the coupled PU chain from
sub-percent noise (D); `polynomialBoundedBelow` and its `detectGhost` verdicts
on non-quadratic `H` (E); the order-inference degeneracy flag on a linear
order-1 system (F); the multi-field pipeline's ghost verdict is now reliably
correct rather than merely valid (G).

---

## Results

- **2nd-order discovery.** With the production selector (`selector="lasso"`),
  `isotropic_quartic_calibration` (reference system): exact recovery at 0 %; the
  correct sparse term set with coefficients to ~2 % relative through ≥ 5 %
  position noise. The *greedy* selector held the term set only to ~1 % (Open
  problem A). `anharmonic_chain_blind` (blind holdout, same frozen tolerances):
  all six chain sites — the five interior sites and the boundary site — recovered
  exactly at 0 noise under the LASSO default (the boundary site failed under
  greedy; see Open problem C, now resolved).
- **Equivalence-class classification** is wired into `compareToExpected` and the
  higher-derivative studies; it flags every failed recovery, including the ones
  whose coefficients look close.
- **Higher-derivative track** works at order 2: single-coordinate PU recovery
  robust to ~3 % noise; multi-coordinate coupled-chain recovery exact on clean
  data and up to a total derivative from ~0.03 % position noise once each
  trajectory is differentiated separately (Open problem D);
  `inferLagrangianOrder` correct on PU and the anharmonic oscillator;
  `endToEndPipeline` correct on order + ghost from noisy positions only.
- **Ghost verdict** on a data-recovered PU Lagrangian is stable to ≥ 35 %
  measurement noise — the signature is qualitative — conditional on correct order
  identification, which is the fragile step. Non-quadratic `H` is no longer an
  automatic `ghost=None`: `polynomialBoundedBelow` resolves the polynomial case
  (the anharmonic oscillator now reads `ghost=False`; a recovered PU with a
  spurious `q² q̇²` term still reads `ghost=True`) — Open problem E.
- **Constrained-Hamiltonian analysis** detects degenerate Lagrangians, runs the
  Dirac–Bergmann secondary-constraint chain to closure, classifies the full
  constraint set, builds the second-class Dirac bracket, and — for a
  fully-second-class system — reduces `H` onto the constraint surface for a real
  ghost verdict (`L = q̇₁q₂ − ½q₂² − ½q₁²` → two second-class constraints,
  reduced `H = ½(q₁² + q₂²)`, `ghost=False`). First-class constraints leave the
  verdict `ghost=None` pending gauge fixing.

Isotropic quartic calibration system, 6 DOF, position noise as % of signal std
(`noise_robustness_sweep`, single seed; greedy columns from item 14 / the
regularised-select comparison):

| noise | greedy term set | LASSO term set | LASSO max&#124;Δcoef&#124; | LASSO equiv-class |
|---|---|---|---|---|
| 0 % | exact | exact | 0.0000 | same theory (`ΔL == 0`) |
| 1 % | exact | exact | 0.0006 | same theory |
| 2 % | 7 missing / 19 spurious | exact | 0.0028 | coeffs close, not same theory |
| 5 % | 20 / 20 | exact | 0.0175 | coeffs close, not same theory |
| 10 % | 21 / 11 | 4 / 33 | 0.199 | failed |

LASSO holds the exact sparse term set to ~5 % noise where greedy's broke from
2 %; the strict equivalence verdict still turns over at ~1–2 % because every EL
column is built from noisy derivatives (the errors-in-variables gap — Open
problem A). A multi-seed × 3-system `model_selection_comparison` run (greedy
vs STLSQ vs LASSO on `isotropic_quartic_calibration`, `anharmonic_chain_blind`,
`coupled_quartic_blind`) confirms LASSO as the more robust selector across
seeds; see `results/model_selection_comparison.json` for the raw numbers.

---

## Open problems

### A. The ~1–2 % noise ceiling was the greedy selector — fixed by switching to debiased LASSO

At ≥ 2 % position noise the *greedy* forward-selection path fills with spurious
velocity-dependent degree-4 terms (`q_i² q̇_i²`, `q_i q_j q̇_i q̇_j`): genuinely
well-correlated with the noise-corrupted residual, and greedy commits to them
before the true cubics. This was thought to be structural; it is not. The
debiased-LASSO path (§12) — full least-squares fit, LASSO path, threshold, refit —
recovers the true sparse Lagrangian further into noise, and is now the production
2nd-order selector (`runDiscoveryStreaming(selector="lasso")`, the default;
validated over multiple seeds and three systems by `model_selection_comparison`).
Greedy stays available as `selector="greedy"`.

**Still the principled endpoint:** an errors-in-variables / total-least-squares
formulation — every EL column is built from noisy derivatives, so the regression
is errors-in-variables, and LASSO only mitigates that. LASSO also degrades
eventually (past ~5–10 % noise); TLS is where the next gain is.

### B. Jerk/snap libraries fail via on-shell EOM degeneracy — mitigated by an order prior

`jerk_snap_distractor_study` handed an order-3 library recovers `s2² − 7/30 s3²`
(jerk-squared) instead of the order-2 PU Lagrangian. On the solution manifold
`q⁽⁶⁾ = −5 q⁽⁴⁾ − 4 q̈` exactly, so the EL column of `q'''²` is a linear
combination of lower-order columns for every trajectory; stacking trajectories
does not lift this — they are all on-shell. Order-2 libraries work; order-≥3
libraries do not. The equivalence-class check flags every failed recovery, so the
failure is detectable.

**Fix (cheap, available now): a hard order prior.**
`recoverHigherOrderLagrangian(..., orderPrior=True)` runs `inferLagrangianOrder`
on the same derivative columns before building the library and, if the inferred
order is below the requested one, truncates the columns and shrinks the library
to the inferred order — the degenerate `q'''²` distractor is never constructed.
On clean PU columns the order-3 request collapses to order 2 and recovery returns
the true Lagrangian (up to a null term); `jerk_snap_distractor_study` now runs
each scenario with and without the prior. The end-to-end pipeline was already
immune (it infers the order and passes it explicitly).

Still open: under ≥ 1 % position noise the noisy `spline`/Savitzky–Golay
derivatives break the on-shell relation *and* fool `inferLagrangianOrder`, so the
prior does not fire and the noisy-derivative recoveries stay wrong — that is the
differentiation gap of problem D, not the degeneracy. Off-manifold data (or a
genuinely forced trajectory) remains the principled lift for a true order-≥3
target where the prior cannot help.

### C. Blind-chain boundary site — resolved by the LASSO switch

Under the greedy selector, even at 0 noise the last coordinate `q5` of the open
chain was mis-recovered (a `q5² q̇5²` term shadowed `q5⁴`) — the same mechanism
as A. The debiased-LASSO default recovers all six sites exactly, boundary
included; `tests/test_blind_chain_boundary.py` pins the true `q5³` / `q5⁴`
coefficients and the absence of the shadow terms. No separate fix is needed
beyond the selector switch of A.

### D. Noisy multi-field higher-order differentiation — mitigated by segment-wise differentiation

Multi-field higher-derivative recovery is exact on clean data. The
noisy-positions → spline-derivatives step used to collapse at ~0.03 % noise —
two orders of magnitude short of the single-field ~3 % — but most of that was an
artifact: `multi_field_discovery_validation` splined **one** curve through all
six concatenated trajectories, so the step discontinuities at the joins blew up
the 3rd/4th derivatives.

**Fix (available now): `numerical_diff.segmentedDerivatives`.** Differentiate
each trajectory segment on its own (a spline per segment), trim the
spline-unstable segment edges, then restack. `multi_field_discovery_validation`
now runs the `glued` and `segmented` paths side by side: on the `segmented` path
the coupled Pais–Uhlenbeck chain is recovered up to a total derivative from
0.03 % position noise (cross-field coupling coefficient to <1 %), and for most
seeds still at 0.1 %, where the `glued` path fails outright even at 0.03 %.

Still open: a real gap to the single-field ~3 % remains — each field's 4th
derivative is taken from short (per-trajectory) segments — but it is now a
differentiation-accuracy gap, not a concatenation artifact. Longer trajectories
or a joint multi-field smoother are the next step.

### E. Non-quadratic Hamiltonians — resolved for polynomial `H`

A recovered Lagrangian with a spurious `q² q̇²` term, or a genuinely nonlinear
system, gives a non-quadratic `H`. `detectGhost` used to return `ghost=None` for
all of them — safe (no false alarms) but incomplete.

`generation/boundedness.polynomialBoundedBelow(H, positions, momenta)` now gives
a verdict for polynomial `H` (see §8 for the mechanism):

- `unbounded_below` + oscillatory dynamics ⇒ `ghost=True`. The Ostrogradski
  linear-momentum term (`H` exactly linear in some `P_i`) is the usual route and
  survives a spurious `q² q̇²` term.
- `bounded_below` ⇒ `ghost=False`. The anharmonic oscillator
  `½q̇² − ½q² − ¼q⁴`, previously undetermined, resolves here; the inverted
  quartic `+¼q⁴` goes the other way (`unbounded_below`, `ghost=True`).
- `unbounded_below` + non-oscillatory dynamics, `inconclusive`, or a genuinely
  rational `H` ⇒ `ghost=None`. The tachyon/ghost split needs a trustworthy
  linear dynamics classification, which a nonlinear EOM does not give; and the
  Legendre transform of a Lagrangian not linearly invertible in `∂L/∂q^(n)`
  produces a rational `H` outside a polynomial boundedness test.

Still `inconclusive`: a positive-semidefinite leading form whose flat directions
are not axis-aligned (e.g. `Q_a² Q_b²`) — a sum-of-squares / Positivstellensatz
certificate is the principled route. In the ghost ROC battery the FP/FN rates
stay 0 and the residual "undetermined" rate (0.38 → 0.57 over 0–1 % noise) is
unchanged, because those cases are rational `H` and `NonUniqueTopDerivativeError`
from the degree-4 pipeline recoveries — recovery-quality gaps (Open problems
A/B), not the quadratic-only limitation this closes.

### F. Linear order-1 systems are degenerate for order inference

For a purely linear order-1 system (a harmonic oscillator), `EL(q̇²) ∝ EL(q²)`
exactly on-shell, so `inferLagrangianOrder`'s feasibility residual for order 1 is
trivially zero — it returns 1, correctly, but by a degenerate route: the library's
Euler–Lagrange columns span only one direction on the data, so *every* order tests
as feasible and the zero residual carries no evidence. Nonlinear order-1 systems
(with a large enough library) infer cleanly.

`_orderFitResidual` now also reports a `degenerate` flag — the numerical rank of
the (column-normalised) kept EL matrix is `≤ 1` — and `inferLagrangianOrder`
threads it into each `perOrder` record. The verdict is unchanged (smallest order
consistent with the data, which for rank-deficient data is the honest Occam
choice), but a degenerate convergence is now labelled as such rather than trusted,
and `reduceOrderToPrior` (the Open problem B order prior) refuses to lower a
requested order when the inference it rests on is degenerate. A genuine
disambiguation still needs data that excites the higher-frequency mode — a
harmonic trajectory simply does not contain the information.

### G. `detectGhost` broke on legitimately-recovered multi-field Lagrangians — resolved, two separate causes

The multi-field pipeline's ghost verdict (§11) was unreliable for a reason distinct from differentiation accuracy (D): `detectGhost` could return the wrong verdict, or crash, on a Lagrangian that `recoverMultiFieldHigherOrderLagrangian` had legitimately recovered — correct up to the total-derivative freedom the equivalence-class check (§6) is built to recognize as the same theory. Two independent bugs, both in `generation/`:

1. **`ostrogradski_hamiltonian.ostrogradskiHamiltonian`'s canonical momenta are not invariant under `L → L + dF/dt`.** A recovered Lagrangian can land on a null-Lagrangian-equivalent-but-non-canonical monomial (e.g. `q q̈` instead of `−q̇²` — related by `d/dt(qq̇) = q̇² + qq̈`, so a same-field cross-derivative-order term either reduces to a diagonal term with a sign flip, or, when the derivative-order gap is odd, is *exactly* a total derivative and contributes nothing) — and the Legendre transform, having no notion of this equivalence, could fail to invert or read a spuriously different boundedness. **Fix:** `generation/ostrogradski.canonicalizeLagrangian(lagrangian, coords, order)` reduces same-field cross-derivative monomials via repeated integration by parts before the Legendre transform runs, verified against the original by a self-contained EL-residual check (`isNullLagrangianLocal`, built on the existing `eulerLagrangeExpression` — not borrowed from `finding_L`, to keep `generation/` independent of it) before being trusted. **Scope, stated rather than oversold:** same-field only. Cross-*field* odd-derivative-order monomials (`q_i^(a) q_j^(b)`) are *not* droppable the same way — only the symmetric combination `q_i^(a)q_j^(b) + q_i^(b)q_j^(a)` is a total derivative; the antisymmetric part is a genuine gyroscopic-type coupling term, and dropping it would be a real error, not a cosmetic one. Left as an explicit follow-on (below), not attempted.
2. **`characteristicRoots` never supported coupled multi-field systems at all.** It substituted only each coordinate's *own* derivatives with powers of the growth-rate symbol — any coupling term left the other coordinate's symbol unresolved in a `Poly` coefficient, raising `Cannot convert expression to float`, independent of canonicalization. **Fix:** the standard normal-mode analysis for coupled linear oscillators — substitute `q_j^{(k)} → A_j·s^k` (one amplitude per coordinate) across the whole Euler–Lagrange system at once, build `M_{ij} = ∂(\text{equation}_i)/∂A_j`, and solve `det(M(s)) = 0`, rather than finding each coordinate's roots independently (only valid once the system is already decoupled). A single coordinate, or an already-decoupled multi-field system, gives the same physical roots as before (verified by a dedicated regression test, not just the math). A genuinely nonlinear self-term — which the pre-existing single-coordinate code never checked for either, and got away with via an ad hoc amplitude-collapse — falls back to that same legacy per-coordinate computation rather than a strict linearity rejection, so existing nonlinear-Lagrangian ghost verdicts are unaffected.

The originally-diagnosed case (a 2-field coupled Pais–Uhlenbeck chain, segmented differentiation, ~3×10⁻⁴ noise) now reads `ghost = True` reliably — confirmed stable across multiple Python hash seeds, since sympy's own hash-order-dependent internals were adding an unrelated source of run-to-run variance during diagnosis. Still open, deliberately: cross-field odd-derivative-order canonicalization (item 1's stated scope limit).

---

## Roadmap

- Errors-in-variables / total-least-squares recovery — the principled endpoint of
  problem A (the debiased-LASSO switch is done; TLS is the next gain past
  ~5–10 % noise).
- Off-manifold / forced-trajectory data generation to lift the on-shell
  degeneracy for a true order-≥3 target (B), where the order prior cannot help.
- A sum-of-squares / Positivstellensatz certificate for non-axis-aligned
  positive-semidefinite leading forms (e.g. `Q_a² Q_b²`) — the residual
  `inconclusive` case of the non-quadratic boundedness test (E).
- A joint multi-field smoother (or longer trajectories) to close the residual
  differentiation-accuracy gap in multi-field higher-order recovery (D).
- ~~Multi-field extension of the end-to-end pipeline~~ — done:
  `endToEndPipeline(..., noFields=…)`. ~~Multi-field ghost-verdict reliability
  through the pipeline~~ — done, see Open problem G: two independent bugs in
  `generation/` (null-Lagrangian-representative sensitivity, and coupled-system
  support in `characteristicRoots`), both resolved.
- Cross-field canonicalization (Open problem G, item 1's stated scope limit):
  odd-derivative-order monomials between *different* fields need an explicit
  symmetric/antisymmetric split before any term can be dropped — same-field
  reduction alone doesn't generalize to this case by just allowing `i≠j`.
- A general-order CSV front door: `build_matrix.py`'s Gram *accumulation* is
  already general over `(order, noFields)` (naming-agnostic, shared with the
  higher-derivative dense path), but its CSV *column reading* is still
  hardcoded to the legacy 2nd-order `q{i}`/`q{i}dot`/`q{i}ddot` names — nothing
  currently needs a streamed higher-derivative CSV path, so this was left
  undone rather than built and untested.
