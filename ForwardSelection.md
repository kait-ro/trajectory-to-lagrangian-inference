# ForwardSelection

The sparse-regression core of `finding_L`: a greedy, orthogonal-matching-pursuit-style
forward selection run entirely against a streamed Gram matrix.

## The streaming Gram-matrix formulation

The discovered Lagrangian must satisfy the observed equations of motion. With
`EL(θ)` the Euler–Lagrange column of candidate monomial `θ` evaluated on the
trajectory data (`generation.eqnofmotion.EulerLagrangeEqn`, with `q̈_i`
substituted as a data column), we want sparse coefficients `c` such that

```
EL(kinetic) + Σ_j c_j · EL(θ_j) ≈ 0        row-wise, over every data point
```

The design matrix `Θ` has one row per (data point × coordinate) and one column
per candidate. It is never materialised — for a degree-4 library on 6 DOF it is
~23 GB. Instead `finding_L/build_matrix.buildGramMatrixChunked`:

- lambdifies each candidate's EL column once, caching across degree expansions
  (`lambdifiedColumnsForTerms`, `build_matrix.py:25`);
- reads the CSV in `chunkRows`-row chunks, sub-chunks each to a cell budget
  (`GRAM_DENSE_CELL_BUDGET = 48e6`, `build_matrix.py:6`) so the transient dense
  block stays ~1 GB;
- accumulates the row count `n`, the column sums, and `G = Θᵀ Θ`.

Everything downstream uses only `G` (`n_candidates × n_candidates`) and
`b = −G[:, kinetic]`.

## The greedy loop — `finding_L/main_streaming.runDiscoveryStreaming`

Per round, with active set `S`:

1. **Refit** — `c_S = solve(G[S,S], b[S])`
   (`gram_forward_select.fitActiveCoefficientsFromGram`,
   `gram_forward_select.py:6`). A singular block falls back to `lstsq` **and
   emits a `RuntimeWarning`** — that only happens when two selected EL columns
   are collinear, and the coefficient split between them is then not
   identifiable.
2. **Residual** — `residualNormSq = targetNormSq − c_S · b[S]`;
   `scaledResidual = sqrt(residualNormSq / targetNormSq)`
   (`residualNormSquaredFromGram`, `checkResidualToleranceFromGram`).
3. **Score reserves** — for each reserve `j`,
   `score_j = (b_j − c_S · G[j, S]) / (‖θ_j‖ · ‖r‖)` — the cosine between
   candidate `j`'s EL column and the current residual
   (`scoreReserveCandidatesFromGram`, `gram_forward_select.py:22`). Take
   `argmax |score|`.
4. **Select or expand** — add the best candidate if
   `|score| ≥ correlationCutoff`; otherwise try a degree expansion.
5. **Prune** — after the loop, `pruneNearZeroCoefficients`
   (`gram_forward_select.py:60`) iteratively drops any active term with
   `|c| < pruneRelativeThreshold · max|c|` and refits.

## The three stopping conditions

| id | fires when | code |
|---|---|---|
| **A — converged** | `scaledResidual < residualRmsTolerance` | `stopping_conditions`… actually `gram_forward_select.checkResidualToleranceFromGram` (`gram_forward_select.py:87`) |
| **B — stalled** | best reserve `|score| < correlationCutoff` **and** the library is already at `degreeCap` | `stopping_conditions.checkCorrelationCutoff` (`stopping_conditions.py:1`) + `checkDegreeExpansionNeeded` (`stopping_conditions.py:7`) |
| **C — stagnated** | `scaledResidual` improved by `< stagnationTolerance` for `stagnationPatience` consecutive rounds | `stopping_conditions.checkResidualStagnation` (`stopping_conditions.py:21`) |

Between B and continuing: if the score stalled but the degree is below
`degreeCap`, the library is widened by one degree, the Gram matrix is re-streamed
for the new columns (reusing the lambdify cache), the residual history is reset,
and the loop continues.

`maxRounds` (a per-system *search budget*, `experiments/systems.py`) is set to
150 — large enough that A / B / C, never the round cap, ends the search. This is
asserted in `tests/test_frozen_tolerances.py`.

## The frozen tolerance set

`experiments/discovery.FROZEN_TOLERANCES` (`discovery.py:39`). These are the
**`finding_L` library default values, frozen verbatim** — not the output of any
calibration or tuning search. They are identical for every system;
`PhysicalSystem` carries no tolerance fields.

| key | value | threaded into `runDiscoveryStreaming` as | consumed at |
|---|---|---|---|
| `correlationCutoff` | `0.1` | `correlationCutoff` (`main_streaming.py:45`) | `checkCorrelationCutoff` default, `stopping_conditions.py:1` |
| `residualRmsTolerance` | `0.01` | `residualRmsTolerance` (`main_streaming.py:42`) | `checkResidualToleranceFromGram` default, `gram_forward_select.py:87` |
| `pruneRelativeThreshold` | `1e-2` | `pruneRelativeThreshold` (`main_streaming.py:46`) | `pruneNearZeroCoefficients` default, `gram_forward_select.py:60` |
| `stagnationTolerance` | `0.01` | `stagnationTolerance` (`main_streaming.py:43`) | `checkResidualStagnation` default, `stopping_conditions.py:21` |
| `stagnationPatience` | `3` | `stagnationPatience` (`main_streaming.py:44`) | `checkResidualStagnation` default, `stopping_conditions.py:21` |
| `degreeCap` | `4` | `degreeCap` (`main_streaming.py:41`) | `checkDegreeExpansionNeeded`, `stopping_conditions.py:7` |

`degreeCap` is `4` because both benchmark Lagrangians are quartic; the
`checkDegreeExpansionNeeded` signature default of `6` is a generic-API fallback
and is never used by the sweep.

Every value is threaded **explicitly** — none falls back to a function default at
runtime — so the number reported as "the tolerance" is provably the number the
pipeline uses. `runSystemDiscovery` returns `(discovered, logFrame,
tolerancesUsed)` and `noise_robustness_sweep.py` asserts
`tolerancesUsed == FROZEN_TOLERANCES` on every run: the blind holdout provably
uses the same set as the reference system.

## Result gate: the equivalence-class check

`compareToExpected` does not trust "same monomials, coefficients within
tolerance". It converts discovered and expected to functionals and calls
`finding_L.equivalence_class.classifyLagrangianPair`, which checks that
`discovered − expected` is annihilated by the Euler–Lagrange operator
identically. The `equiv?` column in the sweep is:

- `exact` — `ΔL == 0`;
- `null-L` — `ΔL ≠ 0` but its EL residual is identically zero (total-derivative
  equivalent, same physics);
- `no` — nonzero EL residual: physically distinct, the recovery failed even if
  the coefficients looked close.

## Known failure mode

Greedy OLS forward selection against on-shell data cannot separate spurious
velocity-dependent degree-4 terms from the real cubic/quartic terms once ≥ ~2 %
position noise biases the residual — their correlation scores overlap. This is a
property of the estimator, not the thresholds; the frozen tolerances make the
failure reproducible and the equivalence-class gate makes it detected. A
regularisation-path / errors-in-variables alternative is roadmap item 14. See
[`PROJECT.md`](PROJECT.md) problem A.

## Alternative selectors — `finding_L/regularized_select.py` (additive)

Two regularisation-path selectors solve the same Gram-only problem, as a
comparison to the greedy path (never a silent replacement):

- **`sequentialThresholdedLeastSquares`** (SINDy STLSQ): start from the full
  least-squares fit `G c = b`, zero the coefficients below
  `relativeThreshold · max|c|`, refit on the survivors, iterate to a fixed point.
- **`lassoSelect`**: a coordinate-descent LASSO path from `G` and `b` (minimising
  `½ cᵀG c − bᵀc + λ‖c‖₁` over a geometric λ sequence), then pick the sparsest
  solution whose refit residual is within 5 % of the densest, then hard-threshold
  and refit (debiased LASSO).

`experiments/model_selection_comparison.py` runs all three on one degree-4
streaming Gram per system/noise. **Result:** greedy fails from ~2 % noise; STLSQ
holds to ~2 %; the debiased LASSO path recovers both benchmark systems exactly
through ~5 % noise. So the ~1–2 % ceiling (PROJECT.md problem A) is a property of
the *greedy selector*, not of least-squares Lagrangian recovery. The production
path is unchanged pending wider validation.
