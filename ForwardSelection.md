# ForwardSelection

The sparse-regression core: a greedy, orthogonal-matching-pursuit-style forward
selection run entirely against a streamed Gram matrix, with a calibrate-then-lock
tolerance discipline and an equivalence-class gate on the result.

## What is being fit

The discovered Lagrangian must satisfy the observed equations of motion. With
`EL(θ)` the Euler–Lagrange column of a candidate monomial `θ` evaluated on the
trajectory data, we want sparse coefficients `c` with

```
EL(kinetic) + Σ_j c_j · EL(θ_j) ≈ 0      (row-wise, over all data points)
```

Only `G = Θᵀ Θ` and `b = −G[:, kinetic]` are needed — `Θ` itself is never
materialised (`finding_L/build_matrix.buildGramMatrixChunked`).

## The greedy loop (`finding_L/main_streaming.runDiscoveryStreaming`)

Per round, with active set `S`:

1. **Refit** — `c_S = solve(G[S,S], b[S])`
   (`gram_forward_select.fitActiveCoefficientsFromGram`). If `G[S,S]` is
   singular it falls back to `lstsq` **and emits a `RuntimeWarning`** — that
   only happens when two selected EL columns are collinear (a velocity alias
   past the `dropKineticAliasColumns` tolerance), and the coefficient split
   between them is then not identifiable.
2. **Residual** — `residualNormSq = targetNormSq − c_S · b[S]`;
   `scaledResidual = sqrt(residualNormSq / targetNormSq)`.
3. **Score reserves** — for each reserve `j`,
   `score_j = (b_j − c_S · G[j, S]) / (‖θ_j‖ · ‖r‖)` — the cosine between
   candidate `j`'s EL column and the current residual
   (`scoreReserveCandidatesFromGram`). Take `argmax |score|`.
4. **Select or expand** — add the best candidate if `|score| ≥ correlationCutoff`;
   otherwise attempt a degree expansion.
5. **Prune** — after the loop, `pruneNearZeroCoefficients` iteratively drops any
   active term with `|c| < pruneRelativeThreshold · max|c|` and refits.

## Stopping conditions

| condition | trigger | meaning |
|---|---|---|
| **A** | `scaledResidual < residualRmsTolerance` | converged — the active set explains the dynamics |
| **B** | best reserve `|score| < correlationCutoff` and the degree cap is reached | no remaining candidate meaningfully helps |
| **C** | `scaledResidual` improved by `< stagnationTolerance` for `stagnationPatience` consecutive rounds | residual has flattened above the noise floor — graceful exit instead of grinding to `maxRounds` |

`checkDegreeExpansionNeeded` sits between B and continuing: if the score stalled
but the library degree is below `degreeCap`, the library is widened by one
degree, the Gram matrix is re-streamed for the new columns (reusing the
lambdify cache), the residual history is reset, and the loop continues.

## Locked tolerances (`experiments/discovery.LOCKED_TOLERANCES`)

Calibrate-then-test: these knobs were tuned **once**, on
`CALIBRATION_SYSTEM = "isotropic_quartic_calibration"`, and are then frozen.

| knob | value | where it lives |
|---|---|---|
| `correlationCutoff` | `0.1` | `stopping_conditions.checkCorrelationCutoff` |
| `residualRmsTolerance` | `0.01` | `gram_forward_select.checkResidualToleranceFromGram` |
| `pruneRelativeThreshold` | `0.01` | `gram_forward_select.pruneNearZeroCoefficients` |
| `stagnationTolerance` | `0.01` | `stopping_conditions.checkResidualStagnation` |
| `stagnationPatience` | `3` | `stopping_conditions.checkResidualStagnation` |
| `degreeCap` | `4` | structural — both benchmark systems are quartic |

`runDiscoveryStreaming` now takes `correlationCutoff` as an explicit parameter
(previously only the function default was reachable), so all six can be threaded
in from one place.

`runSystemDiscovery(system, csvPath, enforceLocked=True)` (the default):

- takes only **structural** choices (`noCoords`, `startingMaxDegree`,
  `maxRounds`) from the `PhysicalSystem`;
- takes every **tolerance** from `LOCKED_TOLERANCES`;
- calls `_assertNoToleranceOverride(system)` for any non-calibration system and
  raises `ValueError` if its dataclass sets `residualRmsTolerance` or
  `degreeCap` to anything other than the locked value.

`experiments/noise_robustness_sweep.py` prints `lockedTolerancesReport()` and a
`ROLE: CALIBRATION SYSTEM` / `ROLE: BLIND HOLDOUT` banner into every `.txt` /
`.json` artifact, so a reader can see the discipline was actually enforced.

## Equivalence-class gate on the result

`compareToExpected` no longer trusts "same monomials, coefficients within
tolerance". It converts the discovered and expected Lagrangians to functionals
and calls `finding_L.equivalence_class.classifyLagrangianPair`, which checks that
`discovered − expected` is annihilated by the Euler–Lagrange operator
(identically, via `simplify`), not merely small. The verdict rides on
`RecoveryComparison.equivalenceVerdict` and shows up as the `equiv?` column in
the noise sweep:

- `exact` — `ΔL == 0`;
- `null-L` — `ΔL ≠ 0` but its EL residual is identically zero (total-derivative
  equivalent — same physics);
- `no` — nonzero EL residual: physically distinct, the recovery genuinely
  failed even if the coefficients looked close.

## Known failure mode

Greedy OLS forward selection against on-shell data cannot separate spurious
velocity-dependent degree-4 terms (`q_i² q̇_i²`, …) from the real cubic/quartic
terms once ≥ ~2 % position noise biases the residual — their correlation scores
overlap. This is a property of the estimator, not the thresholds; the locked
tolerances make the failure *reproducible and visible*, and the equivalence-class
gate makes it *detected*. A regularisation-path / errors-in-variables
alternative is tracked as item 14.
