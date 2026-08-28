# PROJECT

## Goal

Given trajectory data `q_i(t)` (optionally only noisy positions), recover a
Lagrangian `L` that reproduces the observed dynamics, decide whether two
recovered Lagrangians are the same physical theory, and — for higher-derivative
Lagrangians — decide whether the theory carries an Ostrogradski ghost.

## Pipeline (2nd order)

1. **Dataset** — `experiments/generate_dataset.py` integrates a known symbolic
   `L` (from `experiments/systems.py`) into seeded, optionally noisy trajectory
   CSVs.
2. **Candidate library** — `finding_L/candidates.py` builds monomials in
   `(q_i, q̇_i)` up to a degree, dropping pure-velocity terms.
3. **Streaming Gram matrix** — `finding_L/build_matrix.py` evaluates each
   candidate's Euler–Lagrange column on the data in chunks and accumulates
   `Θᵀ Θ` without ever materialising `Θ`.
4. **Forward selection** — `finding_L/main_streaming.py` +
   `finding_L/gram_forward_select.py` greedily add the candidate most correlated
   with the current residual, refit from the Gram block, and stop on one of
   three conditions (see `ForwardSelection.md`). The library degree is expanded
   on demand up to a cap.
5. **Readable report** — `finding_L/report.py` snaps coefficients to
   small-denominator rationals and groups terms for a human-readable `L`.
6. **Equivalence-class check** — `finding_L/equivalence_class.py` verifies that
   `discovered − expected` is a genuine null Lagrangian (Euler–Lagrange operator
   identically zero), not merely numerically close.

## Pipeline (higher order / Ostrogradski)

- `generation/ostrogradski.py` — full Euler–Lagrange operator
  `Σ_k (−d/dt)^k ∂L/∂q^(k)`, top-derivative solve, RK4 state derivative.
- `generation/ostrogradski_hamiltonian.py` — Ostrogradski momenta and
  Hamiltonian. Raises `NonUniqueTopDerivativeError` when `L` is nonlinear in its
  highest derivative (the Legendre transform is then multi-valued and the
  physical branch is a caller decision).
- `generation/numerical_diff.py` — finite-difference / Savitzky–Golay /
  smoothing-spline estimates of higher derivatives from noisy positions.
- `finding_L/higher_order_candidates.py`, `finding_L/higher_order_discovery.py` —
  the single-coordinate higher-derivative analogue of steps 2–5.
- `generation/ghost_detection.py` — `detectGhost`: Ostrogradski `H`, its Hessian
  definiteness, and the EOM characteristic roots.

## Calibrate-then-test discipline (`experiments/discovery.py`)

All discovery tolerances (`LOCKED_TOLERANCES`) are calibrated **once**, on
`CALIBRATION_SYSTEM = "isotropic_quartic_calibration"`, and then frozen. Every
other benchmark system is a **blind holdout**: `runSystemDiscovery(...,
enforceLocked=True)` (the default) threads the locked values into the pipeline
and raises `ValueError` if a holdout system's `PhysicalSystem` overrides any of
them. `experiments/noise_robustness_sweep.py` prints the locked-tolerance banner
and a `ROLE: CALIBRATION` / `ROLE: BLIND HOLDOUT` marker in every artifact, and
adds an `equiv?` column reporting the equivalence-class verdict per noise level.

Locked knobs: `correlationCutoff` (0.1), `residualRmsTolerance` (0.01),
`pruneRelativeThreshold` (0.01), `stagnationTolerance` (0.01),
`stagnationPatience` (3), `degreeCap` (4, structural — both benchmarks are
quartic).

## Current capabilities

- Exact recovery of the isotropic quartic calibration system and the interior
  sites of the anisotropic anharmonic chain at ≤ ~1 % position noise.
- Ostrogradski EL / Hamiltonian for arbitrary-order, arbitrary-`coords` `L`
  (Hamiltonian construction requires `L` at most quadratic in the top
  derivative).
- Single-coordinate higher-derivative recovery: recovers the Pais–Uhlenbeck
  Lagrangian up to a total derivative, robust to ~3 % noise.
- Ghost verdict on a data-recovered PU Lagrangian, stable to ≥ 35 % noise
  (conditional on correct order identification).
- Equivalence-class classifier wired into `compareToExpected` and the
  higher-derivative validation studies; generalised to any Lagrangian order.

## Known limits

- **~1–2 % measurement-noise ceiling** for 2nd-order recovery. Spurious
  velocity-dependent degree-4 terms are genuinely well-correlated with the
  noise-corrupted residual; greedy OLS forward selection against on-shell data
  cannot separate them. The fix is an errors-in-variables / regularisation-path
  estimator (item 14).
- **Higher-order libraries above order 2 fail** via on-shell EOM degeneracy: the
  EL column of `q'''²` is a linear combination of lower-order columns for every
  on-shell trajectory. Needs off-manifold data or a hard order prior.
- **Open-chain boundary site** is mis-recovered even at 0 noise (same mechanism
  as the noise ceiling).
- **Lagrangian order and number of state variables are still hand-specified**
  for the higher-derivative track (item 11).

See `src/REPORT.md` for the detailed mechanism of each limit.
