# Trajectory-to-Lagrangian Inference

Recover a system's Lagrangian from trajectory data: symbolic Euler–Lagrange
generation, Gram-streaming forward-selection sparse regression, a
higher-derivative (Ostrogradski) extension, and Ostrogradski-ghost detection.

## Layout

`src/` is a path-entry set of packages — there is no installable package
(`[tool.uv] package = false`), so everything runs from `src/`.

| package | contents |
|---|---|
| `generation/` | `eqnofmotion` (symbolic EOM), `integrator` / `higher_order_integrator` (RK4), `generate_data` + `noise` (dataset streaming), `ostrogradski` + `ostrogradski_hamiltonian`, `ghost_detection`, `numerical_diff` |
| `finding_L/`  | `candidates` / `higher_order_candidates` (libraries), `build_matrix` (streaming Gram), `gram_forward_select` + `stopping_conditions` (selection), `main_streaming` + `higher_order_discovery` (drivers), `report` (readable output), `equivalence_class` (null-Lagrangian classifier) |
| `experiments/` | `systems` + `pu_system` (benchmarks), `generate_dataset`, `discovery` (frozen-tolerance policy + comparison), and the runnable studies below |

## Dependencies

numpy, scipy, sympy, pandas, matplotlib (see `pyproject.toml`; the checked-in
`.venv` already has them). Tests additionally need `pytest` (the `dev` group).

## Running

```
cd src
PY=../.venv/bin/python

# 1. build a dataset from a known symbolic Lagrangian
$PY -m experiments.generate_dataset anharmonic_chain_blind --noise 0.0 0.05 0.10 0.25
$PY -m generation.main                       # the isotropic calibration system's CSVs

# 2. recover a Lagrangian from a dataset (readable report to stdout)
$PY -m finding_L.main_streaming               # __main__ points at a generated CSV

# 3. 2nd-order discovery vs measurement noise
$PY -m experiments.noise_robustness_sweep isotropic_quartic_calibration   # reference system
$PY -m experiments.noise_robustness_sweep anharmonic_chain_blind          # blind holdout
#   optional: --noise 0.0 0.01 0.02 0.05 0.10   --chunk-rows N

# 4. higher-derivative (Ostrogradski) track
$PY -m finding_L.equivalence_class            # constructed null-Lagrangian checks
$PY -m experiments.pu_oscillator_validation        # Ostrogradski EL + Hamiltonian conservation
$PY -m experiments.differentiation_method_study    # noisy higher-order differentiation
$PY -m experiments.higher_order_discovery_validation
$PY -m experiments.jerk_snap_distractor_study

# 5. ghost detection
$PY -m experiments.ghost_detection_validation      # ghost verdict vs noise
```

Each study writes `.txt` / `.json` (and some `.png`) into `src/experiments/results/`.
Datasets under `src/experiments/data/` and `src/generation/data/` are git-ignored
and regenerable (the generators are seeded).

## Tests

```
uv sync --group dev
uv run pytest
```

## Further reading

- [`PROJECT.md`](PROJECT.md) — goals, pipeline status, open problems.
- [`LOGIC.md`](LOGIC.md) — how the pipeline works, end to end.
- [`ForwardSelection.md`](ForwardSelection.md) — the sparse-regression algorithm and its tolerances.
- [`src/REPORT.md`](src/REPORT.md) — chronological work log.
