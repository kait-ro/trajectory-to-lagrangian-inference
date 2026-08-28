# Trajectory-to-Lagrangian Inference

Recover a system's Lagrangian from trajectory data: symbolic Euler–Lagrange
generation, Gram-streaming forward-selection sparse regression, a
higher-derivative (Ostrogradski) extension, and Ostrogradski-ghost detection.

## Layout

`src/` is a path-entry set of packages (no installable package — `[tool.uv]
package = false`):

| package | role |
|---|---|
| `generation/` | symbolic EOM, trajectory integrators, dataset generation, Ostrogradski Hamiltonian, ghost detection, noisy differentiation |
| `finding_L/`  | candidate libraries, streaming Gram matrix, forward selection, stopping conditions, readable report, equivalence-class classifier |
| `experiments/` | benchmark systems, calibration + blind-holdout sweep, higher-derivative validation studies, ghost-detection validation |

## Running

Everything runs from `src/` (which puts the packages on `sys.path`):

```
cd src
PY=../.venv/bin/python

$PY -m experiments.generate_dataset anharmonic_chain_blind --noise 0.0 0.05 0.10 0.25
$PY -m finding_L.main_streaming                              # discovery on a generated dataset
$PY -m experiments.noise_robustness_sweep isotropic_quartic_calibration   # calibration system
$PY -m experiments.noise_robustness_sweep anharmonic_chain_blind          # blind holdout
$PY -m experiments.pu_oscillator_validation
$PY -m experiments.ghost_detection_validation
```

Datasets under `src/experiments/data/` and `src/generation/data/` are
git-ignored and regenerable (the generators are seeded).

## Tests

```
uv sync --group dev
uv run pytest
```

Covers the Euler–Lagrange / Ostrogradski Hamiltonian machinery, the
Pais–Uhlenbeck EOM and Hamiltonian conservation, the equivalence-class
classifier, and forward selection on a synthetic Gram matrix.

## Documentation

- [`PROJECT.md`](PROJECT.md) — what the project does, current capabilities, known limits.
- [`LOGIC.md`](LOGIC.md) — the end-to-end pipeline, step by step.
- [`ForwardSelection.md`](ForwardSelection.md) — the sparse-regression method, its
  stopping conditions, locked tolerances, and the equivalence-class gate.
- [`src/REPORT.md`](src/REPORT.md) — chronological work log.
