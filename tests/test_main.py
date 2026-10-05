import numpy as np
import sympy as sp
from experiments.pu_system import groundTruthColumns
from finding_L.main import discoverLagrangian, inferNoFieldsFromCsvHeader
from finding_L.pipeline import PipelineResult
from finding_L.report import DiscoveredLagrangian
from generation.eqnofmotion import defineCoordinates
from generation.generate_data import generateDatasetStreaming
from generation.integrator import GetAccelFunctions


def _write_two_coord_dataset(path, noTrajectories=24, noSteps=240):
    t, coords, vels = defineCoordinates(2)
    lagrangian = sp.Rational(1, 2) * sum(v ** 2 for v in vels) - sp.Rational(1, 2) * sum(
        q ** 2 for q in coords
    )
    accel = GetAccelFunctions(lagrangian, coords, vels, t)
    np.random.seed(0)
    generateDatasetStreaming(
        outputPath=str(path),
        noTrajectories=noTrajectories,
        noSteps=noSteps,
        dt=0.01,
        noisePercentage=0.0,
        accelFunctions=accel,
        noCoords=2,
    )


def test_infer_no_fields_from_csv_header(tmp_path):
    csvPath = tmp_path / "two_coord.csv"
    _write_two_coord_dataset(csvPath)
    assert inferNoFieldsFromCsvHeader(csvPath) == 2


def test_discover_lagrangian_from_csv_path_returns_exact_recovery(tmp_path):
    csvPath = tmp_path / "two_coord.csv"
    _write_two_coord_dataset(csvPath)

    result = discoverLagrangian(csvPath)

    assert isinstance(result, DiscoveredLagrangian)
    q0, q1 = sp.symbols("q0 q1")
    coefficients = sp.expand(result.expression).as_coefficients_dict()
    assert abs(float(coefficients[q0 ** 2]) - (-1)) < 1e-6
    assert abs(float(coefficients[q1 ** 2]) - (-1)) < 1e-6


def test_discover_lagrangian_from_noisy_positions_returns_pipeline_result():
    dt, position, _columns = groundTruthColumns(6, dt=0.004, steps=13000)

    result = discoverLagrangian(position, dt=dt, maxOrder=2)

    assert isinstance(result, PipelineResult)
    assert result.lagrangianOrder == 2


def test_discover_lagrangian_requires_dt_for_array_input():
    dt, position, _columns = groundTruthColumns(6, dt=0.004, steps=2000)
    try:
        discoverLagrangian(position)
    except ValueError as error:
        assert "dt" in str(error)
    else:
        raise AssertionError("expected ValueError when dt is missing for array input")
