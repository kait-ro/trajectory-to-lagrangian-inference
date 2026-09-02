import numpy as np
import sympy as sp
from finding_L.main_streaming import runDiscoveryStreaming
from finding_L.report import buildStateSymbolMap
from generation.eqnofmotion import defineCoordinates
from generation.generate_data import generateDatasetStreaming
from generation.integrator import GetAccelFunctions


def _write_sho_dataset(path, noCoords=1, noTrajectories=24, noSteps=240):
    t, coords, vels = defineCoordinates(noCoords)
    lagrangian = sp.Rational(1, 2) * sum(v**2 for v in vels) - sp.Rational(1, 2) * sum(
        q**2 for q in coords
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
        noCoords=noCoords,
    )


def test_round_callback_fires_per_round_and_matches_final_lagrangian(tmp_path):
    csvPath = tmp_path / "sho_n1.csv"
    _write_sho_dataset(csvPath)

    calls = []
    discovered, _log = runDiscoveryStreaming(
        str(csvPath),
        noCoords=1,
        startingMaxDegree=2,
        maxRounds=20,
        chunkRows=50_000,
        roundCallback=calls.append,
        selector="greedy",
    )

    assert len(calls) >= 1
    assert [call["round"] for call in calls] == sorted(call["round"] for call in calls)

    for call in calls:
        assert set(call) == {
            "round",
            "activeTerms",
            "coefficients",
            "kineticTerm",
            "scaledResidual",
            "converged",
            "currentMaxDegree",
        }
        assert len(call["activeTerms"]) == len(call["coefficients"])
        assert all(isinstance(value, float) for value in call["coefficients"])

    _t, coords, vels = defineCoordinates(1)
    stateSymbolMap = buildStateSymbolMap(coords, vels)
    finalActive = {
        str(sp.expand(term.subs(stateSymbolMap, simultaneous=True)))
        for term in calls[-1]["activeTerms"]
    }
    discoveredMonomials = {
        str(monomial)
        for monomial in sp.expand(discovered.rawExpression).as_coefficients_dict()
        if monomial != 1 and "v" not in str(monomial)
    }
    assert finalActive.issubset(discoveredMonomials | {"q0**2"})


def test_round_callback_fires_once_for_the_lasso_selector(tmp_path):
    csvPath = tmp_path / "sho_lasso_n1.csv"
    _write_sho_dataset(csvPath)

    calls = []
    _discovered, log = runDiscoveryStreaming(
        str(csvPath),
        noCoords=1,
        degreeCap=4,
        chunkRows=50_000,
        roundCallback=calls.append,
        selector="lasso",
    )

    assert len(calls) == 1
    assert calls[0]["round"] == 0
    assert set(calls[0]) == {
        "round",
        "activeTerms",
        "coefficients",
        "kineticTerm",
        "scaledResidual",
        "converged",
        "currentMaxDegree",
    }
    assert list(log["selector"]) == ["lasso"]
    assert "scaledResidual" in log
