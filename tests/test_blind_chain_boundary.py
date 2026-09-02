import pytest
import sympy as sp
from experiments.discovery import compareToExpected, runSystemDiscovery
from experiments.generate_dataset import datasetPath, generateSystemDatasets
from experiments.systems import SYSTEMS

SYSTEM_NAME = "anharmonic_chain_blind"


@pytest.fixture(scope="module")
def cleanChainRecovery():
    generateSystemDatasets(SYSTEM_NAME, [0.0])
    system = SYSTEMS[SYSTEM_NAME]
    csvPath = datasetPath(system, 0.0)
    discovered, _logFrame, _tolerances = runSystemDiscovery(system, csvPath, chunkRows=120_000)
    expected = system.expectedScaledLagrangian(system.noCoords)
    comparison = compareToExpected(discovered, expected, noCoords=system.noCoords)
    return discovered, comparison


def test_clean_blind_chain_recovers_every_site_including_the_boundary(cleanChainRecovery):
    _discovered, comparison = cleanChainRecovery
    assert comparison.missingMonomials == []
    assert comparison.spuriousMonomials == []
    assert comparison.success
    assert comparison.structurallyEquivalent


def test_boundary_site_q5_carries_the_true_anharmonic_terms_and_no_ghost_shadow(cleanChainRecovery):
    discovered, _comparison = cleanChainRecovery
    coefficients = sp.expand(discovered.rawExpression).as_coefficients_dict()
    q5, v5 = sp.Symbol("q5"), sp.Symbol("v5")

    assert float(coefficients[q5 ** 3]) == pytest.approx(-0.08, abs=5e-3)
    assert float(coefficients[q5 ** 4]) == pytest.approx(-0.16, abs=5e-3)

    for shadow in (q5 ** 2 * v5 ** 2, q5 * v5 ** 2, q5):
        assert coefficients.get(shadow, 0) == 0, f"Open problem C shadow term back: {shadow}"
