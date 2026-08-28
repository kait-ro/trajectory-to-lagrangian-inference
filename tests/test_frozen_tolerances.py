import dataclasses

from experiments.discovery import FROZEN_TOLERANCES, lockedDiscoveryTolerances
from experiments.systems import SYSTEMS


def test_physical_system_has_no_tolerance_fields():
    fields = {f.name for f in dataclasses.fields(next(iter(SYSTEMS.values())))}
    for forbidden in ("degreeCap", "residualRmsTolerance", "correlationCutoff", "stagnationTolerance"):
        assert forbidden not in fields, f"PhysicalSystem still carries a per-system tolerance: {forbidden}"


def test_every_system_gets_the_identical_frozen_set():
    perSystem = {name: lockedDiscoveryTolerances(system) for name, system in SYSTEMS.items()}
    values = list(perSystem.values())
    assert all(v == FROZEN_TOLERANCES for v in values), perSystem
    assert all(v == values[0] for v in values), "systems disagree on the tolerance set"


def test_frozen_set_matches_library_defaults():
    from finding_L import gram_forward_select, stopping_conditions

    assert FROZEN_TOLERANCES["correlationCutoff"] == stopping_conditions.checkCorrelationCutoff.__defaults__[0]
    assert FROZEN_TOLERANCES["residualRmsTolerance"] == gram_forward_select.checkResidualToleranceFromGram.__defaults__[0]
    assert FROZEN_TOLERANCES["pruneRelativeThreshold"] == gram_forward_select.pruneNearZeroCoefficients.__defaults__[0]
    assert FROZEN_TOLERANCES["stagnationTolerance"] == stopping_conditions.checkResidualStagnation.__defaults__[0]
    assert FROZEN_TOLERANCES["stagnationPatience"] == stopping_conditions.checkResidualStagnation.__defaults__[1]


def test_maxrounds_is_a_generous_budget_not_a_decider():
    for system in SYSTEMS.values():
        assert system.maxRounds >= 120, f"{system.name} maxRounds={system.maxRounds} is too tight"
