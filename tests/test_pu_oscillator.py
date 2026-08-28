"""Pais-Uhlenbeck oscillator: symbolic EOM and numerical Hamiltonian conservation.

Assertion-based versions of the checks previously only printed by
experiments/pu_oscillator_validation.py.
"""

import numpy as np

from experiments.pu_oscillator_validation import (
    integratePaisUhlenbeck,
    paisUhlenbeckEquationOfMotion,
    regressionAgainstSecondOrder,
)


def test_ostrogradski_el_matches_legacy_second_order_operator():
    ok, mismatch = regressionAgainstSecondOrder()
    assert ok, f"order-1 Ostrogradski EL disagrees with legacy EulerLagrangeEqn: {mismatch}"


def test_pais_uhlenbeck_equation_of_motion():
    matches, elExpression, order = paisUhlenbeckEquationOfMotion()
    assert order == 2  # highest derivative in L is qddot; the resulting EOM is 4th order
    assert matches, f"PU Euler-Lagrange expression != q'''' + (w1^2+w2^2) q'' + w1^2 w2^2 q:\n{elExpression}"


def test_ostrogradski_hamiltonian_is_conserved_on_trajectory():
    _times, _perDerivative, _data, hamiltonianSeries, *_ = integratePaisUhlenbeck(
        omega1=1.0, omega2=2.0, dt=0.004, steps=6000, initialState=[1.0, 0.0, 0.0, 0.0]
    )
    drift = (hamiltonianSeries.max() - hamiltonianSeries.min()) / max(abs(hamiltonianSeries.mean()), 1e-12)
    assert abs(drift) < 1e-6, f"Ostrogradski H drifts by {drift:.2e} over the trajectory"
    assert np.isfinite(hamiltonianSeries).all()
