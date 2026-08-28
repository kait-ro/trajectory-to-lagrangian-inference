import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def estimateNoiseStd(signal):
    secondDifference = np.diff(np.asarray(signal, dtype=float), n=2)
    return np.std(secondDifference) / np.sqrt(6.0)


def finiteDifferenceDerivatives(signal, dt, maxOrder):
    derivatives = [np.asarray(signal, dtype=float)]
    current = derivatives[0]
    for _ in range(maxOrder):
        current = np.gradient(current, dt, edge_order=2)
        derivatives.append(current)
    return derivatives


def savitzkyGolayDerivatives(signal, dt, maxOrder, windowLength=None, polyOrder=None):
    signal = np.asarray(signal, dtype=float)
    polyOrder = max(maxOrder + 1, 4) if polyOrder is None else polyOrder
    if windowLength is None:
        windowLength = max(polyOrder + 2, round(len(signal) * 0.02) | 1)
    if windowLength % 2 == 0:
        windowLength += 1
    windowLength = min(windowLength, len(signal) - (1 - len(signal) % 2))

    derivatives = [savgol_filter(signal, windowLength, polyOrder, deriv=0, delta=dt)]
    for order in range(1, maxOrder + 1):
        derivatives.append(savgol_filter(signal, windowLength, polyOrder, deriv=order, delta=dt))
    return derivatives


def smoothingSplineDerivatives(signal, dt, maxOrder, smoothing=None, degree=5):
    signal = np.asarray(signal, dtype=float)
    times = np.arange(len(signal)) * dt
    degree = max(degree, maxOrder + 1)
    if degree > 5:
        raise ValueError("smoothing spline degree capped at 5; use savitzkyGolayDerivatives for orders above 4")

    if smoothing is None:
        noiseStd = estimateNoiseStd(signal)
        smoothing = len(signal) * noiseStd ** 2 if noiseStd > 0 else len(signal) * 1e-24

    spline = UnivariateSpline(times, signal, k=degree, s=smoothing)
    derivatives = [spline(times)]
    for order in range(1, maxOrder + 1):
        derivatives.append(spline.derivative(n=order)(times))
    return derivatives


def relativeL2Error(estimate, truth, trim=0.05):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    start = int(len(truth) * trim)
    stop = len(truth) - start
    numerator = np.linalg.norm(estimate[start:stop] - truth[start:stop])
    denominator = np.linalg.norm(truth[start:stop])
    return numerator / denominator if denominator > 0 else np.nan


DIFFERENTIATION_METHODS = {
    "finite_difference": finiteDifferenceDerivatives,
    "savitzky_golay": savitzkyGolayDerivatives,
    "smoothing_spline": smoothingSplineDerivatives,
}
