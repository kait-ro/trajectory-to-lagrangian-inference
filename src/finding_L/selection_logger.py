import numpy as np
import pandas as pd


def recordSelectionRound(selectionLog: list, roundNumber: int, bestScore: float, residual: np.ndarray):
    residualRms = np.sqrt(np.mean(residual ** 2))
    selectionLog.append(
        {
            "round": roundNumber,
            "bestReserveScore": bestScore,
            "residualRms": residualRms,
        }
    )
    return selectionLog


def toSelectionLogFrame(selectionLog: list) -> pd.DataFrame:
    return pd.DataFrame(selectionLog)