import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from finding_L.main_streaming import runDiscoveryStreaming
from finding_L.pipeline import endToEndPipeline

CSV_HEADER_COORD_PATTERN = re.compile(r"q(\d+)$")


def inferNoFieldsFromCsvHeader(csvPath):
    header = pd.read_csv(csvPath, nrows=0).columns.tolist()
    indices = {int(match.group(1)) for column in header if (match := CSV_HEADER_COORD_PATTERN.fullmatch(column))}
    if not indices:
        raise ValueError(f"could not infer coordinate count from CSV header: {header}")
    return max(indices) + 1


def discoverLagrangian(
    trajectory,
    dt=None,
    noFields=None,
    maxOrder=3,
    libraryMaxDegree=None,
    degreeCap=4,
    chunkRows=200_000,
    selector="lasso",
):
    if isinstance(trajectory, (str, Path)):
        resolvedNoFields = inferNoFieldsFromCsvHeader(trajectory) if noFields is None else noFields
        discovered, _log = runDiscoveryStreaming(
            str(trajectory),
            noCoords=resolvedNoFields,
            degreeCap=degreeCap,
            chunkRows=chunkRows,
            selector=selector,
        )
        return discovered

    positions = np.asarray(trajectory, dtype=float)
    if dt is None:
        raise ValueError("dt is required when trajectory is an array of noisy positions, not a CSV path")
    resolvedNoFields = (1 if positions.ndim == 1 else positions.shape[1]) if noFields is None else noFields
    resolvedLibraryMaxDegree = 2 if libraryMaxDegree is None else libraryMaxDegree
    return endToEndPipeline(
        positions, dt, maxOrder=maxOrder, libraryMaxDegree=resolvedLibraryMaxDegree, noFields=resolvedNoFields
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = discoverLagrangian(sys.argv[1])
    else:
        repoRoot = Path(__file__).resolve().parents[2]
        csvPath = str(repoRoot / "assets/anharmonic_chain_blind_n6_noise0.csv")
        result = discoverLagrangian(csvPath)

    print(result.text if hasattr(result, "text") else result.summary())
