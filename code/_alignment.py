"""Check time and grid alignment before combining datasets positionally."""

import numpy as np


def validate_alignment(*datasets):
    reference = datasets[0]
    for dataset in datasets[1:]:
        for coordinate in ("time", "lat", "lon"):
            expected = reference[coordinate].values
            actual = dataset[coordinate].values
            if expected.shape != actual.shape:
                raise ValueError(f"Dataset {coordinate} coordinates have different shapes")
            if coordinate == "time":
                matches = np.array_equal(expected, actual)
            else:
                matches = np.allclose(expected, actual, rtol=0, atol=1e-4)
            if not matches:
                raise ValueError(f"Dataset {coordinate} coordinates do not match")
