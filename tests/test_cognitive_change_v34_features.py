from __future__ import annotations

import numpy as np

from training.cognitive_change_clue.v34_features import fit_standardizer


def test_fold_standardizer_uses_population_statistics_and_floors_constants() -> None:
    mean, std = fit_standardizer(
        [
            np.asarray([1.0, 5.0], dtype=np.float32),
            np.asarray([3.0, 5.0], dtype=np.float32),
        ],
        dimension=2,
    )
    np.testing.assert_array_equal(mean, np.asarray([2.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(std, np.asarray([1.0, 1.0], dtype=np.float32))
