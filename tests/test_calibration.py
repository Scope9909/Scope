"""Focused checks for the frozen Validation-to-Test calibration contract."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay.calibration import (  # noqa: E402
    fit_and_apply,
    metrics,
    target_time_bucket_ids,
)


class CalibrationTests(unittest.TestCase):
    def test_metrics_mask_then_average_over_horizons(self):
        prediction = np.array([[[2.0], [4.0]], [[9.0], [8.0]]])
        truth = np.array([[[1.0], [2.0]], [[0.0], [4.0]]])
        result = metrics(prediction, truth)
        self.assertAlmostEqual(result["mae"], 2.0)
        self.assertAlmostEqual(result["rmse"], (1.0 + np.sqrt(10.0)) / 2.0)
        self.assertAlmostEqual(result["mape"], 1.0)

    def test_target_time_uses_forecast_offsets(self):
        actual = target_time_bucket_ids(
            np.array([5, 23]), num_horizons=3,
            slices_per_day=24, num_time_bins=8,
        )
        expected = np.array([[2, 2, 2], [0, 0, 0]])
        np.testing.assert_array_equal(actual, expected)

    def test_perfect_predictions_remain_unchanged(self):
        values = np.arange(1, 49, dtype=np.float64).reshape(8, 3, 2)
        endpoints = np.arange(8, dtype=np.int64)
        smooth, calibrated, _ = fit_and_apply(
            values, values, endpoints, values.copy(), endpoints,
            num_bins=4, smooth_shrinkage=0.55,
            num_time_bins=8, time_shrinkage=0.35,
            slices_per_day=24,
        )
        np.testing.assert_array_equal(smooth, values)
        np.testing.assert_array_equal(calibrated, values)


if __name__ == "__main__":
    unittest.main()
