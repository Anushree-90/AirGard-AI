import unittest
from datetime import datetime, timedelta

import numpy as np

from ml_model import MAX_LAG_HOURS, build_feature_row


class FeatureConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = [datetime(2026, 1, 1) + timedelta(hours=index) for index in range(240)]
        self.aqi = np.arange(240, dtype=float)
        self.pm25 = self.aqi * 0.5

    def test_features_use_only_values_at_or_before_forecast_origin(self) -> None:
        origin = MAX_LAG_HOURS + 10
        before = build_feature_row(self.times, self.aqi, self.pm25, origin, 24)
        changed_aqi = self.aqi.copy()
        changed_pm25 = self.pm25.copy()
        changed_aqi[origin + 1 :] = 9999
        changed_pm25[origin + 1 :] = 8888
        after = build_feature_row(self.times, changed_aqi, changed_pm25, origin, 24)
        self.assertEqual(before, after)

    def test_rejects_insufficient_lag_history_and_invalid_horizon(self) -> None:
        with self.assertRaises(ValueError):
            build_feature_row(self.times, self.aqi, self.pm25, MAX_LAG_HOURS - 1, 1)
        with self.assertRaises(ValueError):
            build_feature_row(self.times, self.aqi, self.pm25, MAX_LAG_HOURS, 25)


if __name__ == "__main__":
    unittest.main()
