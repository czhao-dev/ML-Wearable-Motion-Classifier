from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_modeling.metrics import regression_metrics
from climate_modeling.models import RidgeRegressor, SeasonalNaiveModel


class RidgeRegressorTests(unittest.TestCase):
    def test_ridge_learns_simple_linear_relationship(self):
        features = [[float(value)] for value in range(1, 8)]
        target = [2.0 * row[0] + 1.0 for row in features]

        model = RidgeRegressor(alpha=0.01).fit(features, target)
        predictions = model.predict([[8.0], [9.0]])

        self.assertAlmostEqual(predictions[0], 17.0, delta=0.2)
        self.assertAlmostEqual(predictions[1], 19.0, delta=0.2)

    def test_regression_metrics_identical_predictions(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_fit_rejects_mismatched_target_length(self):
        with self.assertRaisesRegex(ValueError, "row counts"):
            RidgeRegressor().fit([[1.0], [2.0]], [1.0])

    def test_fit_rejects_ragged_features(self):
        with self.assertRaisesRegex(ValueError, "same width"):
            RidgeRegressor().fit([[1.0], [2.0, 3.0]], [1.0, 2.0])

    def test_predict_rejects_wrong_feature_width(self):
        model = RidgeRegressor(alpha=0.1).fit([[1.0, 2.0], [2.0, 3.0]], [3.0, 5.0])

        with self.assertRaisesRegex(ValueError, "Expected 2 features"):
            model.predict([[1.0]])

    def test_non_negative_predictions_are_clamped(self):
        model = RidgeRegressor(alpha=0.1, non_negative=True).fit(
            [[1.0], [2.0], [3.0]],
            [-1.0, -2.0, -3.0],
        )

        self.assertEqual(model.predict([[10.0]]), [0.0])


class SeasonalNaiveModelTests(unittest.TestCase):
    def test_fit_rejects_empty_records(self):
        with self.assertRaisesRegex(ValueError, "empty record"):
            SeasonalNaiveModel("TOBS").fit([])


if __name__ == "__main__":
    unittest.main()
