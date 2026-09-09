import numpy as np

from at2_experiments.modeling.evaluate import evaluate_regression


def test_evaluate_regression_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = evaluate_regression(y_true, y_pred, "Test")
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0


def test_evaluate_regression_returns_expected_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.5])
    result = evaluate_regression(y_true, y_pred, "Test")
    assert set(result.keys()) == {"mae", "rmse"}
