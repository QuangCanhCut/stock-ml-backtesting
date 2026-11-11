"""Kiểm tra SVM baseline và chống rò rỉ bằng dữ liệu tổng hợp."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.data import ModelingView
from src.models.evaluation import binary_metrics
from src.models.svm_baseline import (
    MODEL_CONFIGS, evaluate_fold, make_estimator, select_configurations,
)
from src.models.temporal_split import TemporalSplit


@pytest.fixture
def sample():
    """Tạo fold mà Validation có phân phối cách xa Train để phát hiện leakage."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    values = np.arange(50, dtype=float)
    X = pd.DataFrame({"a": values, "b": np.sin(values)}, index=dates)
    X.iloc[35:, :] += 10000
    y = pd.Series((np.arange(50) % 2).astype(int), index=dates)
    metadata = pd.DataFrame({"signal_date": dates,
                             "label_end_date": dates + pd.Timedelta(days=1)}, index=dates)
    train, valid = tuple(range(30)), tuple(range(35, 45))
    fold = TemporalSplit("dev_1", "RQ1", "development", train, valid, train, valid,
                         dates[0], dates[29], dates[35], dates[44], dates[35],
                         0, 0, 5, 0, 5, {"0": 15, "1": 15}, {"0": 5, "1": 5})
    return ModelingView(X, y, metadata), fold


def test_scaler_fits_only_train_and_is_independent(sample, monkeypatch):
    view, fold = sample
    means = []
    original = StandardScaler.fit

    def capture(self, X, y=None, **kwargs):
        result = original(self, X, y, **kwargs)
        means.append((self.mean_.copy(), self.scale_.copy(), id(self)))
        return result

    monkeypatch.setattr(StandardScaler, "fit", capture)
    config = next(c for c in MODEL_CONFIGS if c.model_name == "linear")
    _, first = evaluate_fold(view, "RQ1", fold, config)
    _, second = evaluate_fold(view, "RQ1", fold, config)
    expected = view.X.iloc[list(fold.train_indices)]
    np.testing.assert_allclose(means[0][0], expected.mean().to_numpy())
    np.testing.assert_allclose(means[0][1], expected.std(ddof=0).to_numpy())
    np.testing.assert_allclose(means[0][0], means[1][0])
    pd.testing.assert_frame_equal(first, second)
    assert first.signal_date.tolist() == view.X.index[list(fold.validation_indices)].tolist()
    assert (first.score_type == "decision_score").all()


@pytest.mark.parametrize("model_name", ["dummy", "linear", "rbf"])
def test_model_families_and_fit_scope(sample, model_name, monkeypatch):
    view, fold = sample
    config = next(c for c in MODEL_CONFIGS if c.model_name == model_name)
    estimator1, estimator2 = make_estimator(config), make_estimator(config)
    assert estimator1 is not estimator2
    if model_name != "dummy":
        assert estimator1.named_steps["scaler"] is not estimator2.named_steps["scaler"]
        assert estimator1.named_steps["classifier"].kernel == model_name
        assert estimator1.named_steps["classifier"].probability is False
    record, predictions = evaluate_fold(view, "RQ1", fold, config)
    assert record["train_samples"] == len(fold.train_indices)
    assert len(predictions) == len(fold.validation_indices)
    if model_name == "dummy":
        assert isinstance(estimator1, DummyClassifier)
        assert (predictions.score_type == "train_prior_probability").all()
        assert predictions.ranking_score.eq(view.y.iloc[list(fold.train_indices)].mean()).all()


def test_reject_reserved_and_mismatched_fold(sample):
    view, fold = sample
    with pytest.raises(ValueError, match="development"):
        evaluate_fold(view, "RQ1", replace(fold, role="reserved_test"), MODEL_CONFIGS[0])
    with pytest.raises(ValueError, match="RQ"):
        evaluate_fold(view, "RQ2", fold, MODEL_CONFIGS[0])


def test_fit_receives_only_training_labels(sample, monkeypatch):
    view, fold = sample
    seen = []
    original = SVC.fit

    def capture(self, X, y, **kwargs):
        seen.append(np.asarray(y).copy())
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(SVC, "fit", capture)
    config = next(c for c in MODEL_CONFIGS if c.model_name == "linear")
    _, first = evaluate_fold(view, "RQ1", fold, config)
    view.y.iloc[list(fold.validation_indices)] = 1 - view.y.iloc[list(fold.validation_indices)]
    _, second = evaluate_fold(view, "RQ1", fold, config)
    expected = view.y.iloc[list(fold.train_indices)].to_numpy()
    np.testing.assert_array_equal(seen[0], expected)
    np.testing.assert_array_equal(seen[1], expected)
    np.testing.assert_allclose(first.ranking_score, second.ranking_score)
    np.testing.assert_array_equal(first.y_pred, second.y_pred)


def test_metrics_and_one_class():
    result = binary_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]),
                            np.array([0.1, 0.9, 0.8, 0.2]))
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["balanced_accuracy"] == 0.5
    assert result["roc_auc"] == pytest.approx(0.75)
    single = binary_metrics(np.array([0, 0]), np.array([0, 0]), np.array([0.1, 0.2]))
    assert single["roc_auc"] is None and single["average_precision"] is None
    assert single["balanced_accuracy"] is None and single["warning"]
    assert single["recall"] is None and single["f1"] is None


def test_selection_tie_break_is_deterministic():
    comparison = pd.DataFrame([
        {"rq": "RQ1", "config_id": "rbf_c0.1_none", "model_name": "rbf",
         "mean_balanced_accuracy": .6, "std_balanced_accuracy": .1,
         "delta_mean_balanced_accuracy_vs_dummy": .1},
        {"rq": "RQ1", "config_id": "linear_c1_none", "model_name": "linear",
         "mean_balanced_accuracy": .6, "std_balanced_accuracy": .1,
         "delta_mean_balanced_accuracy_vs_dummy": .1},
        {"rq": "RQ1", "config_id": "linear_c0.1_none", "model_name": "linear",
         "mean_balanced_accuracy": .6, "std_balanced_accuracy": .1,
         "delta_mean_balanced_accuracy_vs_dummy": .1},
    ])
    assert select_configurations(comparison)["RQ1"]["config_id"] == "linear_c0.1_none"
