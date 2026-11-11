"""Kiểm tra đơn vị cho module LightGBM modeling WP3.4 bằng dữ liệu tổng hợp."""

from dataclasses import replace
from pathlib import Path
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.data import ModelingView
from src.models.temporal_split import TemporalSplit
from src.models.train_lightgbm import (
    MODEL_CONFIGS, LGBMModelConfig, aggregate_results, evaluate_fold,
    fit_and_export_final_development_models, make_estimator, select_configurations,
)


@pytest.fixture
def synthetic_sample():
    """Tạo view và fold phát triển tổng hợp để kiểm tra logic modeling độc lập."""
    np.random.seed(42)
    n_samples = 80
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="B")
    feature_cols = [f"feat_{i}" for i in range(10)]
    X_vals = np.random.randn(n_samples, 10)
    # Tạo tín hiệu nhẹ cho đặc trưng đầu tiên
    y_vals = (X_vals[:, 0] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    X = pd.DataFrame(X_vals, columns=feature_cols, index=dates)
    y = pd.Series(y_vals, index=dates)
    metadata = pd.DataFrame({
        "signal_date": dates,
        "label_end_date": dates + pd.Timedelta(days=1),
    }, index=dates)

    train_indices = tuple(range(50))
    valid_indices = tuple(range(55, 75))

    fold = TemporalSplit(
        fold_name="dev_1",
        rq="RQ1",
        role="development",
        train_indices=train_indices,
        validation_indices=valid_indices,
        train_master_positions=train_indices,
        validation_master_positions=valid_indices,
        train_start_date=dates[0],
        train_end_date=dates[49],
        validation_start_date=dates[55],
        validation_end_date=dates[74],
        evaluation_boundary_date=dates[55],
        boundary_excluded_count=0,
        purged_count=0,
        gap_count=5,
        overlap_exclusions=0,
        excluded_count=5,
        train_class_counts={"0": int((y.iloc[list(train_indices)] == 0).sum()),
                            "1": int((y.iloc[list(train_indices)] == 1).sum())},
        validation_class_counts={"0": int((y.iloc[list(valid_indices)] == 0).sum()),
                                "1": int((y.iloc[list(valid_indices)] == 1).sum())},
    )
    return ModelingView(X, y, metadata), fold


def test_make_estimator_families():
    dummy_cfg = MODEL_CONFIGS[0]
    dummy_est = make_estimator(dummy_cfg)
    assert isinstance(dummy_est, DummyClassifier)

    lgbm_cfg = next(c for c in MODEL_CONFIGS if c.model_name == "lightgbm")
    lgbm_est = make_estimator(lgbm_cfg)
    assert isinstance(lgbm_est, lgb.LGBMClassifier)
    assert lgbm_est.max_depth == lgbm_cfg.max_depth
    assert lgbm_est.num_leaves == lgbm_cfg.num_leaves
    assert lgbm_est.learning_rate == lgbm_cfg.learning_rate
    assert lgbm_est.min_child_samples == lgbm_cfg.min_child_samples
    assert lgbm_est.reg_lambda == lgbm_cfg.reg_lambda
    assert lgbm_est.random_state == 42
    assert lgbm_est.n_jobs == 1
    assert lgbm_est.deterministic is True


def test_make_estimator_rejects_invalid_config():
    bad_cfg = LGBMModelConfig("bad", "invalid_name")
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        make_estimator(bad_cfg)

    bad_depth = LGBMModelConfig("bad_depth", "lightgbm", max_depth=-1, num_leaves=7, learning_rate=0.03, min_child_samples=40, reg_lambda=1.0)
    with pytest.raises(ValueError, match="không hợp lệ"):
        make_estimator(bad_depth)


def test_evaluate_fold_lightgbm_outputs(synthetic_sample):
    view, fold = synthetic_sample
    cfg = LGBMModelConfig(
        config_id="test_lgbm",
        model_name="lightgbm",
        max_depth=3,
        num_leaves=7,
        learning_rate=0.05,
        min_child_samples=10,
        reg_lambda=1.0,
        class_weight=None,
        n_estimators=50,
        random_state=42,
    )
    record, pred_df, logs, imp_records = evaluate_fold(
        view, "RQ1", fold, cfg, compute_perm_importance=True
    )

    assert record["rq"] == "RQ1"
    assert record["fold_name"] == "dev_1"
    assert record["train_samples"] == len(fold.train_indices)
    assert record["validation_samples"] == len(fold.validation_indices)
    assert isinstance(record["best_iteration"], int) and record["best_iteration"] >= 1
    assert record["train_balanced_accuracy"] is not None
    assert record["train_roc_auc"] is not None
    assert record["generalization_gap"] is not None

    # Kiểm tra predictions DataFrame
    assert len(pred_df) == len(fold.validation_indices)
    assert pred_df.score_type.iloc[0] == "predicted_probability"
    assert ((pred_df.ranking_score >= 0.0) & (pred_df.ranking_score <= 1.0)).all()
    assert set(pred_df.y_pred.unique()).issubset({0, 1})

    # Kiểm tra training logs
    assert len(logs) >= record["best_iteration"]
    best_log_entries = [log for log in logs if log["is_best_iteration"]]
    assert len(best_log_entries) == 1
    assert best_log_entries[0]["iteration"] == record["best_iteration"]

    # Kiểm tra feature importance
    assert len(imp_records) == len(view.X.columns)
    first_imp = imp_records[0]
    assert "split_importance" in first_imp
    assert "gain_importance" in first_imp
    assert "perm_importance_mean" in first_imp
    assert "perm_importance_std" in first_imp


def test_evaluate_fold_rejects_reserved_and_mismatched(synthetic_sample):
    view, fold = synthetic_sample
    cfg = MODEL_CONFIGS[0]
    with pytest.raises(ValueError, match="development"):
        evaluate_fold(view, "RQ1", replace(fold, role="reserved_test"), cfg)
    with pytest.raises(ValueError, match="RQ"):
        evaluate_fold(view, "RQ2", fold, cfg)


def test_aggregation_and_selection_tie_breaker():
    metrics_data = [
        # RQ1 Config A: BAcc = 0.55, std = 0.02
        {"rq": "RQ1", "config_id": "cfg_a", "model_name": "lightgbm", "fold_name": "dev_1",
         "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 1.0, "class_weight": None,
         "accuracy": 0.55, "balanced_accuracy": 0.55, "precision": 0.55, "recall": 0.55, "f1": 0.55, "roc_auc": 0.55, "average_precision": 0.55, "positive_prediction_rate": 0.5,
         "best_iteration": 20, "train_balanced_accuracy": 0.6, "train_roc_auc": 0.6, "generalization_gap": 0.05},
        {"rq": "RQ1", "config_id": "cfg_a", "model_name": "lightgbm", "fold_name": "dev_2",
         "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 1.0, "class_weight": None,
         "accuracy": 0.55, "balanced_accuracy": 0.55, "precision": 0.55, "recall": 0.55, "f1": 0.55, "roc_auc": 0.55, "average_precision": 0.55, "positive_prediction_rate": 0.5,
         "best_iteration": 25, "train_balanced_accuracy": 0.6, "train_roc_auc": 0.6, "generalization_gap": 0.05},
        {"rq": "RQ1", "config_id": "cfg_a", "model_name": "lightgbm", "fold_name": "dev_3",
         "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 1.0, "class_weight": None,
         "accuracy": 0.55, "balanced_accuracy": 0.55, "precision": 0.55, "recall": 0.55, "f1": 0.55, "roc_auc": 0.55, "average_precision": 0.55, "positive_prediction_rate": 0.5,
         "best_iteration": 30, "train_balanced_accuracy": 0.6, "train_roc_auc": 0.6, "generalization_gap": 0.05},

        # RQ1 Dummy
        {"rq": "RQ1", "config_id": "dummy_prior", "model_name": "dummy", "fold_name": "dev_1",
         "max_depth": None, "num_leaves": None, "learning_rate": None, "min_child_samples": None, "reg_lambda": None, "class_weight": None,
         "accuracy": 0.5, "balanced_accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5, "average_precision": 0.5, "positive_prediction_rate": 0.0,
         "best_iteration": np.nan, "train_balanced_accuracy": 0.5, "train_roc_auc": 0.5, "generalization_gap": 0.0},
        {"rq": "RQ1", "config_id": "dummy_prior", "model_name": "dummy", "fold_name": "dev_2",
         "max_depth": None, "num_leaves": None, "learning_rate": None, "min_child_samples": None, "reg_lambda": None, "class_weight": None,
         "accuracy": 0.5, "balanced_accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5, "average_precision": 0.5, "positive_prediction_rate": 0.0,
         "best_iteration": np.nan, "train_balanced_accuracy": 0.5, "train_roc_auc": 0.5, "generalization_gap": 0.0},
        {"rq": "RQ1", "config_id": "dummy_prior", "model_name": "dummy", "fold_name": "dev_3",
         "max_depth": None, "num_leaves": None, "learning_rate": None, "min_child_samples": None, "reg_lambda": None, "class_weight": None,
         "accuracy": 0.5, "balanced_accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5, "average_precision": 0.5, "positive_prediction_rate": 0.0,
         "best_iteration": np.nan, "train_balanced_accuracy": 0.5, "train_roc_auc": 0.5, "generalization_gap": 0.0},
    ]

    metrics_df = pd.DataFrame(metrics_data)
    comparison = aggregate_results(metrics_df)
    assert len(comparison) == 2
    row_a = comparison[comparison.config_id == "cfg_a"].iloc[0]
    assert row_a["mean_balanced_accuracy"] == pytest.approx(0.55)
    assert row_a["delta_mean_balanced_accuracy_vs_dummy"] == pytest.approx(0.05)

    # Thêm cấu hình B có cùng mean nhưng std thấp hơn để kiểm tra tie-breaker
    metrics_data_tie = metrics_data + [
        {"rq": "RQ1", "config_id": "cfg_b", "model_name": "lightgbm", "fold_name": "dev_1",
         "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 1.0, "class_weight": None,
         "accuracy": 0.55, "balanced_accuracy": 0.55, "precision": 0.55, "recall": 0.55, "f1": 0.55, "roc_auc": 0.55, "average_precision": 0.55, "positive_prediction_rate": 0.5,
         "best_iteration": 20, "train_balanced_accuracy": 0.6, "train_roc_auc": 0.6, "generalization_gap": 0.05},
        {"rq": "RQ1", "config_id": "cfg_b", "model_name": "lightgbm", "fold_name": "dev_2",
         "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 1.0, "class_weight": None,
         "accuracy": 0.55, "balanced_accuracy": 0.55, "precision": 0.55, "recall": 0.55, "f1": 0.55, "roc_auc": 0.55, "average_precision": 0.55, "positive_prediction_rate": 0.5,
         "best_iteration": 25, "train_balanced_accuracy": 0.6, "train_roc_auc": 0.6, "generalization_gap": 0.05},
        {"rq": "RQ1", "config_id": "cfg_b", "model_name": "lightgbm", "fold_name": "dev_3",
         "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 40, "reg_lambda": 1.0, "class_weight": None,
         "accuracy": 0.55, "balanced_accuracy": 0.55, "precision": 0.55, "recall": 0.55, "f1": 0.55, "roc_auc": 0.55, "average_precision": 0.55, "positive_prediction_rate": 0.5,
         "best_iteration": 30, "train_balanced_accuracy": 0.6, "train_roc_auc": 0.6, "generalization_gap": 0.05},
    ]
    comparison_tie = aggregate_results(pd.DataFrame(metrics_data_tie))
    selected = select_configurations(comparison_tie)
    assert selected["RQ1"]["config_id"] in ("cfg_a", "cfg_b")
