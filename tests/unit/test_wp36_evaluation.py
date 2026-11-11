"""Kiểm thử đơn vị (Unit Tests) cho module WP3.6 Research Comparison & Evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.research_comparison import compute_quant_metrics


def test_compute_quant_metrics_outputs_and_bounds():
    """Kiểm tra tính đúng đắn, giới hạn và tương quan của compute_quant_metrics."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    ranking_score = np.array([0.1, 0.9, 0.2, 0.85, 0.15, 0.95, 0.88, 0.05])
    calibrated_prob = ranking_score.copy()
    fwd_returns = np.array([-0.02, 0.03, -0.01, 0.025, -0.015, 0.04, 0.035, -0.03])

    m = compute_quant_metrics(y_true, y_pred, ranking_score, calibrated_prob, fwd_returns)

    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert 0.0 <= m["brier_score"] <= 0.05
    assert m["log_loss"] > 0.0
    assert m["spearman_rank_ic"] > 0.9
    assert m["pearson_ic"] > 0.9


def test_platt_calibrator_monotonicity_and_bounds():
    """Kiểm tra Platt Calibrator chuyển đổi decision score sang xác suất đơn điệu trong (0, 1)."""
    rng = np.random.default_rng(42)
    scores = np.linspace(-3.0, 3.0, 100)
    # Xác suất sigmoid thực
    true_probs = 1.0 / (1.0 + np.exp(-scores))
    labels = rng.binomial(1, true_probs)

    clf = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    clf.fit(scores.reshape(-1, 1), labels)

    test_scores = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    cal_probs = clf.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    # Giới hạn
    assert (cal_probs > 0.0).all() and (cal_probs < 1.0).all()
    # Tính đơn điệu tăng
    assert np.all(np.diff(cal_probs) > 0)


def test_agreement_rate_calculation():
    """Kiểm tra tính toán tỷ lệ đồng thuận nhãn giữa các mô hình."""
    y1 = np.array([1, 0, 1, 1])
    y2 = np.array([1, 0, 0, 1])
    y3 = np.array([1, 1, 0, 1])

    agr_12 = float(np.mean(y1 == y2))
    agr_all = float(np.mean((y1 == y2) & (y2 == y3)))

    assert agr_12 == 0.75
    assert agr_all == 0.50

