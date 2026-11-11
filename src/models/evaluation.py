"""Tính chỉ số phân loại nhị phân cho dữ liệu đánh giá theo thời gian."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Tính chỉ số phân loại và ranking cho lớp dương 1.

    Args:
        y_true: Nhãn thực nhị phân của một tập validation.
        y_pred: Nhãn dự đoán tương ứng, theo cùng thứ tự.
        scores: Điểm xếp hạng hữu hạn cho lớp dương; không nhất thiết là xác suất.

    Returns:
        Chỉ số, confusion matrix và cảnh báo khi validation chỉ có một lớp.

    Raises:
        ValueError: Khi dữ liệu rỗng, sai kích thước, sai nhãn hoặc điểm không hữu hạn.
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    score = np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or pred.ndim != 1 or score.ndim != 1 or not len(truth)
            or len(truth) != len(pred) or len(truth) != len(score)
            or not np.isin(truth, [0, 1]).all() or not np.isin(pred, [0, 1]).all()
            or not np.isfinite(score).all()):
        raise ValueError("Nhãn hoặc điểm đánh giá không hợp lệ")
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(truth)) == 2
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if two_classes else None,
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)) if tp + fn else None,
        "f1": float(f1_score(truth, pred, zero_division=0)) if 2 * tp + fp + fn else None,
        "roc_auc": float(roc_auc_score(truth, score)) if two_classes else None,
        "average_precision": float(average_precision_score(truth, score)) if two_classes else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "positive_prediction_rate": float(np.mean(pred == 1)),
        "support_0": int(np.sum(truth == 0)), "support_1": int(np.sum(truth == 1)),
        "warning": "Validation chỉ có một lớp; Balanced Accuracy và ranking metrics không xác định." if not two_classes else "",
    }
