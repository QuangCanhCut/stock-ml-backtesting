"""Thực nghiệm Dummy và SVM trên các development folds đã được xác minh."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.data import ModelingView
from src.models.evaluation import binary_metrics
from src.models.modeling_safety import require_development_split
from src.models.temporal_split import TEST_START_YEAR, TemporalSplit


@dataclass(frozen=True)
class ModelConfig:
    """Cấu hình cố định trước khi quan sát kết quả validation."""

    config_id: str
    model_name: str
    C: float | None = None
    class_weight: str | None = None
    gamma: str = "scale"


MODEL_CONFIGS = (ModelConfig("dummy_prior", "dummy"),) + tuple(
    ModelConfig(f"{family}_c{c:g}_{weight or 'none'}", family, c, weight)
    for family in ("linear", "rbf")
    for c in (0.1, 1.0, 10.0)
    for weight in (None, "balanced")
)
METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "roc_auc", "average_precision", "positive_prediction_rate")


def make_estimator(config: ModelConfig) -> DummyClassifier | Pipeline:
    """Tạo estimator mới; SVM luôn có scaler được fit cùng Train của fold.

    Args:
        config: Cấu hình Dummy, Linear SVM hoặc RBF SVM đã chốt.

    Returns:
        DummyClassifier hoặc Pipeline mới chưa được fit.

    Raises:
        ValueError: Khi tên mô hình hoặc tham số không được hỗ trợ.
    """
    if config.model_name == "dummy":
        return DummyClassifier(strategy="prior")
    if (config.model_name not in {"linear", "rbf"} or config.C is None
            or not np.isfinite(config.C) or config.C <= 0
            or config.class_weight not in (None, "balanced") or config.gamma != "scale"):
        raise ValueError("Cấu hình SVM không hợp lệ")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel=config.model_name, C=config.C,
                           gamma=config.gamma, class_weight=config.class_weight,
                           probability=False)),
    ])


def evaluate_fold(
    view: ModelingView, rq: str, fold: TemporalSplit, config: ModelConfig,
) -> tuple[dict, pd.DataFrame]:
    """Fit estimator độc lập trên Train và đánh giá đúng Validation của fold.

    Args:
        view: ModelingView cùng thứ tự index với manifest đã xác minh.
        rq: Bài toán nghiên cứu khớp fold.
        fold: Fold development từ API an toàn của WP3.2.1.
        config: Cấu hình estimator được chọn trước khi chạy.

    Returns:
        Bản ghi metrics và bảng dự đoán Validation theo ngày.

    Raises:
        ValueError: Khi fold, index hoặc nhãn không đáp ứng điều kiện an toàn.
    """
    require_development_split(fold)
    if fold.rq != rq or not view.X.index.equals(view.y.index) or not view.X.index.equals(view.metadata.index):
        raise ValueError("RQ hoặc index X, y và metadata không khớp fold")
    train, validation = list(fold.train_indices), list(fold.validation_indices)
    for indices in (train, validation):
        if (not indices or indices != sorted(set(indices))
                or indices[0] < 0 or indices[-1] >= len(view.X)):
            raise ValueError("Indices của fold không hợp lệ")
    if set(train) & set(validation):
        raise ValueError("Train và Validation không được giao nhau")
    X_train, y_train = view.X.iloc[train], view.y.iloc[train]
    X_valid, y_valid = view.X.iloc[validation], view.y.iloc[validation]
    metadata_train = view.metadata.iloc[train]
    metadata_valid = view.metadata.iloc[validation]
    if (X_train.index[-1] >= X_valid.index[0]
            or X_train.index[0] != fold.train_start_date or X_train.index[-1] != fold.train_end_date
            or X_valid.index[0] != fold.validation_start_date or X_valid.index[-1] != fold.validation_end_date
            or metadata_train.label_end_date.max() >= fold.evaluation_boundary_date
            or metadata_valid.label_end_date.max().year >= TEST_START_YEAR
            or X_valid.index.max().year >= TEST_START_YEAR):
        raise ValueError("Fold vượt ranh giới phát triển hoặc có khoảng nhãn chồng lấn")
    if set(y_train.unique()) != {0, 1}:
        raise ValueError(f"{rq}/{fold.fold_name}: Train phải có đủ hai lớp")

    # Pipeline mới cho từng fold bảo đảm thống kê scaler chỉ học từ Train.
    estimator = make_estimator(config)
    estimator.fit(X_train, y_train.astype(int))
    predictions = estimator.predict(X_valid)
    if config.model_name == "dummy":
        class_index = list(estimator.classes_).index(1)
        scores = estimator.predict_proba(X_valid)[:, class_index]
        score_type = "train_prior_probability"
    else:
        scores = estimator.decision_function(X_valid)
        score_type = "decision_score"
    metrics = binary_metrics(y_valid.to_numpy(), predictions, scores)
    record = {"rq": rq, "fold_name": fold.fold_name, "model_name": config.model_name,
              "config_id": config.config_id, "C": config.C,
              "gamma": config.gamma if config.model_name == "rbf" else None,
              "class_weight": config.class_weight,
              "train_samples": len(train), "validation_samples": len(validation),
              "train_start": str(X_train.index[0].date()), "train_end": str(X_train.index[-1].date()),
              "validation_start": str(X_valid.index[0].date()),
              "validation_end": str(X_valid.index[-1].date()), **metrics}
    prediction_frame = pd.DataFrame({
        "rq": rq, "fold_name": fold.fold_name, "model_name": config.model_name,
        "config_id": config.config_id, "signal_date": X_valid.index,
        "label_end_date": metadata_valid.label_end_date.to_numpy(),
        "y_true": y_valid.to_numpy(dtype=int), "y_pred": predictions,
        "ranking_score": scores, "score_type": score_type,
    })
    return record, prediction_frame


def aggregate_results(metrics: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp mean, độ lệch chuẩn tổng thể, min và max theo từng cấu hình.

    Args:
        metrics: Metrics từng development fold, cùng schema cho mọi cấu hình.

    Returns:
        Một dòng cho mỗi RQ/cấu hình, gồm chênh lệch Balanced Accuracy với Dummy.

    Raises:
        ValueError: Khi số fold không đồng nhất hoặc thiếu Dummy cùng RQ.
    """
    groups = []
    for (rq, config_id), frame in metrics.groupby(["rq", "config_id"], sort=False):
        if frame.fold_name.duplicated().any() or len(frame) != 3:
            raise ValueError(f"{rq}/{config_id}: cần đủ ba development folds duy nhất")
        row = {"rq": rq, "config_id": config_id, "model_name": frame.model_name.iloc[0]}
        for name in METRIC_NAMES:
            values = frame[name].astype(float)
            row[f"mean_{name}"] = float(values.mean()) if values.notna().all() else np.nan
            row[f"std_{name}"] = float(values.std(ddof=0)) if values.notna().all() else np.nan
            row[f"min_{name}"] = float(values.min()) if values.notna().all() else np.nan
            row[f"max_{name}"] = float(values.max()) if values.notna().all() else np.nan
        groups.append(row)
    result = pd.DataFrame(groups)
    for rq in result.rq.unique():
        baseline = result[(result.rq == rq) & (result.config_id == "dummy_prior")]
        if len(baseline) != 1:
            raise ValueError(f"{rq}: thiếu Dummy baseline")
        mask = result.rq == rq
        result.loc[mask, "delta_mean_balanced_accuracy_vs_dummy"] = (
            result.loc[mask, "mean_balanced_accuracy"] - baseline.mean_balanced_accuracy.iloc[0]
        )
    return result


def select_configurations(comparison: pd.DataFrame) -> dict[str, dict]:
    """Chọn SVM theo Balanced Accuracy, độ ổn định và thứ tự cấu hình cố định.

    Args:
        comparison: Kết quả tổng hợp qua ba development folds.

    Returns:
        Cấu hình thắng cho từng RQ; Dummy chỉ đóng vai trò đối chứng.

    Raises:
        ValueError: Khi thiếu cấu hình hoặc Balanced Accuracy không xác định.
    """
    order = {config.config_id: position for position, config in enumerate(MODEL_CONFIGS)}
    selected = {}
    for rq in comparison.rq.unique():
        candidates = comparison[(comparison.rq == rq) & (comparison.model_name != "dummy")].copy()
        if candidates.empty or candidates.mean_balanced_accuracy.isna().any():
            raise ValueError(f"{rq}: không thể chọn SVM khi thiếu Balanced Accuracy")
        candidates["complexity"] = candidates.model_name.map({"linear": 0, "rbf": 1})
        candidates["config_order"] = candidates.config_id.map(order)
        if candidates[["complexity", "config_order"]].isna().any().any():
            raise ValueError(f"{rq}: có cấu hình ngoài danh sách đã chốt")
        winner = candidates.sort_values(
            ["mean_balanced_accuracy", "std_balanced_accuracy", "complexity", "config_order"],
            ascending=[False, True, True, True], kind="stable",
        ).iloc[0]
        selected[rq] = {"config_id": winner.config_id, "model_name": winner.model_name,
                        "mean_balanced_accuracy": float(winner.mean_balanced_accuracy),
                        "std_balanced_accuracy": float(winner.std_balanced_accuracy),
                        "delta_mean_balanced_accuracy_vs_dummy": float(winner.delta_mean_balanced_accuracy_vs_dummy)}
    return selected
