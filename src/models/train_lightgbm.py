"""Thực nghiệm LightGBM, trích xuất Feature Importance và lưu trữ Final Development Model.

Module này hiện thực hóa WP3.4:
- Huấn luyện LightGBM và Dummy baseline trên 3 development folds (dev_1, dev_2, dev_3).
- Áp dụng early stopping trên tập Validation của từng fold để ghi nhận best_iteration.
- Đánh giá xác suất dự đoán (predicted probability) và nhãn phân loại bằng binary_metrics.
- Tính toán In-sample Tree Importance (Gain & Split) và Out-of-Sample Validation Permutation Importance.
- Tổng hợp chỉ số, so sánh với Dummy và SVM baseline (WP3.3), và lựa chọn cấu hình thắng cuộc.
- Huấn luyện lại (refit) đúng 1 bản model cuối cùng cho mỗi RQ trên toàn bộ dữ liệu phát triển 2016–2021
  (1.297 mẫu sau khi áp dụng label-aware purging và pre-validation gap 5 phiên trước năm 2022).
- Tuyệt đối không huấn luyện, dự đoán hoặc chạm vào dữ liệu Reserved Test (2022–2026).
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
from typing import Any
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
import sklearn

from src.dataset_builder import expected_feature_columns, load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, ModelingView, load_research_dataset, make_modeling_view
from src.models.evaluation import binary_metrics
from src.models.modeling_safety import (
    MANIFEST_PATH, get_development_splits, get_full_development_train_indices, require_development_split,
)
from src.models.temporal_split import TEST_START_YEAR, TemporalSplit

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LGBMModelConfig:
    """Cấu hình mô hình LightGBM hoặc Dummy cố định trước khi quan sát kết quả."""

    config_id: str
    model_name: str
    max_depth: int | None = None
    num_leaves: int | None = None
    learning_rate: float | None = None
    min_child_samples: int | None = None
    reg_lambda: float | None = None
    class_weight: str | None = None
    n_estimators: int = 300
    random_state: int = 42


def _build_model_configs() -> tuple[LGBMModelConfig, ...]:
    """Tạo danh sách cấu hình LightGBM đại diện và Dummy đối chứng.

    Thiết kế 16 cấu hình LightGBM kết hợp giữa cấu trúc cây (depth 3/leaves 7 vs depth 5/leaves 15),
    tốc độ học (0.03 vs 0.05), ràng buộc mẫu tối thiểu trên lá kết hợp L2 regularization
    (min_child_samples 40 với lambda 1.0 vs min_child_samples 80 với lambda 5.0)
    và trọng số lớp (None vs balanced), cùng dummy_prior làm mốc so sánh.
    """
    configs: list[LGBMModelConfig] = [LGBMModelConfig("dummy_prior", "dummy")]

    tree_architectures = [
        (3, 7),   # Cây nông: max_depth=3, num_leaves=7
        (5, 15),  # Cây trung bình: max_depth=5, num_leaves=15
    ]
    learning_rates = [0.03, 0.05]
    leaf_priors = [
        (40, 1.0),  # Ít bảo thủ hơn: min_child_samples=40, reg_lambda=1.0
        (80, 5.0),  # Rất bảo thủ: min_child_samples=80, reg_lambda=5.0
    ]
    class_weights = [None, "balanced"]

    for max_depth, num_leaves in tree_architectures:
        for lr in learning_rates:
            for min_child, reg_l2 in leaf_priors:
                for weight in class_weights:
                    weight_label = weight or "none"
                    cfg_id = f"lgbm_d{max_depth}_lr{lr}_mc{min_child}_l{int(reg_l2)}_{weight_label}"
                    configs.append(LGBMModelConfig(
                        config_id=cfg_id,
                        model_name="lightgbm",
                        max_depth=max_depth,
                        num_leaves=num_leaves,
                        learning_rate=lr,
                        min_child_samples=min_child,
                        reg_lambda=reg_l2,
                        class_weight=weight,
                        n_estimators=300,
                        random_state=42,
                    ))
    return tuple(configs)


MODEL_CONFIGS: tuple[LGBMModelConfig, ...] = _build_model_configs()
METRIC_NAMES = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "roc_auc", "average_precision", "positive_prediction_rate",
    "best_iteration", "train_balanced_accuracy", "train_roc_auc", "generalization_gap",
)
EARLY_STOPPING_ROUNDS = 30


def make_estimator(config: LGBMModelConfig) -> DummyClassifier | lgb.LGBMClassifier:
    """Khởi tạo estimator mới theo cấu hình; LightGBM chạy đơn luồng tất định.

    Args:
        config: Cấu hình Dummy hoặc LightGBM đã chốt trước.

    Returns:
        DummyClassifier hoặc LGBMClassifier chưa được fit.

    Raises:
        ValueError: Khi thông số cấu hình không hợp lệ.
    """
    if config.model_name == "dummy":
        return DummyClassifier(strategy="prior")
    if config.model_name != "lightgbm":
        raise ValueError(f"Tên mô hình không được hỗ trợ: {config.model_name}")
    if (config.max_depth is None or config.max_depth <= 0
            or config.num_leaves is None or config.num_leaves <= 1
            or config.learning_rate is None or config.learning_rate <= 0
            or config.min_child_samples is None or config.min_child_samples <= 0
            or config.reg_lambda is None or config.reg_lambda < 0
            or config.class_weight not in (None, "balanced")):
        raise ValueError("Thông số siêu tham số của LightGBM không hợp lệ")

    return lgb.LGBMClassifier(
        objective="binary",
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        min_child_samples=config.min_child_samples,
        reg_lambda=config.reg_lambda,
        class_weight=config.class_weight,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=1,
        deterministic=True,
        verbose=-1,
    )


def evaluate_fold(
    view: ModelingView,
    rq: str,
    fold: TemporalSplit,
    config: LGBMModelConfig,
    compute_perm_importance: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit estimator độc lập trên Train của fold và đánh giá đúng Validation tương ứng.

    Args:
        view: ModelingView với ma trận đặc trưng causal và nhãn nghiên cứu.
        rq: RQ1 hoặc RQ2.
        fold: Development fold an toàn từ modeling_safety API.
        config: Cấu hình mô hình được đánh giá.
        compute_perm_importance: Có tính Out-of-sample Permutation Importance hay không.

    Returns:
        Tuple gồm (record_metrics, prediction_frame, training_log_records, feature_importance_records).

    Raises:
        ValueError: Khi fold vi phạm điều kiện an toàn hoặc nhãn một lớp.
    """
    require_development_split(fold)
    if fold.rq != rq or not view.X.index.equals(view.y.index) or not view.X.index.equals(view.metadata.index):
        raise ValueError("RQ hoặc index của view không khớp với fold phát triển")

    train_idx, valid_idx = list(fold.train_indices), list(fold.validation_indices)
    for indices in (train_idx, valid_idx):
        if (not indices or indices != sorted(set(indices))
                or indices[0] < 0 or indices[-1] >= len(view.X)):
            raise ValueError("Chỉ số vị trí của fold không hợp lệ")
    if set(train_idx) & set(valid_idx):
        raise ValueError("Tập Train và Validation không được phép giao nhau")

    X_train, y_train = view.X.iloc[train_idx], view.y.iloc[train_idx]
    X_valid, y_valid = view.X.iloc[valid_idx], view.y.iloc[valid_idx]
    metadata_train = view.metadata.iloc[train_idx]
    metadata_valid = view.metadata.iloc[valid_idx]

    if (X_train.index[-1] >= X_valid.index[0]
            or X_train.index[0] != fold.train_start_date or X_train.index[-1] != fold.train_end_date
            or X_valid.index[0] != fold.validation_start_date or X_valid.index[-1] != fold.validation_end_date
            or metadata_train.label_end_date.max() >= fold.evaluation_boundary_date
            or metadata_valid.label_end_date.max().year >= TEST_START_YEAR
            or X_valid.index.max().year >= TEST_START_YEAR):
        raise ValueError("Fold vi phạm ranh giới thời gian phát triển hoặc chồng lấn nhãn")
    if set(y_train.unique()) != {0, 1}:
        raise ValueError(f"{rq}/{fold.fold_name}: Tập Train phải có đủ hai lớp nhị phân")

    training_logs: list[dict[str, Any]] = []
    feature_importance_records: list[dict[str, Any]] = []

    if config.model_name == "dummy":
        estimator = make_estimator(config)
        estimator.fit(X_train, y_train.astype(int))
        class_1_idx = list(estimator.classes_).index(1)
        valid_scores = estimator.predict_proba(X_valid)[:, class_1_idx]
        valid_preds = estimator.predict(X_valid)
        score_type = "train_prior_probability"

        best_iteration = None
        train_scores = estimator.predict_proba(X_train)[:, class_1_idx]
        train_preds = estimator.predict(X_train)
        train_metrics = binary_metrics(y_train.to_numpy(), train_preds, train_scores)
        valid_metrics = binary_metrics(y_valid.to_numpy(), valid_preds, valid_scores)

        train_balanced_acc = train_metrics["balanced_accuracy"]
        train_roc_auc = train_metrics["roc_auc"]
        train_binary_logloss = None
        valid_binary_logloss = None
        generalization_gap = 0.0

    else:
        estimator = make_estimator(config)
        eval_results: dict[str, dict[str, list[float]]] = {}
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.record_evaluation(eval_results),
        ]

        # Khớp mô hình với tập Train và dừng sớm trên tập Validation của fold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(
                X_train,
                y_train.astype(int),
                eval_set=[(X_train, y_train.astype(int)), (X_valid, y_valid.astype(int))],
                eval_names=["train", "valid"],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )

        best_iteration = int(estimator.best_iteration_)
        valid_scores = estimator.predict_proba(X_valid)[:, 1]
        valid_preds = estimator.predict(X_valid)
        score_type = "predicted_probability"

        valid_metrics = binary_metrics(y_valid.to_numpy(), valid_preds, valid_scores)

        # Tính train metrics tại best_iteration để đo lường overfitting
        train_scores = estimator.predict_proba(X_train)[:, 1]
        train_preds = estimator.predict(X_train)
        train_metrics = binary_metrics(y_train.to_numpy(), train_preds, train_scores)

        train_balanced_acc = train_metrics["balanced_accuracy"]
        train_roc_auc = train_metrics["roc_auc"]
        train_binary_logloss = float(eval_results["train"]["binary_logloss"][best_iteration - 1])
        valid_binary_logloss = float(eval_results["valid"]["binary_logloss"][best_iteration - 1])
        generalization_gap = (float(train_balanced_acc - valid_metrics["balanced_accuracy"])
                              if train_balanced_acc is not None and valid_metrics["balanced_accuracy"] is not None
                              else None)

        # Trích xuất lịch sử quá trình boosting
        n_iters = len(eval_results["train"]["binary_logloss"])
        for i in range(1, n_iters + 1):
            training_logs.append({
                "rq": rq,
                "fold_name": fold.fold_name,
                "config_id": config.config_id,
                "iteration": i,
                "train_binary_logloss": float(eval_results["train"]["binary_logloss"][i - 1]),
                "valid_binary_logloss": float(eval_results["valid"]["binary_logloss"][i - 1]),
                "is_best_iteration": bool(i == best_iteration),
            })

        # Trích xuất In-sample Tree Gain và Split Importance
        booster = estimator.booster_
        gain_imp = booster.feature_importance(importance_type="gain")
        split_imp = booster.feature_importance(importance_type="split")

        # Tính Out-of-sample Permutation Importance trên tập Validation nếu được yêu cầu
        if compute_perm_importance:
            perm = permutation_importance(
                estimator,
                X_valid,
                y_valid.astype(int),
                scoring="balanced_accuracy",
                n_repeats=10,
                random_state=42,
            )
            perm_mean = perm.importances_mean
            perm_std = perm.importances_std
        else:
            perm_mean = np.zeros(len(X_valid.columns), dtype=float)
            perm_std = np.zeros(len(X_valid.columns), dtype=float)

        for col_name, g_val, s_val, p_mean, p_std in zip(
            X_valid.columns, gain_imp, split_imp, perm_mean, perm_std, strict=True
        ):
            feature_importance_records.append({
                "rq": rq,
                "fold_name": fold.fold_name,
                "config_id": config.config_id,
                "feature_name": col_name,
                "split_importance": int(s_val),
                "gain_importance": float(g_val),
                "perm_importance_mean": float(p_mean),
                "perm_importance_std": float(p_std),
            })

    record = {
        "rq": rq,
        "fold_name": fold.fold_name,
        "model_name": config.model_name,
        "config_id": config.config_id,
        "max_depth": config.max_depth,
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "min_child_samples": config.min_child_samples,
        "reg_lambda": config.reg_lambda,
        "class_weight": config.class_weight,
        "best_iteration": best_iteration,
        "train_samples": len(train_idx),
        "validation_samples": len(valid_idx),
        "train_start": str(X_train.index[0].date()),
        "train_end": str(X_train.index[-1].date()),
        "validation_start": str(X_valid.index[0].date()),
        "validation_end": str(X_valid.index[-1].date()),
        "train_balanced_accuracy": train_balanced_acc,
        "train_roc_auc": train_roc_auc,
        "train_binary_logloss": train_binary_logloss,
        "valid_binary_logloss": valid_binary_logloss,
        "generalization_gap": generalization_gap,
        **valid_metrics,
    }

    prediction_frame = pd.DataFrame({
        "rq": rq,
        "fold_name": fold.fold_name,
        "model_name": config.model_name,
        "config_id": config.config_id,
        "signal_date": X_valid.index,
        "label_end_date": metadata_valid.label_end_date.to_numpy(),
        "y_true": y_valid.to_numpy(dtype=int),
        "y_pred": valid_preds,
        "ranking_score": valid_scores,
        "score_type": score_type,
    })

    return record, prediction_frame, training_logs, feature_importance_records


def aggregate_results(metrics: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp mean, std (ddof=0), min và max qua ba development folds cho từng cấu hình.

    Args:
        metrics: DataFrame metrics của từng fold.

    Returns:
        DataFrame so sánh các cấu hình kèm chênh lệch Balanced Accuracy so với Dummy.

    Raises:
        ValueError: Khi thiếu fold hoặc thiếu mốc đối chứng Dummy.
    """
    groups: list[dict[str, Any]] = []
    for (rq, config_id), frame in metrics.groupby(["rq", "config_id"], sort=False):
        if frame.fold_name.duplicated().any() or len(frame) != 3:
            raise ValueError(f"{rq}/{config_id}: cần đúng 3 development folds duy nhất")
        first = frame.iloc[0]
        row: dict[str, Any] = {
            "rq": rq,
            "config_id": config_id,
            "model_name": first["model_name"],
            "max_depth": first.get("max_depth"),
            "num_leaves": first.get("num_leaves"),
            "learning_rate": first.get("learning_rate"),
            "min_child_samples": first.get("min_child_samples"),
            "reg_lambda": first.get("reg_lambda"),
            "class_weight": first.get("class_weight"),
        }
        for name in METRIC_NAMES:
            values = frame[name].dropna().astype(float)
            if len(values) == 3:
                row[f"mean_{name}"] = float(values.mean())
                row[f"std_{name}"] = float(values.std(ddof=0))
                row[f"min_{name}"] = float(values.min())
                row[f"max_{name}"] = float(values.max())
            else:
                row[f"mean_{name}"] = np.nan
                row[f"std_{name}"] = np.nan
                row[f"min_{name}"] = np.nan
                row[f"max_{name}"] = np.nan
        groups.append(row)

    result = pd.DataFrame(groups)
    for rq in result.rq.unique():
        baseline = result[(result.rq == rq) & (result.config_id == "dummy_prior")]
        if len(baseline) != 1:
            raise ValueError(f"{rq}: thiếu đối chứng Dummy baseline")
        dummy_mean = baseline.mean_balanced_accuracy.iloc[0]
        mask = result.rq == rq
        result.loc[mask, "delta_mean_balanced_accuracy_vs_dummy"] = (
            result.loc[mask, "mean_balanced_accuracy"] - dummy_mean
        )
    return result


def select_configurations(comparison: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Lựa chọn cấu hình LightGBM tối ưu theo Balanced Accuracy và tính đơn giản của cây.

    Thứ tự ưu tiên tie-breaker:
    1. Mean Balanced Accuracy cao nhất.
    2. Độ lệch chuẩn std_balanced_accuracy nhỏ nhất (ổn định nhất qua các năm).
    3. Cấu trúc cây đơn giản / kiểm soát overfitting cao hơn:
       - max_depth nhỏ hơn
       - num_leaves nhỏ hơn
       - min_child_samples lớn hơn (bảo thủ hơn)
       - reg_lambda lớn hơn (regularization mạnh hơn)
    4. Thứ tự cấu hình ban đầu đã công bố.

    Args:
        comparison: DataFrame tổng hợp từ aggregate_results.

    Returns:
        Dict chứa thông tin cấu hình thắng cuộc cho từng RQ.

    Raises:
        ValueError: Khi thiếu ứng viên LightGBM hoặc Balanced Accuracy không xác định.
    """
    order = {cfg.config_id: i for i, cfg in enumerate(MODEL_CONFIGS)}
    selected: dict[str, dict[str, Any]] = {}

    for rq in comparison.rq.unique():
        candidates = comparison[(comparison.rq == rq) & (comparison.model_name != "dummy")].copy()
        if candidates.empty or candidates.mean_balanced_accuracy.isna().any():
            raise ValueError(f"{rq}: không thể chọn mô hình khi thiếu Balanced Accuracy")

        candidates["config_order"] = candidates.config_id.map(order)
        candidates["depth_sort"] = candidates.max_depth.fillna(999).astype(int)
        candidates["leaves_sort"] = candidates.num_leaves.fillna(999).astype(int)
        candidates["min_child_sort"] = -candidates.min_child_samples.fillna(0).astype(float)
        candidates["lambda_sort"] = -candidates.reg_lambda.fillna(0).astype(float)

        winner = candidates.sort_values(
            [
                "mean_balanced_accuracy",
                "std_balanced_accuracy",
                "depth_sort",
                "leaves_sort",
                "min_child_sort",
                "lambda_sort",
                "config_order",
            ],
            ascending=[False, True, True, True, True, True, True],
            kind="stable",
        ).iloc[0]

        selected[rq] = {
            "config_id": winner.config_id,
            "model_name": winner.model_name,
            "max_depth": int(winner.max_depth),
            "num_leaves": int(winner.num_leaves),
            "learning_rate": float(winner.learning_rate),
            "min_child_samples": int(winner.min_child_samples),
            "reg_lambda": float(winner.reg_lambda),
            # CSV/DataFrame biểu diễn class_weight=None thành NaN; JSON cần null hợp lệ.
            "class_weight": None if pd.isna(winner.class_weight) else str(winner.class_weight),
            "mean_balanced_accuracy": float(winner.mean_balanced_accuracy),
            "std_balanced_accuracy": float(winner.std_balanced_accuracy),
            "mean_roc_auc": float(winner.mean_roc_auc),
            "mean_best_iteration": float(winner.mean_best_iteration),
            "mean_generalization_gap": float(winner.mean_generalization_gap),
            "delta_mean_balanced_accuracy_vs_dummy": float(winner.delta_mean_balanced_accuracy_vs_dummy),
        }
    return selected


def fit_and_export_final_development_models(
    root: Path,
    selected_configs: dict[str, dict[str, Any]],
    fold_best_iterations: dict[str, list[int]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Huấn luyện và lưu bản model LightGBM cuối của giai đoạn phát triển (2016–2021).

    Số cây final_n_estimators được tính tất định bằng trung bình làm tròn (>= 1)
    của best_iteration_ qua 3 development folds của cấu hình thắng cuộc.
    Mô hình được fit trên đúng 1.297 mẫu phát triển hợp lệ trước năm 2022.
    Tuyệt đối không sử dụng Reserved Test 2022–2026.

    Args:
        root: Thư mục gốc repository.
        selected_configs: Cấu hình thắng cuộc cho từng RQ.
        fold_best_iterations: Danh sách best_iteration qua 3 development folds cho từng RQ.
        output_dir: Thư mục xuất (mặc định 'models/lightgbm').

    Returns:
        Metadata đầy đủ của các mô hình LightGBM phát triển cuối cùng.
    """
    root = Path(root).resolve()
    target_dir = Path(output_dir).resolve() if output_dir else root / "models/lightgbm"
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = root / "configs/config.yaml"
    config = load_project_config(config_path)
    dataset = load_research_dataset(root, config)
    manifest_path = root / MANIFEST_PATH

    config_lookup = {cfg.config_id: cfg for cfg in MODEL_CONFIGS}
    models_metadata: dict[str, Any] = {}

    for rq in ("RQ1", "RQ2"):
        if rq not in selected_configs:
            raise ValueError(f"Thiếu cấu hình LightGBM được chọn cho {rq}")
        cfg_info = selected_configs[rq]
        cfg_id = cfg_info["config_id"]
        if cfg_id not in config_lookup:
            raise ValueError(f"Không tìm thấy cấu hình {cfg_id} trong MODEL_CONFIGS")
        base_config = config_lookup[cfg_id]

        # Xác định số cây cuối cùng bằng trung bình làm tròn của best_iteration_
        iters = fold_best_iterations.get(rq, [])
        if not iters or len(iters) != 3:
            raise ValueError(f"{rq}: Cần đúng 3 giá trị best_iteration từ development folds")
        final_n_estimators = max(1, int(round(float(np.mean(iters)))))

        train_indices = get_full_development_train_indices(root, rq)
        view = make_modeling_view(dataset, config, rq)

        X_train = view.X.iloc[list(train_indices)]
        y_train = view.y.iloc[list(train_indices)]
        metadata_train = view.metadata.iloc[list(train_indices)]

        if (X_train.index.max().year >= TEST_START_YEAR
                or metadata_train.label_end_date.max().year >= TEST_START_YEAR):
            raise ValueError(f"{rq}: Dữ liệu train phát triển vượt ranh giới năm Test")
        if set(y_train.unique()) != {0, 1}:
            raise ValueError(f"{rq}: Tập train phát triển cuối phải có đủ 2 lớp")

        final_estimator = lgb.LGBMClassifier(
            objective="binary",
            max_depth=base_config.max_depth,
            num_leaves=base_config.num_leaves,
            learning_rate=base_config.learning_rate,
            min_child_samples=base_config.min_child_samples,
            reg_lambda=base_config.reg_lambda,
            class_weight=base_config.class_weight,
            n_estimators=final_n_estimators,
            random_state=42,
            n_jobs=1,
            deterministic=True,
            verbose=-1,
        )
        final_estimator.fit(X_train, y_train.astype(int))

        model_filename = f"{rq.lower()}_lgbm_final_dev_2016_2021.joblib"
        model_path = target_dir / model_filename
        joblib.dump(final_estimator, model_path)
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

        booster = final_estimator.booster_
        gain_imp = {name: float(g) for name, g in zip(view.X.columns, booster.feature_importance(importance_type="gain"))}
        split_imp = {name: int(s) for name, s in zip(view.X.columns, booster.feature_importance(importance_type="split"))}

        models_metadata[rq] = {
            "rq": rq,
            "config_id": base_config.config_id,
            "model_name": "lightgbm",
            "max_depth": base_config.max_depth,
            "num_leaves": base_config.num_leaves,
            "learning_rate": base_config.learning_rate,
            "min_child_samples": base_config.min_child_samples,
            "reg_lambda": base_config.reg_lambda,
            "class_weight": base_config.class_weight,
            "fold_best_iterations": iters,
            "final_n_estimators": final_n_estimators,
            "train_samples": len(train_indices),
            "train_start_date": str(X_train.index[0].date()),
            "train_end_date": str(X_train.index[-1].date()),
            "max_label_end_date": str(metadata_train.label_end_date.max().date()),
            "feature_schema": list(view.X.columns),
            "gain_importance": gain_imp,
            "split_importance": split_imp,
            "model_file": model_filename,
            "model_sha256": model_sha256,
        }

    full_metadata = {
        "status": "COMPLETED_WITH_LIMITATIONS",
        "description": "Bản model LightGBM cuối của giai đoạn phát triển (2016–2021) trước ranh giới Test 2022.",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": RESEARCH_PATH.as_posix(),
        "dataset_sha256": hashlib.sha256((root / RESEARCH_PATH).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": (hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                                  if manifest_path.is_file() else None),
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "models": models_metadata,
    }

    (target_dir / "lgbm_final_dev_metadata.json").write_text(
        json.dumps(full_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return full_metadata

