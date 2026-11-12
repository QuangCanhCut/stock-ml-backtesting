"""Thực nghiệm LSTM, phân tích độ nhạy thời gian/đặc trưng và lưu trữ Final Development Model.

Module này hiện thực hóa WP3.5:
- Huấn luyện LSTM và Dummy baseline trên 3 development folds (dev_1, dev_2, dev_3).
- Áp dụng early stopping trên tập Validation của từng fold để ghi nhận best_epoch và khôi phục checkpoint tối ưu.
- Đánh giá xác suất dự đoán (predicted probability) và nhãn phân loại bằng binary_metrics.
- Phân tích Feature Permutation Importance trên chuỗi Validation và Temporal Lag Sensitivity (độ nhạy theo bước trễ).
- Tổng hợp chỉ số, so sánh đối chứng 4 mô hình: Dummy, SVM (WP3.3), LightGBM (WP3.4) và LSTM (WP3.5).
- Huấn luyện lại (refit) đúng 1 bản model cuối cùng cho mỗi RQ trên toàn bộ dữ liệu phát triển 2016–2021
  (1.297 mẫu sau khi áp dụng label-aware purging và pre-validation gap 5 phiên trước năm 2022).
- Tuyệt đối không huấn luyện, dự đoán hoặc chạm vào dữ liệu Reserved Test (2022–2026).
"""

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, ModelingView, load_research_dataset, make_modeling_view
from src.models.evaluation import binary_metrics
from src.models.lstm_network import (
    LSTMClassifier, build_fold_sequences, build_full_development_sequences,
)
from src.models.modeling_safety import (
    MANIFEST_PATH, get_development_splits, get_full_development_train_indices, require_development_split,
)
from src.models.temporal_split import TemporalSplit

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho tính tái lập (reproducibility) của thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class LSTMModelConfig:
    """Cấu hình siêu tham số của mô hình LSTM và baseline Dummy."""
    config_id: str
    model_name: str  # "dummy" hoặc "lstm"
    seq_len: int = 10
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    class_weight: str | None = None  # None hoặc "balanced"
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 10
    grad_clip: float = 1.0
    random_state: int = 42


# Lưới cấu hình định trước: 1 Dummy đối chứng + 8 cấu hình LSTM kiểm soát overfitting
MODEL_CONFIGS: tuple[LSTMModelConfig, ...] = (
    LSTMModelConfig(
        config_id="dummy_prior",
        model_name="dummy",
        seq_len=10,
        hidden_size=0,
        num_layers=0,
        dropout=0.0,
        learning_rate=0.0,
        weight_decay=0.0,
        class_weight=None,
    ),
    LSTMModelConfig(
        config_id="lstm_seq10_h16_lr0.001_none",
        model_name="lstm",
        seq_len=10,
        hidden_size=16,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.001,
        class_weight=None,
    ),
    LSTMModelConfig(
        config_id="lstm_seq10_h16_lr0.001_balanced",
        model_name="lstm",
        seq_len=10,
        hidden_size=16,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.001,
        class_weight="balanced",
    ),
    LSTMModelConfig(
        config_id="lstm_seq10_h32_lr0.0005_none",
        model_name="lstm",
        seq_len=10,
        hidden_size=32,
        num_layers=1,
        dropout=0.3,
        learning_rate=0.0005,
        weight_decay=0.001,
        class_weight=None,
    ),
    LSTMModelConfig(
        config_id="lstm_seq10_h32_lr0.0005_balanced",
        model_name="lstm",
        seq_len=10,
        hidden_size=32,
        num_layers=1,
        dropout=0.3,
        learning_rate=0.0005,
        weight_decay=0.001,
        class_weight="balanced",
    ),
    LSTMModelConfig(
        config_id="lstm_seq20_h16_lr0.001_none",
        model_name="lstm",
        seq_len=20,
        hidden_size=16,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.001,
        class_weight=None,
    ),
    LSTMModelConfig(
        config_id="lstm_seq20_h16_lr0.001_balanced",
        model_name="lstm",
        seq_len=20,
        hidden_size=16,
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=0.001,
        class_weight="balanced",
    ),
    LSTMModelConfig(
        config_id="lstm_seq20_h32_lr0.0005_none",
        model_name="lstm",
        seq_len=20,
        hidden_size=32,
        num_layers=1,
        dropout=0.3,
        learning_rate=0.0005,
        weight_decay=0.001,
        class_weight=None,
    ),
    LSTMModelConfig(
        config_id="lstm_seq20_h32_lr0.0005_balanced",
        model_name="lstm",
        seq_len=20,
        hidden_size=32,
        num_layers=1,
        dropout=0.3,
        learning_rate=0.0005,
        weight_decay=0.001,
        class_weight="balanced",
    ),
)


def evaluate_fold(
    view: ModelingView,
    rq: str,
    fold: TemporalSplit,
    config: LSTMModelConfig,
    *,
    compute_attribution: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Huấn luyện mô hình độc lập trên Train và đánh giá đúng tập Validation của fold.

    Args:
        view: ModelingView của RQ tương ứng.
        rq: Tên bài toán ("RQ1" hoặc "RQ2").
        fold: Development fold đã được kiểm toán an toàn.
        config: Cấu hình mô hình được chọn trước khi chạy.
        compute_attribution: Có tính Feature Permutation Importance và Lag Sensitivity hay không.

    Returns:
        Tuple gồm (record_metrics, predictions_df, training_logs, feature_importance_records, lag_records).
    """
    require_development_split(fold)
    if fold.rq != rq:
        raise ValueError(f"Fold thuộc {fold.rq}, không khớp với tham số rq={rq}")

    val_indices = list(fold.validation_indices)
    signal_dates = view.metadata.signal_date.iloc[val_indices].tolist()
    label_end_dates = view.metadata.label_end_date.iloc[val_indices].tolist()
    y_val_true = view.y.iloc[val_indices].to_numpy(dtype=int)

    # 1. Dummy Baseline
    if config.model_name == "dummy":
        train_indices = list(fold.train_indices)
        prior = float(view.y.iloc[train_indices].mean())
        val_n = len(val_indices)
        probs = np.full(val_n, prior, dtype=float)
        y_pred = np.full(val_n, int(prior >= 0.5), dtype=int)

        metrics = binary_metrics(y_val_true, y_pred, probs)
        record = {
            "rq": rq,
            "fold_name": fold.fold_name,
            "config_id": config.config_id,
            "model_name": "dummy",
            "seq_len": config.seq_len,
            "hidden_size": 0,
            "num_layers": 0,
            "dropout": 0.0,
            "learning_rate": 0.0,
            "weight_decay": 0.0,
            "class_weight": None,
            "train_samples": len(train_indices),
            "train_sequences": len(train_indices),
            "validation_samples": val_n,
            "best_epoch": 1,
            "total_epochs_trained": 1,
            "train_bce_loss": 0.693147,
            "valid_bce_loss": 0.693147,
            "train_accuracy": float(np.mean(view.y.iloc[train_indices] == int(prior >= 0.5))),
            "train_balanced_accuracy": 0.5,
            "train_roc_auc": 0.5,
            "generalization_gap": 0.0,
            **metrics,
        }
        pred_df = pd.DataFrame({
            "rq": rq,
            "fold_name": fold.fold_name,
            "config_id": config.config_id,
            "model_name": "dummy",
            "signal_date": signal_dates,
            "label_end_date": label_end_dates,
            "y_true": y_val_true,
            "y_pred": y_pred,
            "ranking_score": probs,
            "score_type": "train_prior_probability",
        })
        return record, pred_df, [], [], []

    # 2. PyTorch LSTM Model
    set_seed(config.random_state)
    X_train_seq, y_train_seq, X_valid_seq, y_valid_seq, _ = build_fold_sequences(
        view, fold, config.seq_len
    )

    train_tensor_x = torch.from_numpy(X_train_seq)
    train_tensor_y = torch.from_numpy(y_train_seq)
    valid_tensor_x = torch.from_numpy(X_valid_seq)
    valid_tensor_y = torch.from_numpy(y_valid_seq)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    g = torch.Generator().manual_seed(config.random_state)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=g)

    # Xử lý pos_weight cho BCEWithLogitsLoss nếu dùng balanced
    if config.class_weight == "balanced":
        n_pos = float(np.sum(y_train_seq == 1.0))
        n_neg = float(np.sum(y_train_seq == 0.0))
        pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        train_criterion = nn.BCEWithLogitsLoss()

    eval_criterion = nn.BCEWithLogitsLoss()

    model = LSTMClassifier(
        input_size=X_train_seq.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0
    training_logs: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = train_criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

        # Đánh giá loss trên toàn bộ chuỗi ở chế độ eval()
        model.eval()
        with torch.no_grad():
            train_logits = model(train_tensor_x)
            train_loss = float(eval_criterion(train_logits, train_tensor_y).item())

            valid_logits = model(valid_tensor_x)
            valid_loss = float(eval_criterion(valid_logits, valid_tensor_y).item())

        is_best = False
        if valid_loss < best_val_loss - 1e-5:
            best_val_loss = valid_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        training_logs.append({
            "rq": rq,
            "fold_name": fold.fold_name,
            "config_id": config.config_id,
            "epoch": epoch,
            "train_bce_loss": train_loss,
            "valid_bce_loss": valid_loss,
            "is_best_epoch": is_best,
        })

        if patience_counter >= config.early_stopping_patience:
            LOGGER.debug("Early stopping tại epoch %d (best_epoch=%d)", epoch, best_epoch)
            break

    # Khôi phục checkpoint tốt nhất trước khi xuất metrics và dự đoán
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        val_logits = model(valid_tensor_x)
        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
        val_preds = (val_probs >= 0.5).astype(int)

        train_logits = model(train_tensor_x)
        train_probs = torch.sigmoid(train_logits).cpu().numpy().flatten()
        train_preds = (train_probs >= 0.5).astype(int)

    # Tính metrics Train
    y_train_int = y_train_seq.astype(int)
    train_m = binary_metrics(y_train_int, train_preds, train_probs)

    # Tính metrics Validation
    val_m = binary_metrics(y_val_true, val_preds, val_probs)

    gen_gap = (
        float(train_m["balanced_accuracy"] - val_m["balanced_accuracy"])
        if train_m["balanced_accuracy"] is not None and val_m["balanced_accuracy"] is not None
        else 0.0
    )

    record = {
        "rq": rq,
        "fold_name": fold.fold_name,
        "config_id": config.config_id,
        "model_name": "lstm",
        "seq_len": config.seq_len,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "class_weight": config.class_weight,
        "train_samples": len(fold.train_indices),
        "train_sequences": len(X_train_seq),
        "validation_samples": len(val_indices),
        "best_epoch": best_epoch,
        "total_epochs_trained": epoch,
        "train_bce_loss": train_loss,
        "valid_bce_loss": best_val_loss,
        "train_accuracy": train_m["accuracy"],
        "train_balanced_accuracy": train_m["balanced_accuracy"],
        "train_roc_auc": train_m["roc_auc"],
        "generalization_gap": gen_gap,
        **val_m,
    }

    pred_df = pd.DataFrame({
        "rq": rq,
        "fold_name": fold.fold_name,
        "config_id": config.config_id,
        "model_name": "lstm",
        "signal_date": signal_dates,
        "label_end_date": label_end_dates,
        "y_true": y_val_true,
        "y_pred": val_preds,
        "ranking_score": val_probs,
        "score_type": "predicted_probability",
    })

    feature_imp_records: list[dict[str, Any]] = []
    lag_records: list[dict[str, Any]] = []

    if compute_attribution:
        # 1. Feature Permutation Importance trên chuỗi Validation 3D
        baseline_bacc = val_m["balanced_accuracy"] if val_m["balanced_accuracy"] is not None else 0.5
        feature_names = list(view.X.columns)

        for j, feat_name in enumerate(feature_names):
            drops: list[float] = []
            for rep in range(10):
                rng = np.random.default_rng(config.random_state + rep * 100 + j)
                perm_idx = rng.permutation(len(X_valid_seq))
                X_perm = X_valid_seq.copy()
                X_perm[:, :, j] = X_valid_seq[perm_idx, :, j]

                with torch.no_grad():
                    p_logits = model(torch.from_numpy(X_perm))
                    p_probs = torch.sigmoid(p_logits).cpu().numpy().flatten()
                    p_pred = (p_probs >= 0.5).astype(int)
                    p_m = binary_metrics(y_val_true, p_pred, p_probs)
                    p_bacc = p_m["balanced_accuracy"] if p_m["balanced_accuracy"] is not None else 0.5
                    drops.append(baseline_bacc - p_bacc)

            feature_imp_records.append({
                "rq": rq,
                "fold_name": fold.fold_name,
                "config_id": config.config_id,
                "feature_name": feat_name,
                "perm_importance_mean": float(np.mean(drops)),
                "perm_importance_std": float(np.std(drops, ddof=0)),
            })

        # 2. Temporal Lag Sensitivity (Gradient Saliency theo từng bước trễ)
        X_val_grad = torch.tensor(X_valid_seq, requires_grad=True, dtype=torch.float32)
        grad_logits = model(X_val_grad)
        loss = grad_logits.sum()
        model.zero_grad()
        loss.backward()

        # Trung bình độ lớn gradient tuyệt đối qua batch và features theo từng bước thời gian tau
        grad_sens = X_val_grad.grad.abs().mean(dim=(0, 2)).cpu().numpy()  # Shape: (seq_len,)
        sum_grad = float(grad_sens.sum())
        norm_sens = grad_sens / (sum_grad if sum_grad > 1e-8 else 1.0)

        for tau in range(config.seq_len):
            lag_step = tau - (config.seq_len - 1)  # -(seq_len - 1), ..., -1, 0 (0 là phiên hiện tại t)
            lag_records.append({
                "rq": rq,
                "fold_name": fold.fold_name,
                "config_id": config.config_id,
                "lag_index": tau,
                "lag_step": lag_step,
                "gradient_sensitivity": float(grad_sens[tau]),
                "gradient_sensitivity_normalized": float(norm_sens[tau]),
            })

    return record, pred_df, training_logs, feature_imp_records, lag_records


def aggregate_results(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp kết quả của từng cấu hình qua 3 development folds."""
    rows: list[dict[str, Any]] = []

    for (rq, config_id), frame in fold_metrics.groupby(["rq", "config_id"], sort=False):
        row0 = frame.iloc[0]
        row = {
            "rq": rq,
            "config_id": config_id,
            "model_name": row0["model_name"],
            "seq_len": row0["seq_len"],
            "hidden_size": row0["hidden_size"],
            "num_layers": row0["num_layers"],
            "dropout": row0["dropout"],
            "learning_rate": row0["learning_rate"],
            "weight_decay": row0["weight_decay"],
            "class_weight": row0["class_weight"],
            "mean_accuracy": float(frame["accuracy"].mean()),
            "std_accuracy": float(frame["accuracy"].std(ddof=0)),
            "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
            "std_balanced_accuracy": float(frame["balanced_accuracy"].std(ddof=0)),
            "min_balanced_accuracy": float(frame["balanced_accuracy"].min()),
            "max_balanced_accuracy": float(frame["balanced_accuracy"].max()),
            "mean_roc_auc": float(frame["roc_auc"].mean()),
            "std_roc_auc": float(frame["roc_auc"].std(ddof=0)),
            "mean_f1": float(frame["f1"].mean()),
            "mean_best_epoch": float(frame["best_epoch"].mean()),
            "mean_generalization_gap": float(frame["generalization_gap"].mean()),
            "delta_mean_balanced_accuracy_vs_dummy": float(frame["delta_balanced_accuracy_vs_dummy"].mean()),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def select_configurations(comparison: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Lựa chọn cấu hình LSTM thắng cuộc cho RQ1 và RQ2 theo quy tắc nhất quán.

    Quy tắc deterministic:
    1. Mean Balanced Accuracy cao nhất qua 3 development folds.
    2. Nếu bằng nhau, chọn cấu hình có std_balanced_accuracy nhỏ hơn.
    3. Nếu tiếp tục bằng nhau, chọn cấu hình đơn giản hơn (hidden_size nhỏ hơn -> seq_len ngắn hơn -> dropout lớn hơn).
    4. Thứ tự cấu hình định trước trong MODEL_CONFIGS.
    """
    selected: dict[str, dict[str, Any]] = {}
    config_order = {cfg.config_id: i for i, cfg in enumerate(MODEL_CONFIGS)}

    for rq in RQ_TARGETS:
        candidates = comparison[(comparison.rq == rq) & (comparison.model_name == "lstm")].copy()
        if candidates.empty:
            raise ValueError(f"Không có cấu hình LSTM nào cho {rq}")

        candidates["order"] = candidates["config_id"].map(config_order)
        # Sắp xếp theo: mean_bacc giảm, std_bacc tăng, hidden_size tăng, seq_len tăng, order tăng
        sorted_candidates = candidates.sort_values(
            by=["mean_balanced_accuracy", "std_balanced_accuracy", "hidden_size", "seq_len", "order"],
            ascending=[False, True, True, True, True],
        )
        winner = sorted_candidates.iloc[0]
        selected[rq] = {
            "config_id": str(winner["config_id"]),
            "model_name": str(winner["model_name"]),
            "seq_len": int(winner["seq_len"]),
            "hidden_size": int(winner["hidden_size"]),
            "num_layers": int(winner["num_layers"]),
            "dropout": float(winner["dropout"]),
            "learning_rate": float(winner["learning_rate"]),
            "weight_decay": float(winner["weight_decay"]),
            "class_weight": None if pd.isna(winner["class_weight"]) else str(winner["class_weight"]),
            "mean_balanced_accuracy": float(winner["mean_balanced_accuracy"]),
            "std_balanced_accuracy": float(winner["std_balanced_accuracy"]),
            "min_balanced_accuracy": float(winner["min_balanced_accuracy"]),
            "max_balanced_accuracy": float(winner["max_balanced_accuracy"]),
            "mean_roc_auc": float(winner["mean_roc_auc"]),
            "std_roc_auc": float(winner["std_roc_auc"]),
            "mean_f1": float(winner["mean_f1"]),
            "mean_best_epoch": float(winner["mean_best_epoch"]),
            "mean_generalization_gap": float(winner["mean_generalization_gap"]),
            "delta_mean_balanced_accuracy_vs_dummy": float(winner["delta_mean_balanced_accuracy_vs_dummy"]),
        }

    return selected


def fit_and_export_final_development_models(
    root: Path,
    selected_configs: dict[str, dict[str, Any]],
    fold_best_epochs: dict[str, list[int]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Huấn luyện và lưu bản model LSTM cuối của giai đoạn phát triển (2016–2021).

    Số epoch final_epochs được tính tất định bằng trung bình làm tròn (>= 1)
    của best_epoch qua 3 development folds của cấu hình thắng cuộc.
    Mô hình được fit trên đúng 1.297 mẫu phát triển hợp lệ trước năm 2022.
    Tuyệt đối không sử dụng Reserved Test 2022–2026.

    Args:
        root: Thư mục gốc repository.
        selected_configs: Cấu hình thắng cuộc cho từng RQ.
        fold_best_epochs: Danh sách best_epoch qua 3 development folds cho từng RQ.
        output_dir: Thư mục xuất (mặc định 'models/lstm').

    Returns:
        Metadata đầy đủ của các mô hình LSTM phát triển cuối cùng.
    """
    root = Path(root).resolve()
    target_dir = Path(output_dir).resolve() if output_dir else root / "models/lstm"
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = root / "configs/config.yaml"
    config = load_project_config(config_path)
    dataset = load_research_dataset(root, config)
    manifest_path = root / MANIFEST_PATH

    models_meta: dict[str, Any] = {}

    for rq in RQ_TARGETS:
        winner_info = selected_configs[rq]
        cfg_id = winner_info["config_id"]
        model_cfg = next(c for c in MODEL_CONFIGS if c.config_id == cfg_id)

        # Tính final_epochs tất định từ best_epochs qua 3 folds
        epochs_list = fold_best_epochs[rq]
        final_epochs = max(1, int(round(float(np.mean(epochs_list)))))

        train_indices = get_full_development_train_indices(root, rq)
        view = make_modeling_view(dataset, config, rq)

        # Xây dựng chuỗi trên đúng 1.297 mẫu phát triển 2016–2021
        X_train_seq, y_train_seq, fitted_scaler = build_full_development_sequences(
            view, train_indices, model_cfg.seq_len
        )

        set_seed(model_cfg.random_state)
        final_model = LSTMClassifier(
            input_size=X_train_seq.shape[2],
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
        )

        train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
        g = torch.Generator().manual_seed(model_cfg.random_state)
        train_loader = DataLoader(train_dataset, batch_size=model_cfg.batch_size, shuffle=True, generator=g)

        if model_cfg.class_weight == "balanced":
            n_pos = float(np.sum(y_train_seq == 1.0))
            n_neg = float(np.sum(y_train_seq == 0.0))
            pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            final_model.parameters(), lr=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay
        )

        final_model.train()
        for _ in range(final_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = final_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=model_cfg.grad_clip)
                optimizer.step()

        # Tạo gói bundle để lưu
        feature_schema = list(view.X.columns)
        bundle_data = {
            "model_state_dict": final_model.state_dict(),
            "network_config": {
                "input_size": int(X_train_seq.shape[2]),
                "hidden_size": int(model_cfg.hidden_size),
                "num_layers": int(model_cfg.num_layers),
                "dropout": float(model_cfg.dropout),
            },
            "config_id": cfg_id,
            "seq_len": int(model_cfg.seq_len),
            "learning_rate": float(model_cfg.learning_rate),
            "weight_decay": float(model_cfg.weight_decay),
            "class_weight": model_cfg.class_weight,
            "final_epochs": int(final_epochs),
            "fold_best_epochs": [int(x) for x in epochs_list],
            "scaler_mean": fitted_scaler.mean_.tolist(),
            "scaler_scale": fitted_scaler.scale_.tolist(),
            "feature_schema": feature_schema,
            "train_samples": int(len(train_indices)),
            "train_sequences": int(len(X_train_seq)),
            "rq": rq,
        }

        model_file = f"{rq.lower()}_lstm_final_dev_2016_2021.pt"
        model_path = target_dir / model_file
        torch.save(bundle_data, model_path)

        file_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

        train_dates = view.metadata.signal_date.iloc[list(train_indices)]
        label_dates = view.metadata.label_end_date.iloc[list(train_indices)]

        models_meta[rq] = {
            "rq": rq,
            "config_id": cfg_id,
            "model_name": "lstm",
            "seq_len": model_cfg.seq_len,
            "hidden_size": model_cfg.hidden_size,
            "num_layers": model_cfg.num_layers,
            "dropout": model_cfg.dropout,
            "learning_rate": model_cfg.learning_rate,
            "weight_decay": model_cfg.weight_decay,
            "class_weight": model_cfg.class_weight,
            "fold_best_epochs": [int(x) for x in epochs_list],
            "final_epochs": final_epochs,
            "train_samples": len(train_indices),
            "train_sequences": len(X_train_seq),
            "train_start_date": train_dates.min().strftime("%Y-%m-%d"),
            "train_end_date": train_dates.max().strftime("%Y-%m-%d"),
            "max_label_end_date": label_dates.max().strftime("%Y-%m-%d"),
            "feature_schema": feature_schema,
            "scaler_mean": fitted_scaler.mean_.tolist(),
            "scaler_scale": fitted_scaler.scale_.tolist(),
            "total_trainable_parameters": final_model.count_trainable_parameters(),
            "model_file": model_file,
            "model_sha256": file_sha256,
        }

    metadata = {
        "status": "COMPLETED_WITH_LIMITATIONS",
        "description": "Bản model LSTM cuối của giai đoạn phát triển (2016–2021) trước ranh giới Test 2022.",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": RESEARCH_PATH.as_posix(),
        "dataset_sha256": hashlib.sha256((root / RESEARCH_PATH).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "models": models_meta,
    }

    metadata_path = target_dir / "lstm_final_dev_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return metadata


def load_final_lstm_bundle(bundle_path: Path) -> tuple[LSTMClassifier, StandardScaler, dict[str, Any]]:
    """Tải gói model LSTM cuối từ đĩa, khôi phục kiến trúc, trọng số và StandardScaler.

    Args:
        bundle_path: Đường dẫn tệp .pt của mô hình.

    Returns:
        Tuple gồm (model, scaler, bundle_metadata).
    """
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model bundle tại {bundle_path}")

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

    net_cfg = bundle["network_config"]
    model = LSTMClassifier(
        input_size=net_cfg["input_size"],
        hidden_size=net_cfg["hidden_size"],
        num_layers=net_cfg["num_layers"],
        dropout=net_cfg["dropout"],
    )
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(bundle["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(bundle["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return model, scaler, bundle

