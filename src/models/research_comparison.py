"""Nghiên cứu đối chứng tổng hợp, mở khóa Reserved Test có kiểm soát và tính toán Quant IC.

Module này hiện thực hóa WP3.6 và WP3.7:
1. Tổng hợp kết quả Development Walk-Forward CV (2019–2021) trên 4 họ mô hình (Dummy, SVM, LightGBM, LSTM).
2. Phân tích độ đồng thuận nhãn và tương quan điểm số dự báo (Prediction Agreement & Score Correlation).
3. Mở khóa một lần có kiểm soát (One-Shot Unfreezing) tập Reserved Test (2022–2026) theo 2 giao thức:
   - Protocol A (frozen_2016_2021): Dùng mô hình tĩnh đã lưu trong models/ (fit 2016–2021).
   - Protocol B (annual_expanding_retrain): Retrain mở rộng định kỳ đầu mỗi năm (2022..2026) với gap 5 phiên.
4. Tính toán hệ thống chỉ số Quant: Balanced Accuracy, ROC-AUC, Brier Score, Log Loss, Spearman Rank IC, Pearson IC.
5. Hiệu chỉnh xác suất nhân quả (Causal Platt Calibration) cho SVM dựa trên Out-of-Fold của Development Folds.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, ModelingView, load_research_dataset, make_modeling_view
from src.models.evaluation import binary_metrics
from src.models.lstm_network import LSTMClassifier
from src.models.modeling_safety import (
    MANIFEST_PATH, get_final_evaluation_split, get_full_development_train_indices,
)
from src.models.temporal_split import TemporalSplit, _first_session, _validate_view
from src.models.train_lightgbm import make_estimator as make_lgbm_estimator, MODEL_CONFIGS as LGBM_CONFIGS
from src.models.train_lstm import load_final_lstm_bundle, set_seed

LOGGER = logging.getLogger(__name__)


def compute_quant_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ranking_score: np.ndarray,
    calibrated_prob: np.ndarray,
    forward_returns: np.ndarray,
) -> dict[str, Any]:
    """Tính toán toàn diện các chỉ số phân loại, hiệu chuẩn xác suất và Information Coefficient (IC).

    Args:
        y_true: Mảng nhãn thực tế 1D nhị phân (0 hoặc 1).
        y_pred: Mảng nhãn dự đoán 1D (0 hoặc 1).
        ranking_score: Điểm xếp hạng nguyên bản (decision score hoặc predicted probability).
        calibrated_prob: Xác suất đã hiệu chuẩn nằm nghiêm ngặt trong [0, 1].
        forward_returns: Lợi suất tương lai thực tế tương ứng với bài toán.

    Returns:
        Dict chứa đầy đủ các chỉ số định lượng.
    """
    base_m = binary_metrics(y_true, y_pred, ranking_score)

    # Đảm bảo calibrated_prob nằm trong (1e-15, 1 - 1e-15) để tính log loss an toàn
    prob_clipped = np.clip(np.asarray(calibrated_prob, dtype=float), 1e-15, 1.0 - 1e-15)

    # 1. Calibration Metrics
    brier = float(brier_score_loss(y_true, prob_clipped))
    try:
        ll = float(log_loss(y_true, prob_clipped))
    except Exception:
        ll = float("nan")

    # 2. Information Coefficient (IC & Rank IC)
    ret = np.asarray(forward_returns, dtype=float)
    sc = np.asarray(ranking_score, dtype=float)

    if len(np.unique(sc)) > 1 and len(np.unique(ret)) > 1:
        s_res = spearmanr(sc, ret)
        rank_ic = float(s_res.statistic if hasattr(s_res, "statistic") else s_res[0])
        p_res = pearsonr(sc, ret)
        p_ic = float(p_res.statistic if hasattr(p_res, "statistic") else p_res[0])
    else:
        rank_ic = 0.0
        p_ic = 0.0

    return {
        **base_m,
        "brier_score": brier,
        "log_loss": ll,
        "spearman_rank_ic": rank_ic,
        "pearson_ic": p_ic,
    }


def fit_causal_platt_calibrator(root: Path, rq: str, config_id: str) -> LogisticRegression:
    """Huấn luyện bộ hiệu chỉnh Platt Scaling nhân quả 1D cho SVM dựa trên Out-of-Fold của Dev Folds.

    Quy tắc an toàn chống rò rỉ:
    Chỉ sử dụng dự báo ngoài mẫu (OOF) của 3 development folds (2019–2021) từ WP3.3.
    Tuyệt đối không sử dụng bất kỳ mẫu nào của tập Reserved Test (2022–2026).

    Args:
        root: Thư mục gốc repository.
        rq: Tên bài toán ("RQ1" hoặc "RQ2").
        config_id: ID cấu hình SVM thắng cuộc.

    Returns:
        Mô hình LogisticRegression đã được fit trên decision score và nhãn thực tế OOF.
    """
    oof_path = root / "results/modeling/svm/validation_predictions.parquet"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file OOF SVM tại {oof_path}")

    oof_df = pd.read_parquet(oof_path)
    sub = oof_df[(oof_df.rq == rq) & (oof_df.config_id == config_id)]
    if sub.empty:
        raise ValueError(f"Không tìm thấy dự báo OOF của {rq}/{config_id} trong {oof_path}")

    X_oof = sub[["ranking_score"]].to_numpy(dtype=float)
    y_oof = sub["y_true"].to_numpy(dtype=int)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(X_oof, y_oof)
    return calibrator


def evaluate_dev_prediction_agreement(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phân tích độ đồng thuận nhãn và tương quan điểm số dự báo giữa SVM, LightGBM và LSTM trên Dev Validation.

    Args:
        root: Thư mục gốc repository.

    Returns:
        Tuple gồm (agreement_summary_df, merged_dev_predictions_df).
    """
    svm_path = root / "results/modeling/svm/validation_predictions.parquet"
    lgbm_path = root / "results/modeling/lightgbm/validation_predictions.parquet"
    lstm_path = root / "results/modeling/lstm/validation_predictions.parquet"

    svm_sel = json.loads((root / "results/modeling/svm/selected_configs.json").read_text(encoding="utf-8"))["selected"]
    lgbm_sel = json.loads((root / "results/modeling/lightgbm/selected_configs.json").read_text(encoding="utf-8"))["selected"]
    lstm_sel = json.loads((root / "results/modeling/lstm/selected_configs.json").read_text(encoding="utf-8"))["selected"]

    svm_df = pd.read_parquet(svm_path)
    lgbm_df = pd.read_parquet(lgbm_path)
    lstm_df = pd.read_parquet(lstm_path)

    agreement_rows: list[dict[str, Any]] = []
    merged_frames: list[pd.DataFrame] = []

    for rq in RQ_TARGETS:
        s_sub = svm_df[(svm_df.rq == rq) & (svm_df.config_id == svm_sel[rq]["config_id"])].rename(
            columns={"y_pred": "y_pred_svm", "ranking_score": "score_svm"}
        )[["rq", "fold_name", "signal_date", "label_end_date", "y_true", "y_pred_svm", "score_svm"]]

        lg_sub = lgbm_df[(lgbm_df.rq == rq) & (lgbm_df.config_id == lgbm_sel[rq]["config_id"])].rename(
            columns={"y_pred": "y_pred_lgbm", "ranking_score": "score_lgbm"}
        )[["rq", "fold_name", "signal_date", "y_pred_lgbm", "score_lgbm"]]

        ls_sub = lstm_df[(lstm_df.rq == rq) & (lstm_df.config_id == lstm_sel[rq]["config_id"])].rename(
            columns={"y_pred": "y_pred_lstm", "ranking_score": "score_lstm"}
        )[["rq", "fold_name", "signal_date", "y_pred_lstm", "score_lstm"]]

        merged = s_sub.merge(lg_sub, on=["rq", "fold_name", "signal_date"]).merge(
            ls_sub, on=["rq", "fold_name", "signal_date"]
        )
        merged_frames.append(merged)

        # Đánh giá theo từng fold và tổng thể
        eval_groups = [("overall_2019_2021", merged)] + [(f, grp) for f, grp in merged.groupby("fold_name")]
        for grp_name, frame in eval_groups:
            s_lg_agr = float(np.mean(frame["y_pred_svm"] == frame["y_pred_lgbm"]))
            s_ls_agr = float(np.mean(frame["y_pred_svm"] == frame["y_pred_lstm"]))
            lg_ls_agr = float(np.mean(frame["y_pred_lgbm"] == frame["y_pred_lstm"]))
            all_three_agr = float(
                np.mean((frame["y_pred_svm"] == frame["y_pred_lgbm"]) & (frame["y_pred_lgbm"] == frame["y_pred_lstm"]))
            )

            # Tương quan điểm số
            s_lg_spearman = float(spearmanr(frame["score_svm"], frame["score_lgbm"]).statistic)
            s_ls_spearman = float(spearmanr(frame["score_svm"], frame["score_lstm"]).statistic)
            lg_ls_spearman = float(spearmanr(frame["score_lgbm"], frame["score_lstm"]).statistic)

            s_lg_pearson = float(pearsonr(frame["score_svm"], frame["score_lgbm"]).statistic)
            s_ls_pearson = float(pearsonr(frame["score_svm"], frame["score_lstm"]).statistic)
            lg_ls_pearson = float(pearsonr(frame["score_lgbm"], frame["score_lstm"]).statistic)

            agreement_rows.append({
                "rq": rq,
                "scope": grp_name,
                "samples": len(frame),
                "svm_lgbm_agreement": s_lg_agr,
                "svm_lstm_agreement": s_ls_agr,
                "lgbm_lstm_agreement": lg_ls_agr,
                "three_way_agreement": all_three_agr,
                "svm_lgbm_spearman_corr": s_lg_spearman,
                "svm_lstm_spearman_corr": s_ls_spearman,
                "lgbm_lstm_spearman_corr": lg_ls_spearman,
                "svm_lgbm_pearson_corr": s_lg_pearson,
                "svm_lstm_pearson_corr": s_ls_pearson,
                "lgbm_lstm_pearson_corr": lg_ls_pearson,
            })

    return pd.DataFrame(agreement_rows), pd.concat(merged_frames, ignore_index=True)


def evaluate_protocol_a_frozen(
    root: Path,
    view: ModelingView,
    rq: str,
    test_split: TemporalSplit,
    platt_calibrator: LogisticRegression,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Giao thức A (frozen_2016_2021): Đánh giá mô hình cố định đã lưu trong models/ trên Reserved Test 2022–2026.

    Args:
        root: Thư mục gốc repository.
        view: ModelingView của RQ.
        rq: "RQ1" hoặc "RQ2".
        test_split: TemporalSplit của reserved_test.
        platt_calibrator: Bộ hiệu chuẩn Platt scaling đã fit trên OOF Dev.
        dataset: Master dataset chứa các cột forward return.

    Returns:
        DataFrame dự báo Out-of-Sample của Giao thức A.
    """
    test_indices = list(test_split.validation_indices)
    X_test = view.X.iloc[test_indices]
    y_test = view.y.iloc[test_indices].to_numpy(dtype=int)
    signal_dates = view.metadata.signal_date.iloc[test_indices].tolist()
    label_end_dates = view.metadata.label_end_date.iloc[test_indices].tolist()

    # Forward returns
    fwd_log_1 = dataset["forward_log_return_1"].iloc[test_indices].to_numpy(dtype=float)
    fwd_gross = dataset["forward_trade_gross_return"].iloc[test_indices].to_numpy(dtype=float)
    fwd_net = dataset["forward_trade_net_return_proxy"].iloc[test_indices].to_numpy(dtype=float)

    train_indices = list(test_split.train_indices)
    pred_records: list[pd.DataFrame] = []

    # 1. Dummy Baseline
    prior = float(view.y.iloc[train_indices].mean())
    d_score = np.full(len(test_indices), prior, dtype=float)
    d_pred = np.full(len(test_indices), int(prior >= 0.5), dtype=int)
    pred_records.append(pd.DataFrame({
        "protocol": "frozen_2016_2021",
        "rq": rq,
        "model_name": "dummy",
        "config_id": "dummy_prior",
        "signal_date": signal_dates,
        "label_end_date": label_end_dates,
        "y_true": y_test,
        "y_pred": d_pred,
        "ranking_score": d_score,
        "score_type": "train_prior_probability",
        "calibrated_probability": d_score,
        "forward_log_return_1": fwd_log_1,
        "forward_trade_gross_return": fwd_gross,
        "forward_trade_net_return_proxy": fwd_net,
    }))

    # 2. Linear SVM
    svm_file = root / f"models/svm/{rq.lower()}_svm_final_dev_2016_2021.joblib"
    svm_meta = json.loads((root / "models/svm/svm_final_dev_metadata.json").read_text(encoding="utf-8"))["models"][rq]
    svm_model = joblib.load(svm_file)

    s_scores = svm_model.decision_function(X_test)
    s_pred = (s_scores >= 0.0).astype(int)
    s_cal_prob = platt_calibrator.predict_proba(s_scores.reshape(-1, 1))[:, 1]

    pred_records.append(pd.DataFrame({
        "protocol": "frozen_2016_2021",
        "rq": rq,
        "model_name": "svm",
        "config_id": svm_meta["config_id"],
        "signal_date": signal_dates,
        "label_end_date": label_end_dates,
        "y_true": y_test,
        "y_pred": s_pred,
        "ranking_score": s_scores,
        "score_type": "decision_score",
        "calibrated_probability": s_cal_prob,
        "forward_log_return_1": fwd_log_1,
        "forward_trade_gross_return": fwd_gross,
        "forward_trade_net_return_proxy": fwd_net,
    }))

    # 3. LightGBM
    lgbm_file = root / f"models/lightgbm/{rq.lower()}_lgbm_final_dev_2016_2021.joblib"
    lgbm_meta = json.loads((root / "models/lightgbm/lgbm_final_dev_metadata.json").read_text(encoding="utf-8"))["models"][rq]
    lgbm_model = joblib.load(lgbm_file)

    lg_probs = lgbm_model.predict_proba(X_test)[:, 1]
    lg_pred = (lg_probs >= 0.5).astype(int)

    pred_records.append(pd.DataFrame({
        "protocol": "frozen_2016_2021",
        "rq": rq,
        "model_name": "lightgbm",
        "config_id": lgbm_meta["config_id"],
        "signal_date": signal_dates,
        "label_end_date": label_end_dates,
        "y_true": y_test,
        "y_pred": lg_pred,
        "ranking_score": lg_probs,
        "score_type": "predicted_probability",
        "calibrated_probability": lg_probs,
        "forward_log_return_1": fwd_log_1,
        "forward_trade_gross_return": fwd_gross,
        "forward_trade_net_return_proxy": fwd_net,
    }))

    # 4. PyTorch LSTM
    lstm_file = root / f"models/lstm/{rq.lower()}_lstm_final_dev_2016_2021.pt"
    lstm_model, lstm_scaler, lstm_bundle = load_final_lstm_bundle(lstm_file)
    seq_len = int(lstm_bundle["seq_len"])

    # Xây dựng chuỗi test nhân quả với scaler đã lưu từ 2016–2021
    X_scaled_all = lstm_scaler.transform(view.X.to_numpy(dtype=np.float32)).astype(np.float32)
    n_test = len(test_indices)
    X_test_seq = np.empty((n_test, seq_len, view.X.shape[1]), dtype=np.float32)

    for k, t_idx in enumerate(test_indices):
        start_idx = t_idx - seq_len + 1
        X_test_seq[k] = X_scaled_all[start_idx : t_idx + 1]

    with torch.no_grad():
        lstm_logits = lstm_model(torch.from_numpy(X_test_seq))
        ls_probs = torch.sigmoid(lstm_logits).cpu().numpy().flatten()
        ls_pred = (ls_probs >= 0.5).astype(int)

    pred_records.append(pd.DataFrame({
        "protocol": "frozen_2016_2021",
        "rq": rq,
        "model_name": "lstm",
        "config_id": lstm_bundle["config_id"],
        "signal_date": signal_dates,
        "label_end_date": label_end_dates,
        "y_true": y_test,
        "y_pred": ls_pred,
        "ranking_score": ls_probs,
        "score_type": "predicted_probability",
        "calibrated_probability": ls_probs,
        "forward_log_return_1": fwd_log_1,
        "forward_trade_gross_return": fwd_gross,
        "forward_trade_net_return_proxy": fwd_net,
    }))

    return pd.concat(pred_records, ignore_index=True)


def evaluate_protocol_b_expanding(
    root: Path,
    view: ModelingView,
    rq: str,
    test_split: TemporalSplit,
    master_index: pd.DatetimeIndex,
    config: dict,
    platt_calibrator: LogisticRegression,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Giao thức B (annual_expanding_retrain): Retrain mở rộng định kỳ đầu mỗi năm (2022–2026).

    Giữ cố định 100% hyperparameters đã chọn ở Development CV.
    Áp dụng đúng quy tắc label-aware purging và gap 5 phiên trước ngày giao dịch đầu tiên của năm Y.

    Args:
        root: Thư mục gốc repository.
        view: ModelingView của RQ.
        rq: "RQ1" hoặc "RQ2".
        test_split: TemporalSplit của reserved_test.
        master_index: Lịch giao dịch master index của dataset.
        config: Cấu hình hệ thống config.yaml.
        platt_calibrator: Bộ hiệu chuẩn Platt scaling cho SVM.
        dataset: Master dataset chứa forward returns.

    Returns:
        DataFrame dự báo Out-of-Sample của Giao thức B.
    """
    signal_positions, end_positions = _validate_view(master_index, view, rq, config)
    test_indices_all = set(test_split.validation_indices)

    # Đọc thông số cấu hình thắng cuộc từ Dev
    svm_meta = json.loads((root / "models/svm/svm_final_dev_metadata.json").read_text(encoding="utf-8"))["models"][rq]
    lgbm_meta = json.loads((root / "models/lightgbm/lgbm_final_dev_metadata.json").read_text(encoding="utf-8"))["models"][rq]
    lstm_meta = json.loads((root / "models/lstm/lstm_final_dev_metadata.json").read_text(encoding="utf-8"))["models"][rq]

    years_eval = [2022, 2023, 2024, 2025, 2026]
    year_dfs: list[pd.DataFrame] = []

    for year in years_eval:
        boundary = _first_session(master_index, year)
        # 1. Tập Train mở rộng: trước năm year, thỏa mãn purging và gap 5 phiên
        candidates = [i for i, yr in enumerate(view.X.index.year) if yr < year]
        train_indices = [
            i for i in candidates
            if end_positions[i] < boundary and signal_positions[i] < boundary - 5
        ]

        # 2. Tập Test của năm year
        test_indices = [i for i in test_indices_all if view.X.index[i].year == year]
        test_indices.sort()

        if not test_indices or not train_indices:
            continue

        X_train_raw = view.X.iloc[train_indices]
        y_train = view.y.iloc[train_indices].to_numpy(dtype=int)
        X_test_raw = view.X.iloc[test_indices]
        y_test = view.y.iloc[test_indices].to_numpy(dtype=int)

        signal_dates = view.metadata.signal_date.iloc[test_indices].tolist()
        label_end_dates = view.metadata.label_end_date.iloc[test_indices].tolist()

        fwd_log_1 = dataset["forward_log_return_1"].iloc[test_indices].to_numpy(dtype=float)
        fwd_gross = dataset["forward_trade_gross_return"].iloc[test_indices].to_numpy(dtype=float)
        fwd_net = dataset["forward_trade_net_return_proxy"].iloc[test_indices].to_numpy(dtype=float)

        # A. Dummy
        d_prior = float(y_train.mean())
        d_score = np.full(len(test_indices), d_prior, dtype=float)
        d_pred = np.full(len(test_indices), int(d_prior >= 0.5), dtype=int)
        year_dfs.append(pd.DataFrame({
            "protocol": "annual_expanding_retrain",
            "rq": rq,
            "model_name": "dummy",
            "config_id": "dummy_prior",
            "signal_date": signal_dates,
            "label_end_date": label_end_dates,
            "y_true": y_test,
            "y_pred": d_pred,
            "ranking_score": d_score,
            "score_type": "train_prior_probability",
            "calibrated_probability": d_score,
            "forward_log_return_1": fwd_log_1,
            "forward_trade_gross_return": fwd_gross,
            "forward_trade_net_return_proxy": fwd_net,
        }))

        # B. Linear SVM (Retrained)
        from sklearn.svm import SVC
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="linear",
                C=float(svm_meta["C"]),
                class_weight=svm_meta["class_weight"],
                random_state=42,
            )),
        ])
        svm_pipe.fit(X_train_raw, y_train)
        s_scores = svm_pipe.decision_function(X_test_raw)
        s_pred = (s_scores >= 0.0).astype(int)
        s_cal_prob = platt_calibrator.predict_proba(s_scores.reshape(-1, 1))[:, 1]
        year_dfs.append(pd.DataFrame({
            "protocol": "annual_expanding_retrain",
            "rq": rq,
            "model_name": "svm",
            "config_id": svm_meta["config_id"],
            "signal_date": signal_dates,
            "label_end_date": label_end_dates,
            "y_true": y_test,
            "y_pred": s_pred,
            "ranking_score": s_scores,
            "score_type": "decision_score",
            "calibrated_probability": s_cal_prob,
            "forward_log_return_1": fwd_log_1,
            "forward_trade_gross_return": fwd_gross,
            "forward_trade_net_return_proxy": fwd_net,
        }))

        # C. LightGBM (Retrained)
        import lightgbm as lgb
        lgbm_est = lgb.LGBMClassifier(
            objective="binary",
            max_depth=int(lgbm_meta["max_depth"]),
            num_leaves=int(lgbm_meta["num_leaves"]),
            learning_rate=float(lgbm_meta["learning_rate"]),
            min_child_samples=int(lgbm_meta["min_child_samples"]),
            reg_lambda=float(lgbm_meta["reg_lambda"]),
            class_weight=lgbm_meta["class_weight"],
            n_estimators=int(lgbm_meta["final_n_estimators"]),
            random_state=42,
            n_jobs=1,
            deterministic=True,
            verbose=-1,
        )
        lgbm_est.fit(X_train_raw, y_train)
        lg_probs = lgbm_est.predict_proba(X_test_raw)[:, 1]
        lg_pred = (lg_probs >= 0.5).astype(int)
        year_dfs.append(pd.DataFrame({
            "protocol": "annual_expanding_retrain",
            "rq": rq,
            "model_name": "lightgbm",
            "config_id": lgbm_meta["config_id"],
            "signal_date": signal_dates,
            "label_end_date": label_end_dates,
            "y_true": y_test,
            "y_pred": lg_pred,
            "ranking_score": lg_probs,
            "score_type": "predicted_probability",
            "calibrated_probability": lg_probs,
            "forward_log_return_1": fwd_log_1,
            "forward_trade_gross_return": fwd_gross,
            "forward_trade_net_return_proxy": fwd_net,
        }))

        # D. PyTorch LSTM (Retrained)
        seq_len = int(lstm_meta["seq_len"])
        final_epochs = int(lstm_meta["final_epochs"])
        scaler = StandardScaler()
        scaler.fit(X_train_raw.to_numpy(dtype=np.float32))

        X_scaled_all = scaler.transform(view.X.to_numpy(dtype=np.float32)).astype(np.float32)
        y_all = view.y.to_numpy(dtype=np.float32)

        # Train sequences
        n_tr_seq = len(train_indices) - seq_len + 1
        X_tr_seq = np.empty((n_tr_seq, seq_len, view.X.shape[1]), dtype=np.float32)
        y_tr_seq = np.empty(n_tr_seq, dtype=np.float32)
        for i in range(n_tr_seq):
            s_i = train_indices[i]
            e_i = train_indices[i + seq_len - 1]
            X_tr_seq[i] = X_scaled_all[s_i : e_i + 1]
            y_tr_seq[i] = y_all[e_i]

        # Test sequences
        n_te = len(test_indices)
        X_te_seq = np.empty((n_te, seq_len, view.X.shape[1]), dtype=np.float32)
        for k, t_idx in enumerate(test_indices):
            s_i = t_idx - seq_len + 1
            X_te_seq[k] = X_scaled_all[s_i : t_idx + 1]

        set_seed(42)
        lstm_model = LSTMClassifier(
            input_size=view.X.shape[1],
            hidden_size=int(lstm_meta["hidden_size"]),
            num_layers=int(lstm_meta["num_layers"]),
            dropout=float(lstm_meta["dropout"]),
        )
        if lstm_meta["class_weight"] == "balanced":
            pos_weight = torch.tensor([float(np.sum(y_tr_seq == 0.0)) / max(1.0, float(np.sum(y_tr_seq == 1.0)))])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            lstm_model.parameters(),
            lr=float(lstm_meta["learning_rate"]),
            weight_decay=float(lstm_meta["weight_decay"]),
        )
        train_ds = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq))
        g = torch.Generator().manual_seed(42)
        train_ld = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)

        lstm_model.train()
        for _ in range(final_epochs):
            for bx, by in train_ld:
                optimizer.zero_grad()
                out = lstm_model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                optimizer.step()

        lstm_model.eval()
        with torch.no_grad():
            ls_logits = lstm_model(torch.from_numpy(X_te_seq))
            ls_probs = torch.sigmoid(ls_logits).cpu().numpy().flatten()
            ls_pred = (ls_probs >= 0.5).astype(int)

        year_dfs.append(pd.DataFrame({
            "protocol": "annual_expanding_retrain",
            "rq": rq,
            "model_name": "lstm",
            "config_id": lstm_meta["config_id"],
            "signal_date": signal_dates,
            "label_end_date": label_end_dates,
            "y_true": y_test,
            "y_pred": ls_pred,
            "ranking_score": ls_probs,
            "score_type": "predicted_probability",
            "calibrated_probability": ls_probs,
            "forward_log_return_1": fwd_log_1,
            "forward_trade_gross_return": fwd_gross,
            "forward_trade_net_return_proxy": fwd_net,
        }))

    return pd.concat(year_dfs, ignore_index=True)


def compute_test_metrics_summary(
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính bảng tổng hợp metrics toàn kỳ Test (2022–2026) và phân rã theo từng năm.

    Args:
        predictions_df: DataFrame chứa toàn bộ dự báo test của cả 2 giao thức và các mô hình.

    Returns:
        Tuple gồm (test_overall_df, test_yearly_df).
    """
    overall_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    # 1. Phân rã theo từng năm
    for (protocol, rq, model_name, config_id), grp in predictions_df.groupby(
        ["protocol", "rq", "model_name", "config_id"], sort=False
    ):
        fwd_col = "forward_log_return_1" if rq == "RQ1" else "forward_trade_net_return_proxy"
        yearly_rank_ics: list[float] = []

        grp = grp.copy()
        grp["year"] = pd.to_datetime(grp["signal_date"]).dt.year

        for year, y_frame in grp.groupby("year"):
            m = compute_quant_metrics(
                y_frame["y_true"].to_numpy(),
                y_frame["y_pred"].to_numpy(),
                y_frame["ranking_score"].to_numpy(),
                y_frame["calibrated_probability"].to_numpy(),
                y_frame[fwd_col].to_numpy(),
            )
            yearly_rank_ics.append(m["spearman_rank_ic"])
            yearly_rows.append({
                "protocol": protocol,
                "rq": rq,
                "model_name": model_name,
                "config_id": config_id,
                "year": year,
                "samples": len(y_frame),
                **m,
            })

        # Toàn kỳ (Overall 2022–2026)
        overall_m = compute_quant_metrics(
            grp["y_true"].to_numpy(),
            grp["y_pred"].to_numpy(),
            grp["ranking_score"].to_numpy(),
            grp["calibrated_probability"].to_numpy(),
            grp[fwd_col].to_numpy(),
        )

        # Tính IC IR (Information Coefficient Information Ratio) qua các năm
        mean_ic = float(np.mean(yearly_rank_ics)) if yearly_rank_ics else 0.0
        std_ic = float(np.std(yearly_rank_ics, ddof=0)) if len(yearly_rank_ics) > 1 else 0.0
        ic_ir = float(mean_ic / std_ic) if std_ic > 1e-6 else 0.0

        overall_rows.append({
            "protocol": protocol,
            "rq": rq,
            "model_name": model_name,
            "config_id": config_id,
            "evaluation_period": "2022-2026",
            "samples": len(grp),
            **overall_m,
            "mean_annual_rank_ic": mean_ic,
            "std_annual_rank_ic": std_ic,
            "ic_ir": ic_ir,
        })

    overall_df = pd.DataFrame(overall_rows)
    yearly_df = pd.DataFrame(yearly_rows)

    # Bổ sung delta vs Dummy
    for df in (overall_df, yearly_df):
        merge_keys = ["protocol", "rq", "year"] if "year" in df.columns else ["protocol", "rq"]
        dummy_bacc = df[df.model_name == "dummy"].set_index(merge_keys)["balanced_accuracy"]
        df["delta_balanced_accuracy_vs_dummy"] = (
            df["balanced_accuracy"] - pd.MultiIndex.from_frame(df[merge_keys]).map(dummy_bacc).to_numpy(dtype=float)
        )

    return overall_df, yearly_df
