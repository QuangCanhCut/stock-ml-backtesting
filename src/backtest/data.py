"""Data Contract và căn chỉnh lịch giao dịch cho Backtest (WP4.1).

Module này hiện thực hóa các chức năng:
- Kiểm toán mã băm SHA-256 bất biến của Dataset nghiên cứu và Split Manifest.
- Căn chỉnh ngày tín hiệu (signal_date t) với ngày thực thi mở cửa (execution_date t+1)
  và phiên bán sớm nhất theo chu kỳ thanh toán T+2.5 (earliest_sell_date t+4).
- Nạp và xác thực bộ đệm hạt giống (Dev OOF Seed Buffers) từ các mô hình thắng cuộc
  của giai đoạn phát triển (2019–2021) để chống Cold Start trong tính toán phân vị nhân quả.
- Đảm bảo 100% không có rò rỉ dữ liệu tương lai (Zero Look-ahead Bias).
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.rules import HOSEMicrostructureRules

RESEARCH_DATASET_REL_PATH = Path("data/processed/HPG_features_labeled_research.parquet")
SPLIT_MANIFEST_REL_PATH = Path("results/modeling/splits/split_manifest.json")
TEST_PREDICTIONS_REL_PATH = Path("results/modeling/final_evaluation/test_predictions_2022_2026.parquet")

EXPECTED_DATASET_SHA256 = "4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e"
EXPECTED_MANIFEST_SHA256 = "f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42"

WINNING_CONFIGS: dict[str, dict[str, tuple[str, str]]] = {
    "RQ1": {
        "dummy": ("svm", "dummy_prior"),
        "svm": ("svm", "linear_c0.1_balanced"),
        "lightgbm": ("lightgbm", "lgbm_d3_lr0.03_mc80_l5_none"),
        "lstm": ("lstm", "lstm_seq10_h32_lr0.0005_none"),
    },
    "RQ2": {
        "dummy": ("svm", "dummy_prior"),
        "svm": ("svm", "linear_c1_balanced"),
        "lightgbm": ("lightgbm", "lgbm_d3_lr0.05_mc40_l1_balanced"),
        "lstm": ("lstm", "lstm_seq20_h16_lr0.001_balanced"),
    },
}


def compute_file_sha256(file_path: Path) -> str:
    """Tính mã băm SHA-256 của một file trên đĩa.

    Args:
        file_path: Đường dẫn tệp tin cần kiểm tra.

    Returns:
        Chuỗi băm hex 64 ký tự.
    """
    hasher = hashlib.sha256()
    hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def verify_research_artifacts(root: Path) -> dict[str, str]:
    """Kiểm toán tính bất biến của các tệp tin cốt lõi từ Phase 2 và Phase 3.

    Args:
        root: Thư mục gốc của repository.

    Returns:
        Từ điển chứa mã băm thực tế của các artifacts.

    Raises:
        FileNotFoundError: Khi một trong các tệp tin cốt lõi bị thiếu.
        ValueError: Khi mã băm không khớp chính xác với giá trị đã kiểm toán.
    """
    root = Path(root).resolve()
    dataset_file = root / RESEARCH_DATASET_REL_PATH
    manifest_file = root / SPLIT_MANIFEST_REL_PATH

    if not dataset_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset nghiên cứu tại: {dataset_file}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy split manifest tại: {manifest_file}")

    actual_ds_hash = compute_file_sha256(dataset_file)
    actual_mf_hash = compute_file_sha256(manifest_file)

    if actual_ds_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset SHA-256 sai lệch! Thực tế: {actual_ds_hash}, Kỳ vọng: {EXPECTED_DATASET_SHA256}"
        )
    if actual_mf_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA-256 sai lệch! Thực tế: {actual_mf_hash}, Kỳ vọng: {EXPECTED_MANIFEST_SHA256}"
        )

    return {
        "dataset_sha256": actual_ds_hash,
        "manifest_sha256": actual_mf_hash,
    }


def load_market_calendar(root: Path) -> pd.DataFrame:
    """Nạp lịch giao dịch chuẩn (Master Calendar) và các chuỗi giá/biến động HPG.

    Args:
        root: Thư mục gốc của repository.

    Returns:
        DataFrame sắp xếp theo thứ tự thời gian tăng dần với DatetimeIndex.
    """
    root = Path(root).resolve()
    dataset_file = root / RESEARCH_DATASET_REL_PATH
    df = pd.read_parquet(dataset_file)

    # Đảm bảo index là DatetimeIndex chuẩn hóa
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required_cols = [
        "open", "high", "low", "close", "volume",
        "volatility_21", "atr_ratio_14",
    ]
    return df[required_cols].copy()


def load_dev_oof_seed_buffers(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Nạp phân phối điểm số Out-of-Fold của tập Development (2019–2021) làm bộ đệm khởi tạo.

    Phục vụ cho việc tính toán phân vị nhân quả (Causal Expanding Quantiles) từ phiên đầu tiên
    của năm 2022 mà không bị lỗi Cold Start (thiếu mẫu lịch sử ban đầu).

    Args:
        root: Thư mục gốc của repository.

    Returns:
        Từ điển phân tầng [rq][model_name] chứa mảng 1D numpy các điểm số ranking_score OOF Dev.
        Bao gồm cả 'ensemble' là điểm Z-score trung bình trọng số đều của 3 mô hình.
    """
    root = Path(root).resolve()
    buffers: dict[str, dict[str, np.ndarray]] = {"RQ1": {}, "RQ2": {}}

    for rq in ("RQ1", "RQ2"):
        model_scores: dict[str, np.ndarray] = {}
        dates_check: list[pd.Timestamp] | None = None

        for model_name, (folder, cfg_id) in WINNING_CONFIGS[rq].items():
            pred_file = root / f"results/modeling/{folder}/validation_predictions.parquet"
            if not pred_file.is_file():
                raise FileNotFoundError(f"Không tìm thấy file OOF Dev predictions: {pred_file}")

            df_val = pd.read_parquet(pred_file)
            sub = df_val[(df_val["rq"] == rq) & (df_val["config_id"] == cfg_id)].copy()
            sub = sub.sort_values("signal_date")

            curr_dates = sub["signal_date"].tolist()
            if dates_check is None:
                dates_check = curr_dates
            elif curr_dates != dates_check:
                raise ValueError(f"{rq} {model_name}: Lịch ngày OOF Dev không khớp với các mô hình khác!")

            scores = sub["ranking_score"].to_numpy(dtype=float)
            model_scores[model_name] = scores
            buffers[rq][model_name] = scores

        # Tính toán điểm OOF Dev cho 'ensemble'
        # Chuẩn hóa Z-score trên Dev OOF cho từng mô hình thành phần: svm, lightgbm, lstm
        z_models = []
        for m in ("svm", "lightgbm", "lstm"):
            s = model_scores[m]
            mean_s = float(np.mean(s))
            std_s = float(np.std(s))
            if std_s < 1e-12:
                std_s = 1.0
            z_m = (s - mean_s) / std_s
            z_models.append(z_m)

        # Trung bình trọng số đều (1/3)
        z_ensemble = np.mean(z_models, axis=0)
        buffers[rq]["ensemble"] = z_ensemble

    return buffers


def align_predictions_with_market(
    predictions_df: pd.DataFrame,
    market_df: pd.DataFrame,
    rules: HOSEMicrostructureRules,
) -> pd.DataFrame:
    """Căn chỉnh bảng dự đoán Test với dữ liệu thị trường theo đúng luật vi cấu trúc HOSE.

    Quy tắc nhân quả bất biến (Causal Alignment):
    - Tín hiệu tại ngày t (signal_date) chỉ biết giá Close(t) và các chỉ báo tại ngày t.
    - Lệnh thực thi mua sớm nhất tại phiên mở cửa sáng hôm sau (execution_date = t+1, giá exec_open).
    - Chu kỳ thanh toán T+2.5: Phiên mở cửa sớm nhất được phép bán là t+4 (earliest_sell_date, giá earliest_sell_open).
    - Nhận diện kẹt thanh khoản: is_ceiling_locked (không mua được) và is_floor_locked (không bán được).

    Args:
        predictions_df: DataFrame dự đoán từ test_predictions_2022_2026.parquet.
        market_df: DataFrame dữ liệu thị trường từ load_market_calendar.
        rules: Đối tượng quy tắc vi cấu trúc sàn HOSE.

    Returns:
        DataFrame đã căn chỉnh đầy đủ thông tin thực thi và chỉ báo biến động.

    Raises:
        ValueError: Nếu có bất kỳ dòng nào bị thiếu giá khớp lệnh exec_open.
    """
    calendar_dates = list(market_df.index)
    date_to_idx = {d: i for i, d in enumerate(calendar_dates)}
    n_calendar = len(calendar_dates)

    # Thu thập danh sách ngày signal_date duy nhất
    signal_dates = sorted(predictions_df["signal_date"].unique())

    market_lookup: list[dict[str, Any]] = []
    for sig_date in signal_dates:
        sig_ts = pd.to_datetime(sig_date)
        if sig_ts not in date_to_idx:
            raise ValueError(f"Ngày signal_date {sig_ts} không tồn tại trong Master Calendar!")

        idx = date_to_idx[sig_ts]
        row_t = market_df.iloc[idx]

        # Phiên thực thi t+1
        if idx + 1 < n_calendar:
            exec_date = calendar_dates[idx + 1]
            row_next = market_df.iloc[idx + 1]
            exec_open = float(row_next["open"])
            exec_high = float(row_next["high"])
            exec_low = float(row_next["low"])
            exec_close = float(row_next["close"])
            exec_volume = float(row_next["volume"])
        else:
            exec_date = pd.NaT
            exec_open = np.nan
            exec_high = np.nan
            exec_low = np.nan
            exec_close = np.nan
            exec_volume = np.nan

        # Phiên bán sớm nhất theo T+2.5 (phiên t+4)
        if idx + 4 < n_calendar:
            earliest_sell_date = calendar_dates[idx + 4]
            earliest_sell_open = float(market_df.iloc[idx + 4]["open"])
        else:
            earliest_sell_date = pd.NaT
            earliest_sell_open = np.nan

        prev_close = float(row_t["close"])
        is_ceil = rules.is_ceiling_locked(exec_open, prev_close) if not np.isnan(exec_open) else False
        is_floor = rules.is_floor_locked(exec_open, prev_close) if not np.isnan(exec_open) else False

        market_lookup.append({
            "signal_date": sig_ts,
            "execution_date": exec_date,
            "prev_close": prev_close,
            "exec_open": exec_open,
            "exec_high": exec_high,
            "exec_low": exec_low,
            "exec_close": exec_close,
            "exec_volume": exec_volume,
            "earliest_sell_date": earliest_sell_date,
            "earliest_sell_open": earliest_sell_open,
            "volatility_21": float(row_t["volatility_21"]),
            "atr_ratio_14": float(row_t["atr_ratio_14"]),
            "is_ceiling_locked": is_ceil,
            "is_floor_locked": is_floor,
        })

    lookup_df = pd.DataFrame(market_lookup)

    # Đảm bảo kiểu dữ liệu ngày khớp nhau trước khi merge
    preds_copy = predictions_df.copy()
    preds_copy["signal_date"] = pd.to_datetime(preds_copy["signal_date"])

    merged = pd.merge(preds_copy, lookup_df, on="signal_date", how="left")

    # Kiểm tra kiểm toán: 0 dòng bị thiếu exec_open
    missing_exec = merged["exec_open"].isna().sum()
    if missing_exec > 0:
        raise ValueError(f"Phát hiện {missing_exec} dòng bị thiếu giá thực thi exec_open!")

    return merged

