"""Kiểm thử tích hợp End-to-End cho toàn bộ quy trình WP4.1 & WP4.2 (Pipeline & Data Contract)."""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.backtest.data import (
    RESEARCH_DATASET_REL_PATH,
    SPLIT_MANIFEST_REL_PATH,
    compute_file_sha256,
    verify_research_artifacts,
)


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Trả về đường dẫn thư mục gốc của repository."""
    return Path(__file__).resolve().parents[2]


def test_research_artifacts_integrity(repo_root: Path) -> None:
    """Xác nhận mã băm SHA-256 của Phase 2 & Phase 3 không bị thay đổi."""
    hashes = verify_research_artifacts(repo_root)
    assert hashes["dataset_sha256"] == "4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e"
    assert hashes["manifest_sha256"] == "f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42"


def test_backtest_signals_parquet_contract(repo_root: Path) -> None:
    """Kiểm tra hợp đồng dữ liệu của file Parquet tín hiệu backtest."""
    signals_file = repo_root / "results/backtesting/backtest_signals_2022_2026.parquet"
    assert signals_file.is_file(), f"Thiếu file tín hiệu: {signals_file}"

    df = pd.read_parquet(signals_file)

    # 1. Kiểm tra kích thước dữ liệu (56 chuỗi tín hiệu = 67.116 dòng)
    assert len(df) == 67116

    # 2. Kiểm tra các cột bắt buộc
    required_cols = [
        "protocol", "rq", "model_name", "signal_mode", "signal_date", "execution_date",
        "prev_close", "exec_open", "exec_close", "earliest_sell_open",
        "volatility_21", "atr_ratio_14", "ranking_score", "calibrated_probability",
        "entry_threshold_val", "exit_threshold_val", "trigger", "desired_position",
        "is_ceiling_locked", "is_floor_locked", "forward_log_return_1",
        "forward_trade_gross_return", "forward_trade_net_return_proxy", "y_true",
    ]
    for col in required_cols:
        assert col in df.columns, f"Thiếu cột {col} trong Parquet tín hiệu!"

    # 3. Khẳng định 0 dòng bị thiếu giá khớp lệnh exec_open
    assert df["exec_open"].isna().sum() == 0

    # 4. Kiểm tra các giá trị hợp lệ của trigger và position
    assert set(df["trigger"].unique()).issubset({"ENTRY", "EXIT", "BUFFER"})
    assert set(df["desired_position"].unique()).issubset({0, 1})


def test_signal_diagnostics_and_starvation_resolution(repo_root: Path) -> None:
    """Kiểm tra báo cáo chẩn đoán và xác nhận giải quyết triệt để hiện tượng Đói Tín Hiệu."""
    diag_file = repo_root / "results/backtesting/signal_diagnostics_2022_2026.csv"
    assert diag_file.is_file(), f"Thiếu file chẩn đoán: {diag_file}"

    df_diag = pd.read_csv(diag_file)
    assert len(df_diag) == 56  # 56 chuỗi tín hiệu độc lập

    # 1. Kiểm tra SVM RQ2 (annual_expanding_retrain):
    # Ở fixed_053_048, số lần vào lệnh chỉ là 10, trong khi quantile_q80_q30 tăng lên 98 lần
    svm_rq2_fixed = df_diag[
        (df_diag["protocol"] == "annual_expanding_retrain")
        & (df_diag["rq"] == "RQ2")
        & (df_diag["model_name"] == "svm")
        & (df_diag["signal_mode"] == "fixed_053_048")
    ].iloc[0]

    svm_rq2_q80 = df_diag[
        (df_diag["protocol"] == "annual_expanding_retrain")
        & (df_diag["rq"] == "RQ2")
        & (df_diag["model_name"] == "svm")
        & (df_diag["signal_mode"] == "quantile_q80_q30")
    ].iloc[0]

    assert svm_rq2_fixed["entry_trigger_count"] == 10
    assert svm_rq2_q80["entry_trigger_count"] == 98
    assert svm_rq2_q80["desired_state_switches"] > svm_rq2_fixed["desired_state_switches"]

    # 2. Kiểm tra LightGBM RQ1 (annual_expanding_retrain):
    # Ở fixed_053_048, time in market chỉ 7.75%, trong khi quantile_q80_q30 đạt 34.83%
    lgbm_rq1_fixed = df_diag[
        (df_diag["protocol"] == "annual_expanding_retrain")
        & (df_diag["rq"] == "RQ1")
        & (df_diag["model_name"] == "lightgbm")
        & (df_diag["signal_mode"] == "fixed_053_048")
    ].iloc[0]

    lgbm_rq1_q80 = df_diag[
        (df_diag["protocol"] == "annual_expanding_retrain")
        & (df_diag["rq"] == "RQ1")
        & (df_diag["model_name"] == "lightgbm")
        & (df_diag["signal_mode"] == "quantile_q80_q30")
    ].iloc[0]

    assert lgbm_rq1_fixed["desired_time_in_market_pct"] < 10.0
    assert lgbm_rq1_q80["desired_time_in_market_pct"] > 30.0


def test_audit_json_metadata(repo_root: Path) -> None:
    """Kiểm tra tệp tin metadata JSON kiểm toán."""
    audit_file = repo_root / "results/backtesting/wp41_42_contract_and_signal_audit.json"
    assert audit_file.is_file(), f"Thiếu file JSON audit: {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["status"] == "COMPLETED"
    assert meta["aligned_market_data"]["missing_exec_open_count"] == 0
    assert meta["aligned_market_data"]["rq1_sessions"] == 1200
    assert meta["aligned_market_data"]["rq2_sessions"] == 1197
    assert meta["output_signal_rows"] == 67116

