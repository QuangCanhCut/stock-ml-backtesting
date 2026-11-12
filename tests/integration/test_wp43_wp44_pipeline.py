"""Kiểm thử tích hợp End-to-End cho WP4.3 & WP4.4 (Phase 4).

Bao gồm:
1. Xác minh tính bất biến mã băm SHA-256 của Dataset nghiên cứu và Split Manifest.
2. Kiểm tra sự tồn tại và tính hợp lệ cấu trúc của 6 artifacts tại results/backtesting/.
3. Kiểm tra strategy_comparison_2022_2026.csv: Đủ 114 chiến lược, không có NaN/Inf, sắp xếp đúng.
4. Kiểm tra all_daily_portfolios_2022_2026.parquet: Đủ 136.914 dòng, 100% tuân thủ bất biến kế toán.
5. Kiểm tra all_trade_logs_2022_2026.csv: 100% lệnh đóng tuân thủ holding_sessions >= 3 do luật T+2.5.
6. Kiểm tra cost_sensitivity.csv: Đủ 4 kịch bản phí và tính đơn điệu của chi phí giao dịch.
7. Kiểm tra wp43_44_backtest_audit.json: Metadata và Top 10 chiến lược.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest.data import verify_research_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "backtesting"


def test_research_artifacts_integrity() -> None:
    """Xác minh mã băm SHA-256 của dataset và manifest giữ nguyên 100%."""
    hashes = verify_research_artifacts(REPO_ROOT)
    assert hashes["dataset_sha256"] == "4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e"
    assert hashes["manifest_sha256"] == "f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42"


def test_all_six_artifacts_exist() -> None:
    """Kiểm tra sự tồn tại của đầy đủ 6 artifacts bắt buộc của WP4.3 & WP4.4."""
    expected_files = [
        "strategy_comparison_2022_2026.csv",
        "yearly_strategy_performance.csv",
        "cost_sensitivity.csv",
        "all_daily_portfolios_2022_2026.parquet",
        "all_trade_logs_2022_2026.csv",
        "wp43_44_backtest_audit.json",
    ]
    for filename in expected_files:
        filepath = RESULTS_DIR / filename
        assert filepath.is_file(), f"Thiếu artifact bắt buộc: {filepath}"
        assert filepath.stat().st_size > 0, f"Artifact bị rỗng: {filepath}"


def test_strategy_comparison_contract() -> None:
    """Kiểm toán bảng so sánh 114 chiến lược (strategy_comparison_2022_2026.csv)."""
    csv_path = RESULTS_DIR / "strategy_comparison_2022_2026.csv"
    df = pd.read_csv(csv_path)

    # 1. Số lượng chiến lược: đúng 114 (112 ML + 2 Benchmarks)
    assert len(df) == 114, f"Kỳ vọng 114 chiến lược, thực tế: {len(df)}"

    # 2. Hai benchmark bắt buộc phải hiện diện
    benchmarks = set(df["strategy_id"])
    assert "buy_and_hold_hpg" in benchmarks, "Thiếu benchmark buy_and_hold_hpg"
    assert "risk_free_cash" in benchmarks, "Thiếu benchmark risk_free_cash"

    # 3. Kiểm tra không có NaN hoặc Inf trong các cột định lượng quan trọng
    check_cols = [
        "final_nav_vnd", "total_return_pct", "cagr_pct", "annualized_volatility_pct",
        "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
        "total_trades", "win_rate_pct", "annual_turnover", "total_fees_paid_vnd",
    ]
    for col in check_cols:
        assert not df[col].isna().any(), f"Cột {col} trong bảng tổng hợp chứa giá trị NaN!"
        assert not np.isinf(df[col]).any(), f"Cột {col} trong bảng tổng hợp chứa giá trị Inf!"

    # 4. Sắp xếp giảm dần theo Sharpe Ratio
    sharpes = df["sharpe_ratio"].tolist()
    assert sharpes == sorted(sharpes, reverse=True), "Bảng so sánh chưa được sắp xếp giảm dần theo Sharpe Ratio!"


def test_all_daily_portfolios_integrity() -> None:
    """Kiểm toán chuỗi dữ liệu danh mục hằng ngày (all_daily_portfolios_2022_2026.parquet)."""
    parquet_path = RESULTS_DIR / "all_daily_portfolios_2022_2026.parquet"
    df = pd.read_parquet(parquet_path)

    # 114 chiến lược x 1.201 phiên giao dịch = 136.914 dòng
    expected_rows = 114 * 1201
    assert len(df) == expected_rows, f"Kỳ vọng {expected_rows} dòng daily, thực tế: {len(df)}"

    # Kiểm tra bất biến kế toán trên toàn bộ 136.914 dòng
    nav_diff = np.abs(df["nav"] - (df["cash"] + df["stock_value"]))
    max_diff = float(nav_diff.max())
    assert max_diff < 1e-4, f"Vi phạm bất biến kế toán: độ lệch lớn nhất là {max_diff}"

    # Đảm bảo NAV luôn dương và không có giá trị NaN
    assert (df["nav"] > 0).all(), "Phát hiện giá trị NAV <= 0"
    assert not df["nav"].isna().any(), "Phát hiện giá trị NAV bị NaN"


def test_trade_logs_and_t25_compliance() -> None:
    """Kiểm toán nhật ký giao dịch và tuân thủ chu kỳ thanh toán T+2.5."""
    trades_path = RESULTS_DIR / "all_trade_logs_2022_2026.csv"
    df = pd.read_csv(trades_path)

    assert not df.empty, "Bảng trade logs không được rỗng!"

    # Với các lệnh đã đóng bình thường ('CLOSED'), thời gian nắm giữ bắt buộc phải >= 3 phiên
    closed_trades = df[df["status"] == "CLOSED"]
    if not closed_trades.empty:
        min_holding = closed_trades["holding_sessions"].min()
        assert min_holding >= 3, f"Phát hiện lệnh đóng vi phạm luật T+2.5! Holding session nhỏ nhất: {min_holding}"

    # Kiểm tra các cột phí và PnL không có NaN
    pnl_cols = ["gross_pnl", "net_pnl", "buy_fee", "sell_fee"]
    for col in pnl_cols:
        assert not df[col].isna().any(), f"Cột {col} trong trade logs chứa NaN!"


def test_cost_sensitivity_stress_test_integrity() -> None:
    """Kiểm toán kết quả bài kiểm tra độ nhạy chi phí giao dịch (cost_sensitivity.csv)."""
    cost_path = RESULTS_DIR / "cost_sensitivity.csv"
    df = pd.read_csv(cost_path)

    # Kiểm tra đủ 4 mức chi phí: 0.30%, 0.50%, 0.75%, 1.00%
    scenarios = set(df["cost_scenario_pct"])
    assert scenarios == {0.30, 0.50, 0.75, 1.00}, f"Các kịch bản chi phí không khớp: {scenarios}"

    # Kiểm tra tính đơn điệu: Phí giao dịch phải tăng khi mức phí tăng
    grouped = df.groupby(["protocol", "rq", "model_name", "signal_mode", "sizing_mode"])
    for _, grp in grouped:
        sorted_grp = grp.sort_values("cost_scenario_pct")
        fees = sorted_grp["total_fees_paid_vnd"].tolist()
        # Phí của kịch bản 1.00% phải lớn hơn kịch bản 0.30% (nếu có giao dịch)
        if fees[-1] > 0:
            assert fees[-1] > fees[0], "Tổng phí không tăng khi chi phí giao dịch tăng!"


def test_audit_json_metadata() -> None:
    """Kiểm toán file metadata wp43_44_backtest_audit.json."""
    json_path = RESULTS_DIR / "wp43_44_backtest_audit.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_strategies_simulated"] == 114
    assert "top_10_strategies_by_sharpe" in data
    assert len(data["top_10_strategies_by_sharpe"]) == 10
    assert "benchmarks" in data
    assert "cost_sensitivity_summary" in data

