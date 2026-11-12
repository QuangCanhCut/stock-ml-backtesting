"""Kiểm thử đơn vị cho Động cơ khớp lệnh Event-Driven sàn HOSE (WP4.3).

Bao gồm:
1. Bất biến kế toán: NAV_t = Cash_t + Total_Shares_t * Close_t tại mọi phiên.
2. Kiểm tra khóa thanh toán T+2.5: Mua Open(1) -> Bán Open(4), từ chối Open(2) và Open(3).
3. Kiểm tra kẹt trần (+6.7% từ chối mua) và kẹt sàn (-6.7% từ chối bán).
4. Kiểm tra quản trị vị thế Volatility Targeting (sigma_target = 0.25).
5. Kiểm tra quy tắc lô chẵn 100 cổ phiếu sàn HOSE.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    PositionLot,
    SimulationResult,
    simulate_buy_and_hold,
    simulate_risk_free_cash,
    simulate_strategy,
)
from src.backtest.rules import HOSEMicrostructureRules


@pytest.fixture
def dummy_calendar() -> pd.DataFrame:
    """Tạo Master Calendar giả lập gồm 10 phiên giao dịch từ 2022-01-04 đến 2022-01-17."""
    dates = pd.date_range("2022-01-04", periods=10, freq="B")
    data = {
        "open": [30.0, 30.0, 31.0, 30.5, 32.0, 31.5, 33.0, 32.5, 34.0, 33.5],
        "high": [30.5, 31.5, 31.5, 31.0, 32.5, 32.0, 33.5, 33.0, 34.5, 34.0],
        "low": [29.5, 29.5, 30.0, 30.0, 31.0, 31.0, 32.0, 32.0, 33.0, 33.0],
        "close": [30.0, 31.0, 30.5, 31.5, 31.5, 32.5, 32.5, 33.5, 33.5, 34.0],
        "volume": [1_000_000] * 10,
        "volatility_21": [0.25] * 10,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def rules() -> HOSEMicrostructureRules:
    return HOSEMicrostructureRules(
        initial_capital=100_000_000.0,
        commission_buy=0.0015,
        commission_sell=0.0015,
        tax_sell=0.0010,
        slippage=0.0005,
    )


def test_accounting_invariant_all_in_out(dummy_calendar: pd.DataFrame, rules: HOSEMicrostructureRules) -> None:
    """Khẳng định tại mọi phiên d: abs(NAV_d - (Cash_d + Total_Shares_d * Close_d)) < 1e-4."""
    dates = dummy_calendar.index
    # Mua tại Open phiên 1 (signal Day 0 = 1), giữ đến Open phiên 6 rồi bán (signal Day 5 = 0)
    sig_dates = dates[:8]
    exec_dates = dates[1:9]
    positions = [1, 1, 1, 1, 1, 0, 0, 0]

    signals_df = pd.DataFrame({
        "signal_date": sig_dates,
        "execution_date": exec_dates,
        "desired_position": positions,
        "volatility_21": [0.25] * len(sig_dates),
    })

    sim = simulate_strategy(dummy_calendar, signals_df, rules, sizing_mode="all_in_out")

    for _, row in sim.daily_df.iterrows():
        expected_nav = row["cash"] + row["total_shares"] * dummy_calendar.loc[row["date"], "close"]
        assert abs(row["nav"] - expected_nav) < 1e-4, f"Sai lệch NAV tại {row['date']}"


def test_t25_settlement_lock(dummy_calendar: pd.DataFrame, rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra khóa T+2.5: Mua tại Open(1), phát tín hiệu Bán tại t=1 (muốn bán Open(2)).

    Hệ thống KHÔNG được bán tại Open(2) hay Open(3) (t25_delayed_exits > 0), mà phải hoãn
    và chỉ khớp Bán đúng tại Open(4).
    """
    dates = dummy_calendar.index
    # Signal Day 0 = 1 -> Khớp Mua tại Open(1) (index d=1)
    # Signal Day 1 = 0 -> Muốn Bán tại Open(2) (index d=2) -> Kẹt T+2.5!
    # Signal Day 2 = 0 -> Muốn Bán tại Open(3) (index d=3) -> Kẹt T+2.5!
    # Signal Day 3 = 0 -> Muốn Bán tại Open(4) (index d=4) -> Mở khóa bán thành công!
    signals_df = pd.DataFrame({
        "signal_date": [dates[0], dates[1], dates[2], dates[3]],
        "execution_date": [dates[1], dates[2], dates[3], dates[4]],
        "desired_position": [1, 0, 0, 0],
        "volatility_21": [0.25, 0.25, 0.25, 0.25],
    })

    sim = simulate_strategy(dummy_calendar, signals_df, rules, sizing_mode="all_in_out")

    # Xác nhận có trì hoãn do T+2.5
    assert sim.t25_delayed_exits >= 2, "Phải ghi nhận ít nhất 2 lần hoãn bán tại Open(2) và Open(3)"

    # Kiểm tra tại Open(2) và Open(3) cổ phiếu vẫn còn nguyên trong tài khoản
    daily = sim.daily_df.set_index("date")
    assert daily.loc[dates[1], "total_shares"] > 0
    assert daily.loc[dates[2], "total_shares"] > 0
    assert daily.loc[dates[3], "total_shares"] > 0
    # Tại Open(4) (index 4), đã bán xong -> total_shares về 0
    assert daily.loc[dates[4], "total_shares"] == 0

    # Kiểm tra trade log
    assert len(sim.trades_df) == 1
    trade = sim.trades_df.iloc[0]
    assert trade["entry_date"] == dates[1]
    assert trade["exit_date"] == dates[4]
    assert trade["holding_sessions"] == 3  # Giữ trọn 3 phiên: index 1, 2, 3 và bán tại đầu phiên 4
    assert trade["status"] == "CLOSED"


def test_ceiling_and_floor_lock_skips(rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra từ chối khớp lệnh khi kẹt trần (+6.8%) và kẹt sàn (-6.8%)."""
    dates = pd.date_range("2022-01-04", periods=6, freq="B")
    # Giá Open phiên 1 tăng +6.8% so với Close phiên 0 -> Kẹt trần, hủy mua
    # Giá Open phiên 2 bình thường -> Khớp mua bù
    # Giá Open phiên 5 giảm -6.8% so với Close phiên 4 -> Kẹt sàn, hủy bán
    calendar_df = pd.DataFrame(
        {
            "open": [30.0, 32.04, 30.0, 30.0, 30.0, 27.9],  # 32.04 / 30.0 = +6.8%, 27.9 / 30.0 = -7.0%
            "high": [30.0, 32.5, 30.5, 30.5, 30.5, 28.0],
            "low": [29.5, 31.5, 29.5, 29.5, 29.5, 27.0],
            "close": [30.0, 31.0, 30.0, 30.0, 30.0, 27.5],
            "volume": [1_000_000] * 6,
            "volatility_21": [0.25] * 6,
        },
        index=dates,
    )

    signals_df = pd.DataFrame({
        "signal_date": [dates[0], dates[1], dates[2], dates[3], dates[4]],
        "execution_date": [dates[1], dates[2], dates[3], dates[4], dates[5]],
        "desired_position": [1, 1, 1, 1, 0],
        "volatility_21": [0.25] * 5,
    })

    sim = simulate_strategy(calendar_df, signals_df, rules, sizing_mode="all_in_out")

    # Xác nhận đã skip trần và sàn
    assert sim.ceiling_locked_skips >= 1, "Phải ghi nhận ít nhất 1 lần từ chối mua do trần"
    assert sim.floor_locked_skips >= 1, "Phải ghi nhận ít nhất 1 lần từ chối bán do sàn"

    # Tại phiên 1 (kẹt trần), chưa mua được
    daily = sim.daily_df.set_index("date")
    assert daily.loc[dates[1], "total_shares"] == 0
    # Tại phiên 2 (hết trần), đã mua thành công
    assert daily.loc[dates[2], "total_shares"] > 0


def test_volatility_targeting_sizing(dummy_calendar: pd.DataFrame, rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra quản trị vị thế Volatility Targeting (sigma_target = 0.25).

    - vol = 0.50 -> w_target = 0.25 / 0.50 = 0.50 (giải ngân ~50% tiền).
    - vol = 0.10 -> w_target = clip(0.25 / 0.10, 0.10, 1.00) = 1.00 (trần 100%).
    - vol = 5.00 -> w_target = clip(0.25 / 5.00, 0.10, 1.00) = 0.10 (sàn 10%).
    """
    dates = dummy_calendar.index

    # Trường hợp 1: vol = 0.50
    sig_df_50 = pd.DataFrame({
        "signal_date": [dates[0]],
        "execution_date": [dates[1]],
        "desired_position": [1],
        "volatility_21": [0.50],
    })
    sim_50 = simulate_strategy(dummy_calendar, sig_df_50, rules, sizing_mode="vol_targeting")
    daily_50 = sim_50.daily_df.set_index("date")
    # Giá trị cổ phiếu tại phiên 1 phải xấp xỉ 50% NAV
    stock_val_50 = daily_50.loc[dates[1], "stock_value"]
    nav_50 = daily_50.loc[dates[1], "nav"]
    ratio_50 = stock_val_50 / nav_50
    assert 0.45 <= ratio_50 <= 0.55, f"Tỷ trọng vol 0.50 phải quanh 50%, thực tế: {ratio_50:.2f}"

    # Trường hợp 2: vol = 0.10 -> trần 100%
    sig_df_10 = pd.DataFrame({
        "signal_date": [dates[0]],
        "execution_date": [dates[1]],
        "desired_position": [1],
        "volatility_21": [0.10],
    })
    sim_10 = simulate_strategy(dummy_calendar, sig_df_10, rules, sizing_mode="vol_targeting")
    daily_10 = sim_10.daily_df.set_index("date")
    ratio_10 = daily_10.loc[dates[1], "stock_value"] / daily_10.loc[dates[1], "nav"]
    assert ratio_10 >= 0.95, f"Tỷ trọng vol 0.10 phải sát 100%, thực tế: {ratio_10:.2f}"

    # Trường hợp 3: vol = 5.0 -> sàn 10%
    sig_df_high = pd.DataFrame({
        "signal_date": [dates[0]],
        "execution_date": [dates[1]],
        "desired_position": [1],
        "volatility_21": [5.0],
    })
    sim_high = simulate_strategy(dummy_calendar, sig_df_high, rules, sizing_mode="vol_targeting")
    daily_high = sim_high.daily_df.set_index("date")
    ratio_high = daily_high.loc[dates[1], "stock_value"] / daily_high.loc[dates[1], "nav"]
    assert 0.08 <= ratio_high <= 0.12, f"Tỷ trọng vol 5.0 phải sát sàn 10%, thực tế: {ratio_high:.2f}"


def test_lot_size_multiple_of_100(dummy_calendar: pd.DataFrame, rules: HOSEMicrostructureRules) -> None:
    """Khẳng định mọi lệnh mua đều tuân thủ quy chuẩn lô chẵn 100 cổ phiếu HOSE."""
    dates = dummy_calendar.index
    signals_df = pd.DataFrame({
        "signal_date": [dates[0]],
        "execution_date": [dates[1]],
        "desired_position": [1],
        "volatility_21": [0.33],
    })

    sim = simulate_strategy(dummy_calendar, signals_df, rules, sizing_mode="all_in_out")
    daily = sim.daily_df.set_index("date")
    shares = daily.loc[dates[1], "total_shares"]
    assert shares > 0
    assert shares % 100 == 0, f"Khối lượng {shares} không phải là bội số của 100!"


def test_benchmarks_simulation(dummy_calendar: pd.DataFrame, rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra tính nhất quán của Buy & Hold và Risk-Free Cash Benchmarks."""
    sim_bh = simulate_buy_and_hold(dummy_calendar, rules)
    sim_cash = simulate_risk_free_cash(dummy_calendar, rules)

    assert sim_bh.strategy_id == "buy_and_hold_hpg"
    assert sim_cash.strategy_id == "risk_free_cash"

    # Buy & Hold: total_shares > 0 và lô chẵn 100
    assert sim_bh.daily_df.iloc[-1]["total_shares"] % 100 == 0
    assert len(sim_bh.trades_df) == 1
    assert sim_bh.trades_df.iloc[0]["status"] == "OPEN_MTM"

    # Risk-Free Cash: shares luôn bằng 0, NAV tăng đều đặn
    assert (sim_cash.daily_df["total_shares"] == 0).all()
    assert sim_cash.final_nav > sim_cash.initial_capital
    assert len(sim_cash.trades_df) == 0

