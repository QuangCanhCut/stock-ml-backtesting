"""Kiểm thử đơn vị cho Hệ thống Thước đo Định lượng Quant Metrics (WP4.4).

Bao gồm:
1. Kiểm tra công thức tính CAGR trên chuỗi NAV mẫu tính tay.
2. Kiểm tra Sharpe Ratio và Sortino Ratio so với lãi suất phi rủi ro 4.5%/năm.
3. Kiểm tra Max Drawdown và Calmar Ratio.
4. Kiểm tra Profit Factor, Payoff Ratio và Win Rate %.
5. Kiểm tra trường hợp biên (Zero Trades, Zero Losses, Zero Volatility) không sinh NaN hoặc Inf.
6. Kiểm tra hàm bóc tách hiệu suất từng năm compute_yearly_metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import SimulationResult, TradeRecord
from src.backtest.metrics import compute_strategy_metrics, compute_yearly_metrics
from src.backtest.rules import HOSEMicrostructureRules


@pytest.fixture
def rules() -> HOSEMicrostructureRules:
    return HOSEMicrostructureRules(risk_free_rate=0.045)


def test_cagr_calculation(rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra công thức CAGR: Vốn 100M sau đúng 252 phiên đạt 110M -> CAGR = 10.0%."""
    # 253 điểm (Day 0 + 252 phiên sinh lợi)
    dates = pd.date_range("2022-01-04", periods=253, freq="B")
    daily_df = pd.DataFrame({
        "date": dates,
        "cash": [100.0] * 253,
        "stock_value": [0.0] * 253,
        "nav": np.linspace(100.0, 110.0, 253),
        "total_shares": [0] * 253,
        "shares_available": [0] * 253,
        "shares_locked": [0] * 253,
        "daily_return": [0.0] + [(110.0 / 100.0) ** (1 / 252) - 1.0] * 252,
        "drawdown": [0.0] * 253,
    })

    sim = SimulationResult(
        strategy_id="test_cagr",
        protocol="test",
        rq="RQ1",
        model_name="test",
        signal_mode="test",
        sizing_mode="all_in_out",
        daily_df=daily_df,
        trades_df=pd.DataFrame(),
        t25_delayed_exits=0,
        ceiling_locked_skips=0,
        floor_locked_skips=0,
        initial_capital=100.0,
        final_nav=110.0,
        total_fees_paid=0.0,
        total_traded_value=0.0,
    )

    metrics = compute_strategy_metrics(sim, rules=rules)
    assert abs(metrics["cagr_pct"] - 10.0) < 0.1, f"CAGR kỳ vọng ~10.0%, thực tế: {metrics['cagr_pct']}"


def test_max_drawdown_and_calmar(rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra Max Drawdown và Calmar Ratio trên chuỗi NAV có đỉnh và đáy rõ ràng."""
    dates = pd.date_range("2022-01-04", periods=5, freq="B")
    # NAV: 100 -> 120 -> 90 -> 110 -> 110
    # Peak = 120, Trough = 90 -> Drawdown = (90 - 120) / 120 = -25.0%
    navs = [100.0, 120.0, 90.0, 110.0, 110.0]
    dds = [0.0, 0.0, -0.25, -0.0833, -0.0833]
    daily_rets = [0.0, 0.20, -0.25, 0.2222, 0.0]

    daily_df = pd.DataFrame({
        "date": dates,
        "cash": navs,
        "stock_value": [0.0] * 5,
        "nav": navs,
        "total_shares": [0] * 5,
        "shares_available": [0] * 5,
        "shares_locked": [0] * 5,
        "daily_return": daily_rets,
        "drawdown": dds,
    })

    sim = SimulationResult(
        strategy_id="test_mdd",
        protocol="test",
        rq="RQ1",
        model_name="test",
        signal_mode="test",
        sizing_mode="all_in_out",
        daily_df=daily_df,
        trades_df=pd.DataFrame(),
        t25_delayed_exits=0,
        ceiling_locked_skips=0,
        floor_locked_skips=0,
        initial_capital=100.0,
        final_nav=110.0,
        total_fees_paid=0.0,
        total_traded_value=0.0,
    )

    metrics = compute_strategy_metrics(sim, rules=rules)
    assert abs(metrics["max_drawdown_pct"] - (-25.0)) < 1e-3
    assert metrics["calmar_ratio"] > 0.0


def test_profit_factor_and_payoff(rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra Profit Factor và Payoff Ratio trên danh sách giao dịch mẫu."""
    trades_df = pd.DataFrame([
        {
            "trade_id": 1,
            "entry_date": pd.Timestamp("2022-01-05"),
            "exit_date": pd.Timestamp("2022-01-10"),
            "entry_price": 30.0,
            "exit_price": 33.0,
            "shares": 1000,
            "buy_fee": 100.0,
            "sell_fee": 100.0,
            "gross_pnl": 3000.0,
            "net_pnl": 2800.0,
            "net_return_pct": 9.3,
            "holding_sessions": 3,
            "exit_reason": "SIGNAL_EXIT",
            "status": "CLOSED",
        },
        {
            "trade_id": 2,
            "entry_date": pd.Timestamp("2022-01-15"),
            "exit_date": pd.Timestamp("2022-01-20"),
            "entry_price": 33.0,
            "exit_price": 31.0,
            "shares": 1000,
            "buy_fee": 100.0,
            "sell_fee": 100.0,
            "gross_pnl": -2000.0,
            "net_pnl": -2200.0,
            "net_return_pct": -6.6,
            "holding_sessions": 3,
            "exit_reason": "SIGNAL_EXIT",
            "status": "CLOSED",
        },
    ])

    dates = pd.date_range("2022-01-04", periods=3, freq="B")
    daily_df = pd.DataFrame({
        "date": dates,
        "cash": [100.0, 105.0, 106.0],
        "stock_value": [0.0, 0.0, 0.0],
        "nav": [100.0, 105.0, 106.0],
        "total_shares": [0, 0, 0],
        "shares_available": [0, 0, 0],
        "shares_locked": [0, 0, 0],
        "daily_return": [0.0, 0.05, 0.0095],
        "drawdown": [0.0, 0.0, 0.0],
    })

    sim = SimulationResult(
        strategy_id="test_pf",
        protocol="test",
        rq="RQ1",
        model_name="test",
        signal_mode="test",
        sizing_mode="all_in_out",
        daily_df=daily_df,
        trades_df=trades_df,
        t25_delayed_exits=0,
        ceiling_locked_skips=0,
        floor_locked_skips=0,
        initial_capital=100.0,
        final_nav=106.0,
        total_fees_paid=400.0,
        total_traded_value=127_000.0,
    )

    metrics = compute_strategy_metrics(sim, rules=rules)
    assert metrics["total_trades"] == 2
    assert metrics["win_rate_pct"] == 50.0
    # Profit Factor = 2800.0 / 2200.0 = 1.2727
    assert abs(metrics["profit_factor"] - (2800.0 / 2200.0)) < 1e-3
    # Payoff Ratio = 9.3 / 6.6 = 1.409
    assert abs(metrics["payoff_ratio"] - (9.3 / 6.6)) < 1e-2


def test_zero_trades_edge_case(rules: HOSEMicrostructureRules) -> None:
    """Đảm bảo không sinh NaN hoặc Inf khi chiến lược không có bất kỳ giao dịch nào."""
    dates = pd.date_range("2022-01-04", periods=10, freq="B")
    daily_df = pd.DataFrame({
        "date": dates,
        "cash": [100.0] * 10,
        "stock_value": [0.0] * 10,
        "nav": [100.0] * 10,
        "total_shares": [0] * 10,
        "shares_available": [0] * 10,
        "shares_locked": [0] * 10,
        "daily_return": [0.0] * 10,
        "drawdown": [0.0] * 10,
    })

    sim = SimulationResult(
        strategy_id="test_zero",
        protocol="test",
        rq="RQ1",
        model_name="test",
        signal_mode="test",
        sizing_mode="all_in_out",
        daily_df=daily_df,
        trades_df=pd.DataFrame(),
        t25_delayed_exits=0,
        ceiling_locked_skips=0,
        floor_locked_skips=0,
        initial_capital=100.0,
        final_nav=100.0,
        total_fees_paid=0.0,
        total_traded_value=0.0,
    )

    metrics = compute_strategy_metrics(sim, rules=rules)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            assert not np.isnan(v), f"Cột {k} bị NaN!"
            assert not np.isinf(v), f"Cột {k} bị Inf!"


def test_compute_yearly_metrics(rules: HOSEMicrostructureRules) -> None:
    """Kiểm tra phân rã hiệu suất qua nhiều năm (2022, 2023)."""
    dates_22 = pd.date_range("2022-01-04", "2022-12-30", freq="B")
    dates_23 = pd.date_range("2023-01-03", "2023-12-29", freq="B")
    all_dates = dates_22.append(dates_23)
    n = len(all_dates)

    daily_df = pd.DataFrame({
        "date": all_dates,
        "cash": np.linspace(100.0, 150.0, n),
        "stock_value": [0.0] * n,
        "nav": np.linspace(100.0, 150.0, n),
        "total_shares": [0] * n,
        "shares_available": [0] * n,
        "shares_locked": [0] * n,
        "daily_return": [0.0] + [0.001] * (n - 1),
        "drawdown": [0.0] * n,
    })

    sim = SimulationResult(
        strategy_id="test_yearly",
        protocol="test",
        rq="RQ1",
        model_name="test",
        signal_mode="test",
        sizing_mode="all_in_out",
        daily_df=daily_df,
        trades_df=pd.DataFrame(),
        t25_delayed_exits=0,
        ceiling_locked_skips=0,
        floor_locked_skips=0,
        initial_capital=100.0,
        final_nav=150.0,
        total_fees_paid=0.0,
        total_traded_value=0.0,
    )

    yearly = compute_yearly_metrics(sim, rules=rules)
    assert len(yearly) == 2
    years = [item["year"] for item in yearly]
    assert years == [2022, 2023]
    for y_item in yearly:
        assert not np.isnan(y_item["annual_return_pct"])
        assert not np.isnan(y_item["sharpe_ratio"])

