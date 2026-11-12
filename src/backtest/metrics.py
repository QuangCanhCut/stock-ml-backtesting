"""Hệ thống Thước đo Định lượng (Quant Metrics), Đối chuẩn (Benchmarks) và Stress Test (WP4.4).

Module này hiện thực hóa các chức năng:
- Tính toán đầy đủ các chỉ số hiệu suất danh mục tài chính chuẩn mực quốc tế:
  + Tăng trưởng: Final NAV, Total Return %, CAGR %.
  + Rủi ro & Hiệu chỉnh rủi ro: Annualized Volatility %, Sharpe Ratio, Sortino Ratio, Max Drawdown %, Calmar Ratio.
  + Thống kê vi cấu trúc: Win Rate %, Profit Factor, Payoff Ratio, Holding Sessions, Time in Market %, Annual Turnover, Phí thuế đã trả.
  + So sánh trực chuẩn với Buy & Hold HPG: Excess CAGR %, Excess Sharpe, MDD Reduction %.
- Bóc tách hiệu suất từng năm chu kỳ (Yearly Breakdown 2022–2026).
- Kiểm tra độ nhạy chi phí giao dịch (Cost Sensitivity / Friction Stress Test) qua 4 kịch bản ma sát:
  0.30%, 0.50% (chuẩn HOSE), 0.75%, 1.00%.
"""

from typing import Any
import numpy as np
import pandas as pd

from src.backtest.engine import SimulationResult, simulate_strategy
from src.backtest.rules import HOSEMicrostructureRules


def compute_strategy_metrics(
    sim_result: SimulationResult,
    buy_and_hold_result: SimulationResult | None = None,
    rules: HOSEMicrostructureRules | None = None,
) -> dict[str, Any]:
    """Tính toán toàn bộ các thước đo định lượng cho một lượt mô phỏng chiến lược.

    Args:
        sim_result: Kết quả mô phỏng trả về từ engine.
        buy_and_hold_result: Kết quả mô phỏng của chiến lược Buy & Hold để so sánh đối chuẩn.
        rules: Bộ luật vi cấu trúc sàn HOSE (để lấy lãi suất phi rủi ro).

    Returns:
        Từ điển chứa đầy đủ các chỉ số hiệu suất, rủi ro và đối chuẩn.
    """
    initial_cap = sim_result.initial_capital
    final_nav = sim_result.final_nav
    total_ret_pct = (final_nav / initial_cap - 1.0) * 100.0

    daily_df = sim_result.daily_df
    # Bỏ qua ngày Day 0 khi tính toán chuỗi tỷ suất sinh lời hằng ngày
    daily_returns = daily_df["daily_return"].iloc[1:].to_numpy(dtype=float)
    n_days = len(daily_returns)
    years = max(n_days / 252.0, 1e-4)

    # CAGR %
    if final_nav > 0.0:
        cagr_pct = ((final_nav / initial_cap) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = -100.0

    # Biến động thường niên hóa (Annualized Volatility %)
    vol_daily = float(np.std(daily_returns, ddof=1)) if n_days > 1 else 0.0
    ann_vol_pct = float(vol_daily * np.sqrt(252.0) * 100.0)

    # Lãi suất phi rủi ro hằng ngày
    rf_rate = rules.risk_free_rate if rules is not None else 0.045
    daily_rf = (1.0 + rf_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - daily_rf
    mean_excess = float(np.mean(excess_returns)) if n_days > 0 else 0.0

    # Sharpe Ratio
    if vol_daily > 1e-12:
        sharpe_ratio = float((mean_excess / vol_daily) * np.sqrt(252.0))
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (chỉ xét độ lệch chuẩn của phần lợi nhuận âm dưới lãi suất phi rủi ro)
    downside_excess = np.where(excess_returns < 0.0, excess_returns, 0.0)
    downside_var = np.mean(downside_excess ** 2) if n_days > 0 else 0.0
    downside_std = float(np.sqrt(downside_var))
    if downside_std > 1e-12:
        sortino_ratio = float((mean_excess / downside_std) * np.sqrt(252.0))
    else:
        sortino_ratio = 0.0

    # Max Drawdown % (luôn <= 0.0)
    max_dd_pct = float(daily_df["drawdown"].min() * 100.0)

    # Calmar Ratio (CAGR / |Max Drawdown|)
    abs_mdd = abs(daily_df["drawdown"].min())
    if abs_mdd > 1e-6:
        calmar_ratio = float((cagr_pct / 100.0) / abs_mdd)
    else:
        calmar_ratio = 0.0

    # Thống kê giao dịch
    trades_df = sim_result.trades_df
    total_trades = len(trades_df)

    if total_trades > 0:
        win_mask = trades_df["net_pnl"] > 0
        loss_mask = trades_df["net_pnl"] < 0
        win_rate_pct = float(win_mask.mean() * 100.0)

        pos_pnl = float(trades_df.loc[win_mask, "net_pnl"].sum())
        neg_pnl = float(abs(trades_df.loc[loss_mask, "net_pnl"].sum()))

        if neg_pnl > 1e-6:
            profit_factor = float(pos_pnl / neg_pnl)
        else:
            profit_factor = 999.0 if pos_pnl > 0 else 0.0

        win_rets = trades_df.loc[win_mask, "net_return_pct"]
        loss_rets = trades_df.loc[loss_mask, "net_return_pct"]
        if len(win_rets) > 0 and len(loss_rets) > 0:
            avg_win = float(win_rets.mean())
            avg_loss = float(abs(loss_rets.mean()))
            payoff_ratio = float(avg_win / avg_loss) if avg_loss > 1e-6 else 999.0
        else:
            payoff_ratio = 0.0

        avg_holding = float(trades_df["holding_sessions"].mean())
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0
        payoff_ratio = 0.0
        avg_holding = 0.0

    # Tỷ lệ thời gian nắm giữ cổ phiếu (Time in Market %)
    market_sessions = daily_df["total_shares"].iloc[1:] > 0
    time_in_market_pct = float(market_sessions.mean() * 100.0) if n_days > 0 else 0.0

    # Vòng quay hàng năm (Annual Turnover)
    mean_nav = float(daily_df["nav"].mean())
    total_traded = sim_result.total_traded_value
    if mean_nav > 0.0 and years > 0.0:
        annual_turnover = float((total_traded / mean_nav) / years)
    else:
        annual_turnover = 0.0

    # So sánh với Buy & Hold Benchmark
    if buy_and_hold_result is not None:
        bh_final = buy_and_hold_result.final_nav
        bh_cagr = ((bh_final / buy_and_hold_result.initial_capital) ** (1.0 / years) - 1.0) * 100.0
        bh_daily_ret = buy_and_hold_result.daily_df["daily_return"].iloc[1:].to_numpy(dtype=float)
        bh_vol = float(np.std(bh_daily_ret, ddof=1))
        bh_excess = bh_daily_ret - daily_rf
        bh_sharpe = float((np.mean(bh_excess) / bh_vol) * np.sqrt(252.0)) if bh_vol > 1e-12 else 0.0
        bh_mdd = float(buy_and_hold_result.daily_df["drawdown"].min() * 100.0)

        excess_cagr = float(cagr_pct - bh_cagr)
        excess_sharpe = float(sharpe_ratio - bh_sharpe)
        mdd_reduction = float(abs(bh_mdd) - abs(max_dd_pct))
    else:
        excess_cagr = 0.0
        excess_sharpe = 0.0
        mdd_reduction = 0.0

    return {
        "strategy_id": sim_result.strategy_id,
        "protocol": sim_result.protocol,
        "rq": sim_result.rq,
        "model_name": sim_result.model_name,
        "signal_mode": sim_result.signal_mode,
        "sizing_mode": sim_result.sizing_mode,
        "final_nav_vnd": round(final_nav, 2),
        "total_return_pct": round(total_ret_pct, 4),
        "cagr_pct": round(cagr_pct, 4),
        "annualized_volatility_pct": round(ann_vol_pct, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "calmar_ratio": round(calmar_ratio, 4),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 4),
        "payoff_ratio": round(payoff_ratio, 4),
        "avg_actual_holding_sessions": round(avg_holding, 2),
        "actual_time_in_market_pct": round(time_in_market_pct, 2),
        "annual_turnover": round(annual_turnover, 4),
        "total_fees_paid_vnd": round(sim_result.total_fees_paid, 2),
        "t25_delayed_exits": sim_result.t25_delayed_exits,
        "ceiling_locked_skips": sim_result.ceiling_locked_skips,
        "floor_locked_skips": sim_result.floor_locked_skips,
        "excess_cagr_vs_buyhold_pct": round(excess_cagr, 4),
        "excess_sharpe_vs_buyhold": round(excess_sharpe, 4),
        "mdd_reduction_vs_buyhold_pct": round(mdd_reduction, 4),
    }


def compute_yearly_metrics(
    sim_result: SimulationResult,
    rules: HOSEMicrostructureRules | None = None,
) -> list[dict[str, Any]]:
    """Phân tách hiệu suất danh mục theo từng năm chu kỳ (2022, 2023, 2024, 2025, 2026).

    Args:
        sim_result: Kết quả mô phỏng trả về từ engine.
        rules: Luật vi cấu trúc HOSE.

    Returns:
        Danh sách từ điển chứa chỉ số từng năm của chiến lược.
    """
    daily_df = sim_result.daily_df.copy()
    if not isinstance(daily_df["date"].dtype, pd.DatetimeIndex):
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    trades_df = sim_result.trades_df.copy()
    if not trades_df.empty and "exit_date" in trades_df.columns:
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

    rf_rate = rules.risk_free_rate if rules is not None else 0.045
    daily_rf = (1.0 + rf_rate) ** (1.0 / 252.0) - 1.0

    years = sorted(daily_df["date"].dt.year.unique())
    yearly_records: list[dict[str, Any]] = []

    for y in years:
        sub_y = daily_df[daily_df["date"].dt.year == y]
        if sub_y.empty:
            continue

        # Điểm mốc NAV đầu năm
        if y == years[0]:
            # Năm đầu tiên (2022): lấy NAV tại Day 0
            start_nav = float(daily_df.iloc[0]["nav"])
            y_rets = sub_y["daily_return"].iloc[1:].to_numpy(dtype=float)
            y_shares = sub_y["total_shares"].iloc[1:]
        else:
            # Các năm sau: lấy NAV phiên giao dịch cuối cùng của năm trước đó
            prior_df = daily_df[daily_df["date"].dt.year < y]
            start_nav = float(prior_df.iloc[-1]["nav"])
            y_rets = sub_y["daily_return"].to_numpy(dtype=float)
            y_shares = sub_y["total_shares"]

        end_nav = float(sub_y.iloc[-1]["nav"])
        ann_return_pct = ((end_nav / start_nav) - 1.0) * 100.0 if start_nav > 0 else 0.0

        n_y = len(y_rets)
        vol_y = float(np.std(y_rets, ddof=1)) if n_y > 1 else 0.0
        ann_vol_pct = float(vol_y * np.sqrt(252.0) * 100.0)

        excess_y = y_rets - daily_rf
        mean_excess_y = float(np.mean(excess_y)) if n_y > 0 else 0.0
        sharpe_y = float((mean_excess_y / vol_y) * np.sqrt(252.0)) if vol_y > 1e-12 else 0.0

        # Max Drawdown trong năm y tính từ đỉnh trong năm
        nav_series = np.insert(sub_y["nav"].to_numpy(dtype=float), 0, start_nav)
        peaks_y = np.maximum.accumulate(nav_series)
        dd_y = (nav_series / peaks_y - 1.0) * 100.0
        max_dd_y = float(np.min(dd_y))

        # Số giao dịch đóng trong năm y
        if not trades_df.empty and "exit_date" in trades_df.columns:
            trades_count = int((trades_df["exit_date"].dt.year == y).sum())
        else:
            trades_count = 0

        # Tỷ lệ thời gian nắm giữ cổ phiếu trong năm y
        market_time_pct = float((y_shares > 0).mean() * 100.0) if len(y_shares) > 0 else 0.0

        yearly_records.append({
            "strategy_id": sim_result.strategy_id,
            "protocol": sim_result.protocol,
            "rq": sim_result.rq,
            "model_name": sim_result.model_name,
            "signal_mode": sim_result.signal_mode,
            "sizing_mode": sim_result.sizing_mode,
            "year": int(y),
            "annual_return_pct": round(ann_return_pct, 4),
            "annual_volatility_pct": round(ann_vol_pct, 4),
            "sharpe_ratio": round(sharpe_y, 4),
            "max_drawdown_pct": round(max_dd_y, 4),
            "trades_count": trades_count,
            "actual_time_in_market_pct": round(market_time_pct, 2),
        })

    return yearly_records


def run_cost_sensitivity_stress_test(
    calendar_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    rules: HOSEMicrostructureRules,
) -> pd.DataFrame:
    """Chạy bài kiểm tra độ nhạy ma sát (Friction Stress Test) qua 4 kịch bản chi phí.

    Áp dụng cho các mô hình thuộc giao thức `annual_expanding_retrain` (svm, lightgbm,
    lstm, ensemble) trên cả RQ1 và RQ2, qua các chế độ ngưỡng và cả 2 chế độ position sizing.

    Các kịch bản chi phí vòng quay:
    1. 0.30% (mua 0.15%, bán 0.15%)
    2. 0.50% (mua 0.20%, bán 0.30%) [chuẩn HOSE]
    3. 0.75% (mua 0.30%, bán 0.45%)
    4. 1.00% (mua 0.40%, bán 0.60%)

    Args:
        calendar_df: Master Calendar.
        signals_df: Toàn bộ bảng tín hiệu test.
        rules: Luật vi cấu trúc HOSE.

    Returns:
        DataFrame chứa kết quả Stress Test của toàn bộ các cấu hình qua 4 kịch bản.
    """
    cost_scenarios = [
        {"round_trip": 0.0030, "buy": 0.0015, "sell": 0.0015, "label": 0.30, "buy_label": 0.15, "sell_label": 0.15},
        {"round_trip": 0.0050, "buy": 0.0020, "sell": 0.0030, "label": 0.50, "buy_label": 0.20, "sell_label": 0.30},
        {"round_trip": 0.0075, "buy": 0.0030, "sell": 0.0045, "label": 0.75, "buy_label": 0.30, "sell_label": 0.45},
        {"round_trip": 0.0100, "buy": 0.0040, "sell": 0.0060, "label": 1.00, "buy_label": 0.40, "sell_label": 0.60},
    ]

    # Lọc chuỗi tín hiệu thuộc giao thức annual_expanding_retrain và các mô hình machine learning
    target_models = {"svm", "lightgbm", "lstm", "ensemble"}
    sub_signals = signals_df[
        (signals_df["protocol"] == "annual_expanding_retrain") &
        (signals_df["model_name"].isin(target_models))
    ]

    grouped = sub_signals.groupby(["protocol", "rq", "model_name", "signal_mode"])
    results: list[dict[str, Any]] = []

    for (protocol, rq, model_name, signal_mode), group in grouped:
        for sizing_mode in ("all_in_out", "vol_targeting"):
            base_strategy_id = f"{protocol}_{rq}_{model_name}_{signal_mode}_{sizing_mode}"

            for sc in cost_scenarios:
                cost_label = sc["label"]
                sim = simulate_strategy(
                    calendar_df=calendar_df,
                    signals_df=group,
                    rules=rules,
                    strategy_id=f"{base_strategy_id}_cost{cost_label}",
                    protocol=str(protocol),
                    rq=str(rq),
                    model_name=str(model_name),
                    signal_mode=str(signal_mode),
                    sizing_mode=sizing_mode,
                    custom_buy_cost=sc["buy"],
                    custom_sell_cost=sc["sell"],
                )

                metrics = compute_strategy_metrics(sim, buy_and_hold_result=None, rules=rules)

                results.append({
                    "protocol": protocol,
                    "rq": rq,
                    "model_name": model_name,
                    "signal_mode": signal_mode,
                    "sizing_mode": sizing_mode,
                    "cost_scenario_pct": cost_label,
                    "buy_cost_pct": sc["buy_label"],
                    "sell_cost_pct": sc["sell_label"],
                    "final_nav_vnd": metrics["final_nav_vnd"],
                    "total_return_pct": metrics["total_return_pct"],
                    "cagr_pct": metrics["cagr_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "sortino_ratio": metrics["sortino_ratio"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "total_fees_paid_vnd": metrics["total_fees_paid_vnd"],
                    "annual_turnover": metrics["annual_turnover"],
                })

    return pd.DataFrame(results)

