"""Kịch bản điều phối thực thi WP4.3 & WP4.4 (Phase 4).

Chức năng:
1. Xác minh mã băm SHA-256 bất biến của Dataset và Manifest.
2. Nạp Master Calendar và toàn bộ 56 chuỗi tín hiệu từ backtest_signals_2022_2026.parquet.
3. Mô phỏng khớp lệnh Event-Driven HOSE (T+2.5, Lô 100, Trần/Sàn +-6.7%) cho:
   - 2 Chiến lược đối chuẩn: buy_and_hold_hpg, risk_free_cash.
   - 112 Chiến lược ML (56 chuỗi tín hiệu x 2 chế độ Sizing: all_in_out, vol_targeting).
   - Tổng cộng: 114 chiến lược giao dịch.
4. Tính toán toàn bộ hệ thống thước đo định lượng (Sharpe, Sortino, Max Drawdown, Calmar, Turnover, Phí thuế).
5. Phân tách hiệu suất danh mục theo từng năm (2022–2026).
6. Chạy kiểm tra độ nhạy chi phí giao dịch (Friction Stress Test: 0.30%, 0.50%, 0.75%, 1.00%).
7. Xuất 6 artifacts chuẩn hóa vào results/backtesting/ và in báo cáo tóm tắt định lượng.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Đảm bảo import được module src khi chạy trực tiếp từ CLI
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.backtest.data import load_market_calendar, verify_research_artifacts
from src.backtest.engine import (
    SimulationResult,
    simulate_buy_and_hold,
    simulate_risk_free_cash,
    simulate_strategy,
)
from src.backtest.metrics import (
    compute_strategy_metrics,
    compute_yearly_metrics,
    run_cost_sensitivity_stress_test,
)
from src.backtest.rules import HOSEMicrostructureRules


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    results_dir = root / "results" / "backtesting"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KHỞI CHẠY ĐỘNG CƠ KHỚP LỆNH HOSE & HỆ THỐNG THƯỚC ĐO QUANT (WP4.3 & WP4.4)")
    print("=" * 80)

    # 1. Kiểm toán tính bất biến của artifacts Phase 2 & 3
    print("\n[Bước 1/6] Kiểm toán tính bất biến mã băm SHA-256 dữ liệu nguồn...")
    hashes = verify_research_artifacts(root)
    print(f"  ✓ Dataset SHA-256:  {hashes['dataset_sha256']}")
    print(f"  ✓ Manifest SHA-256: {hashes['manifest_sha256']}")

    # 2. Nạp Master Calendar và Signals
    print("\n[Bước 2/6] Nạp Master Calendar và chuỗi tín hiệu Backtest...")
    full_market_df = load_market_calendar(root)
    # Chuẩn hóa Master Test Calendar: 2022-01-04 đến 2026-10-30 (gồm đúng 1.201 phiên giao dịch)
    market_df = full_market_df.loc["2022-01-04":"2026-10-30"].copy()
    rules = HOSEMicrostructureRules()
    signals_path = results_dir / "backtest_signals_2022_2026.parquet"
    if not signals_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file tín hiệu tại {signals_path}!")

    signals_df = pd.read_parquet(signals_path)
    print(f"  ✓ Master Calendar: {len(market_df)} phiên ({market_df.index[0].date()} -> {market_df.index[-1].date()})")
    print(f"  ✓ Tín hiệu Backtest: {len(signals_df):,} dòng từ {signals_path.name}")

    # 3. Chạy mô phỏng 2 Benchmarks
    print("\n[Bước 3/6] Mô phỏng 2 Chiến lược Đối chuẩn (Benchmarks)...")
    sim_bh = simulate_buy_and_hold(market_df, rules)
    sim_cash = simulate_risk_free_cash(market_df, rules)

    all_simulations: list[SimulationResult] = [sim_bh, sim_cash]
    print(f"  ✓ Buy & Hold HPG: Final NAV = {sim_bh.final_nav:,.0f} VNĐ, Return = {((sim_bh.final_nav/sim_bh.initial_capital)-1)*100:.2f}%")
    print(f"  ✓ Risk-Free Cash: Final NAV = {sim_cash.final_nav:,.0f} VNĐ, Return = {((sim_cash.final_nav/sim_cash.initial_capital)-1)*100:.2f}%")

    # 4. Chạy mô phỏng 112 Chiến lược ML
    print("\n[Bước 4/6] Mô phỏng 112 cấu hình chiến lược Machine Learning...")
    grouped = signals_df.groupby(["protocol", "rq", "model_name", "signal_mode"])
    n_series = len(grouped)
    print(f"  Phát hiện {n_series} chuỗi tín hiệu x 2 sizing modes = {n_series * 2} ML strategies.")

    for (protocol, rq, model_name, signal_mode), group in grouped:
        for sizing_mode in ("all_in_out", "vol_targeting"):
            strategy_id = f"{protocol}_{rq}_{model_name}_{signal_mode}_{sizing_mode}"
            sim = simulate_strategy(
                calendar_df=market_df,
                signals_df=group,
                rules=rules,
                strategy_id=strategy_id,
                protocol=str(protocol),
                rq=str(rq),
                model_name=str(model_name),
                signal_mode=str(signal_mode),
                sizing_mode=sizing_mode,
            )
            all_simulations.append(sim)

    total_strategies = len(all_simulations)
    print(f"  ✓ Hoàn tất mô phỏng toàn diện {total_strategies} chiến lược (112 ML + 2 Benchmarks)!")

    # 5. Tính toán Thước đo Tổng thể và Bóc tách Từng năm
    print("\n[Bước 5/6] Tính toán Thước đo Toàn diện & Bóc tách Từng năm...")
    summary_metrics_list: list[dict] = []
    yearly_metrics_list: list[dict] = []
    all_daily_list: list[pd.DataFrame] = []
    all_trades_list: list[pd.DataFrame] = []

    for sim in all_simulations:
        # Tính metrics toàn kỳ (so sánh trực chuẩn với Buy & Hold HPG)
        m = compute_strategy_metrics(sim, buy_and_hold_result=sim_bh, rules=rules)
        summary_metrics_list.append(m)

        # Tính metrics bóc tách từng năm
        y_list = compute_yearly_metrics(sim, rules=rules)
        yearly_metrics_list.extend(y_list)

        # Thu thập chuỗi danh mục hằng ngày
        d_df = sim.daily_df.copy()
        d_df["strategy_id"] = sim.strategy_id
        d_df["protocol"] = sim.protocol
        d_df["rq"] = sim.rq
        d_df["model_name"] = sim.model_name
        d_df["signal_mode"] = sim.signal_mode
        d_df["sizing_mode"] = sim.sizing_mode
        all_daily_list.append(d_df)

        # Thu thập nhật ký giao dịch
        if not sim.trades_df.empty:
            t_df = sim.trades_df.copy()
            t_df["strategy_id"] = sim.strategy_id
            t_df["protocol"] = sim.protocol
            t_df["rq"] = sim.rq
            t_df["model_name"] = sim.model_name
            t_df["signal_mode"] = sim.signal_mode
            t_df["sizing_mode"] = sim.sizing_mode
            all_trades_list.append(t_df)

    # Tạo DataFrame bảng xếp hạng tổng hợp, sắp xếp giảm dần theo Sharpe Ratio
    summary_df = pd.DataFrame(summary_metrics_list).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    yearly_df = pd.DataFrame(yearly_metrics_list)
    all_daily_df = pd.concat(all_daily_list, ignore_index=True)
    all_trades_df = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()

    # 6. Chạy Kiểm tra Độ nhạy Ma sát (Friction Stress Test)
    print("\n[Bước 6/6] Thực thi Kiểm tra Độ nhạy Ma sát Vi cấu trúc (Friction Stress Test)...")
    cost_df = run_cost_sensitivity_stress_test(market_df, signals_df, rules)
    print(f"  ✓ Đã chạy Stress Test qua 4 mức phí cho {len(cost_df) // 4} cấu hình ML retrain = {len(cost_df)} kịch bản.")

    # Xuất các file artifacts vào results/backtesting/
    summary_csv_path = results_dir / "strategy_comparison_2022_2026.csv"
    yearly_csv_path = results_dir / "yearly_strategy_performance.csv"
    cost_csv_path = results_dir / "cost_sensitivity.csv"
    daily_parquet_path = results_dir / "all_daily_portfolios_2022_2026.parquet"
    trades_csv_path = results_dir / "all_trade_logs_2022_2026.csv"
    audit_json_path = results_dir / "wp43_44_backtest_audit.json"

    summary_df.to_csv(summary_csv_path, index=False)
    yearly_df.to_csv(yearly_csv_path, index=False)
    cost_df.to_csv(cost_csv_path, index=False)
    all_daily_df.to_parquet(daily_parquet_path, index=False)
    all_trades_df.to_csv(trades_csv_path, index=False)

    # Tạo Audit JSON
    top_10 = summary_df.head(10).to_dict(orient="records")
    bh_metric = summary_df[summary_df["strategy_id"] == "buy_and_hold_hpg"].iloc[0].to_dict()
    cash_metric = summary_df[summary_df["strategy_id"] == "risk_free_cash"].iloc[0].to_dict()

    audit_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 4 - Strategy Backtesting & Order Execution",
        "work_packages": ["WP4.3 - Event-Driven HOSE Execution Engine", "WP4.4 - Quant Metrics & Cost Sensitivity"],
        "dataset_sha256": hashes["dataset_sha256"],
        "manifest_sha256": hashes["manifest_sha256"],
        "total_strategies_simulated": total_strategies,
        "benchmarks": {
            "buy_and_hold_hpg": bh_metric,
            "risk_free_cash": cash_metric,
        },
        "top_10_strategies_by_sharpe": top_10,
        "cost_sensitivity_summary": {
            "total_scenarios_tested": len(cost_df),
            "cost_levels": [0.30, 0.50, 0.75, 1.00],
            "mean_sharpe_by_cost": cost_df.groupby("cost_scenario_pct")["sharpe_ratio"].mean().to_dict(),
            "mean_cagr_by_cost": cost_df.groupby("cost_scenario_pct")["cagr_pct"].mean().to_dict(),
        },
    }

    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("XUẤT THÀNH CÔNG 6 ARTIFACTS TẠI results/backtesting/:")
    print(f"  1. strategy_comparison_2022_2026.csv ({len(summary_df)} dòng)")
    print(f"  2. yearly_strategy_performance.csv ({len(yearly_df)} dòng)")
    print(f"  3. cost_sensitivity.csv ({len(cost_df)} dòng)")
    print(f"  4. all_daily_portfolios_2022_2026.parquet ({len(all_daily_df):,} dòng)")
    print(f"  5. all_trade_logs_2022_2026.csv ({len(all_trades_df):,} giao dịch)")
    print(f"  6. wp43_44_backtest_audit.json")
    print("=" * 80)

    # In tóm tắt Top 10 chiến lược
    print("\n" + "TOP 10 CHIẾN LƯỢC CÓ SHARPE RATIO CAO NHẤT (2022–2026):")
    cols_to_print = [
        "strategy_id", "cagr_pct", "annualized_volatility_pct",
        "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "total_trades",
    ]
    print(summary_df[cols_to_print].head(10).to_string(index=False))

    print("\n" + "ĐỐI CHUẨN SO VỚI BENCHMARK BUY & HOLD HPG:")
    bh_cols = ["strategy_id", "final_nav_vnd", "total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct"]
    print(summary_df[summary_df["strategy_id"].isin(["buy_and_hold_hpg", "risk_free_cash"])][bh_cols].to_string(index=False))

    print("\n" + "TÓM TẮT TRUNG BÌNH FRICTION STRESS TEST THEO MỨC PHÍ:")
    stress_grp = cost_df.groupby("cost_scenario_pct")[["cagr_pct", "sharpe_ratio", "total_fees_paid_vnd"]].mean()
    print(stress_grp.to_string())
    print("=" * 80)


if __name__ == "__main__":
    main()
