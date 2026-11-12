"""Mô-đun Trực quan hóa Định lượng (300 DPI) và Phân tích Nhật ký Giao dịch (WP4.5).

Chức năng:
1. Xuất bộ 4 biểu đồ nghiên cứu chuẩn publication (300 DPI, font tiếng Việt/Anh chuẩn):
   - 01_equity_curves_vs_buyhold.png: Đường cong tài sản NAV so sánh với Buy & Hold.
   - 02_underwater_drawdown.png: Biểu đồ diện tích sụt giảm dưới đỉnh (Underwater Drawdown).
   - 03_yearly_regime_and_rq_comparison.png: So sánh lợi nhuận từng năm và RQ1 vs RQ2.
   - 04_cost_sensitivity_and_turnover.png: Độ nhạy ma sát thị trường và ảnh hưởng của vòng quay.
2. Trích xuất bảng dữ liệu các ca giao dịch mẫu (results/backtesting/trade_case_studies.csv).
"""

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")  # Backend không cần GUI
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Cấu hình font và phong cách đồ thị chuẩn Quantitative Research
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.autolayout": True,
})


def export_trade_case_studies(results_dir: Path) -> pd.DataFrame:
    """Trích xuất 20 giao dịch của chiến lược quán quân LSTM Retrain RQ2.

    Args:
        results_dir: Thư mục chứa kết quả backtest (results/backtesting/).

    Returns:
        DataFrame chứa 20 giao dịch chi tiết và lưu ra trade_case_studies.csv.
    """
    trades_path = results_dir / "all_trade_logs_2022_2026.csv"
    if not trades_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file nhật ký giao dịch: {trades_path}")

    df_trades = pd.read_csv(trades_path)
    target_id = "annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out"
    top_trades = df_trades[df_trades["strategy_id"] == target_id].copy().reset_index(drop=True)

    out_csv = results_dir / "trade_case_studies.csv"
    top_trades.to_csv(out_csv, index=False)
    return top_trades


def plot_equity_curves(daily_parquet_path: Path, output_path: Path) -> None:
    """Vẽ biểu đồ đường cong tăng trưởng tài sản NAV (01_equity_curves_vs_buyhold.png).

    So sánh 6 đường tiêu biểu:
    1. Buy & Hold HPG (Đối chuẩn thị trường)
    2. Risk-Free Cash (Đối chuẩn phi rủi ro 4.5%/năm)
    3. Top 1 ML: LSTM Retrain RQ2 Fixed All-in-Out (160.18M VNĐ)
    4. Top Risk-Managed: SVM Frozen RQ2 Q70/Q40 Vol-Targeting (131.65M VNĐ, MDD -15.01%)
    5. Top Ensemble: Ensemble Frozen RQ2 Q80/Q30 All-in-Out (141.20M VNĐ)
    6. Top RQ1: SVM Retrain RQ1 Fixed All-in-Out (124.34M VNĐ)
    """
    df_daily = pd.read_parquet(daily_parquet_path)
    if not isinstance(df_daily["date"].dtype, pd.DatetimeIndex):
        df_daily["date"] = pd.to_datetime(df_daily["date"])

    target_configs = {
        "annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out": {
            "label": "Top 1 ML: LSTM Retrain RQ2 All-in (+60.18%, CAGR 10.40%)",
            "color": "#1f77b4",
            "linewidth": 2.2,
            "linestyle": "-",
            "zorder": 6,
        },
        "frozen_2016_2021_RQ2_ensemble_quantile_q80_q30_all_in_out": {
            "label": "Top Ensemble: Ensemble Frozen RQ2 (+41.20%, CAGR 7.51%)",
            "color": "#9467bd",
            "linewidth": 1.7,
            "linestyle": "-",
            "zorder": 5,
        },
        "frozen_2016_2021_RQ2_svm_quantile_q70_q40_vol_targeting": {
            "label": "Top Risk-Managed: SVM Frozen Vol-Targeting (+31.65%, MDD -15.01%)",
            "color": "#2ca02c",
            "linewidth": 1.8,
            "linestyle": "-",
            "zorder": 4,
        },
        "annual_expanding_retrain_RQ1_svm_fixed_053_048_all_in_out": {
            "label": "Top RQ1: SVM Retrain RQ1 (+24.34%, CAGR 4.68%)",
            "color": "#ff7f0e",
            "linewidth": 1.5,
            "linestyle": "--",
            "zorder": 3,
        },
        "risk_free_cash": {
            "label": "Risk-Free Cash 4.5%/năm (+23.32%, CAGR 4.50%)",
            "color": "#333333",
            "linewidth": 1.4,
            "linestyle": ":",
            "zorder": 2,
        },
        "buy_and_hold_hpg": {
            "label": "Buy & Hold HPG (Thị trường) (-12.50%, MDD -67.83%)",
            "color": "#d62728",
            "linewidth": 1.8,
            "linestyle": "-.",
            "zorder": 1,
        },
    }

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Đánh dấu vùng khủng hoảng ngành thép năm 2022
    crisis_start = pd.Timestamp("2022-03-01")
    crisis_end = pd.Timestamp("2022-11-15")
    ax.axvspan(crisis_start, crisis_end, color="#fee2e2", alpha=0.5, label="Khủng hoảng ngành Thép 2022 (HPG -67.8%)")

    for strat_id, cfg in target_configs.items():
        sub = df_daily[df_daily["strategy_id"] == strat_id].sort_values("date")
        if not sub.empty:
            # Quy đổi NAV sang Triệu VNĐ
            nav_million = sub["nav"] / 1_000_000.0
            ax.plot(
                sub["date"],
                nav_million,
                label=cfg["label"],
                color=cfg["color"],
                linewidth=cfg["linewidth"],
                linestyle=cfg["linestyle"],
                zorder=cfg["zorder"],
            )

    ax.set_title("Đường cong Tăng trưởng Tài sản NAV (2022–2026): Machine Learning vs. Benchmarks", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian giao dịch", fontsize=11, labelpad=8)
    ax.set_ylabel("Giá trị Tài sản Ròng (Triệu VNĐ) — Vốn gốc 100M", fontsize=11, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_ylim(40, 185)
    ax.axhline(100.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)

    # Chú thích điểm cuối kỳ của Top 1 LSTM
    ax.annotate(
        "Top 1: 160.18M (+60.2%)",
        xy=(pd.Timestamp("2026-10-30"), 160.18),
        xytext=(pd.Timestamp("2026-01-01"), 172.0),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2),
        fontweight="bold",
        color="#1f77b4",
        fontsize=9.5,
    )
    # Chú thích đáy của Buy & Hold
    ax.annotate(
        "Đáy B&H: 32.17M (-67.8%)",
        xy=(pd.Timestamp("2022-11-15"), 35.0),
        xytext=(pd.Timestamp("2022-07-01"), 50.0),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontweight="bold",
        color="#d62728",
        fontsize=9,
    )

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_underwater_drawdown(daily_parquet_path: Path, output_path: Path) -> None:
    """Vẽ biểu đồ diện tích sụt giảm dưới đỉnh Underwater Drawdown (02_underwater_drawdown.png)."""
    df_daily = pd.read_parquet(daily_parquet_path)
    if not isinstance(df_daily["date"].dtype, pd.DatetimeIndex):
        df_daily["date"] = pd.to_datetime(df_daily["date"])

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    # 1. Buy & Hold HPG
    sub_bh = df_daily[df_daily["strategy_id"] == "buy_and_hold_hpg"].sort_values("date")
    # 2. Top 1 LSTM Retrain
    sub_lstm = df_daily[df_daily["strategy_id"] == "annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out"].sort_values("date")
    # 3. Top Risk-Managed SVM Vol-Targeting
    sub_svm = df_daily[df_daily["strategy_id"] == "frozen_2016_2021_RQ2_svm_quantile_q70_q40_vol_targeting"].sort_values("date")

    # Vẽ diện tích Drawdown %
    ax.fill_between(sub_bh["date"], sub_bh["drawdown"] * 100.0, 0, color="#d62728", alpha=0.35, label="Buy & Hold HPG (Max DD: -67.83%)")
    ax.plot(sub_bh["date"], sub_bh["drawdown"] * 100.0, color="#d62728", linewidth=1.2, linestyle="-.")

    ax.plot(sub_lstm["date"], sub_lstm["drawdown"] * 100.0, color="#1f77b4", linewidth=1.8, label="Top 1 ML: LSTM Retrain RQ2 (Max DD: -21.73%)")
    ax.plot(sub_svm["date"], sub_svm["drawdown"] * 100.0, color="#2ca02c", linewidth=1.6, linestyle="--", label="Top Risk-Managed: SVM Vol-Targeting (Max DD: -15.01%)")

    ax.set_title("Biểu đồ Sụt giảm Dưới Đỉnh (Underwater Drawdown Profile 2022–2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Thời gian", fontsize=11, labelpad=8)
    ax.set_ylabel("Mức sụt giảm từ đỉnh (%)", fontsize=11, labelpad=8)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_ylim(-75, 5)
    ax.axhline(0, color="#333333", linestyle="-", linewidth=0.8)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_regime_and_rq_comparison(
    yearly_csv_path: Path,
    summary_csv_path: Path,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ 2 panel so sánh theo từng năm và RQ1 vs RQ2 (03_yearly_regime_and_rq_comparison.png)."""
    df_yearly = pd.read_csv(yearly_csv_path)
    df_summary = pd.read_csv(summary_csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # --- PANEL A: Grouped Bar Chart So sánh Tỷ suất sinh lời từng năm ---
    target_strats = [
        ("buy_and_hold_hpg", "Buy & Hold HPG", "#d62728"),
        ("annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out", "LSTM Retrain RQ2", "#1f77b4"),
        ("frozen_2016_2021_RQ2_svm_quantile_q70_q40_vol_targeting", "SVM Vol-Targeting RQ2", "#2ca02c"),
        ("frozen_2016_2021_RQ2_ensemble_quantile_q80_q30_all_in_out", "Ensemble RQ2", "#9467bd"),
    ]

    years = [2022, 2023, 2024, 2025, 2026]
    n_strats = len(target_strats)
    x = np.arange(len(years))
    bar_width = 0.18

    for i, (strat_id, label, color) in enumerate(target_strats):
        sub = df_yearly[df_yearly["strategy_id"] == strat_id]
        ret_dict = dict(zip(sub["year"], sub["annual_return_pct"]))
        y_vals = [ret_dict.get(y, 0.0) for y in years]
        pos = x - (bar_width * (n_strats - 1) / 2) + i * bar_width
        ax1.bar(pos, y_vals, width=bar_width, label=label, color=color, alpha=0.9, edgecolor="#333333", linewidth=0.5)

    ax1.axhline(0, color="#333333", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, fontsize=10.5, fontweight="bold")
    ax1.set_title("Panel A: Lợi nhuận Hằng năm theo Chu kỳ Thị trường (%)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Tỷ suất sinh lời năm (%)", fontsize=10.5)
    ax1.set_xlabel("Năm giao dịch", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

    # --- PANEL B: Boxplot So sánh Phân phối Sharpe Ratio & CAGR giữa RQ1 và RQ2 ---
    # Lọc bỏ Dummy và Benchmarks để đối chứng thực chất các mô hình Machine Learning
    ml_models = df_summary[~df_summary["model_name"].isin(["dummy", "buy_and_hold_hpg", "risk_free_cash"])].copy()

    # Tạo biểu đồ phân phối Sharpe Ratio cho RQ1 vs RQ2 tách theo Protocol
    sns.boxplot(
        data=ml_models,
        x="rq",
        y="sharpe_ratio",
        hue="protocol",
        palette=["#aec7e8", "#1f77b4"],
        ax=ax2,
        width=0.55,
        boxprops=dict(alpha=0.85),
    )
    ax2.axhline(0, color="#d62728", linestyle="--", linewidth=1.0, label="Ngưỡng hòa vốn Sharpe = 0")
    ax2.axhline(
        -0.0649, color="#888888", linestyle=":", linewidth=1.0, label="Sharpe Buy & Hold (-0.065)"
    )

    ax2.set_title("Panel B: Phân phối Sharpe Ratio (RQ1 vs. RQ2 theo Protocol)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Câu hỏi Nghiên cứu (RQ1: 1-Day Log-Ret vs. RQ2: 3-Day T+2.5 Net)", fontsize=10.5)
    ax2.set_ylabel("Sharpe Ratio (R_f = 4.5%/năm)", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

    fig.suptitle("Động thái Hiệu suất theo Chu kỳ Kinh tế & Tác động của Biến Mục tiêu (2022–2026)", fontsize=14, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost_sensitivity_and_turnover(
    cost_csv_path: Path,
    summary_csv_path: Path,
    output_path: Path,
) -> None:
    """Vẽ biểu đồ 2 panel phân tích độ nhạy ma sát và vòng quay (04_cost_sensitivity_and_turnover.png)."""
    df_cost = pd.read_csv(cost_csv_path)
    df_summary = pd.read_csv(summary_csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # --- PANEL A: Đường cong suy giảm CAGR khi mức phí tăng từ 0.30% đến 1.00% ---
    target_cost_strats = [
        ("annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out", "LSTM Retrain RQ2 All-in", "#1f77b4", "o-"),
        ("annual_expanding_retrain_RQ2_lstm_fixed_053_048_vol_targeting", "LSTM Retrain RQ2 Vol-Targeting", "#3992d4", "s--"),
        ("annual_expanding_retrain_RQ1_svm_fixed_053_048_all_in_out", "SVM Retrain RQ1 All-in", "#ff7f0e", "^-"),
        ("annual_expanding_retrain_RQ2_svm_quantile_q80_q30_all_in_out", "SVM Retrain RQ2 Q80/Q30", "#2ca02c", "d-."),
        ("annual_expanding_retrain_RQ2_ensemble_quantile_q80_q30_all_in_out", "Ensemble Retrain RQ2", "#9467bd", "x-"),
    ]

    for strat_id, label, color, fmt in target_cost_strats:
        # Tách các thành phần từ strat_id
        parts = strat_id.split("_")
        protocol = "annual_expanding_retrain"
        rq = parts[3]
        model = parts[4]
        signal_mode = "_".join(parts[5:-2]) if "vol_targeting" in strat_id or "all_in_out" in strat_id else "_".join(parts[5:])
        sizing_mode = "vol_targeting" if "vol_targeting" in strat_id else "all_in_out"

        sub = df_cost[
            (df_cost["protocol"] == protocol) &
            (df_cost["rq"] == rq) &
            (df_cost["model_name"] == model) &
            (df_cost["signal_mode"] == signal_mode) &
            (df_cost["sizing_mode"] == sizing_mode)
        ].sort_values("cost_scenario_pct")

        if not sub.empty:
            ax1.plot(sub["cost_scenario_pct"], sub["cagr_pct"], fmt, label=label, color=color, linewidth=1.8, markersize=6)

    ax1.axhline(0, color="#333333", linestyle="-", linewidth=0.8)
    ax1.axhline(-2.76, color="#d62728", linestyle=":", linewidth=1.2, label="Buy & Hold HPG (-2.76%)")
    ax1.axvline(0.50, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, label="Phí chuẩn HOSE (0.50%)")

    ax1.set_title("Panel A: Độ bền vững Alpha qua 4 Kịch bản Ma sát Phí", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Tổng chi phí vòng quay (Mua + Bán + Thuế + Trượt giá, %)", fontsize=10.5)
    ax1.set_ylabel("Tỷ suất sinh lời năm CAGR (%)", fontsize=10.5)
    ax1.set_xticks([0.30, 0.50, 0.75, 1.00])
    ax1.set_xticklabels(["0.30% (Ưu đãi)", "0.50% (Chuẩn)", "0.75% (Trượt cao)", "1.00% (Stress)"])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

    # --- PANEL B: Scatter Plot Vòng quay danh mục vs. CAGR ---
    ml_only = df_summary[~df_summary["model_name"].isin(["dummy", "buy_and_hold_hpg", "risk_free_cash"])].copy()

    # Tô màu theo RQ
    palette = {"RQ1": "#ff7f0e", "RQ2": "#1f77b4"}
    sns.scatterplot(
        data=ml_only,
        x="annual_turnover",
        y="cagr_pct",
        hue="rq",
        style="sizing_mode",
        palette=palette,
        s=70,
        alpha=0.85,
        ax=ax2,
    )

    # Đường xu hướng tuyến tính minh họa chi phí ma sát
    sns.regplot(
        data=ml_only,
        x="annual_turnover",
        y="cagr_pct",
        scatter=False,
        ax=ax2,
        color="#888888",
        line_kws={"linestyle": "--", "linewidth": 1.2},
    )

    ax2.axhline(0, color="#333333", linestyle="-", linewidth=0.8)
    ax2.set_title("Panel B: Mối quan hệ giữa Vòng quay Hằng năm & Lợi nhuận", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Vòng quay danh mục hàng năm (Annual Turnover = Traded Value / NAV / Years)", fontsize=10.5)
    ax2.set_ylabel("Tỷ suất sinh lời năm CAGR (%)", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.8)

    fig.suptitle("Phân tích Ma sát Vi cấu trúc Sàn HOSE & Tác động của Chi phí Giao dịch", fontsize=14, fontweight="bold", y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(results_dir: Path) -> dict[str, Path]:
    """Sinh toàn bộ 4 biểu đồ độ phân giải cao (300 DPI) vào results/backtesting/figures/.

    Args:
        results_dir: Thư mục chứa kết quả backtest (results/backtesting/).

    Returns:
        Từ điển đường dẫn các file ảnh đã xuất.
    """
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily_parquet = results_dir / "all_daily_portfolios_2022_2026.parquet"
    yearly_csv = results_dir / "yearly_strategy_performance.csv"
    cost_csv = results_dir / "cost_sensitivity.csv"
    summary_csv = results_dir / "strategy_comparison_2022_2026.csv"

    out_paths = {
        "fig1": fig_dir / "01_equity_curves_vs_buyhold.png",
        "fig2": fig_dir / "02_underwater_drawdown.png",
        "fig3": fig_dir / "03_yearly_regime_and_rq_comparison.png",
        "fig4": fig_dir / "04_cost_sensitivity_and_turnover.png",
    }

    plot_equity_curves(daily_parquet, out_paths["fig1"])
    plot_underwater_drawdown(daily_parquet, out_paths["fig2"])
    plot_yearly_regime_and_rq_comparison(yearly_csv, summary_csv, out_paths["fig3"])
    plot_cost_sensitivity_and_turnover(cost_csv, summary_csv, out_paths["fig4"])

    return out_paths

