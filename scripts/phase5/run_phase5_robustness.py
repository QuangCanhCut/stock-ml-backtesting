"""Kịch bản thực thi Phase 5: Kiểm định độ bền vững bổ sung (Robustness Testing).

Chức năng:
1. Xác minh tính bất biến SHA-256 của Dataset nghiên cứu và Split Manifest.
2. Thực nghiệm 1A: Feature Group Ablation Study trên 3 Development Folds (2019–2021)
   qua 5 nhóm đặc trưng (Returns, Trend, Momentum, Volatility, Volume) cho 3 họ mô hình (SVM, LightGBM, LSTM)
   trên cả 2 bài toán nghiên cứu RQ1 và RQ2 (Tổng cộng 36 kịch bản).
   Lưu kết quả tại: results/robustness/feature_group_ablation_dev_cv.csv
3. Thực nghiệm 1B: Outlier-Trade Removal Stress Test trên dữ liệu kiểm thử thực tế (2022–2026)
   cho 4 chiến lược tiêu biểu (Top 1 LSTM All-in, Top 2 LSTM Vol-Targeting, Top Ensemble, Top Risk-Managed SVM)
   khi loại bỏ lần lượt Top 1, Top 2, Top 3 lệnh lãi lớn nhất.
   Lưu kết quả tại: results/robustness/outlier_trade_stress_test.csv
4. Xuất biểu đồ trực quan hóa tổng hợp Phase 5 chuẩn publication (300 DPI) tại:
   results/robustness/figures/phase5_robustness_summary.png
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Đảm bảo import được module src
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.data import verify_research_artifacts
from src.dataset_builder import load_project_config
from src.models.data import ModelingView, load_research_dataset, make_modeling_view
from src.models.evaluation import binary_metrics
from src.models.modeling_safety import get_development_splits
from src.models.svm_baseline import (
    ModelConfig as SVMConfig,
    evaluate_fold as evaluate_svm_fold,
)
from src.models.train_lightgbm import (
    LGBMModelConfig,
    evaluate_fold as evaluate_lgbm_fold,
)
from src.models.train_lstm import (
    LSTMModelConfig,
    evaluate_fold as evaluate_lstm_fold,
)

# Định nghĩa 5 nhóm đặc trưng nhân quả từ Phase 2 (Tổng cộng đúng 26 features)
FEATURE_GROUPS: dict[str, list[str]] = {
    "Returns": [
        "log_return", "return_lag_1", "return_lag_2", "return_lag_3",
        "return_lag_5", "return_lag_10", "return_lag_21",
    ],
    "Trend": [
        "sma_ratio_10", "sma_ratio_20", "sma_ratio_50", "sma_ratio_200",
        "ema_ratio_12", "ema_ratio_26", "macd_ratio",
    ],
    "Momentum": [
        "momentum_5", "momentum_10", "momentum_21", "rsi_14",
    ],
    "Volatility": [
        "volatility_21", "volatility_63", "atr_ratio_14",
        "bollinger_width_20", "bollinger_percent_b_20",
    ],
    "Volume": [
        "volume_zscore_20", "volume_change_1", "volume_ratio_20",
    ],
}

ALL_26_FEATURES = [f for grp in FEATURE_GROUPS.values() for f in grp]
assert len(ALL_26_FEATURES) == 26, f"Dự kiến 26 features, thực tế: {len(ALL_26_FEATURES)}"


def run_feature_group_ablation(root: Path, output_dir: Path) -> pd.DataFrame:
    """Thực nghiệm 1A: Feature Group Ablation trên 3 Development Folds (2019–2021)."""
    print("\n" + "=" * 80)
    print("THỰC NGHIỆM 1A: FEATURE GROUP ABLATION STUDY TRÊN DEVELOPMENT FOLDS (2019–2021)")
    print("=" * 80)

    config = load_project_config(root / "configs" / "config.yaml")
    dataset = load_research_dataset(root, config)

    # Đọc baseline chuẩn từ dev_cv_all_models_summary.csv nếu có để đối chiếu
    summary_path = root / "results" / "modeling" / "final_evaluation" / "dev_cv_all_models_summary.csv"
    baseline_lookup: dict[tuple[str, str], dict[str, float]] = {}
    if summary_path.is_file():
        summary_df = pd.read_csv(summary_path)
        for _, row in summary_df.iterrows():
            rq_val = str(row["rq"])
            fam_val = str(row["model_family"])
            baseline_lookup[(rq_val, fam_val)] = {
                "mean_bacc": float(row["mean_balanced_accuracy"]),
                "std_bacc": float(row["std_balanced_accuracy"]),
                "mean_auc": float(row["mean_roc_auc"]),
            }

    # Định nghĩa cấu hình thắng cuộc của từng RQ đã chốt ở Phase 3
    model_configs = {
        "RQ1": {
            "svm": SVMConfig("linear_c0.1_balanced", "linear", 0.1, "balanced"),
            "lightgbm": LGBMModelConfig(
                config_id="lgbm_d3_lr0.03_mc80_l5_none",
                model_name="lightgbm",
                max_depth=3,
                num_leaves=7,
                learning_rate=0.03,
                min_child_samples=80,
                reg_lambda=5.0,
                class_weight=None,
                n_estimators=300,
                random_state=42,
            ),
            "lstm": LSTMModelConfig(
                config_id="lstm_seq10_h32_lr0.0005_none",
                model_name="lstm",
                seq_len=10,
                hidden_size=32,
                num_layers=1,
                dropout=0.3,
                learning_rate=0.0005,
                weight_decay=0.001,
                class_weight=None,
                max_epochs=50,
                early_stopping_patience=10,
            ),
        },
        "RQ2": {
            "svm": SVMConfig("linear_c1_balanced", "linear", 1.0, "balanced"),
            "lightgbm": LGBMModelConfig(
                config_id="lgbm_d3_lr0.05_mc40_l1_balanced",
                model_name="lightgbm",
                max_depth=3,
                num_leaves=7,
                learning_rate=0.05,
                min_child_samples=40,
                reg_lambda=1.0,
                class_weight="balanced",
                n_estimators=300,
                random_state=42,
            ),
            "lstm": LSTMModelConfig(
                config_id="lstm_seq20_h16_lr0.001_balanced",
                model_name="lstm",
                seq_len=20,
                hidden_size=16,
                num_layers=1,
                dropout=0.2,
                learning_rate=0.001,
                weight_decay=0.001,
                class_weight="balanced",
                max_epochs=50,
                early_stopping_patience=10,
            ),
        },
    }

    # Danh sách 6 kịch bản ablation
    scenarios = [
        ("Full_26_Features", None),
        ("Drop_Returns", "Returns"),
        ("Drop_Trend", "Trend"),
        ("Drop_Momentum", "Momentum"),
        ("Drop_Volatility", "Volatility"),
        ("Drop_Volume", "Volume"),
    ]

    records = []

    for rq in ("RQ1", "RQ2"):
        print(f"\n>>> Đang đánh giá bài toán: {rq}...")
        folds = get_development_splits(root, rq)
        view_full = make_modeling_view(dataset, config, rq)

        for model_name in ("svm", "lightgbm", "lstm"):
            print(f"  + Mô hình: {model_name.upper()}...")
            m_cfg = model_configs[rq][model_name]

            # 1. Đo lường Baseline Full 26 features
            full_bacc = baseline_lookup.get((rq, model_name), {}).get("mean_bacc")
            full_std = baseline_lookup.get((rq, model_name), {}).get("std_bacc")
            full_auc = baseline_lookup.get((rq, model_name), {}).get("mean_auc")

            if full_bacc is None or full_auc is None:
                # Nếu chưa có trong cache, tính trực tiếp trên 3 folds
                bacc_list, auc_list = [], []
                for f in folds:
                    if model_name == "svm":
                        rec, _ = evaluate_svm_fold(view_full, rq, f, m_cfg)
                    elif model_name == "lightgbm":
                        rec, _, _, _ = evaluate_lgbm_fold(view_full, rq, f, m_cfg, compute_perm_importance=False)
                    else:
                        rec, _, _, _, _ = evaluate_lstm_fold(view_full, rq, f, m_cfg)
                    bacc_list.append(rec["balanced_accuracy"])
                    auc_list.append(rec["roc_auc"])
                full_bacc = float(np.mean(bacc_list))
                full_std = float(np.std(bacc_list, ddof=1)) if len(bacc_list) > 1 else 0.0
                full_auc = float(np.mean(auc_list))

            # Chạy qua 6 kịch bản
            for sc_name, dropped_grp in scenarios:
                if dropped_grp is None:
                    # Kịch bản Full Baseline
                    records.append({
                        "rq": rq,
                        "model_name": model_name,
                        "ablation_scenario": sc_name,
                        "dropped_group": "none",
                        "num_features": 26,
                        "mean_balanced_accuracy": full_bacc,
                        "std_balanced_accuracy": full_std,
                        "mean_roc_auc": full_auc,
                        "delta_bacc_vs_full": 0.0,
                        "delta_auc_vs_full": 0.0,
                    })
                    print(f"    - {sc_name:20s}: BACC={full_bacc:.4f} (std={full_std:.4f}), AUC={full_auc:.4f}, Delta=0.0000")
                else:
                    # Kịch bản loại bỏ 1 nhóm đặc trưng
                    drop_cols = set(FEATURE_GROUPS[dropped_grp])
                    sub_cols = [c for c in view_full.X.columns if c not in drop_cols]
                    sub_view = ModelingView(
                        X=view_full.X[sub_cols],
                        y=view_full.y,
                        metadata=view_full.metadata,
                    )
                    bacc_list, auc_list = [], []
                    for f in folds:
                        if model_name == "svm":
                            rec, _ = evaluate_svm_fold(sub_view, rq, f, m_cfg)
                        elif model_name == "lightgbm":
                            rec, _, _, _ = evaluate_lgbm_fold(sub_view, rq, f, m_cfg, compute_perm_importance=False)
                        else:
                            rec, _, _, _, _ = evaluate_lstm_fold(sub_view, rq, f, m_cfg)
                        bacc_list.append(rec["balanced_accuracy"])
                        auc_list.append(rec["roc_auc"])

                    sc_bacc = float(np.mean(bacc_list))
                    sc_std = float(np.std(bacc_list, ddof=1)) if len(bacc_list) > 1 else 0.0
                    sc_auc = float(np.mean(auc_list))
                    d_bacc = sc_bacc - full_bacc
                    d_auc = sc_auc - full_auc

                    records.append({
                        "rq": rq,
                        "model_name": model_name,
                        "ablation_scenario": sc_name,
                        "dropped_group": dropped_grp,
                        "num_features": len(sub_cols),
                        "mean_balanced_accuracy": sc_bacc,
                        "std_balanced_accuracy": sc_std,
                        "mean_roc_auc": sc_auc,
                        "delta_bacc_vs_full": d_bacc,
                        "delta_auc_vs_full": d_auc,
                    })
                    print(f"    - {sc_name:20s}: BACC={sc_bacc:.4f} (std={sc_std:.4f}), AUC={sc_auc:.4f}, Delta={d_bacc:+.4f}")

    ablation_df = pd.DataFrame(records)
    out_csv = output_dir / "feature_group_ablation_dev_cv.csv"
    ablation_df.to_csv(out_csv, index=False)
    print(f"\n  ✓ Đã lưu bảng Feature Group Ablation (36 kịch bản): {out_csv}")
    return ablation_df


def run_outlier_trade_stress_test(root: Path, output_dir: Path) -> pd.DataFrame:
    """Thực nghiệm 1B: Outlier-Trade Removal Stress Test trên Backtest 2022–2026."""
    print("\n" + "=" * 80)
    print("THỰC NGHIỆM 1B: OUTLIER-TRADE REMOVAL STRESS TEST TRÊN BACKTEST 2022–2026")
    print("=" * 80)

    trades_path = root / "results" / "backtesting" / "all_trade_logs_2022_2026.csv"
    summary_path = root / "results" / "backtesting" / "strategy_comparison_2022_2026.csv"

    trades_df = pd.read_csv(trades_path)
    summary_df = pd.read_csv(summary_path)

    target_strategies = [
        "annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out",
        "annual_expanding_retrain_RQ2_lstm_fixed_053_048_vol_targeting",
        "frozen_2016_2021_RQ2_ensemble_quantile_q80_q30_all_in_out",
        "frozen_2016_2021_RQ2_svm_quantile_q70_q40_vol_targeting",
    ]

    records = []
    initial_cap = 100_000_000.0
    n_days = 1201.0
    years = n_days / 252.0

    # Lấy CAGR đối chuẩn
    bh_row = summary_df[summary_df["strategy_id"] == "buy_and_hold_hpg"].iloc[0]
    bh_cagr = float(bh_row["cagr_pct"])  # -2.7645%

    for strat_id in target_strategies:
        st_row = summary_df[summary_df["strategy_id"] == strat_id].iloc[0]
        orig_final_nav = float(st_row["final_nav_vnd"])

        sub_trades = trades_df[trades_df["strategy_id"] == strat_id].copy()
        sub_trades_sorted = sub_trades.sort_values("net_return_pct", ascending=False)
        total_trades_count = len(sub_trades)

        print(f"\n Chiến lược: {strat_id} (Tổng số giao dịch: {total_trades_count})")

        scenarios = [
            ("All_Trades", 0),
            ("Excl_Top_1_Best_Trade", 1),
            ("Excl_Top_2_Best_Trades", 2),
            ("Excl_Top_3_Best_Trades", 3),
        ]

        for sc_name, k_remove in scenarios:
            if k_remove == 0:
                pnl_removed = 0.0
            else:
                pnl_removed = float(sub_trades_sorted.head(k_remove)["net_pnl"].sum())

            adj_final_nav = orig_final_nav - pnl_removed
            adj_tot_ret = ((adj_final_nav / initial_cap) - 1.0) * 100.0

            if adj_final_nav > 0:
                adj_cagr = ((adj_final_nav / initial_cap) ** (1.0 / years) - 1.0) * 100.0
            else:
                adj_cagr = -100.0

            beats_bh = bool(adj_cagr > -2.7645)
            beats_rf = bool(adj_cagr > 4.50)
            rem_trades = total_trades_count - k_remove

            records.append({
                "strategy_id": strat_id,
                "scenario": sc_name,
                "removed_trades_count": k_remove,
                "remaining_trades": rem_trades,
                "adjusted_final_nav_vnd": round(adj_final_nav),
                "adjusted_total_return_pct": round(adj_tot_ret, 2),
                "adjusted_cagr_pct": round(adj_cagr, 2),
                "beats_buy_and_hold": beats_bh,
                "beats_risk_free": beats_rf,
            })

            status_str = f"Beats B&H: {beats_bh!s:5s} | Beats RF: {beats_rf!s:5s}"
            print(f"  - {sc_name:23s} | Rem:{k_remove} | NAV: {adj_final_nav:,.0f} | Ret: {adj_tot_ret:+.2f}% | CAGR: {adj_cagr:+.2f}% | {status_str}")

    outlier_df = pd.DataFrame(records)
    out_csv = output_dir / "outlier_trade_stress_test.csv"
    outlier_df.to_csv(out_csv, index=False)
    print(f"\n  ✓ Đã lưu bảng Outlier-Trade Removal Stress Test (16 kịch bản): {out_csv}")
    return outlier_df


def plot_phase5_robustness_summary(
    ablation_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Xuất biểu đồ 2 panel chuẩn publication (300 DPI): phase5_robustness_summary.png."""
    print("\n" + "=" * 80)
    print("XUẤT BIỂU ĐỒ TỔNG HỢP PHASE 5 ROBUSTNESS (300 DPI)")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # -------------------------------------------------------------------------
    # PANEL A: Feature Group Ablation trên Dev CV (RQ2)
    # -------------------------------------------------------------------------
    ax_a = axes[0]
    rq2_ablation = ablation_df[
        (ablation_df["rq"] == "RQ2") & (ablation_df["ablation_scenario"] != "Full_26_Features")
    ].copy()

    # Nhóm đặc trưng theo thứ tự
    groups = ["Returns", "Trend", "Momentum", "Volatility", "Volume"]
    y_pos = np.arange(len(groups))
    bar_height = 0.25

    colors = {
        "svm": "#2b5c8f",      # Xanh lam đậm
        "lightgbm": "#2a9d8f", # Xanh ngọc
        "lstm": "#e76f51",     # Đỏ san hô / cam
    }
    model_labels = {
        "svm": "SVM (linear_c1_bal)",
        "lightgbm": "LightGBM (d3_lr0.05_bal)",
        "lstm": "PyTorch LSTM (seq20_h16_bal)",
    }

    for idx, m_name in enumerate(["svm", "lightgbm", "lstm"]):
        sub_m = rq2_ablation[rq2_ablation["model_name"] == m_name]
        # Lấy delta theo thứ tự nhóm
        delta_vals = [
            float(sub_m[sub_m["dropped_group"] == g]["delta_bacc_vs_full"].iloc[0]) * 100.0
            for g in groups
        ]
        offset = (idx - 1) * bar_height
        bars = ax_a.barh(
            y_pos + offset, delta_vals, height=bar_height,
            color=colors[m_name], label=model_labels[m_name],
            alpha=0.88, edgecolor="black", linewidth=0.6,
        )
        # Thêm nhãn giá trị
        for b, v in zip(bars, delta_vals):
            x_loc = b.get_width()
            ha = "left" if x_loc >= 0 else "right"
            pad = 0.05 if x_loc >= 0 else -0.05
            ax_a.text(
                x_loc + pad, b.get_y() + b.get_height() / 2,
                f"{v:+.2f}%", va="center", ha=ha, fontsize=8, fontweight="bold",
            )

    ax_a.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([f"Drop {g}" for g in groups], fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Độ suy giảm Balanced Accuracy so với Full 26 Features (Δ %)", fontsize=11, fontweight="bold")
    ax_a.set_title("Panel A: Đóng góp của Nhóm Đặc trưng trên Dev CV (RQ2 2019–2021)\n(Mức giảm càng sâu chứng minh nhóm đặc trưng càng quan trọng)", fontsize=12, fontweight="bold", pad=12)
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)
    ax_a.grid(True, linestyle="--", alpha=0.4, axis="x")
    ax_a.set_xlim(-3.5, 1.5)

    # -------------------------------------------------------------------------
    # PANEL B: Outlier-Trade Removal Stress Test
    # -------------------------------------------------------------------------
    ax_b = axes[1]

    strat_order = [
        ("annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out", "Top 1 LSTM (All-in)", "#e76f51"),
        ("annual_expanding_retrain_RQ2_lstm_fixed_053_048_vol_targeting", "Top 2 LSTM (Vol-Target)", "#f4a261"),
        ("frozen_2016_2021_RQ2_ensemble_quantile_q80_q30_all_in_out", "Top Ensemble (All-in)", "#2a9d8f"),
        ("frozen_2016_2021_RQ2_svm_quantile_q70_q40_vol_targeting", "Top SVM (Vol-Target)", "#2b5c8f"),
    ]

    scenarios = ["All_Trades", "Excl_Top_1_Best_Trade", "Excl_Top_2_Best_Trades", "Excl_Top_3_Best_Trades"]
    scenario_labels = ["Toàn bộ Lệnh\n(All Trades)", "Loại Top 1\n(Excl 1)", "Loại Top 2\n(Excl 2)", "Loại Top 3\n(Excl 3)"]
    x_pos = np.arange(len(scenarios))
    w = 0.20

    for idx, (s_id, s_lbl, s_col) in enumerate(strat_order):
        sub_s = outlier_df[outlier_df["strategy_id"] == s_id]
        cagr_vals = [
            float(sub_s[sub_s["scenario"] == sc]["adjusted_cagr_pct"].iloc[0])
            for sc in scenarios
        ]
        offset = (idx - 1.5) * w
        bars = ax_b.bar(
            x_pos + offset, cagr_vals, width=w,
            color=s_col, label=s_lbl,
            alpha=0.9, edgecolor="black", linewidth=0.6,
        )
        # Thêm nhãn số
        for b, v in zip(bars, cagr_vals):
            y_loc = b.get_height()
            va = "bottom" if y_loc >= 0 else "top"
            pad = 0.3 if y_loc >= 0 else -0.8
            ax_b.text(
                b.get_x() + b.get_width() / 2, y_loc + pad,
                f"{v:+.1f}%", ha="center", va=va, fontsize=8, fontweight="bold",
            )

    # Đường tham chiếu đối chuẩn
    ax_b.axhline(4.50, color="#d4a373", linestyle="--", linewidth=1.5, label="Risk-Free Cash (+4.50%)")
    ax_b.axhline(-2.76, color="#e63946", linestyle=":", linewidth=1.5, label="Buy & Hold HPG (-2.76%)")
    ax_b.axhline(0.00, color="black", linestyle="-", linewidth=0.8, alpha=0.6)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(scenario_labels, fontsize=10.5, fontweight="bold")
    ax_b.set_ylabel("Adjusted CAGR (%) Sau Khi Loại Bỏ Giao Dịch Thắng Lớn", fontsize=11, fontweight="bold")
    ax_b.set_title("Panel B: Stress Test Loại Bỏ Giao Dịch Lãi Đột Biến (2022–2026)\n(Kiểm tra độ bền vững Alpha và mức độ phụ thuộc vào may mắn)", fontsize=12, fontweight="bold", pad=12)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.0)
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_b.set_ylim(-10.0, 14.0)

    plt.tight_layout()
    fig_path = figures_dir / "phase5_robustness_summary.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"  ✓ Đã xuất biểu đồ chuẩn publication: {fig_path} ({fig_path.stat().st_size / 1024:.1f} KB)")
    return fig_path


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    rob_dir = root / "results" / "robustness"
    fig_dir = rob_dir / "figures"
    rob_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KHỞI CHẠY QUY TRÌNH KIỂM ĐỊNH ĐỘ BỀN VỮNG BỔ SUNG (PHASE 5 ROBUSTNESS)")
    print("=" * 80)

    # 1. Xác minh tính bất biến mã băm nguồn
    print("\n[Bước 1/4] Kiểm toán tính bất biến mã băm SHA-256 dữ liệu nguồn...")
    hashes = verify_research_artifacts(root)
    print(f"  ✓ Dataset SHA-256:  {hashes['dataset_sha256']}")
    print(f"  ✓ Manifest SHA-256: {hashes['manifest_sha256']}")

    # 2. Thực nghiệm 1A
    print("\n[Bước 2/4] Thực thi Feature Group Ablation Study...")
    ablation_df = run_feature_group_ablation(root, rob_dir)

    # 3. Thực nghiệm 1B
    print("\n[Bước 3/4] Thực thi Outlier-Trade Removal Stress Test...")
    outlier_df = run_outlier_trade_stress_test(root, rob_dir)

    # 4. Xuất biểu đồ Phase 5 (300 DPI)
    print("\n[Bước 4/4] Xuất biểu đồ trực quan hóa tổng hợp Phase 5...")
    fig_path = plot_phase5_robustness_summary(ablation_df, outlier_df, fig_dir)

    print("\n" + "=" * 80)
    print("HOÀN TẤT TOÀN DIỆN PHASE 5 ROBUSTNESS TESTING! 100% KẾT QUẢ ĐÃ SẴN SÀNG.")
    print("=" * 80)


if __name__ == "__main__":
    main()

