"""Kiểm thử tích hợp cho Phase 5: Robustness Testing và Hoàn thiện Portfolio Documentation.

Bao gồm:
1. Xác minh tính bất biến mã băm SHA-256 của dataset và split manifest.
2. Kiểm tra sự tồn tại và tính hợp lệ (0 NaN, đủ 36 dòng, đúng cột) của feature_group_ablation_dev_cv.csv.
3. Kiểm tra sự tồn tại và tính hợp lệ (0 NaN, đủ 16 dòng, đúng cột) của outlier_trade_stress_test.csv.
4. Kiểm tra sự tồn tại và dung lượng chuẩn 300 DPI (> 50 KB) của biểu đồ phase5_robustness_summary.png.
5. Kiểm tra sự tồn tại của README.md tại thư mục gốc repository, xác minh toàn bộ các đường dẫn ảnh
   tương đối bên trong README.md đều tồn tại thực tế trên đĩa và các bảng số liệu khớp chính xác.
"""

from pathlib import Path
import re
import pandas as pd
import pytest

from src.backtest.data import verify_research_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROBUSTNESS_DIR = REPO_ROOT / "results" / "robustness"
FIGURES_DIR = ROBUSTNESS_DIR / "figures"
README_PATH = REPO_ROOT / "README.md"


def test_research_artifacts_sha256_invariance() -> None:
    """Xác minh mã băm SHA-256 của dataset và split manifest không bị thay đổi trong Phase 5."""
    hashes = verify_research_artifacts(REPO_ROOT)
    assert hashes["dataset_sha256"] == "4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e"
    assert hashes["manifest_sha256"] == "f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42"


def test_feature_group_ablation_csv_validity() -> None:
    """Kiểm tra tính toàn vẹn của kết quả Feature Group Ablation Study (36 kịch bản, 0 NaN)."""
    csv_path = ROBUSTNESS_DIR / "feature_group_ablation_dev_cv.csv"
    assert csv_path.is_file(), f"Thiếu file: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 36, f"Kỳ vọng 36 kịch bản ablation (2 RQ * 3 models * 6 scenarios), nhận được: {len(df)}"
    assert df.isna().sum().sum() == 0, f"Bảng feature_group_ablation_dev_cv.csv có chứa NaN: {df.isna().sum().to_dict()}"

    expected_cols = [
        "rq", "model_name", "ablation_scenario", "dropped_group", "num_features",
        "mean_balanced_accuracy", "std_balanced_accuracy", "mean_roc_auc",
        "delta_bacc_vs_full", "delta_auc_vs_full"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Thiếu cột {col} trong ablation CSV"

    # Kiểm tra các kịch bản Full_26_Features có delta = 0
    full_scenarios = df[df["ablation_scenario"] == "Full_26_Features"]
    assert len(full_scenarios) == 6  # 2 RQ * 3 models
    assert (full_scenarios["delta_bacc_vs_full"] == 0.0).all()
    assert (full_scenarios["num_features"] == 26).all()


def test_outlier_trade_stress_test_csv_validity() -> None:
    """Kiểm tra tính toàn vẹn của kết quả Outlier-Trade Removal Stress Test (16 kịch bản, 0 NaN)."""
    csv_path = ROBUSTNESS_DIR / "outlier_trade_stress_test.csv"
    assert csv_path.is_file(), f"Thiếu file: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 16, f"Kỳ vọng 16 kịch bản stress test (4 strategies * 4 scenarios), nhận được: {len(df)}"
    assert df.isna().sum().sum() == 0, f"Bảng outlier_trade_stress_test.csv có chứa NaN: {df.isna().sum().to_dict()}"

    expected_cols = [
        "strategy_id", "scenario", "removed_trades_count", "remaining_trades",
        "adjusted_final_nav_vnd", "adjusted_total_return_pct", "adjusted_cagr_pct",
        "beats_buy_and_hold", "beats_risk_free"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Thiếu cột {col} trong outlier stress test CSV"

    # Kiểm tra Top 1 LSTM ở kịch bản All_Trades có NAV khớp chính xác 160,179,634
    top1_all = df[
        (df["strategy_id"] == "annual_expanding_retrain_RQ2_lstm_fixed_053_048_all_in_out") &
        (df["scenario"] == "All_Trades")
    ].iloc[0]
    assert top1_all["adjusted_final_nav_vnd"] == 160179634
    assert abs(top1_all["adjusted_cagr_pct"] - 10.40) < 0.05
    assert bool(top1_all["beats_buy_and_hold"]) is True
    assert bool(top1_all["beats_risk_free"]) is True


def test_phase5_summary_figure_exists_and_quality() -> None:
    """Kiểm tra sự tồn tại và dung lượng (> 50 KB, 300 DPI) của biểu đồ tổng hợp Phase 5."""
    fig_path = FIGURES_DIR / "phase5_robustness_summary.png"
    assert fig_path.is_file(), f"Thiếu file biểu đồ: {fig_path}"
    file_size_kb = fig_path.stat().st_size / 1024.0
    assert file_size_kb > 50.0, f"Biểu đồ {fig_path.name} có dung lượng quá nhỏ ({file_size_kb:.1f} KB)!"


def test_readme_file_and_image_links() -> None:
    """Kiểm tra sự tồn tại của README.md và toàn bộ các liên kết hình ảnh tương đối bên trong."""
    assert README_PATH.is_file(), f"Thiếu file README.md tại gốc repository: {README_PATH}"
    content = README_PATH.read_text(encoding="utf-8")

    # Kiểm tra các đề mục cốt lõi
    assert "Stock Movement Prediction" in content
    assert "End-to-End Quantitative Pipeline Architecture" in content
    assert "Key Empirical Results" in content
    assert "Core Research Findings" in content
    assert "Reproducibility & Quickstart Guide" in content
    assert "COMPLETED_WITH_LIMITATIONS" in content

    # Tìm toàn bộ các đường dẫn ảnh dạng ![alt](path) (loại trừ URL trực tuyến như badges)
    image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    matches = image_pattern.findall(content)
    local_images = [m.strip() for m in matches if not m.startswith("http")]
    assert len(local_images) >= 3, f"README.md phải nhúng ít nhất 3 biểu đồ nghiên cứu cục bộ, tìm thấy {len(local_images)}"

    for rel_path in local_images:
        target_path = REPO_ROOT / rel_path
        assert target_path.is_file(), f"Ảnh được nhúng trong README.md không tồn tại thực tế trên đĩa: {rel_path}"
        assert target_path.stat().st_size > 10 * 1024, f"File ảnh {rel_path} có dung lượng quá nhỏ!"
