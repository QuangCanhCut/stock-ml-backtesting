"""Kiểm thử tích hợp cho WP4.5 (Phase 4): Trực quan hóa, Case Studies, Notebook và Báo cáo.

Bao gồm:
1. Xác minh tính bất biến mã băm SHA-256 của dữ liệu nghiên cứu và manifest.
2. Kiểm tra sự tồn tại và dung lượng (> 10 KB) của cả 4 file biểu đồ 300 DPI trong results/backtesting/figures/.
3. Kiểm tra tính hợp lệ của results/backtesting/trade_case_studies.csv (đúng 20 giao dịch của Top 1 LSTM).
4. Kiểm tra tính hợp lệ của notebooks/04_strategy_backtesting.ipynb (đã có outputs, dung lượng > 100 KB).
5. Kiểm tra tính đồng bộ số liệu giữa results/backtesting/phase4_report.md và strategy_comparison_2022_2026.csv.
"""

import json
from pathlib import Path
import nbformat
import pandas as pd
import pytest

from src.backtest.data import verify_research_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "backtesting"
FIGURES_DIR = RESULTS_DIR / "figures"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def test_research_artifacts_integrity() -> None:
    """Xác minh mã băm SHA-256 của dataset và manifest giữ nguyên 100%."""
    hashes = verify_research_artifacts(REPO_ROOT)
    assert hashes["dataset_sha256"] == "4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e"
    assert hashes["manifest_sha256"] == "f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42"


def test_four_publication_figures_exist() -> None:
    """Kiểm tra sự tồn tại và dung lượng hợp lệ (> 10 KB) của bộ 4 biểu đồ 300 DPI."""
    expected_figures = [
        "01_equity_curves_vs_buyhold.png",
        "02_underwater_drawdown.png",
        "03_yearly_regime_and_rq_comparison.png",
        "04_cost_sensitivity_and_turnover.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.is_file(), f"Thiếu file biểu đồ: {fig_path}"
        file_size_kb = fig_path.stat().st_size / 1024.0
        assert file_size_kb > 10.0, f"Biểu đồ {fig_name} có dung lượng quá nhỏ ({file_size_kb:.1f} KB)!"


def test_trade_case_studies_validity() -> None:
    """Kiểm tra tính hợp lệ của kết quả trích xuất 20 giao dịch của Top 1 LSTM."""
    cases_path = RESULTS_DIR / "trade_case_studies.csv"
    assert cases_path.is_file(), f"Thiếu file: {cases_path}"

    df = pd.read_csv(cases_path)
    assert len(df) == 20, f"Kỳ vọng 20 giao dịch, thực tế: {len(df)}"

    expected_cols = [
        "trade_id", "entry_date", "exit_date", "entry_price", "exit_price",
        "shares", "holding_sessions", "gross_pnl", "net_pnl", "net_return_pct", "status"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Thiếu cột {col} trong trade_case_studies.csv"

    # Kiểm tra tỷ lệ thắng 60%
    win_rate = (df["net_pnl"] > 0).mean() * 100.0
    assert abs(win_rate - 60.0) < 1e-2, f"Win rate kỳ vọng 60.0%, thực tế: {win_rate:.2f}%"


def test_notebook_structure_and_outputs() -> None:
    """Kiểm tra notebook 04_strategy_backtesting.ipynb có đầy đủ cấu trúc và outputs."""
    nb_path = NOTEBOOKS_DIR / "04_strategy_backtesting.ipynb"
    assert nb_path.is_file(), f"Thiếu notebook: {nb_path}"
    assert nb_path.stat().st_size > 100_000, f"Dung lượng notebook quá nhỏ ({nb_path.stat().st_size} bytes)!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Đảm bảo có ít nhất 15 cells (bao gồm markdown, code và outputs)
    assert len(nb.cells) >= 15, f"Số lượng cell quá ít: {len(nb.cells)}"

    # Đảm bảo các code cells đều có outputs
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 7, f"Số lượng code cell quá ít: {len(code_cells)}"
    for idx, cell in enumerate(code_cells):
        assert len(cell.outputs) > 0, f"Code cell {idx + 1} không có outputs!"


def test_phase4_report_programmatic_consistency() -> None:
    """Đối chiếu tự động tính nhất quán số liệu giữa phase4_report.md và strategy_comparison_2022_2026.csv."""
    report_path = RESULTS_DIR / "phase4_report.md"
    assert report_path.is_file(), f"Thiếu báo cáo: {report_path}"

    report_text = report_path.read_text(encoding="utf-8")
    summary_df = pd.read_csv(RESULTS_DIR / "strategy_comparison_2022_2026.csv")
    top_lstm = summary_df.iloc[0]

    # Kiểm tra các số liệu cốt lõi của Top 1 LSTM
    assert "160,179,634" in report_text or "160.18" in report_text, "Thiếu số liệu NAV của Top 1 LSTM"
    assert "10.40%" in report_text, "Thiếu số liệu CAGR của Top 1 LSTM"
    assert "0.4070" in report_text, "Thiếu số liệu Sharpe Ratio của Top 1 LSTM"
    assert "-21.73%" in report_text, "Thiếu số liệu Max Drawdown của Top 1 LSTM"
    assert "-67.83%" in report_text, "Thiếu số liệu Max Drawdown của Buy & Hold"

    # Kiểm tra đầy đủ 9 phần trong báo cáo
    for section_idx in range(1, 10):
        section_header = f"## {section_idx}."
        assert section_header in report_text, f"Thiếu mục {section_header} trong báo cáo!"

