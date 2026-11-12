"""Script thực thi chính WP4.1 & WP4.2: Backtest Data Contract, Microstructure Rules & Causal Signal Mapping.

Quy trình:
1. Kiểm toán mã băm SHA-256 của Dataset nghiên cứu và Split Manifest.
2. Nạp quy tắc vi cấu trúc sàn HOSE (HOSEMicrostructureRules) từ configs/config.yaml.
3. Nạp Master Market Calendar và căn chỉnh nhân quả với test_predictions_2022_2026.parquet.
4. Nạp bộ đệm điểm số Out-of-Fold của tập Development (2019–2021) chống Cold Start.
5. Sinh toàn bộ tín hiệu giao dịch cho 3 modes: fixed_053_048, quantile_q80_q30, quantile_q70_q40
   và Ensemble kết hợp Z-Score đa mô hình (SVM + LightGBM + LSTM).
6. Xuất 3 artifacts tại results/backtesting/:
   - wp41_42_contract_and_signal_audit.json
   - backtest_signals_2022_2026.parquet
   - signal_diagnostics_2022_2026.csv
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.backtest.data import (
    RESEARCH_DATASET_REL_PATH,
    SPLIT_MANIFEST_REL_PATH,
    TEST_PREDICTIONS_REL_PATH,
    align_predictions_with_market,
    load_dev_oof_seed_buffers,
    load_market_calendar,
    verify_research_artifacts,
)
from src.backtest.rules import HOSEMicrostructureRules
from src.backtest.signals import build_all_backtest_signals, compute_signal_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_DIR_REL = Path("results/backtesting")


def run_pipeline(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Chạy toàn bộ pipeline WP4.1 & WP4.2 và lưu trữ artifacts.

    Args:
        root: Thư mục gốc của repository.

    Returns:
        tuple gồm (signals_df, diagnostics_df, audit_metadata).
    """
    root = Path(root).resolve()
    output_dir = root / OUTPUT_DIR_REL
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("1. Bắt đầu kiểm toán tính bất biến của artifacts Phase 2 & Phase 3...")
    hashes = verify_research_artifacts(root)
    LOGGER.info("   -> Dataset SHA-256: %s (HỢP LỆ)", hashes["dataset_sha256"][:12])
    LOGGER.info("   -> Manifest SHA-256: %s (HỢP LỆ)", hashes["manifest_sha256"][:12])

    LOGGER.info("2. Nạp cấu hình vi cấu trúc sàn HOSE từ configs/config.yaml...")
    config_file = root / "configs/config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rules = HOSEMicrostructureRules.from_config(cfg)
    LOGGER.info(
        "   -> Biểu phí: Mua %.2f%%, Bán %.2f%%, Tổng vòng quay: %.2f%%",
        rules.buy_cost_rate * 100,
        rules.sell_cost_rate * 100,
        rules.effective_round_trip_cost * 100,
    )
    LOGGER.info(
        "   -> Biên độ trần/sàn: ±%.1f%%, Ngưỡng nhận diện kẹt: ±%.1f%%",
        rules.daily_price_limit * 100,
        rules.limit_hit_threshold * 100,
    )

    LOGGER.info("3. Nạp Master Market Calendar và căn chỉnh với Test Predictions...")
    market_df = load_market_calendar(root)
    LOGGER.info("   -> Tổng số phiên giao dịch Master Calendar: %d phiên", len(market_df))

    test_preds_file = root / TEST_PREDICTIONS_REL_PATH
    if not test_preds_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file dự đoán Test tại: {test_preds_file}")
    preds_df = pd.read_parquet(test_preds_file)
    LOGGER.info("   -> Số dòng dự đoán Test nạp vào: %d dòng", len(preds_df))

    aligned_df = align_predictions_with_market(preds_df, market_df, rules)
    LOGGER.info("   -> Căn chỉnh thành công! 0 dòng bị thiếu giá thực thi exec_open.")

    LOGGER.info("4. Nạp bộ đệm khởi tạo OOF Dev (2019–2021) chống Cold Start...")
    dev_oof_buffers = load_dev_oof_seed_buffers(root)
    for rq in ("RQ1", "RQ2"):
        LOGGER.info(
            "   -> %s: OOF Dev samples = %d phiên (SVM, LGBM, LSTM, Ensemble)",
            rq,
            len(dev_oof_buffers[rq]["svm"]),
        )

    LOGGER.info("5. Sinh toàn bộ tín hiệu giao dịch nhân quả (3 modes + Ensemble)...")
    signals_df = build_all_backtest_signals(aligned_df, dev_oof_buffers)
    LOGGER.info("   -> Tổng số dòng tín hiệu giao dịch đã tạo: %d dòng", len(signals_df))

    LOGGER.info("6. Tính toán chẩn đoán hành vi tín hiệu (Signal Diagnostics)...")
    diagnostics_df = compute_signal_diagnostics(signals_df)

    LOGGER.info("7. Xuất các artifacts kết quả...")
    # 7.1. File Parquet tín hiệu
    parquet_out = output_dir / "backtest_signals_2022_2026.parquet"
    signals_df.to_parquet(parquet_out, index=False)
    LOGGER.info("   -> Đã lưu: %s", parquet_out)

    # 7.2. File CSV chẩn đoán
    csv_out = output_dir / "signal_diagnostics_2022_2026.csv"
    diagnostics_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    LOGGER.info("   -> Đã lưu: %s", csv_out)

    # 7.3. File JSON kiểm toán
    audit_metadata = {
        "status": "COMPLETED",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": RESEARCH_DATASET_REL_PATH.as_posix(),
        "dataset_sha256": hashes["dataset_sha256"],
        "manifest_path": SPLIT_MANIFEST_REL_PATH.as_posix(),
        "manifest_sha256": hashes["manifest_sha256"],
        "rules": {
            "exchange": rules.exchange,
            "initial_capital": rules.initial_capital,
            "risk_free_rate": rules.risk_free_rate,
            "trading_days_per_year": rules.trading_days_per_year,
            "daily_price_limit": rules.daily_price_limit,
            "limit_hit_threshold": rules.limit_hit_threshold,
            "settlement_cycle": rules.settlement_cycle,
            "buy_cost_rate": rules.buy_cost_rate,
            "sell_cost_rate": rules.sell_cost_rate,
            "effective_round_trip_cost": rules.effective_round_trip_cost,
        },
        "input_test_predictions_rows": len(preds_df),
        "aligned_market_data": {
            "total_calendar_sessions": len(market_df),
            "test_start_date": str(aligned_df["signal_date"].min().date()),
            "test_end_date_rq1": str(aligned_df[aligned_df["rq"] == "RQ1"]["signal_date"].max().date()),
            "test_end_date_rq2": str(aligned_df[aligned_df["rq"] == "RQ2"]["signal_date"].max().date()),
            "rq1_sessions": int(len(aligned_df[aligned_df["rq"] == "RQ1"]["signal_date"].unique())),
            "rq2_sessions": int(len(aligned_df[aligned_df["rq"] == "RQ2"]["signal_date"].unique())),
            "missing_exec_open_count": int(aligned_df["exec_open"].isna().sum()),
        },
        "dev_oof_seed_buffers": {
            rq: {m: len(buf) for m, buf in dev_oof_buffers[rq].items()}
            for rq in ("RQ1", "RQ2")
        },
        "output_signal_rows": len(signals_df),
        "diagnostics_summary": {
            "total_series": len(diagnostics_df),
            "series_list": diagnostics_df[["protocol", "rq", "model_name", "signal_mode"]].to_dict(orient="records"),
        },
    }

    json_out = output_dir / "wp41_42_contract_and_signal_audit.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(audit_metadata, f, indent=2, ensure_ascii=False)
    LOGGER.info("   -> Đã lưu: %s", json_out)

    return signals_df, diagnostics_df, audit_metadata


def print_comparison_highlights(diagnostics_df: pd.DataFrame) -> None:
    """In bảng so sánh nổi bật chứng minh đã giải quyết hiện tượng Đói Tín Hiệu (Signal Starvation)."""
    print("\n" + "=" * 95)
    print("BẢNG ĐỐI CHỨNG MỨC ĐỘ THAM GIA THỊ TRƯỜNG (% TIME IN MARKET) GIỮA CÁC CHẾ ĐỘ TÍN HIỆU")
    print("=" * 95)

    focus_models = [
        ("annual_expanding_retrain", "RQ2", "svm"),
        ("annual_expanding_retrain", "RQ1", "lightgbm"),
        ("annual_expanding_retrain", "RQ2", "lstm"),
        ("annual_expanding_retrain", "RQ2", "ensemble"),
    ]

    highlight_rows = []
    for proto, rq, m in focus_models:
        sub = diagnostics_df[
            (diagnostics_df["protocol"] == proto)
            & (diagnostics_df["rq"] == rq)
            & (diagnostics_df["model_name"] == m)
        ]
        for _, r in sub.iterrows():
            highlight_rows.append({
                "Protocol": r["protocol"],
                "RQ": r["rq"],
                "Model": r["model_name"],
                "Mode": r["signal_mode"],
                "Time in Market (%)": f"{r['desired_time_in_market_pct']:.2f}%",
                "Entries": r["entry_trigger_count"],
                "Exits": r["exit_trigger_count"],
                "Switches": r["desired_state_switches"],
                "Avg Hold (Sessions)": r["avg_desired_holding_sessions"],
            })

    df_hl = pd.DataFrame(highlight_rows)
    print(df_hl.to_string(index=False))
    print("=" * 95 + "\n")


def main() -> None:
    """Hàm chạy thực thi dòng lệnh."""
    signals_df, diagnostics_df, _ = run_pipeline(ROOT)
    print_comparison_highlights(diagnostics_df)


if __name__ == "__main__":
    main()

