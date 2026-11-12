"""Script thực hiện kiểm toán định lượng bổ sung WP3.6 & WP3.7 trước Phase 4.

Tạo 2 artifacts kiểm toán tại results/modeling/final_evaluation/:
1. test_probability_threshold_audit.csv: Thống kê phân phối xác suất hiệu chuẩn và tỷ lệ kích hoạt ngưỡng 0.53 / 0.48.
2. rq2_quintile_return_audit.csv: Phân tích 5 phân vị lợi suất và tỷ lệ thắng của LSTM vs SVM ở RQ2 (Protocol B).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def generate_probability_threshold_audit(
    predictions_path: Path, output_path: Path
) -> pd.DataFrame:
    """Kiểm toán phân phối calibrated_probability và ranking_score cho mọi tổ hợp.

    Args:
        predictions_path: Đường dẫn file parquet test_predictions_2022_2026.parquet.
        output_path: Đường dẫn file xuất CSV.

    Returns:
        DataFrame kết quả kiểm toán ngưỡng xác suất.
    """
    df = pd.read_parquet(predictions_path)
    records = []

    for (protocol, rq, model_name), grp in df.groupby(
        ["protocol", "rq", "model_name"], sort=False
    ):
        prob = grp["calibrated_probability"]
        score = grp["ranking_score"]
        config_id = grp["config_id"].iloc[0]
        n_samples = len(grp)

        # Ngưỡng giao dịch
        pct_above_053 = float((prob >= 0.53).mean())
        pct_below_048 = float((prob <= 0.48).mean())
        pct_in_band = float(((prob > 0.48) & (prob < 0.53)).mean())

        records.append({
            "protocol": protocol,
            "rq": rq,
            "model_name": model_name,
            "config_id": config_id,
            "samples": n_samples,
            "prob_min": float(prob.min()),
            "prob_p10": float(prob.quantile(0.10)),
            "prob_p25": float(prob.quantile(0.25)),
            "prob_median": float(prob.median()),
            "prob_p75": float(prob.quantile(0.75)),
            "prob_p90": float(prob.quantile(0.90)),
            "prob_max": float(prob.max()),
            "pct_above_053": pct_above_053,
            "pct_below_048": pct_below_048,
            "pct_in_band_048_053": pct_in_band,
            "score_min": float(score.min()),
            "score_p10": float(score.quantile(0.10)),
            "score_p25": float(score.quantile(0.25)),
            "score_median": float(score.median()),
            "score_p75": float(score.quantile(0.75)),
            "score_p90": float(score.quantile(0.90)),
            "score_max": float(score.max()),
        })

    audit_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return audit_df


def generate_rq2_quintile_return_audit(
    predictions_path: Path, output_path: Path
) -> pd.DataFrame:
    """Phân tích 5 phân vị lợi suất và tỷ lệ thắng cho LSTM và SVM ở RQ2 (Protocol B).

    Args:
        predictions_path: Đường dẫn file parquet test_predictions_2022_2026.parquet.
        output_path: Đường dẫn file xuất CSV.

    Returns:
        DataFrame phân tích quintile lợi suất ròng liên tục.
    """
    df = pd.read_parquet(predictions_path)
    rq2_retrain = df[(df["rq"] == "RQ2") & (df["protocol"] == "annual_expanding_retrain")]

    records = []
    for model_name in ["svm", "lstm"]:
        m_df = rq2_retrain[rq2_retrain["model_name"] == model_name].copy()
        m_df["quintile"] = pd.qcut(
            m_df["ranking_score"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )

        for q_name, q_grp in m_df.groupby("quintile", observed=False):
            net_ret = q_grp["forward_trade_net_return_proxy"]
            records.append({
                "rq": "RQ2",
                "protocol": "annual_expanding_retrain",
                "model_name": model_name,
                "quintile": str(q_name),
                "samples": len(q_grp),
                "ranking_score_min": float(q_grp["ranking_score"].min()),
                "ranking_score_max": float(q_grp["ranking_score"].max()),
                "calibrated_prob_mean": float(q_grp["calibrated_probability"].mean()),
                "win_rate": float((q_grp["y_true"] == 1).mean()),
                "mean_net_return": float(net_ret.mean()),
                "median_net_return": float(net_ret.median()),
                "std_net_return": float(net_ret.std()),
                "p05_net_return": float(np.percentile(net_ret, 5)),
                "min_net_return": float(net_ret.min()),
                "max_net_return": float(net_ret.max()),
            })

    quintile_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quintile_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return quintile_df


def main() -> None:
    """Chạy toàn bộ quy trình kiểm toán định lượng."""
    preds_file = ROOT / "results/modeling/final_evaluation/test_predictions_2022_2026.parquet"
    t2_file = ROOT / "results/modeling/final_evaluation/test_probability_threshold_audit.csv"
    t3_file = ROOT / "results/modeling/final_evaluation/rq2_quintile_return_audit.csv"

    print("Đang kiểm toán phân phối xác suất và ngưỡng kích hoạt (Task 2)...")
    t2_df = generate_probability_threshold_audit(preds_file, t2_file)
    print(f"Đã xuất: {t2_file} ({len(t2_df)} dòng)")

    print("Đang kiểm toán phân vị lợi suất RQ2 LSTM vs SVM (Task 3)...")
    t3_df = generate_rq2_quintile_return_audit(preds_file, t3_file)
    print(f"Đã xuất: {t3_file} ({len(t3_df)} dòng)")


if __name__ == "__main__":
    main()

