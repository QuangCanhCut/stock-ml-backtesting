"""Kiểm tra tích hợp cho WP3.6 & WP3.7 trên dataset HPG Research thực tế và artifacts Phase 3."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset
from src.models.modeling_safety import (
    MANIFEST_PATH, get_final_evaluation_split,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có dataset Research trong môi trường kiểm thử")
def test_final_evaluation_unfreezing_and_sample_counts():
    """Kiểm tra việc mở khóa tập Reserved Test trả về đúng số lượng mẫu và ranh giới thời gian."""
    # 1. RQ1
    s1 = get_final_evaluation_split(ROOT, "RQ1", final_evaluation=True)
    assert len(s1.train_indices) == 1297
    assert len(s1.validation_indices) == 1200
    assert str(s1.train_start_date.date()) == "2016-10-20"
    assert str(s1.train_end_date.date()) == "2021-12-24"
    assert str(s1.validation_start_date.date()) == "2022-01-04"
    assert str(s1.validation_end_date.date()) == "2026-10-29"

    # 2. RQ2
    s2 = get_final_evaluation_split(ROOT, "RQ2", final_evaluation=True)
    assert len(s2.train_indices) == 1297
    assert len(s2.validation_indices) == 1197
    assert str(s2.train_start_date.date()) == "2016-10-20"
    assert str(s2.train_end_date.date()) == "2021-12-24"
    assert str(s2.validation_start_date.date()) == "2022-01-04"
    assert str(s2.validation_end_date.date()) == "2026-10-26"


@pytest.mark.skipif(not (ROOT / "results/modeling/final_evaluation/test_predictions_2022_2026.parquet").is_file(),
                    reason="Chưa chạy script run_wp36_evaluation.py để tạo artifacts")
def test_final_evaluation_artifacts_and_predictions_integrity():
    """Kiểm tra tính toàn vẹn của artifacts đầu ra của WP3.6/WP3.7."""
    eval_dir = ROOT / "results/modeling/final_evaluation"
    preds_file = eval_dir / "test_predictions_2022_2026.parquet"
    assert preds_file.is_file()

    preds_df = pd.read_parquet(preds_file)
    expected_cols = {
        "protocol", "rq", "model_name", "config_id", "signal_date", "label_end_date",
        "y_true", "y_pred", "ranking_score", "score_type", "calibrated_probability",
        "forward_log_return_1", "forward_trade_gross_return", "forward_trade_net_return_proxy",
    }
    assert expected_cols.issubset(set(preds_df.columns))

    # Kiểm tra xác suất hiệu chuẩn
    assert ((preds_df["calibrated_probability"] >= 0.0) & (preds_df["calibrated_probability"] <= 1.0)).all()

    # Kiểm tra số lượng mẫu test của từng mô hình theo từng giao thức
    for proto in ("frozen_2016_2021", "annual_expanding_retrain"):
        for m in ("dummy", "svm", "lightgbm", "lstm"):
            sub_rq1 = preds_df[(preds_df.protocol == proto) & (preds_df.rq == "RQ1") & (preds_df.model_name == m)]
            assert len(sub_rq1) == 1200
            sub_rq2 = preds_df[(preds_df.protocol == proto) & (preds_df.rq == "RQ2") & (preds_df.model_name == m)]
            assert len(sub_rq2) == 1197

    # Kiểm tra các file CSV
    assert (eval_dir / "dev_cv_all_models_summary.csv").is_file()
    assert (eval_dir / "dev_prediction_agreement.csv").is_file()
    assert (eval_dir / "test_metrics_overall_2022_2026.csv").is_file()
    assert (eval_dir / "test_metrics_by_year_2022_2026.csv").is_file()
    assert (eval_dir / "final_evaluation_metadata.json").is_file()
    assert (ROOT / "results/modeling/model_comparison.csv").is_file()
    assert (ROOT / "results/modeling/phase3_report.md").is_file()

    # Kiểm tra 3 hình ảnh
    fig_dir = ROOT / "results/modeling/figures"
    assert (fig_dir / "01_dev_vs_test_balanced_accuracy.png").is_file()
    assert (fig_dir / "02_yearly_regime_performance.png").is_file()
    assert (fig_dir / "03_model_score_distributions.png").is_file()


def test_research_dataset_and_manifest_hash_integrity():
    """Xác minh các file nguồn không bị sửa đổi hay rò rỉ sau quá trình mở khóa Test."""
    paths = [ROOT / RESEARCH_PATH, ROOT / "configs/config.yaml", ROOT / MANIFEST_PATH]
    expected_sha256 = {
        RESEARCH_PATH.as_posix(): "4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e",
        MANIFEST_PATH.as_posix(): "f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42",
    }
    for p in [paths[0], paths[2]]:
        computed = hashlib.sha256(p.read_bytes()).hexdigest()
        rel_key = p.relative_to(ROOT).as_posix()
        assert computed == expected_sha256[rel_key]

