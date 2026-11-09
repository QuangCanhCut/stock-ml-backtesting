"""Chạy Phase 2 trên Raw Research, lưu riêng và không ghi đè kết quả cũ."""

import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from src.dataset_builder import LABEL_COLUMNS, assemble_dataset, load_project_config, quality_report
from src.eda_analyzer import compute_quant_features, generate_and_save_plots, run_statistical_tests
from src.features import build_features


@contextmanager
def staging_directory(root):
    """Tạo thư mục riêng, kế thừa ACL của results để người dùng đọc được báo cáo."""
    path = root / "results" / f".research-stage-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def run(root):
    root = Path(root).resolve()
    raw_path = root / "data/raw/HPG_10y_daily_research.parquet"
    provenance_path = root / "data/raw/HPG_10y_daily_research.provenance.json"
    processed = root / "data/processed/HPG_features_labeled_research.parquet"
    reports = root / "results/research_phase2"
    if processed.exists() or reports.exists():
        raise FileExistsError("Processed Research hoặc thư mục báo cáo đã tồn tại; không ghi đè")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("gate") != "PASS_WITH_LIMITATIONS" or hashlib.sha256(raw_path.read_bytes()).hexdigest() != provenance.get("dataset_sha256"):
        raise ValueError("Raw Research không khớp provenance hoặc research gate")
    cfg = load_project_config(root / "configs/config.yaml")
    raw = pd.read_parquet(raw_path)
    if len(raw) != provenance["sessions"] or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Raw Research sai số dòng hoặc thứ tự thời gian")
    dataset, names = assemble_dataset(raw, cfg)
    if any(name.startswith(("target_", "forward_", "next_", "future_")) for name in names):
        raise ValueError("Schema feature chứa cột tương lai")
    expected = np.log(raw.close.shift(-1) / raw.close)
    np.testing.assert_allclose(dataset.forward_log_return_1.to_numpy(), expected.to_numpy(), equal_nan=True, rtol=1e-12)
    if not pd.isna(dataset.target_direction.iloc[-1]):
        raise ValueError("Nhãn phiên cuối phải là NA")
    expected_direction = (expected.iloc[:-1] > 0).astype(int).to_numpy()
    np.testing.assert_array_equal(dataset.target_direction.iloc[:-1].astype(int).to_numpy(), expected_direction)
    cutoff = len(raw) * 4 // 5
    prefix = build_features(raw.iloc[:cutoff], cfg)
    pd.testing.assert_frame_equal(prefix, dataset.loc[prefix.index, prefix.columns], check_exact=True)
    old = pd.read_parquet(root / cfg["data"]["processed_path"])
    feature_cols = dataset[names]
    # Phân tích liên hệ target chỉ thực hiện trên train, có purge theo phiên.
    val_start = pd.Timestamp(raw.index.min().year + cfg["validation"]["initial_train_years"], 1, 1)
    train = dataset.loc[dataset.index < val_start].iloc[:-1]
    ic = pd.DataFrame({"spearman_train": [train[name].corr(train.forward_log_return_1, method="spearman") for name in names],
                       "n_train": [len(train[[name, "forward_log_return_1"]].dropna()) for name in names]}, index=names)
    train_stats = train[names].describe().T
    correlations = train[names].corr(method="spearman")
    comparison = {"raw_research_sha256": provenance["dataset_sha256"], "research_gate": provenance["gate"],
                  "dataset_cu": {"rows": len(old), "columns": len(old.columns), "first": str(old.index.min().date()), "last": str(old.index.max().date())},
                  "dataset_research": {"rows": len(dataset), "columns": len(dataset.columns), "features": len(names), "labels": len(LABEL_COLUMNS),
                                       "first": str(dataset.index.min().date()), "last": str(dataset.index.max().date()),
                                       "nan_features": {key: int(value) for key, value in feature_cols.isna().sum().items()},
                                       "nan_labels": {key: int(value) for key, value in dataset[LABEL_COLUMNS].isna().sum().items()}},
                  "comparison": {"rows_added": len(dataset.index.difference(old.index)), "rows_removed": len(old.index.difference(dataset.index)),
                                 "common_sessions": len(dataset.index.intersection(old.index))},
                  "checks": {"chronological_unique": True, "t_plus_1_matches_next_row": True, "last_target_missing": True,
                             "feature_prefix_invariant": True, "feature_name_leakage": False},
                  "limitations": provenance["limitations"] + [provenance["t_plus_1_risk"],
                                  "EDA mô tả dùng toàn bộ giai đoạn, gồm cả test; không được xem test là chưa từng quan sát trong EDA"]}
    with staging_directory(root) as stage:
        stage_parquet = stage / "dataset.parquet"
        dataset.to_parquet(stage_parquet)
        restored = pd.read_parquet(stage_parquet)
        pd.testing.assert_frame_equal(dataset, restored, check_exact=True, check_freq=False)
        if dataset.attrs != restored.attrs:
            raise ValueError("Không bảo toàn metadata dataset")
        comparison["processed_sha256"] = hashlib.sha256(stage_parquet.read_bytes()).hexdigest()
        folder = stage / "reports"
        folder.mkdir()
        (folder / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "dataset_quality.json").write_text(json.dumps(quality_report(dataset, names), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        train_stats.to_csv(folder / "feature_statistics_train.csv")
        correlations.to_csv(folder / "feature_correlations_train.csv")
        ic.to_csv(folder / "feature_ic_train.csv")
        eda = compute_quant_features(raw, cfg)
        eda_report = run_statistical_tests(eda, cfg)
        (folder / "eda_report.txt").write_text(eda_report, encoding="utf-8")
        generate_and_save_plots(eda, str(folder / "figures"))
        os.link(stage_parquet, processed)
        try:
            os.rename(folder, reports)
        except Exception:
            processed.unlink(missing_ok=True)
            raise
    return comparison


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run("."), ensure_ascii=False, indent=2))
