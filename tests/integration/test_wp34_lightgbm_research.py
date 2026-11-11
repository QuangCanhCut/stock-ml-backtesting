"""Kiểm tra tích hợp cho WP3.4 trên dataset HPG Research thực tế và artifacts models/."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset, make_modeling_view
from src.models.modeling_safety import (
    MANIFEST_PATH, get_development_splits, get_full_development_train_indices,
)
from src.models.train_lightgbm import MODEL_CONFIGS, evaluate_fold

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có dataset Research trong môi trường kiểm thử")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_real_development_fold_lightgbm(rq):
    """Kiểm tra đánh giá fold dev_1 trên dữ liệu Research thực tế và bảo toàn tính toàn vẹn file nguồn."""
    paths = [ROOT / RESEARCH_PATH, ROOT / "configs/config.yaml", ROOT / MANIFEST_PATH]
    before_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

    config = load_project_config(paths[1])
    dataset = load_research_dataset(ROOT, config)
    view = make_modeling_view(dataset, config, rq)
    folds = get_development_splits(ROOT, rq)

    cfg = next(c for c in MODEL_CONFIGS if c.model_name == "lightgbm")
    record, pred_df, logs, _ = evaluate_fold(view, rq, folds[0], cfg, compute_perm_importance=False)

    assert record["rq"] == rq
    assert record["fold_name"] == "dev_1"
    assert record["best_iteration"] is not None and record["best_iteration"] >= 1
    assert pred_df.signal_date.dt.year.max() < 2022
    assert pred_df.label_end_date.dt.year.max() < 2022
    assert pred_df.score_type.iloc[0] == "predicted_probability"
    assert ((pred_df.ranking_score >= 0.0) & (pred_df.ranking_score <= 1.0)).all()

    # Bảo đảm các hash nguồn không bị thay đổi
    after_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert before_hashes == after_hashes


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có dataset Research trong môi trường kiểm thử")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_get_full_development_train_indices_and_isolation(rq):
    """Kiểm tra chỉ số train full development 2016–2021 có đúng 1.297 mẫu và cô lập hoàn toàn Test."""
    indices = get_full_development_train_indices(ROOT, rq)
    assert len(indices) == 1297
    assert sorted(set(indices)) == list(indices)

    config = load_project_config(ROOT / "configs/config.yaml")
    dataset = load_research_dataset(ROOT, config)
    view = make_modeling_view(dataset, config, rq)

    train_dates = view.X.index[list(indices)]
    assert train_dates.year.max() <= 2021
    assert str(train_dates.min().date()) == "2016-10-20"
    assert str(train_dates.max().date()) == "2021-12-24"

    metadata_train = view.metadata.iloc[list(indices)]
    assert metadata_train.label_end_date.dt.year.max() < 2022


@pytest.mark.skipif(not (ROOT / "models/svm/rq1_svm_final_dev_2016_2021.joblib").is_file(),
                    reason="Chưa có artifacts SVM trong models/svm/")
def test_saved_svm_final_development_models():
    """Kiểm tra tính toàn vẹn và khả năng dự báo của Final Development Models SVM."""
    for rq in ("rq1", "rq2"):
        model_file = ROOT / f"models/svm/{rq}_svm_final_dev_2016_2021.joblib"
        assert model_file.is_file()
        model = joblib.load(model_file)
        assert hasattr(model, "predict")
        assert hasattr(model, "decision_function")

    meta_file = ROOT / "models/svm/svm_final_dev_metadata.json"
    assert meta_file.is_file()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert meta["models"]["RQ1"]["train_samples"] == 1297
    assert meta["models"]["RQ2"]["train_samples"] == 1297

    weights_file = ROOT / "models/svm/svm_final_dev_feature_weights.csv"
    assert weights_file.is_file()
    weights_df = pd.read_csv(weights_file)
    assert len(weights_df) == 52  # 26 features x 2 RQs
    assert set(weights_df.columns) == {"rq", "feature_name", "scaler_mean", "scaler_scale", "coef", "abs_coef_rank"}


@pytest.mark.skipif(not (ROOT / "models/lightgbm/rq1_lgbm_final_dev_2016_2021.joblib").is_file(),
                    reason="Chưa có artifacts LightGBM trong models/lightgbm/")
def test_saved_lightgbm_final_development_models():
    """Kiểm tra tính toàn vẹn và khả năng dự báo của Final Development Models LightGBM."""
    for rq in ("rq1", "rq2"):
        model_file = ROOT / f"models/lightgbm/{rq}_lgbm_final_dev_2016_2021.joblib"
        assert model_file.is_file()
        model = joblib.load(model_file)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    meta_file = ROOT / "models/lightgbm/lgbm_final_dev_metadata.json"
    assert meta_file.is_file()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert meta["models"]["RQ1"]["train_samples"] == 1297
    assert meta["models"]["RQ2"]["train_samples"] == 1297
    assert meta["models"]["RQ1"]["final_n_estimators"] >= 1
    assert meta["models"]["RQ2"]["final_n_estimators"] >= 1

    lstm_keep = ROOT / "models/lstm/.gitkeep"
    assert lstm_keep.is_file()

