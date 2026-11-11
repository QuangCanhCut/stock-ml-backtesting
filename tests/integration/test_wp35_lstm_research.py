"""Kiểm tra tích hợp cho WP3.5 trên dataset HPG Research thực tế và artifacts models/lstm/."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset, make_modeling_view
from src.models.lstm_network import build_fold_sequences
from src.models.modeling_safety import MANIFEST_PATH, get_development_splits
from src.models.train_lstm import (
    MODEL_CONFIGS, evaluate_fold, load_final_lstm_bundle,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có dataset Research trong môi trường kiểm thử")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_real_development_fold_lstm(rq):
    """Kiểm tra đánh giá fold dev_1 trên dữ liệu Research thực tế và bảo toàn tính toàn vẹn file nguồn."""
    paths = [ROOT / RESEARCH_PATH, ROOT / "configs/config.yaml", ROOT / MANIFEST_PATH]
    before_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

    config = load_project_config(paths[1])
    dataset = load_research_dataset(ROOT, config)
    view = make_modeling_view(dataset, config, rq)
    folds = get_development_splits(ROOT, rq)

    cfg = next(c for c in MODEL_CONFIGS if c.model_name == "lstm")
    record, pred_df, logs, _, _ = evaluate_fold(
        view, rq, folds[0], cfg, compute_attribution=False
    )

    assert record["rq"] == rq
    assert record["fold_name"] == "dev_1"
    assert record["best_epoch"] is not None and record["best_epoch"] >= 1
    assert pred_df.signal_date.dt.year.max() < 2022
    assert pred_df.label_end_date.dt.year.max() < 2022
    assert pred_df.score_type.iloc[0] == "predicted_probability"
    assert ((pred_df.ranking_score >= 0.0) & (pred_df.ranking_score <= 1.0)).all()

    # Khớp đúng số lượng mẫu validation
    assert len(pred_df) == len(folds[0].validation_indices)

    # Bảo đảm các hash nguồn không bị thay đổi
    after_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert before_hashes == after_hashes


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có dataset Research trong môi trường kiểm thử")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_validation_samples_align_with_svm_and_lightgbm(rq):
    """Xác minh số mẫu Validation của LSTM qua 3 folds khớp 100% với WP3.3 và WP3.4."""
    config = load_project_config(ROOT / "configs/config.yaml")
    dataset = load_research_dataset(ROOT, config)
    view = make_modeling_view(dataset, config, rq)
    folds = get_development_splits(ROOT, rq)

    expected_val_counts = {"RQ1": [249, 251, 249], "RQ2": [246, 248, 246]}
    for idx, fold in enumerate(folds):
        _, _, X_val, y_val, _ = build_fold_sequences(view, fold, seq_len=10)
        assert len(y_val) == expected_val_counts[rq][idx]
        assert X_val.shape[0] == expected_val_counts[rq][idx]

    # Nếu có file validation_predictions của WP3.4, kiểm tra khớp ngày signal_date
    lgbm_preds_file = ROOT / "results/modeling/lightgbm/validation_predictions.parquet"
    if lgbm_preds_file.is_file():
        lgbm_preds = pd.read_parquet(lgbm_preds_file)
        lgbm_sub = lgbm_preds[(lgbm_preds.rq == rq) & (lgbm_preds.config_id == "dummy_prior")]
        for idx, fold in enumerate(folds):
            f_lgbm = lgbm_sub[lgbm_sub.fold_name == fold.fold_name]
            expected_dates = view.metadata.signal_date.iloc[list(fold.validation_indices)].tolist()
            assert f_lgbm.signal_date.tolist() == expected_dates


@pytest.mark.skipif(not (ROOT / "models/lstm/rq1_lstm_final_dev_2016_2021.pt").is_file(),
                    reason="Chưa có artifacts LSTM trong models/lstm/")
def test_saved_lstm_final_development_models():
    """Kiểm tra tính toàn vẹn và khả năng nạp/suy luận của Final Development Models LSTM."""
    for rq in ("rq1", "rq2"):
        model_file = ROOT / f"models/lstm/{rq}_lstm_final_dev_2016_2021.pt"
        assert model_file.is_file()
        model, scaler, meta = load_final_lstm_bundle(model_file)
        assert isinstance(model, torch.nn.Module)
        assert hasattr(scaler, "transform")
        assert meta["train_samples"] == 1297
        assert meta["final_epochs"] >= 1

        # Chạy thử 1 forward pass
        sample = torch.randn(2, meta["seq_len"], meta["network_config"]["input_size"])
        with torch.no_grad():
            out = model(sample)
            probs = torch.sigmoid(out).numpy()
            assert probs.shape == (2,)
            assert ((probs >= 0.0) & (probs <= 1.0)).all()

    meta_file = ROOT / "models/lstm/lstm_final_dev_metadata.json"
    assert meta_file.is_file()
    meta_json = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_json["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert meta_json["models"]["RQ1"]["train_samples"] == 1297
    assert meta_json["models"]["RQ2"]["train_samples"] == 1297

