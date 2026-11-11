"""Kiểm thử đơn vị (Unit Tests) cho module LSTM Sequence Modeling WP3.5 bằng dữ liệu synthetic."""

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from src.models.data import ModelingView
from src.models.lstm_network import (
    LSTMClassifier, build_fold_sequences, build_full_development_sequences,
)
from src.models.temporal_split import TemporalSplit
from src.models.train_lstm import (
    LSTMModelConfig, MODEL_CONFIGS, evaluate_fold, load_final_lstm_bundle,
)


@pytest.fixture
def synthetic_sample():
    """Tạo mẫu ModelingView và TemporalSplit tổng hợp để kiểm tra logic mà không chạm vào dữ liệu thật."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    n_features = 26
    rng = np.random.default_rng(42)
    X_vals = rng.standard_normal((100, n_features))
    X = pd.DataFrame(X_vals, index=dates, columns=[f"feat_{i}" for i in range(n_features)])
    y = pd.Series(rng.integers(0, 2, size=100), index=dates)
    metadata = pd.DataFrame({
        "signal_date": dates,
        "label_end_date": dates + pd.Timedelta(days=1),
    }, index=dates)

    train_indices = tuple(range(60))
    val_indices = tuple(range(65, 85))  # 20 mẫu validation, cách train 5 phiên gap (60..64)
    fold = TemporalSplit(
        "dev_1", "RQ1", "development", train_indices, val_indices, train_indices, val_indices,
        dates[0], dates[59], dates[65], dates[84], dates[65],
        0, 0, 5, 0, 5, {"0": 30, "1": 30}, {"0": 10, "1": 10},
    )
    return ModelingView(X, y, metadata), fold


def test_lstm_classifier_architecture():
    """Kiểm tra shape đầu ra, tính toán tham số và chế độ eval của LSTMClassifier."""
    model = LSTMClassifier(input_size=26, hidden_size=16, num_layers=1, dropout=0.2)
    assert model.count_trainable_parameters() > 0

    batch_x = torch.randn(8, 10, 26)
    logits = model(batch_x)
    assert logits.shape == (8,)

    # Chế độ eval
    model.eval()
    with torch.no_grad():
        out1 = model(batch_x)
        out2 = model(batch_x)
        torch.testing.assert_close(out1, out2)


def test_sequence_builder_train_only_standardization(synthetic_sample):
    """Kiểm tra StandardScaler chỉ fit trên tập Train và 100% mẫu Validation được giữ nguyên."""
    view, fold = synthetic_sample
    seq_len = 10

    X_train_seq, y_train_seq, X_val_seq, y_val_seq, scaler = build_fold_sequences(view, fold, seq_len)

    # 1. Số lượng chuỗi Train và Validation
    expected_train_seqs = len(fold.train_indices) - seq_len + 1  # 60 - 10 + 1 = 51
    assert X_train_seq.shape == (expected_train_seqs, seq_len, 26)
    assert len(y_train_seq) == expected_train_seqs

    # 100% mẫu Validation được giữ lại
    assert X_val_seq.shape == (len(fold.validation_indices), seq_len, 26)
    assert len(y_val_seq) == len(fold.validation_indices)

    # 2. Kiểm tra scaler chỉ fit trên Train
    expected_scaler = StandardScaler()
    expected_scaler.fit(view.X.iloc[list(fold.train_indices)].to_numpy(dtype=np.float32))
    np.testing.assert_allclose(scaler.mean_, expected_scaler.mean_, rtol=1e-5)
    np.testing.assert_allclose(scaler.scale_, expected_scaler.scale_, rtol=1e-5)


def test_sequence_builder_raises_on_invalid_bounds(synthetic_sample):
    """Kiểm tra ngoại lệ khi độ dài seq_len vượt quá kích thước dữ liệu."""
    view, fold = synthetic_sample
    with pytest.raises(ValueError, match="không đủ"):
        build_fold_sequences(view, fold, seq_len=100)

    # Kiểm tra khi val index không đủ lịch sử phía trước
    short_fold = replace(fold, validation_indices=(2,))  # Chỉ số 2 mà seq_len=10 thì không đủ
    with pytest.raises(ValueError, match="không có đủ"):
        build_fold_sequences(view, short_fold, seq_len=10)


def test_build_full_development_sequences(synthetic_sample):
    """Kiểm tra xây dựng chuỗi trên toàn bộ tập phát triển hợp lệ."""
    view, _ = synthetic_sample
    train_indices = tuple(range(50))
    seq_len = 10

    X_seq, y_seq, scaler = build_full_development_sequences(view, train_indices, seq_len)
    assert X_seq.shape == (50 - seq_len + 1, seq_len, 26)
    assert len(y_seq) == 50 - seq_len + 1

    expected_scaler = StandardScaler().fit(view.X.iloc[list(train_indices)])
    np.testing.assert_allclose(scaler.mean_, expected_scaler.mean_, rtol=1e-5)


def test_evaluate_fold_dummy_and_lstm(synthetic_sample):
    """Kiểm tra evaluate_fold cho cả Dummy và LSTM cấu hình nhỏ."""
    view, fold = synthetic_sample

    # Dummy
    dummy_cfg = MODEL_CONFIGS[0]
    rec_d, pred_d, logs_d, _, _ = evaluate_fold(view, "RQ1", fold, dummy_cfg)
    assert rec_d["model_name"] == "dummy"
    assert rec_d["balanced_accuracy"] == 0.5
    assert len(pred_d) == len(fold.validation_indices)
    assert (pred_d.score_type == "train_prior_probability").all()

    # LSTM mini config
    mini_lstm_cfg = LSTMModelConfig(
        config_id="lstm_test",
        model_name="lstm",
        seq_len=5,
        hidden_size=8,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=3,
        early_stopping_patience=2,
    )
    rec_l, pred_l, logs_l, feat_recs, lag_recs = evaluate_fold(
        view, "RQ1", fold, mini_lstm_cfg, compute_attribution=True
    )
    assert rec_l["model_name"] == "lstm"
    assert len(pred_l) == len(fold.validation_indices)
    assert (pred_l.score_type == "predicted_probability").all()
    assert len(logs_l) >= 1
    assert len(feat_recs) == 26
    assert len(lag_recs) == 5  # seq_len = 5


def test_reject_reserved_test_fold(synthetic_sample):
    """Bảo vệ an toàn: evaluate_fold từ chối fold reserved_test."""
    view, fold = synthetic_sample
    reserved_fold = replace(fold, role="reserved_test")
    with pytest.raises(ValueError, match="development"):
        evaluate_fold(view, "RQ1", reserved_fold, MODEL_CONFIGS[0])


def test_save_and_load_lstm_bundle(synthetic_sample):
    """Kiểm tra lưu trữ và tải lại model bundle (.pt)."""
    view, _ = synthetic_sample
    model = LSTMClassifier(input_size=26, hidden_size=16, num_layers=1, dropout=0.2)
    scaler = StandardScaler().fit(view.X.iloc[:30])

    bundle_data = {
        "model_state_dict": model.state_dict(),
        "network_config": {
            "input_size": 26,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.2,
        },
        "config_id": "test_cfg",
        "seq_len": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "class_weight": None,
        "final_epochs": 10,
        "fold_best_epochs": [10, 10, 10],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_schema": list(view.X.columns),
        "train_samples": 30,
        "train_sequences": 21,
        "rq": "RQ1",
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir) / "test_model.pt"
        torch.save(bundle_data, bundle_path)

        loaded_model, loaded_scaler, meta = load_final_lstm_bundle(bundle_path)
        assert meta["config_id"] == "test_cfg"
        np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_)

        model.eval()
        sample_x = torch.randn(2, 10, 26)
        with torch.no_grad():
            torch.testing.assert_close(model(sample_x), loaded_model(sample_x))
