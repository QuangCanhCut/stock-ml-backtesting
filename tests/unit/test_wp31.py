"""Kiểm tra view và kiểm toán WP3.1 bằng dữ liệu tổng hợp xác định."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.dataset_builder import assemble_dataset
from src.models.data import make_modeling_view
from src.models.data_validation import audit_dataset


@pytest.fixture
def sample():
    """Tạo giá và cấu hình nhỏ, đủ dài để kiểm tra tiền tố nhân quả."""
    config = {
        "features": {"return_lags": [1], "momentum_windows": [2],
                     "sma_windows": [2], "ema_windows": [2, 3], "rsi_window": 3,
                     "atr_window": 3, "bollinger_window": 3,
                     "bollinger_std": 2.0, "volume_zscore_window": 3},
        "eda": {"rolling_vol_windows": [3]},
        "market_microstructure": {"trading_days_per_year": 252,
                                  "commission_buy": .0015, "commission_sell": .0015,
                                  "tax_sell": .001, "slippage": .0005},
        "labeling": {"fixed_holding_sessions": 3},
    }
    n = 240
    close = 100 + np.arange(n) * .08 + np.sin(np.arange(n) * .7)
    bars = pd.DataFrame({"open": close * (1 + .002 * np.cos(np.arange(n))),
                         "high": close + 2, "low": close - 2,
                         "close": close, "volume": 1000 + np.arange(n) * 3},
                        index=pd.date_range("2020-01-01", periods=n, freq="B"))
    data, _ = assemble_dataset(bars, config)
    return data, config


def test_audit_and_independent_views(sample):
    data, config = sample
    original = data.copy(deep=True)
    report = audit_dataset(data, config)
    rq1 = make_modeling_view(data, config, "RQ1")
    rq2 = make_modeling_view(data, config, "RQ2")
    assert report["views"]["RQ1"]["eligible"] == len(rq1.y)
    assert report["views"]["RQ2"]["eligible"] == len(rq2.y)
    assert rq1.X.index.equals(rq1.y.index)
    assert rq2.X.index.equals(rq2.metadata.index)
    assert rq1.metadata.label_end_date.iloc[-1] == data.index[-1]
    assert rq2.metadata.label_end_date.iloc[-1] == data.index[-1]
    signal = data.index[20]
    assert rq1.metadata.loc[signal, "label_end_date"] == data.index[21]
    assert rq2.metadata.loc[signal, "label_end_date"] == data.index[24]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("column", ["forward_log_return_1", "forward_trade_net_return_proxy",
                                      "target_direction", "target_fixed_hold_net_proxy"])
def test_reject_temporal_corruption(sample, column):
    data, config = sample
    corrupt = data.copy(deep=True)
    index = corrupt.index[20]
    if column.startswith("target_"):
        corrupt.loc[index, column] = 1 - int(corrupt.loc[index, column])
    else:
        corrupt.loc[index, column] += .01
    with pytest.raises(ValueError, match=column):
        audit_dataset(corrupt, config)


@pytest.mark.parametrize("fault", ["missing_feature", "future_feature", "duplicate_date",
                                        "unsorted", "inf", "text", "invalid_class"])
def test_reject_invalid_dataset(sample, fault):
    data, config = sample
    bad = data.copy(deep=True)
    feature = bad.attrs["feature_columns"][0]
    if fault == "missing_feature":
        bad = bad.drop(columns=feature)
    elif fault == "future_feature":
        bad.attrs = deepcopy(data.attrs)
        bad.attrs["feature_columns"][0] = "forward_log_return_1"
    elif fault == "duplicate_date":
        dates = bad.index.to_list()
        dates[1] = dates[0]
        bad.index = pd.DatetimeIndex(dates)
    elif fault == "unsorted":
        bad = bad.iloc[::-1]
    elif fault == "inf":
        bad.loc[bad.index[20], feature] = np.inf
    elif fault == "text":
        bad[feature] = "không hợp lệ"
    else:
        bad.loc[bad.index[20], "target_direction"] = 2
    message = "chứa Inf" if fault == "inf" else None
    with pytest.raises(ValueError, match=message):
        audit_dataset(bad, config)


def test_warmup_and_future_missing_are_separate(sample):
    data, config = sample
    report = audit_dataset(data, config)
    for rq in ("RQ1", "RQ2"):
        view = report["views"][rq]
        assert view["excluded"] == (view["excluded_missing_features"]
                                    + view["excluded_missing_target_with_features"])
    assert pd.isna(data.target_direction.iloc[-1])
    assert data.target_fixed_hold_net_proxy.iloc[-4:].isna().all()


def test_reject_feature_corruption(sample):
    data, config = sample
    bad = data.copy(deep=True)
    bad.loc[bad.index[20], "log_return"] += .01
    with pytest.raises(ValueError, match="Feature khác"):
        audit_dataset(bad, config)


def test_reject_index_mismatch(sample, monkeypatch):
    import src.models.data as module

    data, config = sample
    original = module.get_training_view

    def wrong_index(*args):
        X, y = original(*args)
        y.index = y.index.shift(1, freq="B")
        return X, y

    monkeypatch.setattr(module, "get_training_view", wrong_index)
    with pytest.raises(ValueError, match="Index"):
        make_modeling_view(data, config, "RQ1")
