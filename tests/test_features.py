"""Kiểm tra công thức độc lập và tính nhân quả của các feature tài chính."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.features import build_features, validate_ohlcv


@pytest.fixture
def cfg():
    return {"features": {"return_lags": [1, 2], "momentum_windows": [2, 3],
                         "sma_windows": [2, 3], "ema_windows": [2, 3],
                         "rsi_window": 3, "atr_window": 3,
                         "bollinger_window": 3, "bollinger_std": 2.0,
                         "volume_zscore_window": 3},
            "eda": {"rolling_vol_windows": [2, 3]},
            "market_microstructure": {"trading_days_per_year": 252}}


@pytest.fixture
def prices():
    close = [10., 12., 11., 13., 12., 14., 15., 13., 16., 17.]
    return pd.DataFrame({"open": close, "high": np.array(close) + 1,
                         "low": np.array(close) - 1, "close": close,
                         "volume": [10., 20., 30., 40., 50., 40., 30., 20., 10., 20.]},
                        index=pd.date_range("2020-01-01", periods=len(close)))


@pytest.mark.parametrize("case", ["missing", "duplicate", "zero", "negative", "high", "low",
                                   "volume", "nan", "inf", "index", "nat", "nonnumeric", "duplicate_column"])
def test_validation_rejects_invalid(prices, case):
    df = prices.copy()
    if case == "missing": df = df.drop(columns="open")
    elif case == "duplicate": df.index = [df.index[0]] * len(df)
    elif case == "zero": df.iloc[0, df.columns.get_loc("close")] = 0
    elif case == "negative": df.iloc[0, df.columns.get_loc("open")] = -1
    elif case == "high": df.iloc[0, df.columns.get_loc("high")] = 9
    elif case == "low": df.iloc[0, df.columns.get_loc("low")] = 11
    elif case == "volume": df.iloc[0, df.columns.get_loc("volume")] = -1
    elif case == "nan": df.iloc[0, df.columns.get_loc("close")] = np.nan
    elif case == "inf": df.iloc[0, df.columns.get_loc("volume")] = np.inf
    elif case == "index": df.index = range(len(df))
    elif case == "nat": df.index = pd.DatetimeIndex([pd.NaT] + list(df.index[1:]))
    elif case == "nonnumeric": df["volume"] = "1"
    elif case == "duplicate_column": df.columns = ["open", "high", "low", "close", "close"]
    with pytest.raises(ValueError): validate_ohlcv(df)


def test_non_dataframe():
    with pytest.raises(TypeError): validate_ohlcv([])


def test_sort_and_no_input_mutation(prices, cfg):
    unsorted = prices.iloc[::-1].copy()
    original = unsorted.copy(deep=True)
    out = build_features(unsorted, cfg)
    assert_frame_equal(unsorted, original)
    assert out.index.equals(prices.index)
    assert len(out) == len(prices)


def test_returns_momentum_and_trend(prices, cfg):
    out = build_features(prices, cfg)
    assert out.log_return.iloc[2] == pytest.approx(np.log(11 / 12))
    assert out.return_lag_1.iloc[2] == pytest.approx(np.log(12 / 10))
    assert out.return_lag_2.iloc[3] == pytest.approx(np.log(12 / 10))
    assert out.momentum_3.iloc[3] == pytest.approx(np.log(13 / 10))
    assert out.sma_ratio_3.iloc[2] == pytest.approx(11 / 11 - 1)
    # EMA với adjust=False: cửa sổ 2 có alpha=2/3, cửa sổ 3 có alpha=1/2.
    fast = (2 / 3) * 11 + (1 / 3) * ((2 / 3) * 12 + (1 / 3) * 10)
    slow = 0.5 * 11 + 0.5 * (0.5 * 12 + 0.5 * 10)
    assert out.ema_ratio_2.iloc[2] == pytest.approx(11 / fast - 1)
    assert out.macd_ratio.iloc[2] == pytest.approx((fast - slow) / 11)


def test_rsi_wilder_and_atr(prices, cfg):
    out = build_features(prices, cfg)
    # Biến động +2, -1, +2: tăng trung bình 4/3, giảm 1/3; tiếp theo là -1.
    assert np.isnan(out.rsi_3.iloc[2])
    assert out.rsi_3.iloc[3] == pytest.approx(80)
    assert out.rsi_3.iloc[4] == pytest.approx(100 * (8 / 9) / (8 / 9 + 5 / 9))
    # Các TR đầu là 2, 3, 2; TR tiếp theo là 3. Khởi tạo và cập nhật theo Wilder.
    assert out.atr_ratio_3.iloc[2] == pytest.approx((7 / 3) / 11)
    assert out.atr_ratio_3.iloc[3] == pytest.approx(((7 / 3 * 2 + 3) / 3) / 13)


def test_bollinger_volatility_volume(prices, cfg):
    out = build_features(prices, cfg)
    sd = np.sqrt(((10 - 11)**2 + (12 - 11)**2 + (11 - 11)**2) / 3)
    assert out.bollinger_width_3.iloc[2] == pytest.approx(4 * sd / 11)
    assert out.bollinger_percent_b_3.iloc[2] == pytest.approx(0.5)
    r1, r2 = np.log(12 / 10), np.log(11 / 12)
    assert out.volatility_2.iloc[2] == pytest.approx(abs(r1-r2) / np.sqrt(2) * np.sqrt(252))
    assert out.volume_zscore_3.iloc[2] == pytest.approx(1.0)
    assert out.volume_change_1.iloc[2] == pytest.approx(0.5)
    assert out.volume_ratio_3.iloc[2] == pytest.approx(1.5)


def test_flat_and_zero_volume(prices, cfg):
    df = prices.copy()
    df[["open", "high", "low", "close"]] = 10.
    df["volume"] = 0.
    out = build_features(df, cfg)
    assert out.rsi_3.iloc[3] == 50
    assert out.atr_ratio_3.iloc[2] == 0
    assert out.bollinger_width_3.iloc[2] == 0
    assert out.bollinger_percent_b_3.isna().all()
    assert out.volume_zscore_3.isna().all()
    assert out.volume_change_1.isna().all()
    assert out.volume_ratio_3.isna().all()
    assert not np.isinf(out.to_numpy()).any()


@pytest.mark.parametrize("direction,expected", [(1, 100), (-1, 0)])
def test_rsi_one_sided(prices, cfg, direction, expected):
    df = prices.copy()
    close = 20 + direction * np.arange(len(df))
    df["open"] = df["close"] = close
    df["high"] = close + 1
    df["low"] = close - 1
    assert build_features(df, cfg).rsi_3.iloc[3] == expected


def test_short_data_warmup(prices, cfg):
    out = build_features(prices.iloc[:2], cfg)
    assert len(out) == 2
    assert out.rsi_3.isna().all() and out.sma_ratio_3.isna().all()
    assert np.isnan(out.log_return.iloc[0])
    assert not np.isinf(out.to_numpy()).any()


def test_prefix_and_future_modification(prices, cfg):
    full = build_features(prices, cfg)
    for cutoff in (2, 5, 8):
        prefix = build_features(prices.iloc[:cutoff + 1], cfg)
        assert_frame_equal(full.iloc[:cutoff + 1], prefix, check_exact=False, rtol=1e-12, atol=1e-12)
        changed = prices.copy()
        changed.iloc[cutoff + 1:, :4] *= 1.5
        changed.iloc[cutoff + 1:, 4] *= 3
        assert_frame_equal(full.iloc[:cutoff + 1], build_features(changed, cfg).iloc[:cutoff + 1])
    assert not any(c.startswith(("target_", "next_", "future_")) for c in full.columns)


def test_config_changes_features(prices, cfg):
    changed = deepcopy(cfg)
    changed["features"].update(sma_windows=[4], ema_windows=[3, 4], rsi_window=4)
    out = build_features(prices, changed)
    assert {"sma_ratio_4", "ema_ratio_3", "ema_ratio_4", "rsi_4"} <= set(out.columns)
    assert out.rsi_4.iloc[4] != pytest.approx(build_features(prices, cfg).rsi_3.iloc[4])
    assert out.sma_ratio_4.iloc[4] == pytest.approx(12 / ((12+11+13+12)/4) - 1)


@pytest.mark.parametrize("section,key,value", [
    ("features", "return_lags", [0]), ("features", "return_lags", [1, 1]),
    ("features", "momentum_windows", []), ("features", "sma_windows", [True]),
    ("features", "ema_windows", [3, 2]), ("features", "ema_windows", [2]),
    ("features", "rsi_window", 0), ("features", "atr_window", -1),
    ("features", "bollinger_std", np.nan), ("features", "bollinger_window", 1),
    ("features", "volume_zscore_window", 1), ("eda", "rolling_vol_windows", [1]),
    ("market_microstructure", "trading_days_per_year", 0)])
def test_invalid_config(prices, cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises(ValueError): build_features(prices, cfg)


def test_rejects_future_labels_and_collisions(prices, cfg):
    for name in ("target_direction", "next_log_return", "future_return", "log_return"):
        df = prices.assign(**{name: 1})
        with pytest.raises(ValueError): build_features(df, cfg)


def test_exact_default_feature_names(prices):
    config = {"features": {"return_lags": [1, 2, 3, 5, 10, 21],
                           "sma_windows": [10, 20, 50, 200], "ema_windows": [12, 26],
                           "rsi_window": 14, "atr_window": 14, "bollinger_window": 20,
                           "bollinger_std": 2., "volume_zscore_window": 20},
              "eda": {"rolling_vol_windows": [21, 63]},
              "market_microstructure": {"trading_days_per_year": 252}}
    expected = {"log_return", "macd_ratio", "rsi_14", "atr_ratio_14",
                "bollinger_width_20", "bollinger_percent_b_20", "volume_change_1",
                "volume_ratio_20", "volume_zscore_20"}
    expected |= {f"return_lag_{k}" for k in [1, 2, 3, 5, 10, 21]}
    expected |= {f"momentum_{k}" for k in [5, 10, 21]}
    expected |= {f"sma_ratio_{k}" for k in [10, 20, 50, 200]}
    expected |= {f"ema_ratio_{k}" for k in [12, 26]}
    expected |= {f"volatility_{k}" for k in [21, 63]}
    assert set(build_features(prices, config).columns) - set(prices.columns) == expected


def test_zero_previous_volume_and_unclipped_bollinger(prices, cfg):
    df = prices.copy()
    df.iloc[1, df.columns.get_loc("volume")] = 0
    cfg["features"]["bollinger_std"] = 0.1
    out = build_features(df, cfg)
    assert pd.isna(out.volume_change_1.iloc[2])
    assert out.bollinger_percent_b_3.iloc[3] > 1
    assert not np.isinf(out.to_numpy()).any()
