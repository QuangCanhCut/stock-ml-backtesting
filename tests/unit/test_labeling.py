"""Kiểm tra công thức, phí và căn chỉnh thời gian của các nhãn."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.features import build_features
from src.labeling import (add_direction_labels, add_fixed_hold_trade_labels,
                          add_return_threshold_proxy, build_labels)


@pytest.fixture
def cfg():
    return {"labeling": {"fixed_holding_sessions": 3},
            "market_microstructure": {"commission_buy": 0.01,
                                      "commission_sell": 0.02,
                                      "tax_sell": 0.03, "slippage": 0.04}}


@pytest.fixture
def bars():
    opens = np.array([100., 110., 120., 130., 150., 90., 110.])
    closes = np.array([100., 110., 100., 100., 120., 110., 130.])
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 1,
                         "low": np.minimum(opens, closes) - 1, "close": closes,
                         "volume": np.arange(1., 8.)},
                        index=pd.date_range("2024-01-01", periods=7, freq="B"))


def test_direction_alignment_dtype_and_missing(bars):
    labels = add_direction_labels(bars)
    assert labels.forward_log_return_1.iloc[0] == pytest.approx(np.log(110 / 100))
    assert labels.target_direction.iloc[:4].tolist() == [1, 0, 0, 1]
    assert str(labels.target_direction.dtype) == "Int8"
    assert pd.isna(labels.forward_log_return_1.iloc[-1])
    assert pd.isna(labels.target_direction.iloc[-1])


def test_threshold_strict_and_configured(bars, cfg):
    # Ngưỡng bằng đúng log return được tính độc lập; phép so sánh là >.
    cfg["market_microstructure"].update(commission_buy=np.log(1.1), commission_sell=0., tax_sell=0., slippage=0.)
    labels = add_return_threshold_proxy(bars, cfg)
    assert labels.iloc[0, 0] == 0
    assert labels.iloc[3, 0] == 1
    assert labels.iloc[1, 0] == 0
    assert pd.isna(labels.iloc[-1, 0])
    assert str(labels.dtypes.iloc[0]) == "Int8"
    cfg["market_microstructure"]["commission_buy"] = 0.2
    assert add_return_threshold_proxy(bars, cfg).iloc[3, 0] == 0


def test_eda_uses_same_cost_threshold(bars, cfg):
    from src.eda_analyzer import compute_quant_features

    cfg["market_microstructure"].update(commission_buy=.01, commission_sell=.02,
                                          tax_sell=.01, slippage=.005)
    eda = compute_quant_features(bars, cfg)
    labels = add_return_threshold_proxy(bars, cfg)
    pd.testing.assert_series_equal(eda["target_net_profit"],
                                   labels.loc[eda.index, "target_return_above_cost_proxy"].astype(int),
                                   check_names=False)
    cfg["market_microstructure"]["commission_buy"] = .2
    changed = compute_quant_features(bars, cfg)
    assert changed.loc[bars.index[3], "target_net_profit"] == 0
    assert eda.loc[bars.index[3], "target_net_profit"] == 1


def test_economic_formula_and_signal_alignment(bars, cfg):
    out = add_fixed_hold_trade_labels(bars, cfg)
    # Tín hiệu tại hàng 0, mua 110 tại hàng 1, bán 150 tại hàng 4.
    expected = (150 * (1 - .04) * (1 - .02 - .03)) / (110 * (1 + .04) * (1 + .01)) - 1
    assert out.forward_trade_gross_return.iloc[0] == pytest.approx(150 / 110 - 1)
    assert out.forward_trade_net_return_proxy.iloc[0] == pytest.approx(expected)
    assert out.target_fixed_hold_net_proxy.iloc[0] == 1
    assert out.target_fixed_hold_net_proxy.iloc[1] == 0
    assert out.iloc[-4:].isna().all().all()
    assert str(out.target_fixed_hold_net_proxy.dtype) == "Int8"
    cfg["labeling"]["fixed_holding_sessions"] = 1
    out = add_fixed_hold_trade_labels(bars, cfg)
    assert out.forward_trade_gross_return.iloc[0] == pytest.approx(120 / 110 - 1)
    assert out.iloc[-2:].isna().all().all()


@pytest.mark.parametrize("fee", ["commission_buy", "commission_sell", "tax_sell", "slippage"])
def test_individual_costs(bars, cfg, fee):
    costs = cfg["market_microstructure"]
    for name in ("commission_buy", "commission_sell", "tax_sell", "slippage"):
        costs[name] = 0.0
    costs[fee] = .1
    net = add_fixed_hold_trade_labels(bars, cfg).forward_trade_net_return_proxy.iloc[0]
    gross_ratio = 150 / 110
    multiplier = {"commission_buy": 1 / 1.1, "commission_sell": .9,
                  "tax_sell": .9, "slippage": .9 / 1.1}[fee]
    assert net == pytest.approx(gross_ratio * multiplier - 1)


def test_zero_cost_equal_gross_and_zero_class(bars, cfg):
    for key in cfg["market_microstructure"]:
        cfg["market_microstructure"][key] = 0.
    bars.loc[:, "open"] = 100.
    bars.loc[:, "high"] = 200.
    bars.loc[:, "low"] = 1.
    out = add_fixed_hold_trade_labels(bars, cfg)
    pd.testing.assert_series_equal(out.forward_trade_gross_return,
                                   out.forward_trade_net_return_proxy, check_names=False)
    assert (out.target_fixed_hold_net_proxy.dropna() == 0).all()


def test_rejects_separate_round_trip_cost(bars, cfg):
    cfg["market_microstructure"]["round_trip_cost"] = .1
    with pytest.raises(ValueError, match="round_trip_cost"):
        build_labels(bars, cfg)


@pytest.mark.parametrize("length", [1, 2, 4])
def test_short_input_and_no_inf(bars, cfg, length):
    result = build_labels(bars.iloc[:length], cfg)
    assert len(result) == length
    assert result.target_fixed_hold_net_proxy.isna().all()
    assert result.target_direction.isna().sum() == 1
    assert not np.isinf(result.select_dtypes(include=[float]).to_numpy()).any()


def test_sorting_and_no_mutation(bars, cfg):
    source = bars.iloc[::-1].copy()
    original = source.copy(deep=True)
    result = build_labels(source, cfg)
    assert_frame_equal(source, original)
    assert result.index.equals(bars.index)
    assert_frame_equal(result, build_labels(bars, cfg))


@pytest.mark.parametrize("case", ["duplicate", "bad_price", "bad_high", "bad_volume", "missing"])
def test_invalid_input(bars, cfg, case):
    if case == "duplicate": bars.index = [bars.index[0]] * len(bars)
    elif case == "bad_price": bars.iloc[0, bars.columns.get_loc("close")] = 0
    elif case == "bad_high": bars.iloc[0, bars.columns.get_loc("high")] = 1
    elif case == "bad_volume": bars.iloc[0, bars.columns.get_loc("volume")] = -1
    elif case == "missing": bars = bars.drop(columns="open")
    with pytest.raises(ValueError): build_labels(bars, cfg)


@pytest.mark.parametrize("section,key,value", [
    ("labeling", "fixed_holding_sessions", 0), ("labeling", "fixed_holding_sessions", 1.5),
    ("labeling", "fixed_holding_sessions", True),
    ("market_microstructure", "commission_buy", -0.1),
    ("market_microstructure", "commission_sell", 1.0),
    ("market_microstructure", "tax_sell", np.nan),
    ("market_microstructure", "slippage", np.inf),
    ("market_microstructure", "slippage", "0.1")])
def test_invalid_config(bars, cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises(ValueError): build_labels(bars, cfg)


def test_sell_and_tax_sum_invalid(bars, cfg):
    cfg["market_microstructure"].update(commission_sell=.6, tax_sell=.4)
    with pytest.raises(ValueError, match="commission_sell"):
        build_labels(bars, cfg)


def test_prefix_labels_and_later_modification(bars, cfg):
    full = build_labels(bars, cfg)
    for length in (1, 3, 5, 6):
        partial = build_labels(bars.iloc[:length], cfg)
        assert_frame_equal(partial.iloc[:max(0, length-4)],
                           full.iloc[:max(0, length-4)])
        assert_frame_equal(partial.iloc[:max(0, length-1), :3],
                           full.iloc[:max(0, length-1), :3])
        assert partial.iloc[max(0, length-4):, 3:].isna().all().all()
        assert partial.iloc[-1, :3].isna().all()
    changed = bars.copy()
    changed.iloc[5:, :4] *= 2
    assert_frame_equal(build_labels(changed, cfg).iloc[:1], full.iloc[:1])


def test_features_remain_separate(bars, cfg):
    bars.columns.name = "Price"
    feature_cfg = deepcopy(cfg)
    feature_cfg.update({"features": {"return_lags": [1], "momentum_windows": [2],
                       "sma_windows": [2], "ema_windows": [2, 3], "rsi_window": 2,
                       "atr_window": 2, "bollinger_window": 2, "bollinger_std": 2,
                       "volume_zscore_window": 2}, "eda": {"rolling_vol_windows": [2]}})
    feature_cfg["market_microstructure"]["trading_days_per_year"] = 252
    features = build_features(bars, feature_cfg)
    original = features.copy(deep=True)
    labels = build_labels(bars, cfg)
    combined = features.join(labels)
    assert_frame_equal(features, original)
    assert_frame_equal(combined[features.columns], features)
    assert not set(features.columns) & set(labels.columns)
    assert all(c.startswith(("forward_", "target_")) for c in labels.columns)
    with pytest.raises(ValueError): build_features(combined, feature_cfg)
