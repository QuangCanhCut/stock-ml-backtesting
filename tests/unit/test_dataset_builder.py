"""Kiểm tra ghép dataset, chống leakage và lưu Parquet an toàn."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

import src.dataset_builder as builder
from src.dataset_builder import (LABEL_COLUMNS, assemble_dataset, get_training_view,
                                 load_project_config, load_raw_data, quality_report,
                                 save_processed_dataset, validate_dataset)


@pytest.fixture
def config():
    return {"data": {"raw_parquet_path": "data/raw.parquet", "processed_path": "data/out.parquet"},
            "features": {"return_lags": [1, 2], "momentum_windows": [2, 3],
                         "sma_windows": [2, 3], "ema_windows": [2, 3], "rsi_window": 3,
                         "atr_window": 3, "bollinger_window": 3, "bollinger_std": 2.,
                         "volume_zscore_window": 3},
            "eda": {"rolling_vol_windows": [2, 3]},
            "market_microstructure": {"trading_days_per_year": 252, "commission_buy": .001,
                                      "commission_sell": .001, "tax_sell": .001,
                                      "slippage": .001},
            "labeling": {"fixed_holding_sessions": 3}}


@pytest.fixture
def bars():
    close = np.array([10., 12., 11., 13., 12., 14., 15., 13., 16., 17.])
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                       "close": close, "volume": np.arange(10, 20)},
                      index=pd.date_range("2020-01-01", periods=len(close), freq="B", name="date"))
    df.columns.name = "Price"
    return df


def test_config_and_cwd(config, tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    path = root / "configs" / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_project_config(path) == config
    (root / "data").mkdir()
    with pytest.raises(FileNotFoundError): load_raw_data(config, root)
    with pytest.raises(FileNotFoundError): load_project_config(root / "missing.yaml")


@pytest.mark.parametrize("text", ["- not\n- a mapping", "data: {}", "[bad yaml"])
def test_invalid_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises((ValueError, yaml.YAMLError)): load_project_config(path)


def test_raw_load_and_sort(bars, config, tmp_path):
    path = tmp_path / config["data"]["raw_parquet_path"]
    path.parent.mkdir(parents=True)
    bars.iloc[::-1].to_parquet(path)
    assert_frame_equal(load_raw_data(config, tmp_path), bars, check_freq=False)


@pytest.mark.parametrize("case", ["extra", "duplicate", "missing", "invalid"])
def test_invalid_raw(bars, config, case):
    if case == "extra": bars["forward_return"] = 1
    elif case == "duplicate": bars.index = [bars.index[0]] * len(bars)
    elif case == "missing": bars = bars.drop(columns="open")
    elif case == "invalid": bars.loc[bars.index[0], "high"] = 1
    with pytest.raises(ValueError): assemble_dataset(bars, config)


def test_assembly_schema_missing_alignment(bars, config):
    original = bars.copy(deep=True)
    data, names = assemble_dataset(bars.iloc[::-1], config)
    assert_frame_equal(bars, original)
    assert data.index.equals(bars.index)
    assert len(data) == len(bars)
    assert list(data.columns) == list(builder.OHLCV) + names + LABEL_COLUMNS
    assert names == builder.expected_feature_columns(config)
    assert data["log_return"].isna().iloc[0]
    assert pd.isna(data.target_direction.iloc[-1])
    assert data.target_fixed_hold_net_proxy.iloc[-4:].isna().all()
    assert data.forward_trade_gross_return.iloc[0] == pytest.approx(12 / 12 - 1)
    assert data.forward_log_return_1.iloc[0] == pytest.approx(np.log(12 / 10))
    assert_frame_equal(data[list(builder.OHLCV)], bars)


@pytest.mark.parametrize("target", ["target_direction", "target_fixed_hold_net_proxy",
                                     "target_return_above_cost_proxy"])
def test_eligible_view(bars, config, target):
    data, names = assemble_dataset(bars, config)
    before = data.copy(deep=True)
    x, y = get_training_view(data, names, target)
    expected = data[names].notna().all(axis=1) & data[target].notna()
    assert x.index.equals(data.index[expected]) and x.index.equals(y.index)
    assert x.columns.tolist() == names
    assert not any(c.startswith(builder.FORBIDDEN) or c in builder.OHLCV for c in x.columns)
    assert str(y.dtype) == "Int8"
    assert_frame_equal(data, before)


def test_reject_leakage_and_wrong_schema(bars, config):
    data, names = assemble_dataset(bars, config)
    for bad in (names + ["forward_log_return_1"], names + ["target_direction"],
                names + ["next_return"], names + ["close"], names[:-1],
                names + ["unapproved_feature"], names + [names[0]]):
        with pytest.raises(ValueError): get_training_view(data, bad, "target_direction")
    with pytest.raises(ValueError): get_training_view(data, names, "forward_log_return_1")
    with pytest.raises(ValueError): validate_dataset(data.assign(fake=1), names)


def test_wrong_feature_output_and_label_index(bars, config, monkeypatch):
    original = builder.build_features
    monkeypatch.setattr(builder, "build_features", lambda d, c: original(d, c).assign(forward_fake=1))
    with pytest.raises(ValueError): assemble_dataset(bars, config)
    monkeypatch.setattr(builder, "build_features", original)
    labels = builder.build_labels
    monkeypatch.setattr(builder, "build_labels", lambda d, c: labels(d, c).iloc[:-1])
    with pytest.raises(ValueError): assemble_dataset(bars, config)


def test_duplicate_or_overlapping_outputs(bars, config, monkeypatch):
    make_labels = builder.build_labels

    def duplicate_labels(data, cfg):
        labels = make_labels(data, cfg)
        return pd.concat([labels, labels.iloc[:, :1]], axis=1)

    monkeypatch.setattr(builder, "build_labels", duplicate_labels)
    with pytest.raises(ValueError):
        assemble_dataset(bars, config)

    def overlapping_labels(data, cfg):
        labels = make_labels(data, cfg)
        labels["close"] = data["close"]
        return labels

    monkeypatch.setattr(builder, "build_labels", overlapping_labels)
    with pytest.raises(ValueError):
        assemble_dataset(bars, config)


def test_bad_dtypes_and_inf(bars, config):
    data, names = assemble_dataset(bars, config)
    modified = data.copy()
    modified.loc[modified.index[-1], names[0]] = np.inf
    with pytest.raises(ValueError): validate_dataset(modified, names)
    modified = data.copy()
    modified["target_direction"] = modified.target_direction.astype("float64")
    with pytest.raises(ValueError): validate_dataset(modified, names)
    modified = data.copy()
    modified.loc[modified.index[0], "target_direction"] = 2
    with pytest.raises(ValueError): validate_dataset(modified, names)


def test_parquet_roundtrip_reproducible(bars, config, tmp_path):
    data, names = assemble_dataset(bars, config)
    out = save_processed_dataset(data, config, tmp_path)
    read = pd.read_parquet(out)
    assert_frame_equal(data, read, check_exact=True, check_freq=False)
    assert read.attrs == data.attrs
    assert str(read.target_direction.dtype) == "Int8"
    x, y = get_training_view(read, names, "target_direction")
    assert len(x) == len(y)
    assert_frame_equal(data, assemble_dataset(bars, config)[0])


def test_failed_write_does_not_replace_output(bars, config, tmp_path, monkeypatch):
    data, _ = assemble_dataset(bars, config)
    output = save_processed_dataset(data, config, tmp_path)
    before = output.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("write failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError): save_processed_dataset(data, config, tmp_path)
    assert output.read_bytes() == before
    assert list(output.parent.glob(".dataset-*.parquet")) == []


def test_cannot_overwrite_raw(bars, config, tmp_path):
    config["data"]["processed_path"] = config["data"]["raw_parquet_path"]
    data, _ = assemble_dataset(bars, config)
    with pytest.raises(ValueError): save_processed_dataset(data, config, tmp_path)


def test_quality_report(bars, config):
    data, names = assemble_dataset(bars, config)
    report = quality_report(data, names)
    assert report["rows"] == len(bars)
    assert report["label_count"] == 6
    assert report["eligible_views"]["target_direction"]["eligible"] == len(get_training_view(data, names, "target_direction")[0])
    assert report["target_stats"]["target_direction"]["missing"] == 1
