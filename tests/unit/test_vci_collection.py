import hashlib
import json
import warnings

import pandas as pd

from src.data_loader import collect_vci


def test_snapshot_and_rate_spacing_without_network(tmp_path, monkeypatch):
    calls, sleeps = [], []

    class FakeQuote:
        def __init__(self, **kwargs):
            assert kwargs == {"symbol": "HPG", "source": "vci"}

        def history(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"time": [pd.Timestamp(kwargs["start"])], "open": [1.], "high": [2.],
                                 "low": [.5], "close": [1.], "volume": [10]})

    monkeypatch.setattr(collect_vci, "Quote", FakeQuote)
    monkeypatch.setattr(collect_vci.time, "sleep", sleeps.append)
    output = tmp_path / "snapshot"
    collect_vci.collect(output)
    assert len(calls) == 11 and sleeps == [15] * 10
    file = output / "HPG_VCI_2016-01-01_2017-01-01.parquet"
    meta = json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert meta["basis"] == "chưa xác minh"


def test_stop_when_provider_warns(tmp_path, monkeypatch):
    calls = []

    class FakeQuote:
        def __init__(self, **kwargs):
            pass

        def history(self, **kwargs):
            calls.append(kwargs)
            warnings.warn("Nguồn cảnh báo giới hạn")
            return pd.DataFrame({"time": [pd.Timestamp("2016-01-04")], "open": [1], "high": [2],
                                 "low": [1], "close": [1], "volume": [10]})

    monkeypatch.setattr(collect_vci, "Quote", FakeQuote)
    collect_vci.collect(tmp_path / "snapshot")
    assert len(calls) == 1
    assert len(list((tmp_path / "snapshot").glob("*.parquet"))) == 1
