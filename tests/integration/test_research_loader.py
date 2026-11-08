"""Kiểm thử research profile bằng snapshot giả lập, không dùng làm chứng cứ HPG."""

import hashlib
import json

import pandas as pd
import pytest

from src.data_loader import research


def prepare(root, monkeypatch):
    snapshot = root / "snapshots"
    snapshot.mkdir()
    (root / "data/raw").mkdir(parents=True)
    early = list(pd.bdate_range("2016-01-04", periods=94))
    valid_dates = early + [pd.Timestamp("2016-05-25")] + [pd.Timestamp(f"{y}-02-01") for y in range(2017, 2027)]
    for year in range(2016, 2027):
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01" if year < 2026 else "2026-11-01"
        dates = [date for date in valid_dates if date.year == year]
        if year == 2018:
            dates += [pd.Timestamp("2018-01-23"), pd.Timestamp("2018-01-24")]
        df = pd.DataFrame({"time": dates, "open": 10., "high": 11., "low": 9., "close": 10., "volume": 100})
        for day in ("2018-01-23", "2018-01-24"):
            mask = df.time.eq(pd.Timestamp(day))
            df.loc[mask, ["open", "high", "low", "close"]] = 10.
            df.loc[mask, "volume"] = 0
        path = snapshot / f"HPG_VCI_{start}_{end}.parquet"
        df.to_parquet(path, index=False)
        meta = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "warnings": [], "symbol": "HPG",
                "start_requested": start, "end_requested": end, "source_attrs": {"source": "vci"},
                "rows": len(df), "retrieved_at_utc": "2026-01-01T00:00:00+00:00", "vnstock_version": "test"}
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    dates = pd.Series(valid_dates)
    monkeypatch.setattr(research, "read_kbs_cache", lambda *_: (pd.DataFrame({"date": dates[dates.ge("2016-05-25")]}), []))
    monkeypatch.setattr(research, "read_raw", lambda *_: pd.DataFrame({"date": dates}))
    return snapshot


def test_research_profile_writes_new_file_and_preserves_volume(tmp_path, monkeypatch):
    snapshot = prepare(tmp_path, monkeypatch)
    target, report = research.export_research(tmp_path, snapshot, expected_sessions=105)
    result = pd.read_parquet(target)
    assert len(result) == 105 and result.close.eq(10000).all() and result.volume.eq(100).all()
    assert report["gate"] == "PASS_WITH_LIMITATIONS" and len(report["excluded"]) == 2
    with pytest.raises(FileExistsError):
        research.export_research(tmp_path, snapshot, expected_sessions=105)
    assert len(pd.read_parquet(target)) == 105


def test_research_profile_rejects_tampered_snapshot(tmp_path, monkeypatch):
    snapshot = prepare(tmp_path, monkeypatch)
    path = snapshot / "HPG_VCI_2018-01-01_2019-01-01.parquet"
    data = pd.read_parquet(path)
    data.loc[0, "volume"] = 0
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="hash"):
        research.export_research(tmp_path, snapshot, expected_sessions=105)
    assert not (tmp_path / "data/raw/HPG_10y_daily_research.parquet").exists()
