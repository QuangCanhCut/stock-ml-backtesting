import pandas as pd
import pytest

from src.data_loader.adapters import canonicalize
from src.data_loader.alignment import diagnostic_2025
from src.data_loader.dates import normalize_dates
from src.data_loader.export import export_dataset, write_reports
from src.data_loader.reconcile import reconcile, volume_diagnostics
from src.data_loader.validation import validate


def bars(dates, volumes=None):
    return pd.DataFrame({"date": pd.to_datetime(dates), "open": 10., "high": 11.,
                         "low": 9., "close": 10., "volume": volumes or [100] * len(dates)})


def test_normalization_keeps_local_calendar_date():
    values = pd.Series(["2025-01-02T00:30:00+07:00", "2025-01-03T09:00:00+07:00"])
    assert normalize_dates(values).tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]


def test_duplicate_dates_fail_after_normalization():
    frame = bars(["2025-01-02 01:00", "2025-01-02 15:00"])
    with pytest.raises(ValueError, match="trùng"):
        canonicalize(frame, "date")


def test_validation_reports_errors_without_repair():
    frame = bars(["2025-01-01", "2025-01-02", "2025-01-03"], [100, 0, 100])
    frame.loc[2, "high"] = 8
    original = frame.copy(deep=True)
    result = validate(frame)
    assert not result.iloc[0].any()
    assert result.loc[1, "nonpositive_volume"]
    assert result.loc[2, "invalid_ohlc_geometry"]
    pd.testing.assert_frame_equal(frame, original)


def test_validation_rejects_missing_or_nonfinite_values():
    frame = bars(["2025-01-01", "2025-01-02"])
    frame["volume"] = frame["volume"].astype(float)
    frame.loc[0, "close"] = float("nan")
    frame.loc[1, "volume"] = float("inf")
    result = validate(frame)
    assert result.nonfinite_or_missing.tolist() == [True, True]


def test_2025_previous_session_uses_full_kbs_calendar():
    kbs = bars(["2025-05-05", "2025-05-06", "2025-05-07"], [100, 200, 300])
    kbs["close"] = [10., 20., 30.]
    raw = bars(["2025-05-05", "2025-05-07"], [100, 264])
    raw.loc[1, "close"] = 20000.
    result = diagnostic_2025(raw, kbs)
    assert result.iloc[0].prev_kbs_date == pd.Timestamp("2025-05-06")
    assert result.iloc[0].alignment == "PREVIOUS_SESSION"


def test_raw_without_kbs_is_preserved_and_unverified():
    raw = bars(["2016-01-04"])
    kbs = bars(["2016-05-25"])
    evidence = pd.DataFrame(columns=["date", "present_in_yahoo", *["open", "high", "low", "close", "volume"]])
    alignment = pd.DataFrame(columns=["date", "alignment"])
    result = reconcile(raw, kbs, evidence, alignment)
    row = result.loc[result.source == "raw_yahoo_archive"].iloc[0]
    assert row.decision == "retain_unverified"
    assert row.verification_status == "unverified_no_kbs_session"
    assert row.close == 10.


def test_flat_bar_is_held_for_review_even_with_positive_volume():
    raw = bars(["2021-01-04"])
    raw.loc[0, ["open", "high", "low", "close"]] = 10.
    kbs = bars(["2021-01-04"])
    evidence = pd.DataFrame(columns=["date", "present_in_yahoo", "open", "high", "low", "close", "volume"])
    result = reconcile(raw, kbs, evidence, pd.DataFrame(columns=["date", "alignment"]))
    row = result.loc[result.source == "raw_yahoo_archive"].iloc[0]
    assert row.decision == "hold_review"
    assert "flat_ohlc_possible_placeholder" in row.anomaly_types


def test_exports_are_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="raw or processed"):
        write_reports(tmp_path / "data/raw/new", tmp_path, {}, {})
    dest = tmp_path / "reports"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        write_reports(dest, tmp_path, {}, {})


def test_volume_diagnostics_both_directions_use_only_past_values():
    dates = pd.bdate_range("2021-01-01", periods=15)
    raw = bars(dates, [1000] * 12 + [100000, 1, 1000])
    kbs = bars(dates, [1000] * 15)
    flags = volume_diagnostics(raw, kbs)
    assert "volume_source_high_exploratory" in flags[dates[12]]
    assert "volume_source_low_exploratory" in flags[dates[13]]
    assert "volume_cross_source_ratio_low_exploratory" in flags[dates[13]]
    raw.loc[14, "volume"] = 99999999
    assert volume_diagnostics(raw, kbs)[dates[13]] == flags[dates[13]]
