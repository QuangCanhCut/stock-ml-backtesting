"""Đọc snapshot hiện có; các adapter này không gọi API."""

from pathlib import Path

import pandas as pd

from .dates import normalize_dates

OHLCV = ["open", "high", "low", "close", "volume"]


def canonicalize(frame, date_column):
    required = {date_column, *OHLCV}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột nguồn: {sorted(missing)}")
    result = frame[[date_column, *OHLCV]].copy().rename(columns={date_column: "date"})
    result["date"] = normalize_dates(result["date"])
    if result["date"].duplicated().any():
        raise ValueError("Ngày nguồn bị trùng; không thể reconciliation an toàn")
    return result.sort_values("date").reset_index(drop=True)


def read_raw(path):
    return canonicalize(pd.read_csv(path), "date")


def read_kbs_cache(directory, years):
    parts, absent = [], []
    for year in years:
        path = Path(directory) / f"HPG_KBS_{year}.parquet"
        if not path.exists():
            absent.append(year)
            continue
        part = canonicalize(pd.read_parquet(path), "time")
        if part.empty or not (part.date.dt.year == year).all():
            raise ValueError(f"Cache KBS rỗng hoặc không hợp lệ cho năm {year}: {path}")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", *OHLCV])
    result["date"] = pd.to_datetime(result["date"])
    if result.date.duplicated().any():
        raise ValueError("Ngày KBS bị trùng giữa các file cache")
    return result, absent


def read_yahoo_rejection_evidence(path):
    """Snapshot chỉ chứa các ngày thiếu đã audit, không phải toàn bộ Yahoo."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["date", *OHLCV, "present_in_yahoo", "reason"])
    evidence = pd.read_csv(path)
    evidence["date"] = normalize_dates(evidence["date"])
    if evidence.date.duplicated().any():
        raise ValueError("Ngày bằng chứng Yahoo bị trùng")
    return evidence
