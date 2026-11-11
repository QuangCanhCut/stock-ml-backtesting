"""Xây Raw Research từ snapshot VCI, không thay đổi profile validated."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .adapters import canonicalize, read_kbs_cache, read_raw
from .validation import validate

PRICE_DOCUMENTATION = "https://vnstocks.com/docs/vnstock/thi-truong-market"
VOLUME_DOCUMENTATION = "https://vnstocks.com/docs/vnstock/vnstock-data-v3/cau-truc-du-lieu/market"
EXCLUDED = {"2018-01-23", "2018-01-24"}


def assess_research(root, snapshot_dir=None, expected_sessions=2702):
    """Trả Raw và báo cáo nếu qua research gate; không ghi file."""
    root = Path(root).resolve()
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else root / "results/data_audit/vci_annual_20261108"
    snapshots, source_metadata = [], []
    for year in range(2016, 2027):
        start = f"{year}-01-01"
        end = f"{year + 1}-01-01" if year < 2026 else "2026-11-01"
        path = snapshot_dir / f"HPG_VCI_{start}_{end}.parquet"
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]
                or metadata.get("warnings") or metadata.get("symbol") != "HPG"
                or metadata.get("start_requested") != start or metadata.get("end_requested") != end
                or metadata.get("source_attrs", {}).get("source") != "vci"):
            raise ValueError(f"Snapshot {year} không khớp hash, nguồn, khoảng hoặc có cảnh báo")
        df = canonicalize(pd.read_parquet(path), "time")
        if len(df) != metadata["rows"] or df.empty or not df.date.between(start, end, inclusive="left").all():
            raise ValueError(f"Snapshot {year} có số dòng hoặc ngày không hợp lệ")
        snapshots.append(df)
        source_metadata.append({"year": year, "rows": len(df), "first": str(df.date.min().date()),
                                "last": str(df.date.max().date()), "sha256": metadata["sha256"],
                                "retrieved_at_utc": metadata["retrieved_at_utc"], "vnstock_version": metadata["vnstock_version"]})
    if len({entry["vnstock_version"] for entry in source_metadata}) != 1:
        raise ValueError("Snapshot VCI có nhiều phiên bản vnstock; cần kiểm tra basis")
    all_rows = pd.concat(snapshots, ignore_index=True).sort_values("date").reset_index(drop=True)
    if all_rows.date.duplicated().any():
        raise ValueError("Ngày bị trùng giữa các snapshot VCI")
    flags = validate(all_rows)
    invalid = all_rows.loc[flags.any(axis=1)].copy()
    if set(invalid.date.dt.strftime("%Y-%m-%d")) != EXCLUDED:
        raise ValueError("Có OHLCV không hợp lệ ngoài hai ngày đã được ghi nhận")
    if not ((invalid.volume == 0) & invalid[["open", "high", "low", "close"]].nunique(axis=1).eq(1)).all():
        raise ValueError("Hai ngày bị loại không khớp điều kiện placeholder đã ghi nhận")
    kbs, absent = read_kbs_cache(root / "results/data_audit/kbs_cache", range(2016, 2027))
    if absent or set(invalid.date) & set(kbs.date):
        raise ValueError("Thiếu cache KBS hoặc ngày volume bằng 0 xuất hiện trong KBS")
    raw = all_rows.loc[~flags.any(axis=1)].copy().reset_index(drop=True)
    if len(raw) != expected_sessions or not raw.date.is_monotonic_increasing or raw.date.duplicated().any():
        raise ValueError(f"Số phiên hoặc thứ tự không đạt: {len(raw)}")
    if set(kbs.date) - set(raw.date):
        raise ValueError("VCI thiếu ngày có trong KBS; cần kiểm tra lịch trước khi build")
    vci_only = raw.loc[~raw.date.isin(kbs.date)]
    prior = read_raw(root / "data/raw/HPG_10y_daily.csv")
    if len(vci_only) != 94 or not vci_only.date.lt(pd.Timestamp("2016-05-25")).all() or not set(vci_only.date).issubset(set(prior.date)):
        raise ValueError("Calendar VCI có ngày ngoài vùng KBS chưa được đối chiếu")
    returns = raw.close.pct_change()
    volume_baseline = raw.volume.shift(1).rolling(30, min_periods=10).median()
    volume_ratio = raw.volume / volume_baseline
    price_outliers = raw.loc[returns.abs().gt(.12), "date"]
    volume_outliers = raw.loc[volume_ratio.gt(10) | volume_ratio.lt(.1), "date"]
    if not price_outliers.empty or not volume_outliers.empty:
        raise ValueError(f"Có outlier cần kiểm tra: price={len(price_outliers)}, volume={len(volume_outliers)}")
    excluded = [{"date": str(row.date.date()), "reason": "VCI volume=0, OHLC phẳng; không có trong Raw hoặc KBS"}
                for row in invalid.itertuples()]
    if set(invalid.date) & set(prior.date):
        raise ValueError("Ngày dự kiến loại có trong Raw; cần kiểm tra")
    for col in ("open", "high", "low", "close"):
        raw[col] = pd.to_numeric(raw[col]) * 1000
    output = raw.set_index("date")[["open", "high", "low", "close", "volume"]]
    if validate(output.reset_index()).any(axis=None):
        raise ValueError("OHLCV sau chuyển đơn vị không hợp lệ")
    report = {"profile": "research", "gate": "PASS_WITH_LIMITATIONS", "source": "VCI qua vnstock 5.2.2",
              "sessions": len(output), "first": str(output.index.min().date()), "last": str(output.index.max().date()),
              "yearly_sources": source_metadata, "excluded": excluded,
              "price_unit": "VND; giá VCI theo nghìn đồng, OHLC nhân 1000", "price_unit_reference": PRICE_DOCUMENTATION,
              "volume": "Theo VCI; không áp dụng hệ số Yahoo/KBS", "volume_reference": VOLUME_DOCUMENTATION,
              "price_adjustment": "VCI có điều chỉnh theo sự kiện quyền; công thức và lịch sử hệ số chưa xác minh",
              "calendar": "Khớp ngày KBS trong phạm vi KBS có dữ liệu; 94 phiên đầu 2016 chỉ đối chiếu ngày Raw",
              "t_plus_1_risk": "Hai ngày bị loại không có trong KBS/Raw nhưng chưa đối chiếu lịch HOSE chính thức; T+1 có thể không phải phiên kế tiếp nếu calendar thiếu",
              "point_in_time": "Chưa chứng minh điều chỉnh point-in-time; không phù hợp kết luận backtest thực thi",
              "outlier_checks": "|return| > 12% hoặc volume so với median 30 phiên trước >10 lần hoặc <0,1 lần: không có",
              "limitations": ["Không phải dataset validated toàn diện", "Calendar chính thức và 94 phiên đầu 2016 chưa xác minh độc lập",
                              "Corporate actions và điều chỉnh point-in-time chưa xác minh", "Không dùng giá adjusted làm giá khớp lệnh"]}
    return output, report


def export_research(root, snapshot_dir=None, expected_sessions=2702):
    raw, report = assess_research(root, snapshot_dir, expected_sessions)
    directory = Path(root) / "data/raw"
    target = directory / "HPG_10y_daily_research.parquet"
    sidecar = directory / "HPG_10y_daily_research.provenance.json"
    tmp = None
    created_report = False
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".research-", suffix=".parquet", delete=False) as stream:
            tmp = Path(stream.name)
            raw.to_parquet(stream)
        pd.testing.assert_frame_equal(raw, pd.read_parquet(tmp), check_exact=True, check_freq=False)
        report["dataset_sha256"] = hashlib.sha256(tmp.read_bytes()).hexdigest()
        with sidecar.open("x", encoding="utf-8") as stream:
            created_report = True
            json.dump(report, stream, ensure_ascii=False, indent=2)
        try:
            os.link(tmp, target)
        except (PermissionError, OSError):
            os.replace(tmp, target)
            tmp = None
    except Exception:
        if created_report:
            sidecar.unlink(missing_ok=True)
        raise
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return target, report
