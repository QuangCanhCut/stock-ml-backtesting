"""Đối chiếu snapshot VCI với Raw/KBS; chỉ tạo báo cáo, không tạo dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data_loader.adapters import canonicalize, read_kbs_cache, read_raw
from src.data_loader.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.snapshot)
    parts = []
    for year in range(2016, 2027):
        end = f"{year + 1}-01-01" if year < 2026 else "2026-11-01"
        path = directory / f"HPG_VCI_{year}-01-01_{end}.parquet"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"] or meta["warnings"]:
            raise ValueError(f"Snapshot VCI sai hash hoặc có cảnh báo: {path}")
        frame = canonicalize(pd.read_parquet(path), "time")
        frame = frame[frame.date.ge(pd.Timestamp(f"{year}-01-01")) & frame.date.lt(pd.Timestamp(end))]
        parts.append(frame)
    vci = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    if vci.date.duplicated().any():
        raise ValueError("Snapshot VCI có ngày trùng")
    raw = read_raw("data/raw/HPG_10y_daily.csv")
    kbs, missing_years = read_kbs_cache("results/data_audit/kbs_cache", range(2016, 2027))
    if missing_years:
        raise ValueError(f"Thiếu cache KBS: {missing_years}")
    raw = raw[(raw.date >= "2016-01-01") & (raw.date < "2026-11-01")]
    kbs = kbs[(kbs.date >= "2016-01-01") & (kbs.date < "2026-11-01")]
    vci["vci_invalid_ohlcv"] = validate(vci).any(axis=1)
    kbs["kbs_invalid_ohlcv"] = validate(kbs).any(axis=1)
    joined = vci.merge(kbs, on="date", how="outer", suffixes=("_vci", "_kbs"), indicator="vci_kbs")
    joined = joined.merge(raw, on="date", how="outer", validate="one_to_one", indicator="raw_merge")
    joined = joined.sort_values("date").reset_index(drop=True)
    joined = joined.rename(columns={col: f"{col}_raw" for col in ("open", "high", "low", "close", "volume")})
    # Ratio chỉ mô tả khác biệt, không xác định hay áp dụng adjustment.
    joined["vci_kbs_close_ratio"] = joined.close_vci / joined.close_kbs
    joined["vci_kbs_volume_ratio"] = joined.volume_vci / joined.volume_kbs
    missing = pd.read_csv("results/data_audit/HPG_full_history_missing.csv", parse_dates=["date"])
    early = raw[raw.date.lt(pd.Timestamp("2016-05-25"))]
    data = {"vci_rows": len(vci), "vci_first": str(vci.date.min().date()), "vci_last": str(vci.date.max().date()),
            "raw_missing_vs_kbs": len(missing), "missing_kbs_dates_present_vci": int(missing.date.isin(vci.date).sum()),
            "raw_early_2016": len(early), "early_raw_dates_present_vci": int(early.date.isin(vci.date).sum()),
            "vci_invalid_ohlcv": int(vci.vci_invalid_ohlcv.sum()), "kbs_invalid_ohlcv": int(kbs.kbs_invalid_ohlcv.sum()),
            "warning": "Đây là đối chiếu thăm dò; ngày trùng không chứng minh basis, OHLCV hay quyền sử dụng."}
    output = Path(args.output)
    output.mkdir(parents=False, exist_ok=False)
    joined.to_csv(output / "vci_kbs_raw.csv", index=False)
    (output / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
