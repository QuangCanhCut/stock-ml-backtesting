"""Thu thập snapshot VCI; dừng khi nguồn từ chối, không ghi đè cache cũ."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from vnstock import Quote


def collect(output):
    destination = Path(output)
    destination.mkdir(parents=False, exist_ok=False)
    quote = Quote(symbol="HPG", source="vci")
    periods = [(f"{year}-01-01", f"{year + 1}-01-01" if year < 2026 else "2026-11-01")
               for year in range(2016, 2027)]
    for index, (start, end) in enumerate(periods):
        if index:
            time.sleep(15)
        # Tôn trọng phản hồi rate limit hoặc từ chối; không tự động retry.
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            snapshot = quote.history(start=start, end=end, interval="1D")
        if snapshot is None or snapshot.empty:
            raise RuntimeError(f"VCI trả dữ liệu rỗng cho {start} đến {end}")
        path = destination / f"HPG_VCI_{start}_{end}.parquet"
        snapshot.to_parquet(path, index=False)
        metadata = {"provider": "VCI qua vnstock", "symbol": "HPG", "start_requested": start,
                    "end_requested": end, "end_convention": "chưa xác minh; lọc theo ngày khi dùng",
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "vnstock_version": importlib.metadata.version("vnstock"),
                    "rows": len(snapshot), "columns": list(snapshot.columns),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "snapshot_type": "DataFrame gốc do vnstock trả về, không phải HTTP response nguyên bản",
                    "source_attrs": snapshot.attrs,
                    "warnings": [str(message.message) for message in messages],
                    "basis": "chưa xác minh", "license": "cần xác minh quyền sử dụng dữ liệu nguồn"}
        (destination / f"{path.stem}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"Đã lưu {path}: {len(snapshot)} dòng", flush=True)
        if messages:
            print("Có cảnh báo từ nguồn; dừng tải thêm để kiểm tra quyền và giới hạn.", flush=True)
            break


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="thư mục snapshot mới; không được tồn tại")
    collect(parser.parse_args().output)
