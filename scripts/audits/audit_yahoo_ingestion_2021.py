
import pandas as pd

from src.data_loader import (
    fetch_from_yfinance,
    clean_and_validate_ohlcv,
)

START = "2021-01-01"
END = "2022-01-01"

# 1. Đọc dataset cũ, không sửa
archive = pd.read_csv(
    "data/raw/HPG_10y_daily.csv",
    parse_dates=["date"]
).set_index("date")

archive = archive.loc["2021"]

# 2. Tải lại dữ liệu Yahoo, chưa cleaning
fetched = fetch_from_yfinance(
    "HPG.VN", START, END
)

# 3. Cleaning trên bản sao trong RAM
cleaned = clean_and_validate_ohlcv(
    fetched.copy()
)

# 4. Lấy danh sách 26 phiên thiếu
reference = pd.read_csv(
    "results/data_audit/"
    "HPG_missing_2021_KBS_reference.csv",
    parse_dates=["date"]
)

missing_dates = set(reference["date"])

fetched_dates = set(fetched.index.normalize())
cleaned_dates = set(cleaned.index.normalize())

print("\n=== YAHOO INGESTION AUDIT ===")
print("Archive:", len(archive))
print("Yahoo fetched:", len(fetched))
print("After cleaning:", len(cleaned))

print("\n=== 26 MISSING DATES ===")
print("Present in Yahoo:", len(
    missing_dates & fetched_dates
))
print("Present after cleaning:", len(
    missing_dates & cleaned_dates
))

removed = sorted(fetched_dates - cleaned_dates)
print("Removed by cleaning:", len(removed))
print("Removed dates:", removed)

# 5. Kiểm tra Volume của hpython -m tests.audit_yahoo_ingestion_2021ai phiên cực đoan
print("\n=== EXTREME VOLUME CHECK ===")

for date in ["2021-09-24", "2021-10-04"]:
    day = pd.Timestamp(date)

    print(f"\nDate: {date}")
    print(
        "Archive Volume:",
        archive["volume"].get(day, "MISSING")
    )
    print(
        "Yahoo Volume:",
        fetched["volume"].get(day, "MISSING")
    )
