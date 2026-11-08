
import pandas as pd
from vnstock import Quote

# 1. Đọc dữ liệu Raw
raw = pd.read_csv("data/raw/HPG_10y_daily.csv")

# Chuẩn hóa về datetime.date
raw["date"] = pd.to_datetime(raw["date"]).dt.date

# 2. Lấy dữ liệu KBS
kbs = Quote(symbol="HPG", source="kbs").history(
    start="2021-01-01",
    end="2022-01-01",
    interval="1D"
)

# Chuẩn hóa cùng kiểu với Raw
kbs["date"] = pd.to_datetime(kbs["time"]).dt.date

# 3. Lọc năm 2021
raw_2021 = raw[
    pd.to_datetime(raw["date"]).dt.year == 2021
]

kbs_2021 = kbs[
    pd.to_datetime(kbs["date"]).dt.year == 2021
]

# 4. So sánh lịch giao dịch
raw_dates = set(raw_2021["date"])
kbs_dates = set(kbs_2021["date"])

common_dates = sorted(raw_dates & kbs_dates)
missing_dates = sorted(kbs_dates - raw_dates)
extra_dates = sorted(raw_dates - kbs_dates)

# 5. In kết quả
print("\n=== CALENDAR AUDIT 2021 ===")
print("Raw sessions:", len(raw_dates))
print("KBS sessions:", len(kbs_dates))
print("Common sessions:", len(common_dates))
print("Missing in Raw:", len(missing_dates))
print("Raw-only sessions:", len(extra_dates))

print("\n=== MISSING DATES ===")
for date in missing_dates:
    print(date)

print("\n=== RAW-ONLY DATES ===")
for date in extra_dates:
    print(date)

# 6. Sanity check
assert len(common_dates) > 0, (
    "No common dates found. Check date parsing!"
)
