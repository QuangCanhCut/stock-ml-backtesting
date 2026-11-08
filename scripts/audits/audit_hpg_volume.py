
import pandas as pd
from vnstock import Quote

# 1. Đọc dữ liệu hiện tại
raw = pd.read_csv("data/raw/HPG_10y_daily.csv")
raw["date"] = pd.to_datetime(raw["date"]).dt.date

# 2. Lấy dữ liệu đối chứng từ KBS
kbs = Quote(symbol="HPG", source="kbs").history(
    start="2021-10-01",
    end="2021-10-16",
    interval="1D"
)

kbs["date"] = pd.to_datetime(kbs["time"]).dt.date

# 3. Lọc khoảng thời gian cần kiểm tra
start = pd.Timestamp("2021-10-01").date()
end = pd.Timestamp("2021-10-15").date()

raw = raw.loc[
    raw["date"].between(start, end),
    ["date", "volume"]
]

kbs = kbs.loc[
    kbs["date"].between(start, end),
    ["date", "volume"]
]

# 4. Đối chiếu theo từng ngày
comparison = pd.merge(
    raw,
    kbs,
    on="date",
    how="outer",
    suffixes=("_raw", "_kbs"),
    validate="one_to_one"
).sort_values("date")

comparison["volume_ratio"] = (
    comparison["volume_raw"] /
    comparison["volume_kbs"]
)

print("\n=== HPG VOLUME AUDIT ===")
print(comparison.to_string(index=False))
