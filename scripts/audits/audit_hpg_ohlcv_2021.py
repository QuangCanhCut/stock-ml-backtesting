
from pathlib import Path
import pandas as pd
from vnstock import Quote

# Đọc dữ liệu hiện tại
raw = pd.read_csv("data/raw/HPG_10y_daily.csv")
raw["date"] = pd.to_datetime(raw["date"]).dt.date

# Lấy dữ liệu đối chứng
kbs = Quote(symbol="HPG", source="kbs").history(
    start="2021-01-01",
    end="2022-01-01",
    interval="1D"
)
kbs["date"] = pd.to_datetime(kbs["time"]).dt.date

columns = ["date", "open", "high", "low", "close", "volume"]

raw_2021 = raw.loc[
    pd.to_datetime(raw["date"]).dt.year == 2021,
    columns
].copy()

kbs_2021 = kbs[columns].copy()

assert raw_2021["date"].is_unique
assert kbs_2021["date"].is_unique

# A. Xuất các phiên thiếu để đối chiếu
missing = kbs_2021.loc[
    ~kbs_2021["date"].isin(raw_2021["date"])
].sort_values("date")

output_dir = Path("results/data_audit")
output_dir.mkdir(parents=True, exist_ok=True)

missing.to_csv(
    output_dir / "HPG_missing_2021_KBS_reference.csv",
    index=False
)

print("=== MISSING SESSIONS ===")
print("Count:", len(missing))
print(missing.to_string(index=False))

# B. Kiểm tra quy ước điều chỉnh giá và volume
comparison = pd.merge(
    raw_2021,
    kbs_2021,
    on="date",
    how="inner",
    suffixes=("_raw", "_kbs"),
    validate="one_to_one"
)

comparison["close_ratio"] = (
    comparison["close_raw"] / comparison["close_kbs"]
)
comparison["volume_ratio"] = (
    comparison["volume_raw"] / comparison["volume_kbs"]
)

# Kiểm tra khoảng phát hiện bất thường
window = comparison.loc[
    (comparison["date"] >= pd.Timestamp("2021-10-01").date()) &
    (comparison["date"] <= pd.Timestamp("2021-10-15").date())
]

print("\n=== OHLCV ADJUSTMENT AUDIT ===")
print(
    window[
        ["date", "close_ratio", "volume_ratio"]
    ].to_string(index=False)
)

comparison.to_csv(
    output_dir / "HPG_overlap_2021_KBS_comparison.csv",
    index=False
)

print("\nAudit files saved to:", output_dir)
