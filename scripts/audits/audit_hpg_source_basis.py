
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/HPG_10y_daily.csv")
CACHE = Path("results/data_audit/kbs_cache")
OUT = Path("results/data_audit")

raw = pd.read_csv(RAW)
raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

refs = []

for year in range(2016, 2027):
    path = CACHE / f"HPG_KBS_{year}.parquet"

    if not path.exists():
        raise FileNotFoundError(path)

    kbs = pd.read_parquet(path).copy()
    dates = pd.to_datetime(kbs["time"])

    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    kbs["date"] = dates.dt.normalize()
    kbs = kbs[
        (kbs["close"] > 0) & (kbs["volume"] > 0)
    ]

    refs.append(kbs[["date", "close", "volume"]])

reference = pd.concat(refs, ignore_index=True)

assert raw["date"].is_unique
assert reference["date"].is_unique

comparison = raw.merge(
    reference,
    on="date",
    suffixes=("_yahoo", "_kbs"),
    validate="one_to_one"
)

comparison = comparison[
    (comparison["close_yahoo"] > 0) &
    (comparison["close_kbs"] > 0) &
    (comparison["volume_kbs"] > 0)
].copy()

# x1000 is exploratory: not a proven adjustment rule.
comparison["price_ratio"] = (
    comparison["close_yahoo"] /
    (comparison["close_kbs"] * 1000)
)

comparison["volume_ratio"] = (
    comparison["volume_yahoo"] /
    comparison["volume_kbs"]
)

comparison["year"] = comparison["date"].dt.year
comparison["month"] = comparison["date"].dt.to_period("M")

yearly = comparison.groupby("year").agg(
    sessions=("date", "size"),
    price_ratio_median=("price_ratio", "median"),
    price_ratio_min=("price_ratio", "min"),
    price_ratio_max=("price_ratio", "max"),
    volume_ratio_median=("volume_ratio", "median"),
    volume_ratio_min=("volume_ratio", "min"),
    volume_ratio_max=("volume_ratio", "max"),
)

monthly = comparison.groupby("month").agg(
    sessions=("date", "size"),
    price_ratio_median=("price_ratio", "median"),
    volume_ratio_median=("volume_ratio", "median"),
)

OUT.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUT / "HPG_source_basis_daily.csv",
    index=False
)
yearly.to_csv(
    OUT / "HPG_source_basis_yearly.csv"
)
monthly.to_csv(
    OUT / "HPG_source_basis_monthly.csv"
)

print("\n=== SOURCE BASIS AUDIT BY YEAR ===")
print(yearly.to_string())

print("\n=== 2021 MONTHLY RATIOS ===")
print(monthly.loc["2021"].to_string())

print("\nAudit saved successfully.")
