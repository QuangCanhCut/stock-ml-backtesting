
from pathlib import Path
import pandas as pd

from src.data_loader import fetch_from_yfinance

# 1. Download Yahoo data before cleaning
df = fetch_from_yfinance(
    "HPG.VN",
    "2021-01-01",
    "2022-01-01"
)

# 2. Load 26 missing sessions from KBS audit
reference = pd.read_csv(
    "results/data_audit/HPG_missing_2021_KBS_reference.csv"
)

dates = pd.DatetimeIndex(
    pd.to_datetime(reference["date"])
).normalize()

cols = ["open", "high", "low", "close", "volume"]

# 3. Inspect the missing sessions
audit = df.reindex(dates)[cols].copy()
audit.index.name = "date"

audit["missing_from_yahoo"] = ~audit.index.isin(
    df.index.normalize()
)

audit["has_nan"] = audit[cols].isna().any(axis=1)

audit["invalid_volume"] = (
    audit["volume"].notna() &
    (audit["volume"] <= 0)
)

audit["invalid_close"] = (
    audit["close"].notna() &
    (audit["close"] <= 0)
)

print("\n=== YAHOO REJECTION AUDIT ===")
print(audit.to_string())

print("\n=== SUMMARY ===")
print("Missing from Yahoo index:",
      audit["missing_from_yahoo"].sum())
print("Rows with NaN:", audit["has_nan"].sum())
print("Invalid Volume:", audit["invalid_volume"].sum())
print("Invalid Close:", audit["invalid_close"].sum())

# 4. Save evidence, without modifying raw data
output = Path(
    "results/data_audit/HPG_yahoo_rejected_2021.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)

audit.to_csv(output)

print("\nSaved:", output)
