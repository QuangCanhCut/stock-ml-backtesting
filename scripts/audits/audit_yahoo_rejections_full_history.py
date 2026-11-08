
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import fetch_from_yfinance

INPUT = Path(
    "results/data_audit/HPG_full_history_missing.csv"
)
OUTPUT = Path(
    "results/data_audit/HPG_yahoo_rejections_full_history.csv"
)

# 1. Read all 63 missing dates
reference = pd.read_csv(INPUT, parse_dates=["date"])
dates = pd.DatetimeIndex(reference["date"]).normalize()

assert dates.is_unique, "Duplicate reference dates"

# 2. Download Yahoo data before cleaning
yahoo = fetch_from_yfinance(
    "HPG.VN",
    "2016-01-01",
    "2026-11-01"
)

yahoo.index = pd.DatetimeIndex(
    yahoo.index
).normalize()

assert yahoo.index.is_unique, "Duplicate Yahoo dates"

# 3. Compare the missing sessions
audit = yahoo.reindex(dates).copy()
audit.index.name = "date"

cols = ["open", "high", "low", "close", "volume"]

audit["year"] = audit.index.year
audit["kbs_volume"] = reference["volume"].to_numpy()
audit["present_in_yahoo"] = audit.index.isin(yahoo.index)

audit["has_nan"] = audit[cols].isna().any(axis=1)

audit["invalid_volume"] = (
    audit["volume"].notna()
    & (audit["volume"] <= 0)
)

audit["invalid_close"] = (
    audit["close"].notna()
    & (audit["close"] <= 0)
)

audit["reason"] = np.select(
    [
        ~audit["present_in_yahoo"],
        audit["has_nan"],
        audit["invalid_volume"],
        audit["invalid_close"],
    ],
    [
        "ABSENT_FROM_YAHOO",
        "MISSING_OHLCV",
        "INVALID_VOLUME",
        "INVALID_CLOSE",
    ],
    default="VALID_IN_CURRENT_YAHOO"
)

# 4. Print summary by year
summary = (
    audit.groupby(["year", "reason"])
    .size()
    .unstack(fill_value=0)
)

print("\n=== FULL HISTORY REJECTION AUDIT ===")
print(summary.to_string())

print("\n=== TOTALS BY REASON ===")
print(audit["reason"].value_counts().to_string())

# 5. Save audit evidence
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT)

print("\nSaved:", OUTPUT)
