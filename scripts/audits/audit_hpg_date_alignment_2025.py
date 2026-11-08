
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(
    "results/data_audit/HPG_source_basis_daily.csv"
)
OUTPUT = Path(
    "results/data_audit/HPG_date_alignment_2025.csv"
)

df = pd.read_csv(SOURCE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Previous trading session in the KBS reference.
df["prev_kbs_date"] = df["date"].shift(1)
df["prev_kbs_close"] = df["close_kbs"].shift(1)
df["prev_kbs_volume"] = df["volume_kbs"].shift(1)

window = df.loc[
    df["date"].between(
        "2025-05-06", "2025-10-31"
    )
].copy()

# Historical factors are used ONLY for diagnostics.
window["volume_factor"] = np.where(
    window["date"] < pd.Timestamp("2025-06-26"),
    1.32,
    1.10,
)

# Compare Yahoo against KBS on the same date.
window["same_price_error"] = (
    window["close_yahoo"]
    / (window["close_kbs"] * 1000) - 1
).abs()

window["same_volume_error"] = (
    window["volume_yahoo"]
    / (
        window["volume_kbs"]
        * window["volume_factor"]
    ) - 1
).abs()

# Compare Yahoo against the previous KBS session.
window["prev_price_error"] = (
    window["close_yahoo"]
    / (window["prev_kbs_close"] * 1000) - 1
).abs()

window["prev_volume_error"] = (
    window["volume_yahoo"]
    / (
        window["prev_kbs_volume"]
        * window["volume_factor"]
    ) - 1
).abs()

# Price tolerance: 0.2%.
# Volume tolerance: 0.1%.
same_match = (
    (window["same_price_error"] < 0.002)
    & (window["same_volume_error"] < 0.001)
)

prev_match = (
    (window["prev_price_error"] < 0.002)
    & (window["prev_volume_error"] < 0.001)
)

window["alignment"] = np.select(
    [
        same_match & ~prev_match,
        prev_match & ~same_match,
        same_match & prev_match,
    ],
    [
        "SAME_DAY",
        "PREVIOUS_SESSION",
        "AMBIGUOUS",
    ],
    default="UNMATCHED",
)

window["weekday"] = window["date"].dt.day_name()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
window.to_csv(OUTPUT, index=False)

print("\n=== DATE ALIGNMENT AUDIT ===")
print(window["alignment"].value_counts())

print("\n=== ALIGNMENT BY WEEKDAY ===")
print(
    pd.crosstab(
        window["weekday"],
        window["alignment"],
    )
)

print("\nSaved:", OUTPUT)
