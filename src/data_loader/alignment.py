"""Đối chiếu thăm dò năm 2025 dựa trên toàn bộ lịch KBS trong cache."""

import numpy as np
import pandas as pd


def diagnostic_2025(raw, kbs):
    # Xác định phiên trước từ lịch KBS trước khi ghép với Raw.
    calendar = kbs.loc[(kbs.date.dt.year == 2025) & (kbs.volume > 0), ["date", "close", "volume"]].sort_values("date").copy()
    calendar["prev_kbs_date"] = calendar.date.shift(1)
    calendar["prev_kbs_close"] = calendar.close.shift(1)
    calendar["prev_kbs_volume"] = calendar.volume.shift(1)
    rows = raw.merge(calendar, on="date", suffixes=("_raw", "_kbs"), validate="one_to_one")
    rows = rows.loc[rows.date.between("2025-05-06", "2025-10-31")].copy()
    # Tái lập giả định audit cũ, không áp dụng chuyển đổi cho dataset.
    factor = np.where(rows.date < pd.Timestamp("2025-06-26"), 1.32, 1.10)
    for label, close, volume in [("same", "close_kbs", "volume_kbs"), ("prev", "prev_kbs_close", "prev_kbs_volume")]:
        rows[f"{label}_price_error"] = (rows.close_raw / (rows[close] * 1000) - 1).abs()
        rows[f"{label}_volume_error"] = (rows.volume_raw / (rows[volume] * factor) - 1).abs()
    same = (rows.same_price_error < .002) & (rows.same_volume_error < .001)
    prev = (rows.prev_price_error < .002) & (rows.prev_volume_error < .001)
    rows["alignment"] = np.select([same & ~prev, prev & ~same, same & prev],
                                   ["SAME_DAY", "PREVIOUS_SESSION", "AMBIGUOUS"], default="UNMATCHED")
    rows["interpretation"] = "exploratory_only_unverified_price_and_volume_basis"
    return rows
