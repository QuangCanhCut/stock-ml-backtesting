"""Chuẩn hóa timestamp về ngày lịch theo múi giờ của nguồn."""

import pandas as pd


def normalize_dates(values):
    # Giữ ngày theo đồng hồ địa phương của nguồn, không đổi ngày theo UTC.
    def normalize(value):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("Thiếu ngày trong nguồn dữ liệu")
        return stamp.tz_localize(None).normalize() if stamp.tzinfo else stamp.normalize()

    return pd.to_datetime(values.map(normalize))
