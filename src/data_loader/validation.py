"""Kiểm định mà không sửa giá trị nguồn."""

import numpy as np
import pandas as pd

from .adapters import OHLCV


def validate(frame):
    numbers = frame[OHLCV].apply(pd.to_numeric, errors="coerce").astype(float)
    finite = pd.DataFrame(np.isfinite(numbers), index=frame.index, columns=OHLCV)
    missing = ~finite.all(axis=1)
    positive = (numbers[["open", "high", "low", "close"]] > 0).all(axis=1)
    volume = numbers.volume > 0
    scale = numbers[["open", "high", "low", "close"]].abs().max(axis=1)
    tolerance = scale * 1e-10
    geometry = ((numbers.high + tolerance >= numbers[["open", "low", "close"]].max(axis=1)) &
                (numbers.low - tolerance <= numbers[["open", "high", "close"]].min(axis=1)))
    flags = pd.DataFrame({"nonfinite_or_missing": missing, "nonpositive_price": ~positive,
                          "nonpositive_volume": ~volume, "invalid_ohlc_geometry": ~geometry}, index=frame.index)
    return flags
