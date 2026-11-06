"""Các đặc trưng OHLCV chỉ dùng dữ liệu đến cuối phiên, không tạo nhãn.

OHLC đầu vào cần cùng quy ước điều chỉnh. Volume có thể vẫn theo quy ước gốc
của nhà cung cấp; sự kiện doanh nghiệp có thể ảnh hưởng đến việc so sánh volume
lịch sử. Tính bất biến theo đoạn đầu chuỗi không chứng minh giá điều chỉnh là
chính xác tại từng thời điểm. NaN biểu thị giai đoạn khởi tạo hoặc phép chia
không xác định.
"""
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

OHLCV = ("open", "high", "low", "close", "volume")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra OHLCV và trả về bản sao đã sắp xếp; không sửa hoặc xóa hàng.

    OHLC phải cùng hệ điều chỉnh, nhưng không thể xác minh chỉ từ OHLCV.
    NaN/Inf đầu vào là lỗi, khác với NaN do khởi tạo feature.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("OHLCV input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("OHLCV input must not be empty")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    missing = set(OHLCV) - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("Index must be a DatetimeIndex without NaT")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed")
    for name in OHLCV:
        col = df[name]
        if (not is_numeric_dtype(col.dtype) or is_bool_dtype(col.dtype)
                or is_complex_dtype(col.dtype)):
            raise ValueError(f"{name} must be numeric")
        if not np.isfinite(col.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"{name} must contain only finite values")
        if name != "volume" and (col <= 0).any():
            raise ValueError(f"{name} must be positive")
        if name == "volume" and (col < 0).any():
            raise ValueError("volume must be non-negative")
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)).any()
            or (df["low"] > df[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC geometry: high/low violate candle bounds")
    return df.sort_index().copy()


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping) or not isinstance(config.get(key), Mapping):
        raise ValueError(f"Configuration requires a '{key}' mapping")
    return config[key]


def _window(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _windows(value: Any, name: str, minimum: int = 1) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list of windows")
    result = [_window(v, name, minimum) for v in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _positive_number(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _wilder(values: pd.Series, window: int) -> pd.Series:
    """Khởi tạo bằng trung bình `window` giá trị hợp lệ đầu tiên, sau đó
    cập nhật: (giá trị trước * (window - 1) + giá trị hiện tại) / window.

    Cho phép NaN ở đầu chuỗi; các giá trị sau đó phải hữu hạn.
    """
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if len(valid) < window:
        return output
    if len(valid) != len(values.loc[valid.index[0]:]):
        raise ValueError("Wilder input must not contain gaps after initialization")
    seed_index = valid.index[window - 1]
    seeded = pd.concat([pd.Series([valid.iloc[:window].mean()], index=[seed_index]),
                        valid.iloc[window:]])
    output.loc[seeded.index] = seeded.ewm(alpha=1 / window, adjust=False).mean()
    return output


def add_return_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Thêm log return ngày và các độ trễ, không dùng return tương lai."""
    out = validate_ohlcv(df)
    cfg = _section(config, "features")
    lags = _windows(cfg.get("return_lags"), "features.return_lags")
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    for lag in lags:
        out[f"return_lag_{lag}"] = out["log_return"].shift(lag)
    return out


def add_trend_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Tính tỷ lệ SMA và EMA từ dữ liệu hiện tại và quá khứ.

    EMA dùng adjust=False và min_periods=span. MACD bằng
    (EMA nhanh - EMA chậm) / close; ema_windows phải chứa đúng hai cửa sổ
    tăng dần để xác định EMA nhanh và EMA chậm.
    """
    out = validate_ohlcv(df)
    cfg = _section(config, "features")
    smas = _windows(cfg.get("sma_windows"), "features.sma_windows")
    emas = _windows(cfg.get("ema_windows"), "features.ema_windows")
    if len(emas) != 2 or emas[0] >= emas[1]:
        raise ValueError("features.ema_windows must contain [fast, slow] with fast < slow")
    close = out["close"]
    for window in smas:
        out[f"sma_ratio_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1
    ema_values = {}
    for window in emas:
        ema_values[window] = close.ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"ema_ratio_{window}"] = close / ema_values[window] - 1
    out["macd_ratio"] = (ema_values[emas[0]] - ema_values[emas[1]]) / close
    return out


def add_momentum_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Thêm momentum log tích lũy và RSI theo phương pháp Wilder.

    RSI khởi tạo mức tăng và giảm trung bình từ n thay đổi giá đầu tiên,
    sau đó làm mượt theo Wilder. Giá không đổi cho RSI = 50; không giảm cho
    100; không tăng cho 0. Với config cũ, cửa sổ momentum mặc định là [5, 10, 21].
    """
    out = validate_ohlcv(df)
    cfg = _section(config, "features")
    windows = _windows(cfg.get("momentum_windows", [5, 10, 21]), "features.momentum_windows")
    rsi_window = _window(cfg.get("rsi_window"), "features.rsi_window")
    close = out["close"]
    for window in windows:
        out[f"momentum_{window}"] = np.log(close / close.shift(window))
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), rsi_window)
    loss = _wilder(-delta.clip(upper=0), rsi_window)
    denominator = gain + loss
    rsi = 100 * gain / denominator.where(denominator != 0)
    out[f"rsi_{rsi_window}"] = rsi.mask((denominator == 0), 50.0)
    return out


def add_volatility_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Tính volatility log return năm hóa (độ lệch chuẩn mẫu, ddof=1),
    ATR Wilder / close và Bollinger (độ lệch chuẩn tổng thể, ddof=0).

    ATR bắt đầu với TR_0 = high - low và khởi tạo bằng n giá trị TR đầu tiên.
    Khi giá không đổi trong cửa sổ, độ rộng Bollinger bằng 0 còn percent B
    không xác định (NaN). Năm hóa dùng quy ước căn bậc hai của thời gian.
    """
    out = validate_ohlcv(df)
    cfg = _section(config, "features")
    windows = _windows(_section(config, "eda").get("rolling_vol_windows"),
                       "eda.rolling_vol_windows", minimum=2)
    days = _positive_number(_section(config, "market_microstructure").get("trading_days_per_year"),
                            "market_microstructure.trading_days_per_year")
    atr_window = _window(cfg.get("atr_window"), "features.atr_window")
    boll_window = _window(cfg.get("bollinger_window"), "features.bollinger_window", 2)
    multiplier = _positive_number(cfg.get("bollinger_std"), "features.bollinger_std")
    close = out["close"]
    returns = np.log(close / close.shift(1))
    for window in windows:
        out[f"volatility_{window}"] = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(days)
    prev = close.shift(1)
    tr = pd.concat([out["high"] - out["low"], (out["high"] - prev).abs(),
                    (out["low"] - prev).abs()], axis=1).max(axis=1)
    out[f"atr_ratio_{atr_window}"] = _wilder(tr, atr_window) / close
    middle = close.rolling(boll_window, min_periods=boll_window).mean()
    std = close.rolling(boll_window, min_periods=boll_window).std(ddof=0)
    band_span = 2 * multiplier * std
    out[f"bollinger_width_{boll_window}"] = band_span / middle
    out[f"bollinger_percent_b_{boll_window}"] = (close - (middle - multiplier * std)) / band_span.where(band_span != 0)
    return out


def add_volume_features(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Tính z-score volume trên cửa sổ quá khứ (độ lệch chuẩn mẫu, ddof=1),
    mức thay đổi và tỷ lệ so với volume trung bình.

    Volume phiên trước, trung bình trượt hoặc độ lệch chuẩn trượt bằng 0 sẽ
    khiến tỷ lệ tương ứng không xác định và nhận NaN; không điền giá trị thiếu.
    """
    out = validate_ohlcv(df)
    window = _window(_section(config, "features").get("volume_zscore_window"),
                     "features.volume_zscore_window", 2)
    volume = out["volume"].astype(float)
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std(ddof=1)
    out[f"volume_zscore_{window}"] = (volume - mean) / std.where(std != 0)
    out["volume_change_1"] = volume / volume.shift(1).where(volume.shift(1) != 0) - 1
    out[f"volume_ratio_{window}"] = volume / mean.where(mean != 0)
    return out


def build_features(df_raw: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Trả về OHLCV đã sắp xếp cùng feature nhân quả, giữ mọi hàng đầu vào.

    Giá trị khởi tạo và tỷ lệ không xác định vẫn là NaN. OHLCV thiếu hoặc
    không hợp lệ sẽ gây lỗi. Hàm không tạo nhãn, return tương lai, điền giá
    trị thiếu hay học tham số trên toàn bộ dữ liệu. Các cột đầu vào khác được
    giữ lại; cột target/tương lai và tên feature trùng bị từ chối.
    """
    out = validate_ohlcv(df_raw)
    unsafe = [c for c in out.columns if isinstance(c, str)
              and (c.startswith("target_") or c.startswith("next_") or c.startswith("future_"))]
    if unsafe:
        raise ValueError(f"Input contains target/future columns: {unsafe}")
    original = set(out.columns)
    for function in (add_return_features, add_trend_features, add_momentum_features,
                     add_volatility_features, add_volume_features):
        # Chỉ dùng OHLCV để tính từng nhóm, tránh cột có sẵn âm thầm ghi đè
        # feature mới hoặc tạo tên cột trùng lặp.
        group = function(out.loc[:, list(OHLCV)], config)
        additions = group.drop(columns=list(OHLCV))
        collision = original.intersection(additions.columns)
        if collision:
            raise ValueError(f"Input already contains feature columns: {sorted(collision)}")
        out = out.join(additions)
    numeric = out.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Feature calculation produced Inf; check input magnitudes")
    return out
