"""Tạo nhãn nghiên cứu, tách biệt với ma trận đặc trưng.

Tín hiệu xuất hiện sau Close(t). Nhãn hướng dùng Close(t+1); proxy giao dịch
dùng Open(t+1) và Open(t+1+h). Giá OHLC có thể được điều chỉnh hồi tố, không
phải giá khớp lịch sử. Proxy không phản ánh thanh khoản, lệnh, corporate actions
hay khả năng bán thực tế và không phải PnL đã thực hiện. Khi chia walk-forward,
cần purge các nhãn có khoảng thông tin tương lai giao với giai đoạn kiểm thử.
"""
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.features import validate_ohlcv


def validate_labeling_input(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra OHLCV và trả về bản sao sắp xếp, không xóa hoặc sửa dữ liệu."""
    return validate_ohlcv(df)


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping) or not isinstance(config.get(key), Mapping):
        raise ValueError(f"Cấu hình cần nhóm {key!r}")
    return config[key]


def _fraction(value: Any, name: str) -> float:
    """Kiểm tra tỷ lệ hữu hạn thuộc [0, 1), từ chối bool và kiểu không phải số."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value < 1):
        raise ValueError(f"{name} phải là số hữu hạn trong [0, 1)")
    return float(value)


def _threshold(config: Mapping[str, Any]) -> float:
    return _fraction(_section(config, "market_microstructure").get("round_trip_cost"),
                     "market_microstructure.round_trip_cost")


def _positive_label(values: pd.Series) -> pd.Series:
    """Giữ missing khi không có dữ liệu tương lai, thay vì biến thành class 0."""
    return (values > 0).where(values.notna()).astype("Int8")


def add_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Nhãn RQ1: log return Close(t) → Close(t+1), tăng là 1, còn lại là 0."""
    data = validate_labeling_input(df)
    forward = np.log(data["close"].shift(-1) / data["close"])
    return pd.DataFrame({"forward_log_return_1": forward,
                         "target_direction": _positive_label(forward)}, index=data.index)


def add_return_threshold_proxy(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """So sánh log return kế tiếp với ngưỡng chi phí dạng tỷ lệ trong config.

    Đây là heuristic thống kê, không phải lợi nhuận ròng của một giao dịch.
    """
    data = validate_labeling_input(df)
    forward = np.log(data["close"].shift(-1) / data["close"])
    target = _positive_label(forward - _threshold(config))
    return pd.DataFrame({"target_return_above_cost_proxy": target}, index=data.index)


def add_fixed_hold_trade_labels(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Proxy long-only: mua Open(t+1), bán Open(t+1+h), sau phí tỷ lệ.

    Slippage áp dụng cả hai chiều; commission mua trên giá mua sau slippage,
    commission bán và thuế trên giá bán sau slippage. Return là tỷ lệ đơn,
    không nhân 100. Giá điều chỉnh không đảm bảo khả năng giao dịch thực tế.
    """
    data = validate_labeling_input(df)
    labeling = _section(config, "labeling")
    holding = labeling.get("fixed_holding_sessions")
    if isinstance(holding, bool) or not isinstance(holding, Integral) or holding < 1:
        raise ValueError("labeling.fixed_holding_sessions phải là số nguyên >= 1")
    costs = _section(config, "market_microstructure")
    buy = _fraction(costs.get("commission_buy"), "commission_buy")
    sell = _fraction(costs.get("commission_sell"), "commission_sell")
    tax = _fraction(costs.get("tax_sell"), "tax_sell")
    slippage = _fraction(costs.get("slippage"), "slippage")
    threshold = _threshold(config)
    if sell + tax >= 1:
        raise ValueError("Tổng commission_sell và tax_sell phải nhỏ hơn 1")
    total = buy + sell + tax + 2 * slippage
    if not np.isclose(total, threshold, rtol=0, atol=1e-12):
        warnings.warn(
            f"round_trip_cost proxy {threshold:.6f} khác tổng phí thành phần "
            f"hai chiều {total:.6f}; nhãn giao dịch dùng từng thành phần.",
            UserWarning, stacklevel=2,
        )
    entry = data["open"].shift(-1)
    exit_price = data["open"].shift(-(1 + int(holding)))
    gross = exit_price / entry - 1
    net = (exit_price * (1 - slippage) * (1 - sell - tax)
           / (entry * (1 + slippage) * (1 + buy)) - 1)
    if np.isinf(gross.to_numpy()).any() or np.isinf(net.to_numpy()).any():
        raise ValueError("Return vượt phạm vi số hữu hạn; kiểm tra dữ liệu giá")
    return pd.DataFrame({"forward_trade_gross_return": gross,
                         "forward_trade_net_return_proxy": net,
                         "target_fixed_hold_net_proxy": _positive_label(net)},
                        index=data.index)


def build_labels(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Trả về sáu cột nhãn/forward return với index đã sắp xếp, không xóa hàng.

    Các cột forward_* và target_* dùng thông tin tương lai và không được đưa
    vào X. Giữ NaN/NA khi thiếu dữ liệu tương lai; không chỉnh input.
    """
    data = validate_labeling_input(df)
    result = pd.concat([add_direction_labels(data), add_return_threshold_proxy(data, config),
                        add_fixed_hold_trade_labels(data, config)], axis=1)
    # Giữ metadata tên trục cột để join không thay đổi metadata của features.
    result.columns.name = data.columns.name
    return result
