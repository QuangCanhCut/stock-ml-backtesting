"""Kiểm toán độc lập schema, căn chỉnh nhãn và tính nhân quả của feature."""

from typing import Any

import numpy as np
import pandas as pd

from src.dataset_builder import LABEL_COLUMNS, expected_feature_columns, validate_dataset
from src.features import OHLCV, build_features
from src.labeling import round_trip_cost_proxy, transaction_costs
from src.models.data import RQ_TARGETS, make_modeling_view


def _compare(actual: pd.Series, expected: pd.Series, name: str) -> None:
    """Đối chiếu số thực cùng vị trí, bao gồm vị trí thiếu dữ liệu tương lai."""
    if not actual.index.equals(expected.index) or not actual.isna().equals(expected.isna()):
        raise ValueError(f"{name}: index hoặc vị trí thiếu không khớp")
    if not np.allclose(actual.to_numpy(dtype=float, na_value=np.nan),
                       expected.to_numpy(dtype=float, na_value=np.nan),
                       rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError(f"{name}: giá trị không khớp OHLCV hoặc cấu hình")


def _check_target(actual: pd.Series, returns: pd.Series, name: str) -> None:
    """Bảo đảm nhãn nhị phân chỉ tồn tại khi lợi nhuận tương lai tồn tại."""
    expected = (returns > 0).where(returns.notna()).astype("Int8")
    if not actual.equals(expected):
        raise ValueError(f"{name}: nhãn không khớp lợi nhuận tương lai")


def audit_dataset(dataset: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Kiểm tra dataset đã lưu so với giá, nhãn và feature nhân quả.

    Args:
        dataset: Dataset Research chưa biến đổi.
        config: Cấu hình Phase 2, gồm chi phí và thời hạn giữ.

    Returns:
        Thống kê kiểm toán thực tế cho RQ1 và RQ2.

    Raises:
        ValueError: Khi phát hiện sai schema, leakage hoặc căn chỉnh thời gian.
    """
    names = expected_feature_columns(config)
    validate_dataset(dataset, names)
    if dataset.attrs.get("fixed_holding_sessions") != config["labeling"]["fixed_holding_sessions"]:
        raise ValueError("Thời hạn giữ trong dataset khác cấu hình")
    bars = dataset.loc[:, list(OHLCV)]
    forward = np.log(bars.close.shift(-1) / bars.close)
    _compare(dataset.forward_log_return_1, forward, "forward_log_return_1")
    _check_target(dataset.target_direction, forward, "target_direction")
    heuristic = forward - round_trip_cost_proxy(config)
    _check_target(dataset.target_return_above_cost_proxy, heuristic,
                  "target_return_above_cost_proxy")

    holding = config["labeling"]["fixed_holding_sessions"]
    entry = bars.open.shift(-1)
    exit_price = bars.open.shift(-(1 + holding))
    buy, sell, tax, slippage = transaction_costs(config)
    gross = exit_price / entry - 1
    net = (exit_price * (1 - slippage) * (1 - sell - tax)
           / (entry * (1 + slippage) * (1 + buy)) - 1)
    _compare(dataset.forward_trade_gross_return, gross, "forward_trade_gross_return")
    _compare(dataset.forward_trade_net_return_proxy, net, "forward_trade_net_return_proxy")
    _check_target(dataset.target_fixed_hold_net_proxy, net, "target_fixed_hold_net_proxy")

    # Tính lại trên nhiều tiền tố: thay đổi dữ liệu về sau không được đổi feature quá khứ.
    for length in sorted({min(len(dataset), n) for n in (220, len(dataset) // 2, len(dataset))}):
        prefix = build_features(bars.iloc[:length], config)
        try:
            pd.testing.assert_frame_equal(prefix[names], dataset.iloc[:length][names],
                                          check_exact=False, rtol=1e-12, atol=1e-12,
                                          check_freq=False)
        except AssertionError as exc:
            raise ValueError(f"Feature khác phép tính nhân quả tại tiền tố {length}") from exc

    feature_complete = dataset[names].notna().all(axis=1)
    views = {}
    for rq, target in RQ_TARGETS.items():
        view = make_modeling_view(dataset, config, rq)
        eligible = feature_complete & dataset[target].notna()
        if not view.X.index.equals(dataset.index[eligible]):
            raise ValueError(f"{rq}: lọc mẫu không khớp điều kiện độc lập")
        counts = view.y.value_counts().to_dict()
        views[rq] = {
            "target": target,
            "eligible": len(view.y),
            "excluded": len(dataset) - len(view.y),
            "excluded_missing_features": int((~feature_complete).sum()),
            "excluded_missing_target_with_features": int((feature_complete & dataset[target].isna()).sum()),
            "class_counts": {str(label): int(counts.get(label, 0)) for label in (0, 1)},
            "class_proportions": {str(label): float(counts.get(label, 0) / len(view.y))
                                  if len(view.y) else None for label in (0, 1)},
            "first": str(view.X.index.min().date()) if len(view.y) else None,
            "last": str(view.X.index.max().date()) if len(view.y) else None,
        }
    return {
        "checks": {"schema": "PASS", "feature_leakage": "PASS",
                   "temporal_alignment_rq1": "PASS", "temporal_alignment_rq2": "PASS",
                   "feature_prefix_invariance": "PASS", "view_eligibility": "PASS"},
        "feature_missing": {name: int(count) for name, count in dataset[names].isna().sum().items()},
        "target_missing": {name: int(dataset[name].isna().sum()) for name in RQ_TARGETS.values()},
        "label_missing": {name: int(dataset[name].isna().sum()) for name in LABEL_COLUMNS},
        "warmup_rows": int((~feature_complete).cumprod().sum()),
        "feature_incomplete_rows": int((~feature_complete).sum()),
        "nonfinite_values": int(np.isinf(dataset[names].to_numpy(dtype=float, na_value=np.nan)).sum()),
        "outliers": {"absolute_close_return_above_12pct": int(bars.close.pct_change().abs().gt(.12).sum()),
                     "volume_ratio_outside_0_1_to_10": int((bars.volume / bars.volume.shift(1).rolling(30, min_periods=10).median()).pipe(
                         lambda s: (s.gt(10) | s.lt(.1))).sum())},
        "cost_proxy_approximation": round_trip_cost_proxy(config),
        "views": views,
    }
