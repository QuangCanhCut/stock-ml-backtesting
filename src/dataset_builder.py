"""Ghép dataset nghiên cứu; không fit biến đổi, mô hình hay chia tập thời gian.

Master dataset giữ NaN và toàn bộ phiên. View đủ điều kiện không phải tập train
cuối cùng: cần chia theo thời gian và purge khoảng nhãn trước khi modeling.
"""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import yaml

from src.features import OHLCV, build_features, validate_ohlcv
from src.labeling import build_labels

ROOT = Path(__file__).resolve().parent.parent
LABEL_COLUMNS = ["forward_log_return_1", "target_direction",
                 "target_return_above_cost_proxy", "forward_trade_gross_return",
                 "forward_trade_net_return_proxy", "target_fixed_hold_net_proxy"]
TARGETS = ["target_direction", "target_return_above_cost_proxy", "target_fixed_hold_net_proxy"]
FORBIDDEN = ("target_", "forward_", "next_", "future_")
SCHEMA_VERSION = 1


def _path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Đường dẫn cấu hình phải là chuỗi không rỗng")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    """Đọc YAML; đường dẫn tương đối tới config được tính từ repository root."""
    path = _path(str(config_path), ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Root của YAML phải là mapping")
    for section in ("data", "features", "eda", "market_microstructure", "labeling"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Thiếu mapping cấu hình: {section}")
    for key in ("raw_parquet_path", "processed_path"):
        _path(config["data"].get(key), path.parent.parent)
    return config


def load_raw_data(config: Mapping[str, Any], project_root: str | Path) -> pd.DataFrame:
    """Đọc parquet cấu hình, kiểm tra OHLCV và từ chối mọi cột bổ sung."""
    path = _path(config["data"]["raw_parquet_path"], Path(project_root))
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy parquet nguồn: {path}")
    return _raw_ohlcv(pd.read_parquet(path))


def _raw_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    data = validate_ohlcv(df)
    extra = [c for c in data.columns if c not in OHLCV]
    if extra:
        raise ValueError(f"Dữ liệu raw chứa cột ngoài OHLCV: {extra}")
    return data.loc[:, list(OHLCV)].copy()


def expected_feature_columns(config: Mapping[str, Any]) -> list[str]:
    """Schema feature duy nhất cho cấu hình, theo thứ tự module features hiện có."""
    try:
        f, e = config["features"], config["eda"]
        return (["log_return"] + [f"return_lag_{n}" for n in f["return_lags"]]
                + [f"sma_ratio_{n}" for n in f["sma_windows"]]
                + [f"ema_ratio_{n}" for n in f["ema_windows"]] + ["macd_ratio"]
                + [f"momentum_{n}" for n in f.get("momentum_windows", [5, 10, 21])]
                + [f"rsi_{f['rsi_window']}"]
                + [f"volatility_{n}" for n in e["rolling_vol_windows"]]
                + [f"atr_ratio_{f['atr_window']}", f"bollinger_width_{f['bollinger_window']}",
                   f"bollinger_percent_b_{f['bollinger_window']}",
                   f"volume_zscore_{f['volume_zscore_window']}", "volume_change_1",
                   f"volume_ratio_{f['volume_zscore_window']}"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Cấu hình feature thiếu hoặc không hợp lệ: {exc}") from exc


def _feature_list(feature_columns: Sequence[str]) -> list[str]:
    if isinstance(feature_columns, (str, bytes)) or not isinstance(feature_columns, Sequence):
        raise ValueError("feature_columns phải là danh sách tên cột")
    names = list(feature_columns)
    if (not names or any(not isinstance(c, str) or c in OHLCV or c.startswith(FORBIDDEN)
                         for c in names) or len(set(names)) != len(names)):
        raise ValueError("Schema feature rỗng, trùng tên, chứa OHLCV hoặc thông tin tương lai")
    return names


def validate_dataset(dataset: pd.DataFrame, feature_columns: Sequence[str],
                     expected_rows: int | None = None) -> None:
    """Kiểm tra cấu trúc, metadata schema, kiểu số, Inf và class của master."""
    names = _feature_list(feature_columns)
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("Dataset phải là DataFrame không rỗng")
    if (not isinstance(dataset.index, pd.DatetimeIndex) or dataset.index.hasnans
            or dataset.index.has_duplicates or not dataset.index.is_monotonic_increasing):
        raise ValueError("Index phải là DatetimeIndex hợp lệ, duy nhất, tăng dần")
    if not dataset.columns.is_unique:
        raise ValueError("Dataset có cột trùng")
    if expected_rows is not None and len(dataset) != expected_rows:
        raise ValueError("Số hàng dataset khác dữ liệu nguồn")
    if dataset.attrs.get("feature_columns") != names or dataset.attrs.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Feature schema không khớp metadata dataset")
    if list(dataset.columns) != list(OHLCV) + names + LABEL_COLUMNS:
        raise ValueError("Cột dataset hoặc thứ tự không khớp schema")
    validate_ohlcv(dataset.loc[:, list(OHLCV)])
    for col in names + [c for c in LABEL_COLUMNS if c.startswith("forward_")]:
        if not is_numeric_dtype(dataset[col].dtype) or is_bool_dtype(dataset[col].dtype):
            raise ValueError(f"Cột {col} phải là số")
        if np.isinf(dataset[col].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Cột {col} chứa Inf")
    for col in TARGETS:
        if str(dataset[col].dtype) != "Int8" or not dataset[col].dropna().isin([0, 1]).all():
            raise ValueError(f"Target {col} phải là Int8 và chỉ chứa 0, 1 hoặc NA")


def assemble_dataset(df_raw: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Ghép một-một theo index; trả về master và danh sách feature theo thứ tự."""
    data = _raw_ohlcv(df_raw)
    features = build_features(data, config)
    labels = build_labels(data, config)
    expected = _feature_list(expected_feature_columns(config))
    if not features.columns.is_unique or not labels.columns.is_unique:
        raise ValueError("Feature hoặc label có cột trùng")
    if list(features.columns) != list(OHLCV) + expected or list(labels.columns) != LABEL_COLUMNS:
        raise ValueError("Output feature/label không khớp schema cấu hình")
    if set(features.columns) & set(labels.columns):
        raise ValueError("Cột feature và label bị trùng")
    if (not features.index.equals(data.index) or not labels.index.equals(data.index)
            or not features.index.is_unique or not labels.index.is_unique
            or not features.index.is_monotonic_increasing or not labels.index.is_monotonic_increasing):
        raise ValueError("Index feature và label không khớp dữ liệu nguồn")
    dataset = features.join(labels, how="inner", validate="one_to_one")
    dataset.attrs = {"schema_version": SCHEMA_VERSION, "feature_columns": expected,
                     "label_columns": LABEL_COLUMNS.copy(),
                     "fixed_holding_sessions": config["labeling"]["fixed_holding_sessions"]}
    validate_dataset(dataset, expected, expected_rows=len(data))
    return dataset, expected


def get_training_view(dataset: pd.DataFrame, feature_columns: Sequence[str],
                      target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Trả X/y đủ điều kiện cho một mục tiêu; chưa chia train/validation/test.

    target_return_above_cost_proxy chỉ là nhãn nghiên cứu heuristic, không phải
    lợi nhuận thực tế. Không dùng bất kỳ cột forward/target hoặc OHLCV trong X.
    """
    if target_column not in TARGETS:
        raise ValueError(f"Target không được hỗ trợ: {target_column}")
    names = _feature_list(feature_columns)
    validate_dataset(dataset, names)
    valid = dataset[names].notna().all(axis=1) & dataset[target_column].notna()
    return dataset.loc[valid, names].copy(), dataset.loc[valid, target_column].copy()


def quality_report(dataset: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, Any]:
    """Thống kê mô tả và số mẫu đủ điều kiện; không dùng để chọn features."""
    names = _feature_list(feature_columns)
    validate_dataset(dataset, names)
    complete = dataset[names].notna().all(axis=1)
    views = {}
    for target in TARGETS:
        x, _ = get_training_view(dataset, names, target)
        views[target] = {"eligible": len(x), "first": str(x.index.min()) if len(x) else None,
                         "last": str(x.index.max()) if len(x) else None}
    return {"rows": len(dataset), "columns": len(dataset.columns),
            "feature_count": len(names), "label_count": len(LABEL_COLUMNS),
            "missing": dataset[names + LABEL_COLUMNS].isna().sum().to_dict(),
            "first_complete_features": str(dataset.index[complete][0]) if complete.any() else None,
            "eligible_views": views,
            "target_stats": {col: {"positive": int((dataset[col] == 1).sum()),
                                    "negative": int((dataset[col] == 0).sum()),
                                    "missing": int(dataset[col].isna().sum()),
                                    "positive_rate": float(dataset[col].mean())}
                             for col in TARGETS},
            "forward_stats": dataset[[c for c in LABEL_COLUMNS if c.startswith("forward_")]]
                             .agg(["mean", "median", "std", "min", "max"]).to_dict()}


def save_processed_dataset(dataset: pd.DataFrame, config: Mapping[str, Any],
                           project_root: str | Path) -> Path:
    """Ghi file tạm cùng thư mục, kiểm tra read-back rồi thay thế nguyên tử."""
    root = Path(project_root)
    raw = _path(config["data"]["raw_parquet_path"], root)
    target = _path(config["data"]["processed_path"], root)
    if raw == target or (raw.exists() and target.exists() and os.path.samefile(raw, target)):
        raise ValueError("Không được ghi đè dữ liệu raw")
    names = dataset.attrs.get("feature_columns")
    if names != expected_feature_columns(config):
        raise ValueError("Schema feature không khớp cấu hình hiện tại")
    validate_dataset(dataset, names)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".dataset-", suffix=".parquet", dir=target.parent)
    os.close(descriptor)
    tmp = Path(filename)
    try:
        dataset.to_parquet(tmp)
        restored = pd.read_parquet(tmp)
        validate_dataset(restored, names, len(dataset))
        # Parquet không lưu thuộc tính freq của DatetimeIndex, nhưng vẫn lưu
        # chính xác các timestamp, tên index, thứ tự và dtype.
        pd.testing.assert_frame_equal(dataset, restored, check_exact=True, check_freq=False)
        if dataset.attrs != restored.attrs:
            raise ValueError("Metadata schema không được bảo toàn qua Parquet")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def build_and_save_dataset(config_path: str | Path = "configs/config.yaml") -> tuple[pd.DataFrame, list[str], Path]:
    """Chạy pipeline; project root là thư mục cha của thư mục chứa config."""
    path = _path(str(config_path), ROOT)
    config = load_project_config(path)
    root = path.parent.parent
    dataset, names = assemble_dataset(load_raw_data(config, root), config)
    output = save_processed_dataset(dataset, config, root)
    return dataset, names, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo master dataset tài chính")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    dataset, names, path = build_and_save_dataset(args.config)
    print(f"Đã lưu và kiểm tra Parquet: {path}")
    print(f"Danh sách features: {names}")
    from pprint import pprint
    pprint(quality_report(dataset, names), sort_dicts=False)


if __name__ == "__main__":
    main()
