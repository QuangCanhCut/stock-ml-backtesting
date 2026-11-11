"""Đọc dataset Research và tạo view đủ điều kiện cho từng câu hỏi nghiên cứu."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.dataset_builder import expected_feature_columns, get_training_view, validate_dataset

RESEARCH_PATH = Path("data/processed/HPG_features_labeled_research.parquet")
RQ_TARGETS = {"RQ1": "target_direction", "RQ2": "target_fixed_hold_net_proxy"}


@dataclass(frozen=True)
class ModelingView:
    """Dữ liệu đủ điều kiện và mốc kết thúc thông tin nhãn để chia thời gian."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


def load_research_dataset(root: Path, config: dict) -> pd.DataFrame:
    """Đọc Parquet Research và kiểm tra schema; không thay đổi tệp nguồn.

    Args:
        root: Thư mục gốc repository.
        config: Cấu hình Phase 2 dùng để xác định thứ tự feature.

    Returns:
        Bản sao dataset đã được kiểm tra.

    Raises:
        FileNotFoundError: Khi thiếu Parquet Research.
        ValueError: Khi index, schema hoặc giá trị không hợp lệ.
    """
    path = root / RESEARCH_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset Research: {path}")
    data = pd.read_parquet(path)
    names = expected_feature_columns(config)
    validate_dataset(data, names)
    if data.attrs.get("fixed_holding_sessions") != config["labeling"]["fixed_holding_sessions"]:
        raise ValueError("Thời hạn giữ trong dataset khác cấu hình")
    return data.copy(deep=True)


def make_modeling_view(dataset: pd.DataFrame, config: dict, rq: str) -> ModelingView:
    """Tạo X/y và thời điểm hết thông tin nhãn, giữ nguyên thứ tự phiên.

    Args:
        dataset: Master dataset Phase 2 đã có metadata schema.
        config: Cấu hình xác định feature và số phiên giữ lệnh.
        rq: Một trong RQ1 hoặc RQ2.

    Returns:
        View đủ điều kiện với ngày tín hiệu và ngày kết thúc nhãn.

    Raises:
        ValueError: Khi RQ, schema hoặc căn chỉnh index không hợp lệ.
    """
    if rq not in RQ_TARGETS:
        raise ValueError(f"Câu hỏi nghiên cứu không được hỗ trợ: {rq}")
    names = expected_feature_columns(config)
    X, y = get_training_view(dataset, names, RQ_TARGETS[rq])
    if not X.index.equals(y.index):
        raise ValueError("Index feature và target không khớp")
    horizon = 1 if rq == "RQ1" else 1 + config["labeling"]["fixed_holding_sessions"]
    positions = dataset.index.get_indexer(X.index)
    if (positions < 0).any() or (positions + horizon >= len(dataset)).any():
        raise ValueError("Nhãn đủ điều kiện vượt quá ngày giao dịch cuối")
    metadata = pd.DataFrame({
        "signal_date": X.index,
        "label_end_date": dataset.index.take(positions + horizon),
    }, index=X.index)
    return ModelingView(X, y, metadata)
