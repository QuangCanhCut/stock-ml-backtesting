"""Chia thời gian mở rộng với purging theo khoảng nhãn và gap theo phiên."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.data import ModelingView, RQ_TARGETS

PRE_VALIDATION_GAP_SESSIONS = 5
TEST_START_YEAR = 2022


@dataclass(frozen=True)
class SplitPeriod:
    """Quy định các năm huấn luyện và đánh giá; năm cuối rỗng nghĩa là đến hết dữ liệu."""

    name: str
    train_end_year: int
    evaluation_start_year: int
    evaluation_end_year: int | None
    role: str


DEFAULT_PERIODS = (
    SplitPeriod("dev_1", 2018, 2019, 2019, "development"),
    SplitPeriod("dev_2", 2019, 2020, 2020, "development"),
    SplitPeriod("dev_3", 2020, 2021, 2021, "development"),
    SplitPeriod("reserved_test", 2021, 2022, None, "reserved_test"),
)


@dataclass(frozen=True)
class TemporalSplit:
    """Chỉ số vị trí trong ModelingView; dùng trực tiếp với X.iloc và y.iloc."""

    fold_name: str
    rq: str
    role: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_master_positions: tuple[int, ...]
    validation_master_positions: tuple[int, ...]
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp
    evaluation_boundary_date: pd.Timestamp
    boundary_excluded_count: int
    purged_count: int
    gap_count: int
    overlap_exclusions: int
    excluded_count: int
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int] | None


def build_temporal_splits(
    master_index: pd.DatetimeIndex,
    view: ModelingView,
    rq: str,
    config: dict,
    periods: tuple[SplitPeriod, ...] = DEFAULT_PERIODS,
    gap_sessions: int = PRE_VALIDATION_GAP_SESSIONS,
) -> tuple[TemporalSplit, ...]:
    """Tạo các fold từ phiên master, không dùng thông tin kết quả của Test.

    Args:
        master_index: Toàn bộ ngày giao dịch đã kiểm tra của master dataset.
        view: X, y và metadata đủ điều kiện của một RQ.
        rq: RQ1 hoặc RQ2, dùng để xác minh horizon nhãn.
        config: Cấu hình Phase 2 chứa thời hạn giữ lệnh.
        periods: Các khoảng năm nghiên cứu đã chốt trước khi xem kết quả.
        gap_sessions: Số phiên master ngay trước vùng đánh giá cần loại khỏi train.

    Returns:
        Các fold với chỉ số vị trí trong ModelingView và vị trí master tương ứng.

    Raises:
        ValueError: Khi metadata, ranh giới hoặc bất biến chống rò rỉ không hợp lệ.
    """
    if rq not in RQ_TARGETS:
        raise ValueError(f"Câu hỏi nghiên cứu không được hỗ trợ: {rq}")
    if isinstance(gap_sessions, bool) or not isinstance(gap_sessions, int) or gap_sessions < 0:
        raise ValueError("Gap phải là số nguyên không âm theo phiên giao dịch")
    if not periods:
        raise ValueError("Cần ít nhất một khoảng chia thời gian")
    signal_positions, end_positions = _validate_view(master_index, view, rq, config)
    return tuple(
        _build_period(master_index, view, rq, period, signal_positions,
                      end_positions, gap_sessions)
        for period in periods
    )


def _validate_view(
    master_index: pd.DatetimeIndex, view: ModelingView, rq: str, config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Xác minh index, eligibility và horizon trước khi tạo bất kỳ fold nào."""
    if (not isinstance(master_index, pd.DatetimeIndex) or master_index.empty
            or master_index.hasnans or master_index.has_duplicates
            or not master_index.is_monotonic_increasing):
        raise ValueError("Lịch phiên master phải là DatetimeIndex duy nhất, tăng dần")
    if (view.X.empty or not view.X.index.equals(view.y.index)
            or not view.X.index.equals(view.metadata.index)
            or view.X.index.has_duplicates or not view.X.index.is_monotonic_increasing):
        raise ValueError("Index X, y và metadata phải khớp, duy nhất và tăng dần")
    if (view.X.isna().any().any()
            or not np.isfinite(view.X.to_numpy(dtype=float)).all()
            or view.y.isna().any()
            or not view.y.isin([0, 1]).all()):
        raise ValueError("ModelingView chứa feature hoặc nhãn không đủ điều kiện")
    if list(view.metadata.columns) != ["signal_date", "label_end_date"]:
        raise ValueError("Metadata thiếu ngày tín hiệu hoặc ngày kết thúc nhãn")
    signal_dates = pd.DatetimeIndex(view.metadata["signal_date"])
    end_dates = pd.DatetimeIndex(view.metadata["label_end_date"])
    if signal_dates.hasnans or end_dates.hasnans or not signal_dates.equals(view.X.index):
        raise ValueError("Ngày tín hiệu trong metadata không khớp ModelingView")
    signal_positions = master_index.get_indexer(signal_dates)
    end_positions = master_index.get_indexer(end_dates)
    if (signal_positions < 0).any() or (end_positions < 0).any():
        raise ValueError("Ngày tín hiệu hoặc ngày kết thúc nhãn không thuộc lịch master")
    holding = config.get("labeling", {}).get("fixed_holding_sessions")
    if isinstance(holding, bool) or not isinstance(holding, int) or holding < 1:
        raise ValueError("Cấu hình fixed_holding_sessions phải là số nguyên dương")
    horizon = 1 if rq == "RQ1" else 1 + holding
    if not np.all(end_positions - signal_positions == horizon):
        raise ValueError(f"Khoảng nhãn {rq} không khớp horizon theo phiên master")
    return signal_positions, end_positions


def _first_session(master_index: pd.DatetimeIndex, year: int) -> int:
    """Tìm phiên đầu tiên trong năm; không suy diễn ngày bằng lịch thông thường."""
    positions = np.flatnonzero(master_index.year == year)
    if not len(positions):
        raise ValueError(f"Không có phiên master trong năm {year}")
    return int(positions[0])


def _class_counts(labels: pd.Series) -> dict[str, int]:
    """Đếm lớp của tập phát triển để giai đoạn modeling nhận diện một lớp."""
    counts = labels.value_counts()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def _build_period(
    master_index: pd.DatetimeIndex, view: ModelingView, rq: str,
    period: SplitPeriod, signal_positions: np.ndarray,
    end_positions: np.ndarray, gap_sessions: int,
) -> TemporalSplit:
    """Áp dụng độc lập điều kiện nhãn và gap, rồi lấy hợp phần bị loại."""
    if (period.role not in {"development", "reserved_test"}
            or period.train_end_year >= period.evaluation_start_year
            or (period.evaluation_end_year is not None
                and period.evaluation_end_year < period.evaluation_start_year)
            or (period.role == "development"
                and (period.evaluation_end_year is None
                     or period.evaluation_end_year >= TEST_START_YEAR))
            or (period.role == "reserved_test"
                and period.evaluation_start_year != TEST_START_YEAR)):
        raise ValueError(f"Thiết kế khoảng {period.name} không hợp lệ hoặc chạm vùng Test")
    boundary = _first_session(master_index, period.evaluation_start_year)
    if boundary < gap_sessions:
        raise ValueError(f"Không đủ phiên master cho gap của {period.name}")
    years = view.X.index.year.to_numpy()
    candidates = np.flatnonzero(years <= period.train_end_year)
    if period.evaluation_end_year is None:
        evaluation = np.flatnonzero(years >= period.evaluation_start_year)
        boundary_excluded = 0
    else:
        next_boundary = _first_session(master_index, period.evaluation_end_year + 1)
        within_years = ((years >= period.evaluation_start_year)
                        & (years <= period.evaluation_end_year))
        boundary_excluded = int((within_years & (end_positions >= next_boundary)).sum())
        evaluation = np.flatnonzero(within_years & (end_positions < next_boundary))

    purged = end_positions[candidates] >= boundary
    gap = signal_positions[candidates] >= boundary - gap_sessions
    excluded = purged | gap
    train = candidates[~excluded]
    if not len(train) or not len(evaluation):
        raise ValueError(f"{period.name}/{rq}: Train hoặc vùng đánh giá không có mẫu hợp lệ")
    if (train[-1] >= evaluation[0] or end_positions[train].max() >= boundary
            or signal_positions[train].max() >= boundary - gap_sessions
            or np.intersect1d(train, evaluation).size):
        raise ValueError(f"{period.name}/{rq}: phát hiện chồng lấn hoặc thiếu gap")
    if period.role == "development" and (years[train] >= TEST_START_YEAR).any():
        raise ValueError(f"{period.name}/{rq}: Train phát sinh trong vùng Test")
    train_counts = _class_counts(view.y.iloc[train])
    evaluation_counts = (_class_counts(view.y.iloc[evaluation])
                         if period.role == "development" else None)
    return TemporalSplit(
        fold_name=period.name, rq=rq, role=period.role,
        train_indices=tuple(map(int, train)),
        validation_indices=tuple(map(int, evaluation)),
        train_master_positions=tuple(map(int, signal_positions[train])),
        validation_master_positions=tuple(map(int, signal_positions[evaluation])),
        train_start_date=view.X.index[train[0]], train_end_date=view.X.index[train[-1]],
        validation_start_date=view.X.index[evaluation[0]],
        validation_end_date=view.X.index[evaluation[-1]],
        evaluation_boundary_date=master_index[boundary],
        boundary_excluded_count=boundary_excluded,
        purged_count=int(purged.sum()), gap_count=int(gap.sum()),
        overlap_exclusions=int((purged & gap).sum()), excluded_count=int(excluded.sum()),
        train_class_counts=train_counts, validation_class_counts=evaluation_counts,
    )
