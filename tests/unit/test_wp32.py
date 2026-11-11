"""Kiểm thử chia thời gian, purging và gap bằng lịch phiên tổng hợp."""

import numpy as np
import pandas as pd
import pytest

from src.models.data import ModelingView
from src.models.temporal_split import (
    DEFAULT_PERIODS, SplitPeriod, build_temporal_splits,
)


@pytest.fixture
def calendar():
    """Lịch có cuối tuần và ngày nghỉ giữa các phiên để thử gap theo vị trí."""
    dates = pd.date_range("2016-01-01", "2022-02-28", freq="B")
    return dates.drop(pd.to_datetime(["2018-12-31", "2019-01-01", "2021-12-31"]))


def make_view(calendar, horizon, excluded=()):
    """Tạo view nhị phân; có thể bỏ vài phiên mà vẫn giữ lịch master nguyên vẹn."""
    positions = np.array([i for i in range(len(calendar) - horizon) if i not in excluded])
    index = calendar.take(positions)
    metadata = pd.DataFrame({"signal_date": index,
                             "label_end_date": calendar.take(positions + horizon)}, index=index)
    X = pd.DataFrame({"feature": positions.astype(float)}, index=index)
    y = pd.Series((positions % 2).astype("int8"), index=index)
    return ModelingView(X, y, metadata)


CONFIG = {"labeling": {"fixed_holding_sessions": 3}}


@pytest.mark.parametrize("rq,horizon", [("RQ1", 1), ("RQ2", 4)])
def test_default_folds_and_holdout(calendar, rq, horizon):
    view = make_view(calendar, horizon)
    original = view.X.copy(deep=True)
    folds = build_temporal_splits(calendar, view, rq, CONFIG)
    assert [fold.fold_name for fold in folds] == [period.name for period in DEFAULT_PERIODS]
    assert [fold.validation_start_date.year for fold in folds] == [2019, 2020, 2021, 2022]
    assert [fold.train_end_date.year for fold in folds] == [2018, 2019, 2020, 2021]
    assert [len(fold.train_indices) for fold in folds[:3]] == sorted(
        len(fold.train_indices) for fold in folds[:3])
    test_boundary = calendar[calendar.year == 2022][0]
    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        assert list(fold.train_indices) == sorted(set(fold.train_indices))
        assert list(fold.validation_indices) == sorted(set(fold.validation_indices))
        boundary = calendar.get_loc(fold.evaluation_boundary_date)
        assert max(fold.train_master_positions) <= boundary - 6
        assert view.metadata.label_end_date.iloc[list(fold.train_indices)].max() < fold.evaluation_boundary_date
        assert fold.excluded_count == fold.purged_count + fold.gap_count - fold.overlap_exclusions
        assert fold.validation_start_date > fold.train_end_date
    assert all(fold.validation_end_date < test_boundary for fold in folds[:3])
    assert folds[-1].validation_start_date == test_boundary
    assert folds[-1].validation_class_counts is None
    pd.testing.assert_frame_equal(view.X, original)
    assert folds == build_temporal_splits(calendar, view, rq, CONFIG)


def test_purging_touches_boundary_and_rq_difference(calendar):
    period = (SplitPeriod("dev", 2018, 2019, 2019, "development"),)
    boundary = calendar.get_loc(calendar[calendar.year == 2019][0])
    one = build_temporal_splits(calendar, make_view(calendar, 1), "RQ1", CONFIG,
                                periods=period, gap_sessions=0)[0]
    four = build_temporal_splits(calendar, make_view(calendar, 4), "RQ2", CONFIG,
                                 periods=period, gap_sessions=0)[0]
    assert one.purged_count == 1
    assert four.purged_count == 4
    assert one.train_master_positions[-1] == boundary - 2
    assert four.train_master_positions[-1] == boundary - 5
    assert boundary - 1 not in one.train_master_positions
    assert boundary - 4 not in four.train_master_positions
    assert boundary - 5 in four.train_master_positions


def test_gap_uses_master_not_eligible_rows(calendar):
    period = (SplitPeriod("dev", 2018, 2019, 2019, "development"),)
    boundary = calendar.get_loc(calendar[calendar.year == 2019][0])
    missing = (boundary - 3, boundary - 2)
    view = make_view(calendar, 1, excluded=missing)
    fold = build_temporal_splits(calendar, view, "RQ1", CONFIG, periods=period)[0]
    assert fold.evaluation_boundary_date == calendar[boundary]
    assert fold.gap_count == 3
    assert fold.overlap_exclusions == 1
    assert fold.train_master_positions[-1] == boundary - 6
    assert (calendar[boundary] - calendar[boundary - 5]).days > 5
    assert calendar[boundary - 1] not in view.X.index or calendar[boundary - 1] not in view.X.iloc[list(fold.train_indices)].index


def test_boundary_excludes_development_labels_crossing_test(calendar):
    view = make_view(calendar, 4)
    fold = build_temporal_splits(calendar, view, "RQ2", CONFIG)[2]
    test_start = calendar[calendar.year == 2022][0]
    assert fold.boundary_excluded_count > 0
    assert view.metadata.label_end_date.iloc[list(fold.validation_indices)].max() < test_start
    assert fold.validation_end_date.year == 2021


def test_development_does_not_use_test_labels(calendar):
    view = make_view(calendar, 4)
    original = build_temporal_splits(calendar, view, "RQ2", CONFIG)[:3]
    view.y.loc[view.y.index.year >= 2022] = 1 - view.y.loc[view.y.index.year >= 2022]
    assert build_temporal_splits(calendar, view, "RQ2", CONFIG)[:3] == original


@pytest.mark.parametrize("rq,horizon", [("RQ1", 1), ("RQ2", 4)])
def test_final_training_labels_end_before_test(calendar, rq, horizon):
    view = make_view(calendar, horizon)
    fold = build_temporal_splits(calendar, view, rq, CONFIG)[-1]
    assert view.metadata.label_end_date.iloc[list(fold.train_indices)].max() < fold.evaluation_boundary_date
    assert min(fold.validation_master_positions) == calendar.get_loc(fold.evaluation_boundary_date)


@pytest.mark.parametrize("case", ["empty_train", "empty_validation", "misaligned",
                                       "wrong_horizon", "duplicate_master", "missing_label",
                                       "infinite_feature"])
def test_reject_invalid_inputs(calendar, case):
    view = make_view(calendar, 1)
    periods = (SplitPeriod("dev", 2018, 2019, 2019, "development"),)
    if case == "empty_train":
        periods = (SplitPeriod("dev", 2015, 2019, 2019, "development"),)
    elif case == "empty_validation":
        periods = (SplitPeriod("dev", 2018, 2023, 2023, "development"),)
    elif case == "misaligned":
        view.y.index = view.y.index + pd.Timedelta(days=1)
    elif case == "wrong_horizon":
        view.metadata.iloc[0, 1] = calendar[5]
    elif case == "duplicate_master":
        calendar = calendar.insert(1, calendar[0])
    elif case == "infinite_feature":
        view.X.iloc[0, 0] = np.inf
    else:
        view.y.iloc[0] = np.nan
    with pytest.raises(ValueError):
        build_temporal_splits(calendar, view, "RQ1", CONFIG, periods=periods)


def test_reject_invalid_period_and_gap(calendar):
    view = make_view(calendar, 1)
    with pytest.raises(ValueError, match="Gap"):
        build_temporal_splits(calendar, view, "RQ1", CONFIG, gap_sessions=-1)
    bad = (SplitPeriod("test_in_dev", 2021, 2022, 2022, "development"),)
    with pytest.raises(ValueError, match="vùng Test"):
        build_temporal_splits(calendar, view, "RQ1", CONFIG, periods=bad)
