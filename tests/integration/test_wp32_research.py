"""Kiểm tra split thực tế trên HPG Research mà không sửa dataset nguồn."""

from pathlib import Path

import pytest

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset, make_modeling_view
from src.models.temporal_split import build_temporal_splits

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có Parquet Research trong môi trường kiểm thử")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_research_splits_and_reserved_test(rq):
    """Xác minh hai RQ có fold hợp lệ và nhãn phát triển không chạm Test."""
    config = load_project_config(ROOT / "configs/config.yaml")
    dataset = load_research_dataset(ROOT, config)
    view = make_modeling_view(dataset, config, rq)
    folds = build_temporal_splits(dataset.index, view, rq, config)
    test_boundary = dataset.index[dataset.index.year == 2022][0]
    assert len(folds) == 4
    assert folds[-1].evaluation_boundary_date == test_boundary
    assert folds[-1].validation_start_date == test_boundary
    for fold in folds[:3]:
        assert fold.validation_end_date < test_boundary
        assert view.metadata.label_end_date.iloc[list(fold.validation_indices)].max() < test_boundary
    for fold in folds:
        assert view.metadata.label_end_date.iloc[list(fold.train_indices)].max() < fold.evaluation_boundary_date
        assert fold.excluded_count == fold.purged_count + fold.gap_count - fold.overlap_exclusions
