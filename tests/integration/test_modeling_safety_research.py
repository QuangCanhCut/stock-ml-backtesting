"""Kiểm tra API an toàn trên Parquet và manifest HPG hiện hành."""

import hashlib
from pathlib import Path

import pytest

from src.models.data import RESEARCH_PATH
from src.models.modeling_safety import MANIFEST_PATH, get_development_splits

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file()
                    or not (ROOT / MANIFEST_PATH).is_file(),
                    reason="Chưa có dataset hoặc manifest Research")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_verified_development_preserves_sources(rq):
    """Chỉ nhận fold phát triển và không thay đổi tệp nguồn khi xác minh."""
    paths = [ROOT / RESEARCH_PATH, ROOT / "configs/config.yaml", ROOT / MANIFEST_PATH]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    folds = get_development_splits(ROOT, rq)
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    assert [fold.fold_name for fold in folds] == ["dev_1", "dev_2", "dev_3"]
    assert all(fold.role == "development" for fold in folds)
    assert before == after
