"""Kiểm thử fingerprint, tính toàn vẹn indices và chặn Test ở cấp API."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset
from src.models.modeling_safety import (
    MANIFEST_PATH, get_development_split, get_development_splits,
    get_final_evaluation_split, require_development_split, _validate_split_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sources():
    """Đọc dữ liệu thật một lần; mỗi test chỉ sửa bản sao trong bộ nhớ."""
    config_path = ROOT / "configs/config.yaml"
    dataset_path = ROOT / RESEARCH_PATH
    config = load_project_config(config_path)
    dataset = load_research_dataset(ROOT, config)
    manifest = json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
    return (manifest, dataset, config,
            hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            hashlib.sha256(config_path.read_bytes()).hexdigest())


def verify(sources, manifest=None, dataset=None, config=None, dataset_hash=None,
           config_hash=None, rq="RQ1"):
    """Gọi validator với nguồn hiện tại, cho phép thay đúng một đầu vào trong test."""
    original_manifest, original_dataset, original_config, original_data_hash, original_config_hash = sources
    return _validate_split_manifest(
        manifest if manifest is not None else original_manifest,
        dataset if dataset is not None else original_dataset,
        config if config is not None else original_config,
        dataset_hash if dataset_hash is not None else original_data_hash,
        config_hash if config_hash is not None else original_config_hash,
        rq,
    )


def test_current_fingerprint_and_folds(sources):
    """Manifest hiện tại phải tái dựng đúng ba fold phát triển và một Test."""
    for rq in ("RQ1", "RQ2"):
        folds = verify(sources, rq=rq)
        assert [fold.fold_name for fold in folds] == ["dev_1", "dev_2", "dev_3", "reserved_test"]
        assert all(fold.role == "development" for fold in folds[:3])
        assert folds[-1].role == "reserved_test"


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64), ("config_sha256", "0" * 64),
    ("path", "data/processed/khac.parquet"), ("master_sessions", 1),
    ("first", "2000-01-01"), ("last", "2030-01-01"),
])
def test_reject_dataset_fingerprint(sources, field, value):
    manifest = deepcopy(sources[0])
    manifest["dataset"][field] = value
    with pytest.raises(ValueError, match=field):
        verify(sources, manifest=manifest)


@pytest.mark.parametrize("change", ["reorder", "missing"])
def test_reject_feature_schema(sources, change):
    manifest = deepcopy(sources[0])
    names = manifest["dataset"]["feature_schema"]
    if change == "reorder":
        names[0], names[1] = names[1], names[0]
    else:
        names.pop()
    with pytest.raises(ValueError, match="feature_schema"):
        verify(sources, manifest=manifest)


@pytest.mark.parametrize("change", ["gap", "year", "holding", "name", "rq", "role"])
def test_reject_split_specification(sources, change):
    manifest = deepcopy(sources[0])
    if change == "gap":
        manifest["split_configuration"]["pre_validation_gap_sessions"] = 4
    elif change == "year":
        manifest["split_configuration"]["periods"][0]["evaluation_start_year"] = 2020
    elif change == "holding":
        manifest["split_configuration"]["fixed_holding_sessions"] = 2
    elif change == "name":
        manifest["folds"][0]["fold_name"] = "khac"
    elif change == "rq":
        manifest["folds"][0]["rq"] = "RQ2"
    else:
        manifest["folds"][0]["role"] = "reserved_test"
    with pytest.raises(ValueError):
        verify(sources, manifest=manifest)


@pytest.mark.parametrize("change", ["negative", "out_of_range", "duplicate",
                                    "overlap", "wrong_date", "wrong_count"])
def test_reject_corrupt_indices(sources, change):
    manifest = deepcopy(sources[0])
    fold = manifest["folds"][0]
    if change == "negative":
        fold["train_indices"][0] = -1
    elif change == "out_of_range":
        fold["train_indices"][-1] = manifest["rq_views"]["RQ1"]["eligible_samples"]
    elif change == "duplicate":
        fold["train_indices"][1] = fold["train_indices"][0]
    elif change == "overlap":
        fold["validation_indices"][0] = fold["train_indices"][0]
    elif change == "wrong_date":
        fold["train_master_positions"][0] += 1
    else:
        fold["train_samples"] -= 1
    with pytest.raises(ValueError):
        verify(sources, manifest=manifest)


def test_reject_dataset_and_config_changes(sources):
    dataset = sources[1].iloc[1:].copy()
    with pytest.raises(ValueError, match="master_sessions"):
        verify(sources, dataset=dataset)
    config = deepcopy(sources[2])
    config["labeling"]["fixed_holding_sessions"] = 2
    with pytest.raises(ValueError):
        verify(sources, config=config)


def test_development_guard_and_final_opt_in(sources):
    folds = get_development_splits(ROOT, "RQ1")
    assert len(folds) == 3
    assert all(require_development_split(fold) is fold for fold in folds)
    with pytest.raises(ValueError, match="Reserved Test"):
        get_development_split(ROOT, "RQ1", "reserved_test")
    with pytest.raises(ValueError, match="final_evaluation=True"):
        get_final_evaluation_split(ROOT, "RQ1")
    test_fold = get_final_evaluation_split(ROOT, "RQ1", final_evaluation=True)
    with pytest.raises(ValueError, match="development"):
        require_development_split(test_fold)
    assert get_development_split(ROOT, "RQ1", "dev_1") == folds[0]
