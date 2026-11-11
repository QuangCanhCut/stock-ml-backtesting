"""Xác minh manifest WP3.2 và tách quyền truy cập fold phát triển với Test."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset_builder import expected_feature_columns, load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, load_research_dataset, make_modeling_view
from src.models.temporal_split import (
    DEFAULT_PERIODS, PRE_VALIDATION_GAP_SESSIONS, TEST_START_YEAR,
    TemporalSplit, build_temporal_splits,
)

MANIFEST_PATH = Path("results/modeling/splits/split_manifest.json")


def get_development_splits(root: Path, rq: str) -> tuple[TemporalSplit, ...]:
    """Trả các fold phát triển đã xác minh, không trao chỉ số Reserved Test.

    Args:
        root: Thư mục gốc repository chứa dataset, config và manifest.
        rq: RQ1 hoặc RQ2.

    Returns:
        Ba fold development với indices trong ModelingView của cùng RQ.

    Raises:
        ValueError: Khi fingerprint, manifest hoặc fold không nhất quán.
        FileNotFoundError: Khi thiếu một tệp nguồn bắt buộc.
    """
    return tuple(split for split in _load_verified_splits(root, rq)
                 if split.role == "development")


def get_development_split(root: Path, rq: str, fold_name: str) -> TemporalSplit:
    """Lấy một fold phát triển và từ chối tên fold Reserved Test."""
    if fold_name == "reserved_test":
        raise ValueError("Không được truy cập Reserved Test trong chế độ phát triển")
    for split in get_development_splits(root, rq):
        if split.fold_name == fold_name:
            return split
    raise ValueError(f"Không có fold phát triển tên {fold_name!r} cho {rq}")


def require_development_split(split: TemporalSplit) -> TemporalSplit:
    """Chặn việc truyền fold Test vào một bước chỉ dành cho Development."""
    if not isinstance(split, TemporalSplit) or split.role != "development":
        raise ValueError("Bước phát triển chỉ nhận fold có role='development'")
    return split


def get_final_evaluation_split(
    root: Path, rq: str, *, final_evaluation: bool = False,
) -> TemporalSplit:
    """Chỉ trao fold Test sau yêu cầu Final Evaluation tường minh.

    Hàm chỉ trả indices; không huấn luyện, dự đoán hoặc tính hiệu suất Test.
    """
    if final_evaluation is not True:
        raise ValueError("Reserved Test chỉ được truy cập khi final_evaluation=True")
    matches = [split for split in _load_verified_splits(root, rq)
               if split.role == "reserved_test"]
    if len(matches) != 1:
        raise ValueError(f"{rq}: manifest phải có đúng một fold Reserved Test")
    return matches[0]


def _validate_split_manifest(
    manifest: dict[str, Any], dataset: pd.DataFrame, config: dict,
    dataset_sha256: str, config_sha256: str, rq: str,
) -> tuple[TemporalSplit, ...]:
    """Đối chiếu fingerprint và toàn bộ indices với logic WP3.2 hiện hành.

    Args:
        manifest: JSON manifest đã đọc từ WP3.2.
        dataset: Master dataset Research đã qua kiểm tra schema.
        config: Cấu hình Phase 2 hiện tại.
        dataset_sha256: Hash của bytes Parquet hiện tại.
        config_sha256: Hash của bytes YAML hiện tại.
        rq: Bài toán cần chuẩn bị modeling.

    Returns:
        Các fold tính lại và đã đối chiếu với manifest, gồm cả Test nội bộ.

    Raises:
        ValueError: Khi bất kỳ fingerprint, chỉ số hoặc ranh giới nào lệch.
    """
    if rq not in RQ_TARGETS:
        raise ValueError(f"Câu hỏi nghiên cứu không được hỗ trợ: {rq}")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Phiên bản split manifest không được hỗ trợ")
    source = manifest.get("dataset")
    if not isinstance(source, dict):
        raise ValueError("Manifest thiếu thông tin dataset")
    expected_source = {
        "path": RESEARCH_PATH.as_posix(),
        "sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "feature_schema": expected_feature_columns(config),
        "master_sessions": len(dataset),
        "first": str(dataset.index[0].date()),
        "last": str(dataset.index[-1].date()),
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"Fingerprint dataset/config không khớp trường {key}")
    if (dataset.attrs.get("feature_columns") != expected_source["feature_schema"]
            or list(dataset[expected_source["feature_schema"]].columns)
            != expected_source["feature_schema"]):
        raise ValueError("Thứ tự feature của dataset không khớp cấu hình")

    specification = manifest.get("split_configuration")
    expected_specification = {
        "pre_validation_gap_sessions": PRE_VALIDATION_GAP_SESSIONS,
        "test_start_year": TEST_START_YEAR,
        "fixed_holding_sessions": config["labeling"]["fixed_holding_sessions"],
        "periods": [vars(period) for period in DEFAULT_PERIODS],
    }
    if specification != expected_specification:
        raise ValueError("Thông số split, gap, horizon hoặc ranh giới năm đã thay đổi")
    validation = manifest.get("validation")
    if (manifest.get("status") != "COMPLETED_WITH_LIMITATIONS"
            or not isinstance(validation, dict) or validation.get("status") != "PASS"):
        raise ValueError("Manifest WP3.2 chưa có trạng thái kiểm định hợp lệ")

    all_folds = manifest.get("folds")
    if (not isinstance(all_folds, list)
            or len(all_folds) != len(DEFAULT_PERIODS) * len(RQ_TARGETS)
            or not all(isinstance(fold, dict) for fold in all_folds)):
        raise ValueError("Số lượng fold trong manifest không khớp thiết kế WP3.2")
    if {fold.get("rq") for fold in all_folds} != set(RQ_TARGETS):
        raise ValueError("Danh sách RQ của manifest không khớp thiết kế")
    views = manifest.get("rq_views")
    if not isinstance(views, dict) or set(views) != set(RQ_TARGETS):
        raise ValueError("Manifest thiếu metadata modeling view của RQ")
    verified = {}
    for current_rq, target in RQ_TARGETS.items():
        view = make_modeling_view(dataset, config, current_rq)
        expected_view = {"target": target, "eligible_samples": len(view.X),
                         "first": str(view.X.index[0].date()),
                         "last": str(view.X.index[-1].date())}
        if views[current_rq] != expected_view:
            raise ValueError(f"{current_rq}: metadata view không khớp dataset hiện tại")
        expected_folds = build_temporal_splits(dataset.index, view, current_rq, config)
        stored = [fold for fold in all_folds if fold.get("rq") == current_rq]
        if len(stored) != len(expected_folds):
            raise ValueError(f"{current_rq}: số fold không khớp")
        for record, expected in zip(stored, expected_folds, strict=True):
            _validate_fold_record(record, expected, len(view.X), dataset.index, view.X.index)
        verified[current_rq] = expected_folds
    return verified[rq]


def _validate_fold_record(
    record: dict, expected: TemporalSplit, view_size: int,
    master_index: pd.DatetimeIndex, view_index: pd.DatetimeIndex,
) -> None:
    """Kiểm tra hình dạng index trước, sau đó so với fold tái tính từ nguồn."""
    if not isinstance(record, dict):
        raise ValueError("Bản ghi fold trong manifest phải là object")
    for key in ("train_indices", "validation_indices"):
        indices = record.get(key)
        if (not isinstance(indices, list)
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or i < 0 or i >= view_size for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError(f"{expected.rq}/{expected.fold_name}: {key} không hợp lệ")
    if set(record["train_indices"]) & set(record["validation_indices"]):
        raise ValueError(f"{expected.rq}/{expected.fold_name}: Train và Validation giao nhau")
    if (record.get("train_samples") != len(record["train_indices"])
            or record.get("validation_samples") != len(record["validation_indices"])):
        raise ValueError(f"{expected.rq}/{expected.fold_name}: số mẫu không khớp indices")
    fields = {
        "rq": expected.rq, "fold_name": expected.fold_name, "role": expected.role,
        "train_indices": list(expected.train_indices),
        "validation_indices": list(expected.validation_indices),
        "train_master_positions": list(expected.train_master_positions),
        "validation_master_positions": list(expected.validation_master_positions),
        "train_start": str(expected.train_start_date.date()),
        "train_end": str(expected.train_end_date.date()),
        "validation_start": str(expected.validation_start_date.date()),
        "validation_end": str(expected.validation_end_date.date()),
        "evaluation_boundary": str(expected.evaluation_boundary_date.date()),
        "purged_samples": expected.purged_count, "gap_samples": expected.gap_count,
        "overlap_exclusions": expected.overlap_exclusions,
        "total_train_exclusions": expected.excluded_count,
        "boundary_exclusions": expected.boundary_excluded_count,
        "train_samples": len(expected.train_indices),
        "validation_samples": len(expected.validation_indices),
        "train_class_counts": expected.train_class_counts,
        "validation_class_counts": expected.validation_class_counts,
        "train_one_class": min(expected.train_class_counts.values()) == 0,
        "validation_one_class": (min(expected.validation_class_counts.values()) == 0
                                 if expected.validation_class_counts is not None else None),
        "status": "PASS",
    }
    for key, value in fields.items():
        if record.get(key) != value:
            raise ValueError(f"{expected.rq}/{expected.fold_name}: trường {key} không khớp")
    for indices_key, positions_key in (("train_indices", "train_master_positions"),
                                       ("validation_indices", "validation_master_positions")):
        actual_dates = view_index.take(record[indices_key])
        master_dates = master_index.take(record[positions_key])
        if not actual_dates.equals(master_dates):
            raise ValueError(f"{expected.rq}/{expected.fold_name}: ngày giao dịch không khớp master")


def _load_verified_splits(root: Path, rq: str) -> tuple[TemporalSplit, ...]:
    """Đọc nguồn hiện tại, từ chối manifest cũ mà không tự tái tạo tệp."""
    root = Path(root).resolve()
    config_path = root / "configs/config.yaml"
    config = load_project_config(config_path)
    dataset = load_research_dataset(root, config)
    manifest_path = root / MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy split manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Split manifest không phải JSON hợp lệ") from exc
    return _validate_split_manifest(
        manifest, dataset, config,
        hashlib.sha256((root / RESEARCH_PATH).read_bytes()).hexdigest(),
        hashlib.sha256(config_path.read_bytes()).hexdigest(), rq,
    )
