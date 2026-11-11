"""Xuất báo cáo và chặn export dataset khi Quality Gate chưa đạt."""

import json
import hashlib
import os
import tempfile
from pathlib import Path

import pandas as pd

from .adapters import OHLCV
from .gate import blockers
from .validation import validate
from .verification import EvidenceError, verify_bundle


def write_reports(output, root, reports, metadata):
    destination = Path(output).resolve()
    root = Path(root).resolve()
    for protected in (root / "data" / "raw", root / "data" / "processed"):
        if destination == protected or protected in destination.parents:
            raise ValueError("Không được ghi báo cáo vào raw or processed data")
    # Không ghi đè thư mục đã tồn tại, kể cả khi chạy lại.
    destination.mkdir(parents=False, exist_ok=False)
    for name, frame in reports.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def export_dataset(reports, metadata, root, evidence_path=None):
    issues = blockers(reports, metadata, root, evidence_path)
    if issues:
        raise RuntimeError("Quality Gate chưa đạt:\n- " + "\n- ".join(issues))
    try:
        raw, _, provenance = verify_bundle(root, evidence_path)
    except EvidenceError as exc:
        raise RuntimeError(f"Quality Gate chưa đạt: {exc}") from exc
    if metadata.get("verified_provenance", {}).get("manifest_sha256") != provenance["manifest_sha256"]:
        raise RuntimeError("Quality Gate: manifest không khớp audit trước khi xuất")
    if raw.empty or raw.date.duplicated().any() or not raw.date.is_monotonic_increasing or validate(raw).any(axis=None):
        raise RuntimeError("Quality Gate: dữ liệu xuất không đạt validation")
    target = Path(root) / "data/raw/HPG_10y_daily_validated.parquet"
    sidecar = target.with_suffix(".provenance.json")
    # Ghi file tạm, đọc kiểm tra, sau đó liên kết độc quyền để không ghi đè.
    created_sidecar = False
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".validated-", suffix=".parquet", delete=False) as stream:
            temp_path = Path(stream.name)
            raw.set_index("date").to_parquet(stream)
        restored = pd.read_parquet(temp_path)
        pd.testing.assert_frame_equal(raw.set_index("date"), restored, check_exact=True, check_freq=False)
        provenance["dataset_sha256"] = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        with sidecar.open("x", encoding="utf-8") as stream:
            created_sidecar = True
            json.dump(provenance, stream, ensure_ascii=False, indent=2)
        try:
            os.link(temp_path, target)
        except (PermissionError, OSError):
            os.replace(temp_path, target)
            temp_path = None
    except Exception:
        if created_sidecar:
            sidecar.unlink(missing_ok=True)
        raise
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return target
