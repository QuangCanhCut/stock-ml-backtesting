"""Xây candidate từ các snapshot đã có quyết định kiểm chứng từng phiên."""

import hashlib

import numpy as np
import pandas as pd
import yaml

from .adapters import OHLCV, canonicalize, read_raw
from .dates import normalize_dates
from .validation import validate


def verify_candidate(root, manifest_path, manifest_bytes, spec):
    # Import tại đây để tránh phụ thuộc vòng khi verify_bundle chọn schema.
    from .verification import EvidenceError, _artifact, _text

    reviewer = _text(spec.get("reviewer"), "reviewer")
    reviewed_at = pd.Timestamp(_text(spec.get("reviewed_at"), "reviewed_at"))
    if pd.isna(reviewed_at) or reviewed_at.tzinfo is None:
        raise EvidenceError("reviewed_at cần timestamp có timezone")
    start = pd.Timestamp(_text(spec.get("start_inclusive"), "start_inclusive"))
    end = pd.Timestamp(_text(spec.get("end_exclusive"), "end_exclusive"))
    if start.tzinfo or end.tzinfo or start != start.normalize() or end != end.normalize() or start >= end:
        raise EvidenceError("Phạm vi candidate không hợp lệ")
    config = yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8"))["data"]
    if start < pd.Timestamp(config["start_date"]) or end > pd.Timestamp(config["end_date"]):
        raise EvidenceError("Phạm vi candidate nằm ngoài cấu hình")

    basis = spec["basis"]
    for field in ("price_adjustment", "volume_unit", "volume_adjustment", "corporate_actions_version", "evidence_explanation"):
        _text(basis.get(field), f"basis.{field}")
    if basis.get("price_unit") != "VND" or basis.get("volume_unit") != "shares":
        raise EvidenceError("Candidate yêu cầu đơn vị VND và shares")
    if basis.get("point_in_time") != "verified" or spec.get("same_basis_confirmed") is not True:
        raise EvidenceError("Chưa xác minh basis hoặc point-in-time")
    if spec.get("calendar_complete_confirmed") is not True:
        raise EvidenceError("Chưa xác minh độ đầy đủ lịch")
    _text(spec.get("calendar_evidence_explanation"), "calendar_evidence_explanation")
    _text(spec.get("excluded_scope_reason"), "excluded_scope_reason")

    artifacts = spec["artifacts"]
    paths = {name: _artifact(root, artifacts.get(name), name)
             for name in ("reference", "calendar", "decisions", "basis_evidence")}
    sources = spec["sources"]
    if not isinstance(sources, dict) or not sources:
        raise EvidenceError("Không có snapshot nguồn cho candidate")
    frames = {}
    for key, record in sources.items():
        _text(key, "source id")
        path = _artifact(root, record, f"sources.{key}")
        if path in paths.values():
            raise EvidenceError("Snapshot nguồn không được trùng artifact đối chiếu")
        paths[f"source:{key}"] = path
        frames[key] = _load_normalized(path, record, basis, _text, EvidenceError)
    if len(set(paths.values())) != len(paths):
        raise EvidenceError("Các artifact phải là file riêng biệt")
    reference_provider = _text(artifacts["reference"].get("provider"), "reference.provider").casefold()
    if any(_text(s.get("provider"), "source.provider").casefold() == reference_provider for s in sources.values()):
        raise EvidenceError("Reference phải khác provider của mọi nguồn candidate")
    calendar_provider = _text(artifacts["calendar"].get("provider"), "calendar.provider").casefold()
    if any(_text(s.get("provider"), "source.provider").casefold() == calendar_provider for s in sources.values()):
        raise EvidenceError("Calendar phải khác provider của mọi nguồn candidate")
    reference = _load_normalized(paths["reference"], artifacts["reference"], basis, _text, EvidenceError)

    calendar = pd.read_csv(paths["calendar"])
    calendar["date"] = normalize_dates(calendar["date"])
    if calendar.date.duplicated().any():
        raise EvidenceError("Ngày trong calendar bị trùng")
    dates = set(calendar.loc[calendar.date.ge(start) & calendar.date.lt(end), "date"])
    decisions = pd.read_csv(paths["decisions"], dtype=str, keep_default_na=False)
    needed = {"date", "decision", "source", "reason", "reviewer", "evidence_reference"}
    if not needed.issubset(decisions.columns):
        raise EvidenceError(f"Quyết định candidate thiếu cột: {sorted(needed - set(decisions.columns))}")
    decisions["date"] = normalize_dates(decisions["date"])
    if decisions.date.duplicated().any():
        raise EvidenceError("Quyết định candidate bị trùng ngày")
    selected = decisions[decisions.date.ge(start) & decisions.date.lt(end)].copy()
    excluded = decisions[~(decisions.date.ge(start) & decisions.date.lt(end))]
    if not dates or set(selected.date) != dates:
        raise EvidenceError("Quyết định không bao phủ đầy đủ calendar trong phạm vi")
    if not selected.decision.eq("accept").all() or not excluded.decision.eq("exclude").all():
        raise EvidenceError("Phiên trong phạm vi cần accept; ngoài phạm vi cần exclude")
    if not selected.evidence_reference.eq("reference").all():
        raise EvidenceError("Mỗi phiên accept phải tham chiếu reference")
    for column in ("reason", "reviewer", "evidence_reference"):
        if decisions[column].str.strip().str.lower().isin(["", "unknown", "unverified", "n/a"]).any():
            raise EvidenceError(f"Quyết định thiếu {column}")
    if not set(selected.source).issubset(frames):
        raise EvidenceError("Quyết định sử dụng snapshot nguồn chưa khai báo")

    parts = []
    for source, chosen in selected.groupby("source"):
        data = frames[source].merge(chosen[["date", "source"]], on="date", validate="one_to_one")
        if len(data) != len(chosen):
            raise EvidenceError(f"Snapshot {source} thiếu phiên đã chọn")
        parts.append(data)
    candidate = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    reference = reference[reference.date.ge(start) & reference.date.lt(end)]
    if set(reference.date) != dates:
        raise EvidenceError("Reference không bao phủ đầy đủ calendar")
    if validate(candidate).any(axis=None) or validate(reference).any(axis=None):
        raise EvidenceError("Candidate hoặc reference không đạt OHLCV validation")
    compare = candidate.merge(reference, on="date", suffixes=("_candidate", "_reference"), validate="one_to_one")
    for col in OHLCV:
        if not np.array_equal(compare[f"{col}_candidate"].to_numpy(dtype=float), compare[f"{col}_reference"].to_numpy(dtype=float)):
            raise EvidenceError(f"{col} candidate khác reference sau chuẩn hóa đơn vị")

    original = read_raw(root / config["raw_csv_path"])
    outside = original.loc[original.date.lt(start) | original.date.ge(end), "date"]
    provenance = {"manifest": str(manifest_path.relative_to(root)),
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "reviewer": reviewer,
                  "reviewed_at": spec["reviewed_at"], "artifacts": artifacts, "sources": sources, "basis": basis,
                  "candidate_builder": "schema_2", "transform": "VND giữ nguyên; 1000_VND nhân 1000; volume giữ nguyên",
                  "scope": {"start_inclusive": str(start.date()), "end_exclusive": str(end.date()),
                            "sessions": len(candidate), "excluded_raw_sessions": len(outside),
                            "excluded_raw_dates": outside.dt.strftime("%Y-%m-%d").tolist(),
                            "excluded_scope_reason": spec["excluded_scope_reason"]},
                  "limitation": "Hash và xác nhận thủ công không tự chứng minh tính độc lập hay độ đúng của nguồn."}
    return candidate[["date", *OHLCV]], decisions.sort_values("date").reset_index(drop=True), provenance


def _load_normalized(path, record, basis, text, error):
    for field in ("price_adjustment", "volume_unit", "volume_adjustment", "corporate_actions_version"):
        if text(record.get(field), field) != basis[field]:
            raise error(f"Nguồn có {field} khác basis candidate")
    if record.get("point_in_time") != "verified":
        raise error("Nguồn chưa xác minh point-in-time")
    unit = record.get("price_unit")
    if unit not in {"VND", "1000_VND"}:
        raise error("Chưa xác minh đơn vị giá của nguồn")
    date_column = text(record.get("date_column"), "date_column")
    if path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        raise error("Snapshot phải là CSV hoặc Parquet")
    frame = canonicalize(data, date_column)
    # Chỉ đổi đơn vị tiền; không thay đổi adjustment hay volume.
    factor = 1000 if unit == "1000_VND" else 1
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="raise") * factor
    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise")
    return frame
