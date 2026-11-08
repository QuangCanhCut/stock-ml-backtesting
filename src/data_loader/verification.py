"""Kiểm tra hồ sơ đối chiếu độc lập trước khi chấp nhận một phạm vi Raw.

Hash bảo đảm tính toàn vẹn file, không chứng minh nguồn hay người duyệt đáng tin.
Các xác nhận về đơn vị và nguồn vẫn cần được con người thẩm định độc lập.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .adapters import OHLCV, canonicalize, read_raw
from .dates import normalize_dates
from .validation import validate


class EvidenceError(ValueError):
    """Hồ sơ thiếu bằng chứng hoặc không nhất quán."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"unknown", "unverified", "n/a", "unknown_source"}:
        raise EvidenceError(f"Thiếu hoặc chưa xác minh: {name}")
    return value.strip()


def _artifact(root, record, label):
    if not isinstance(record, dict):
        raise EvidenceError(f"Thiếu artifact {label}")
    rel = Path(_text(record.get("path"), f"{label}.path"))
    if rel.is_absolute():
        raise EvidenceError(f"Đường dẫn {label} phải tương đối với project root")
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_file():
        raise EvidenceError(f"Không tìm thấy artifact {label} trong project root")
    expected = _text(record.get("sha256"), f"{label}.sha256").lower()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(expected) != 64 or digest != expected:
        raise EvidenceError(f"SHA-256 không khớp: {label}")
    _text(record.get("provider"), f"{label}.provider")
    _text(record.get("version"), f"{label}.version")
    return path


def verify_bundle(root, manifest_path):
    """Trả dữ liệu gốc đã chọn và provenance nếu hồ sơ qua mọi kiểm tra.

    Quy trình không sửa ngày, không nội suy và không đổi đơn vị. Chỉ chấp nhận
    OHLCV Raw khớp tuyệt đối với snapshot độc lập cùng cơ sở trong phạm vi chọn.
    """
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    manifest_path = (manifest_path if manifest_path.is_absolute() else root / manifest_path).resolve()
    if root not in manifest_path.parents or not manifest_path.is_file():
        raise EvidenceError("Manifest phải là file trong project root")
    try:
        manifest_bytes = manifest_path.read_bytes()
        spec = json.loads(manifest_bytes)
        if isinstance(spec, dict) and spec.get("schema_version") == 2:
            from .candidate import verify_candidate
            return verify_candidate(root, manifest_path, manifest_bytes, spec)
        if not isinstance(spec, dict) or spec.get("schema_version") != 1:
            raise EvidenceError("Manifest cần schema_version bằng 1 hoặc 2")
        reviewer = _text(spec.get("reviewer"), "reviewer")
        _text(spec.get("reviewed_at"), "reviewed_at")
        reviewed_at = pd.Timestamp(spec["reviewed_at"])
        if pd.isna(reviewed_at) or reviewed_at.tzinfo is None:
            raise EvidenceError("reviewed_at cần timestamp có timezone")
        start = pd.Timestamp(_text(spec.get("start_inclusive"), "start_inclusive"))
        end = pd.Timestamp(_text(spec.get("end_exclusive"), "end_exclusive"))
        if start.tzinfo or end.tzinfo or start != start.normalize() or end != end.normalize() or start >= end:
            raise EvidenceError("Phạm vi phải là ngày không timezone, start < end")
        config = yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8"))["data"]
        if start < pd.Timestamp(config["start_date"]) or end > pd.Timestamp(config["end_date"]):
            raise EvidenceError("Phạm vi kiểm chứng nằm ngoài cấu hình gốc")

        artifacts = spec["artifacts"]
        paths = {name: _artifact(root, artifacts.get(name), name)
                 for name in ("raw", "reference", "calendar", "decisions", "basis_evidence")}
        if paths["raw"] != (root / config["raw_csv_path"]).resolve():
            raise EvidenceError("Artifact Raw không khớp Raw trong cấu hình")
        if len({paths[name] for name in paths}) != len(paths):
            raise EvidenceError("Các artifact phải là file riêng biệt")
        raw_provider = artifacts["raw"]["provider"].strip().casefold()
        if artifacts["reference"]["provider"].strip().casefold() == raw_provider:
            raise EvidenceError("Nguồn đối chiếu phải khác nhà cung cấp Raw")
        if artifacts["calendar"]["provider"].strip().casefold() == raw_provider:
            raise EvidenceError("Lịch đối chiếu phải khác nhà cung cấp Raw")

        basis = spec["basis"]
        for key in ("price_unit", "price_adjustment", "volume_unit", "volume_adjustment", "corporate_actions_version", "evidence_explanation"):
            _text(basis.get(key), f"basis.{key}")
        if basis.get("point_in_time") != "verified":
            raise EvidenceError("Chưa có xác nhận point-in-time cho basis và corporate actions")
        if spec.get("same_basis_confirmed") is not True or spec.get("calendar_complete_confirmed") is not True:
            raise EvidenceError("Chưa xác nhận cùng basis và độ đầy đủ lịch bằng chứng")
        _text(spec.get("calendar_evidence_explanation"), "calendar_evidence_explanation")
        _text(spec.get("excluded_scope_reason"), "excluded_scope_reason")

        raw = read_raw(paths["raw"])
        ref = canonicalize(pd.read_csv(paths["reference"]), "date")
        calendar = pd.read_csv(paths["calendar"])
        if "date" not in calendar:
            raise EvidenceError("Lịch thiếu cột date")
        calendar["date"] = normalize_dates(calendar["date"])
        if calendar.date.duplicated().any():
            raise EvidenceError("Lịch có ngày trùng")
        decisions = pd.read_csv(paths["decisions"], dtype=str, keep_default_na=False)
        if not {"date", "decision", "reason", "reviewer", "evidence_reference"}.issubset(decisions.columns):
            raise EvidenceError("Quyết định thiếu date, decision, reason, reviewer hoặc evidence_reference")
        decisions["date"] = normalize_dates(decisions["date"])
        if decisions.date.duplicated().any() or set(decisions.date) != set(raw.date):
            raise EvidenceError("Quyết định phải bao phủ một lần mỗi phiên Raw")
        for column in ("reason", "reviewer", "evidence_reference"):
            if decisions[column].str.strip().str.lower().isin(["", "unknown", "unverified", "n/a"]).any():
                raise EvidenceError(f"Quyết định thiếu {column}")
        inside = decisions.date.ge(start) & decisions.date.lt(end)
        if not decisions.loc[inside, "decision"].eq("accept").all() or not decisions.loc[~inside, "decision"].eq("exclude").all():
            raise EvidenceError("Mỗi phiên trong phạm vi phải accept; ngoài phạm vi phải exclude")
        if not decisions.loc[inside, "evidence_reference"].eq("reference").all():
            raise EvidenceError("Mỗi phiên accept phải tham chiếu artifact reference đã đối chiếu")
        candidate = raw[raw.date.ge(start) & raw.date.lt(end)].reset_index(drop=True)
        ref = ref[ref.date.ge(start) & ref.date.lt(end)].reset_index(drop=True)
        dates = set(calendar.loc[calendar.date.ge(start) & calendar.date.lt(end), "date"])
        if candidate.empty or set(candidate.date) != dates or set(ref.date) != dates:
            raise EvidenceError("Lịch, Raw và nguồn độc lập không khớp đầy đủ trong phạm vi")
        if validate(candidate).any(axis=None) or validate(ref).any(axis=None):
            raise EvidenceError("OHLCV Raw hoặc nguồn đối chiếu không hợp lệ")
        compared = candidate.merge(ref, on="date", suffixes=("_raw", "_reference"), validate="one_to_one")
        for col in OHLCV:
            a = pd.to_numeric(compared[f"{col}_raw"]).to_numpy(dtype=float)
            b = pd.to_numeric(compared[f"{col}_reference"]).to_numpy(dtype=float)
            if not np.array_equal(a, b):
                raise EvidenceError(f"{col} khác nguồn đối chiếu; không tự chuyển đổi")
        raw_dates = raw.date.sort_values()
        excluded = raw[~raw.date.isin(candidate.date)]
        provenance = {"manifest": str(manifest_path.relative_to(root)),
                      "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                      "reviewer": reviewer, "reviewed_at": spec["reviewed_at"],
                      "artifacts": artifacts, "basis": basis,
                      "scope": {"start_inclusive": str(start.date()), "end_exclusive": str(end.date()),
                                "sessions": len(candidate), "excluded_raw_sessions": len(excluded),
                                "excluded_raw_dates": excluded.date.dt.strftime("%Y-%m-%d").tolist(),
                                "excluded_scope_reason": spec["excluded_scope_reason"],
                                "original_first": str(raw_dates.iloc[0].date()), "original_last": str(raw_dates.iloc[-1].date())},
                      "calendar_evidence_explanation": spec["calendar_evidence_explanation"],
                      "limitation": "Hash và xác nhận thủ công không tự chứng minh tính độc lập hay độ đúng của nguồn."}
        decisions = decisions.sort_values("date").reset_index(drop=True)
        return candidate, decisions, provenance
    except EvidenceError:
        raise
    except (AttributeError, KeyError, TypeError, OSError, ValueError, UnicodeError, OverflowError, pd.errors.ParserError) as exc:
        raise EvidenceError(f"Hồ sơ bằng chứng không hợp lệ: {exc}") from exc
