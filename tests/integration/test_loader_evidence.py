"""Kiểm tra hồ sơ giả lập; không sử dụng dữ liệu giá thực của dự án."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data_loader.__main__ import run
from src.data_loader.export import export_dataset
from src.data_loader.gate import blockers
from src.data_loader.verification import EvidenceError, verify_bundle


def build_fixture(root):
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    (root / "evidence").mkdir()
    config = {"data": {"start_date": "2020-01-01", "end_date": "2021-01-01", "raw_csv_path": "data/raw/input.csv"}}
    (root / "configs/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    raw = pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-06"],
                        "open": [10., 11., 12.], "high": [12., 13., 14.],
                        "low": [9., 10., 11.], "close": [11., 12., 13.], "volume": [100, 200, 300]})
    raw.to_csv(root / "data/raw/input.csv", index=False)
    raw.iloc[:2].to_csv(root / "evidence/reference.csv", index=False)
    raw.iloc[:2][["date"]].to_csv(root / "evidence/calendar.csv", index=False)
    pd.DataFrame({"date": raw.date, "decision": ["accept", "accept", "exclude"],
                  "reason": ["Đối chiếu", "Đối chiếu", "Ngoài phạm vi"], "reviewer": "Người duyệt",
                  "evidence_reference": ["reference", "reference", "ngoài phạm vi"]}).to_csv(root / "evidence/decisions.csv", index=False)
    (root / "evidence/basis.txt").write_text("Bằng chứng giả lập cho unit test", encoding="utf-8")

    def artifact(path, provider):
        return {"path": path, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                "provider": provider, "version": "bản thử nghiệm 1"}

    spec = {"schema_version": 1, "start_inclusive": "2020-01-01", "end_exclusive": "2020-01-04",
            "reviewer": "Người duyệt", "reviewed_at": "2026-01-01T12:00:00+07:00",
            "same_basis_confirmed": True, "calendar_complete_confirmed": True,
            "calendar_evidence_explanation": "Lịch giả lập đã được đối chiếu", "excluded_scope_reason": "Ngoài phạm vi giả lập",
            "basis": {"price_unit": "VND", "price_adjustment": "unadjusted", "volume_unit": "shares",
                      "volume_adjustment": "unadjusted", "corporate_actions_version": "test-1",
                      "point_in_time": "verified", "evidence_explanation": "Chính sách giả lập"},
            "artifacts": {"raw": artifact("data/raw/input.csv", "nguồn A"),
                          "reference": artifact("evidence/reference.csv", "nguồn B"),
                          "calendar": artifact("evidence/calendar.csv", "nguồn B"),
                          "decisions": artifact("evidence/decisions.csv", "người duyệt"),
                          "basis_evidence": artifact("evidence/basis.txt", "nguồn B")}}
    path = root / "evidence/manifest.json"
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return path, spec


def test_verified_scope_passes_and_export_has_provenance(tmp_path):
    path, _ = build_fixture(tmp_path)
    reports, metadata = run(tmp_path, path)
    assert metadata["quality_gate_blockers"] == []
    assert blockers(reports, metadata, tmp_path, path) == []
    assert reports["sessions"].query("source == 'raw_yahoo_archive'").decision.tolist() == ["accept", "accept", "exclude"]
    target = export_dataset(reports, metadata, tmp_path, path)
    result = pd.read_parquet(target)
    assert len(result) == 2 and list(result.index) == list(pd.to_datetime(["2020-01-02", "2020-01-03"]))
    provenance = json.loads(target.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["scope"]["excluded_raw_sessions"] == 1
    assert provenance["scope"]["excluded_raw_dates"] == ["2020-01-06"]
    assert provenance["dataset_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_dataset(reports, metadata, tmp_path, path)


@pytest.mark.parametrize("problem", ["hash", "basis", "calendar", "mismatch", "decision", "same_source", "ohlc", "missing_session"])
def test_invalid_evidence_blocks_build(tmp_path, problem):
    path, spec = build_fixture(tmp_path)
    if problem == "hash":
        spec["artifacts"]["raw"]["sha256"] = "0" * 64
    elif problem == "basis":
        spec["basis"]["point_in_time"] = "unverified"
    elif problem == "same_source":
        spec["artifacts"]["reference"]["provider"] = "nguồn A"
    else:
        artifact_name = {"calendar": "calendar", "mismatch": "reference", "decision": "decisions", "ohlc": "reference", "missing_session": "reference"}[problem]
        artifact = tmp_path / spec["artifacts"][artifact_name]["path"]
        frame = pd.read_csv(artifact)
        if problem in {"calendar", "missing_session"}:
            frame = frame.iloc[:1]
        elif problem == "mismatch":
            frame.loc[0, "volume"] += 1
        elif problem == "ohlc":
            frame.loc[0, "high"] = 1
        else:
            frame.loc[0, "decision"] = "reject"
        frame.to_csv(artifact, index=False)
        spec["artifacts"][artifact_name]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_bundle(tmp_path, path)
    reports, metadata = run(tmp_path, path)
    assert metadata["quality_gate_blockers"]
    with pytest.raises(RuntimeError, match="Quality Gate"):
        export_dataset(reports, metadata, tmp_path, path)
    assert not (tmp_path / "data/raw/HPG_10y_daily_validated.parquet").exists()


def test_evidence_changed_after_audit_cannot_be_exported(tmp_path):
    path, _ = build_fixture(tmp_path)
    reports, metadata = run(tmp_path, path)
    (tmp_path / "evidence/reference.csv").write_text("date,open\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Quality Gate"):
        export_dataset(reports, metadata, tmp_path, path)
    assert not (tmp_path / "data/raw/HPG_10y_daily_validated.parquet").exists()


def test_cli_builds_only_verified_fixture(tmp_path):
    build_fixture(tmp_path)
    result = subprocess.run([sys.executable, "-m", "src.data_loader", "build", "--root", str(tmp_path),
                             "--evidence", "evidence/manifest.json"], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr + result.stdout
    target = tmp_path / "data/raw/HPG_10y_daily_validated.parquet"
    assert target.exists() and target.with_suffix(".provenance.json").exists()


def test_candidate_can_add_missing_and_replace_a_raw_session(tmp_path):
    path, spec = build_fixture(tmp_path)
    original = pd.read_csv(tmp_path / "data/raw/input.csv")
    original.loc[0, "close"] = 10.5
    original.iloc[[0, 2]].to_csv(tmp_path / "data/raw/input.csv", index=False)
    # Snapshot giả lập có giá theo nghìn VND; giữ nguyên volume.
    candidate_source = pd.read_csv(tmp_path / "evidence/reference.csv")
    candidate_source.loc[:, ["open", "high", "low", "close"]] /= 1000
    source_path = tmp_path / "evidence/nguon_moi.csv"
    candidate_source.to_csv(source_path, index=False)
    decisions = pd.read_csv(tmp_path / "evidence/decisions.csv")
    decisions["source"] = ["nguon_moi", "nguon_moi", "nguon_moi"]
    decisions.to_csv(tmp_path / "evidence/decisions.csv", index=False)
    spec["schema_version"] = 2
    raw_basis = {"price_adjustment": "unadjusted", "volume_unit": "shares",
                 "volume_adjustment": "unadjusted", "corporate_actions_version": "test-1", "point_in_time": "verified"}
    spec["sources"] = {"nguon_moi": {"path": "evidence/nguon_moi.csv", "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                    "provider": "nguon B", "version": "1", "date_column": "date", "price_unit": "1000_VND", **raw_basis}}
    spec["artifacts"]["reference"].update({"date_column": "date", "price_unit": "VND", **raw_basis})
    spec["artifacts"]["reference"]["provider"] = "nguon C"
    spec["artifacts"]["calendar"]["provider"] = "nguon C"
    spec["artifacts"]["decisions"]["sha256"] = hashlib.sha256((tmp_path / "evidence/decisions.csv").read_bytes()).hexdigest()
    spec["basis"]["price_unit"] = "VND"
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    reports, metadata = run(tmp_path, path)
    assert metadata["quality_gate_blockers"] == []
    candidate = reports["candidate_sessions"]
    assert len(candidate) == 2 and candidate.source.tolist() == ["nguon_moi", "nguon_moi"]
    assert candidate.close.tolist() == [11, 12]
    assert pd.read_csv(tmp_path / "data/raw/input.csv").loc[0, "close"] == 10.5
    target = export_dataset(reports, metadata, tmp_path, path)
    assert len(pd.read_parquet(target)) == 2
    candidate_source.loc[0, "volume"] = 999
    candidate_source.to_csv(source_path, index=False)
    spec["sources"]["nguon_moi"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceError, match="khác reference"):
        verify_bundle(tmp_path, path)


def test_candidate_rejects_unverified_basis_and_mismatched_reference(tmp_path):
    path, spec = build_fixture(tmp_path)
    spec["schema_version"] = 2
    spec["basis"]["price_unit"] = "VND"
    spec["sources"] = {"nguon": {**spec["artifacts"]["raw"], "date_column": "date", "price_unit": "1000_VND",
                                 "price_adjustment": "unknown", "volume_unit": "shares", "volume_adjustment": "unadjusted",
                                 "corporate_actions_version": "test-1", "point_in_time": "verified"}}
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_bundle(tmp_path, path)
