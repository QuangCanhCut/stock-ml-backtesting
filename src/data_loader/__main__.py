"""Audit offline và thực thi Quality Gate trước khi build dataset."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from .adapters import read_kbs_cache, read_raw, read_yahoo_rejection_evidence
from .alignment import diagnostic_2025
from .export import export_dataset, write_reports
from .gate import blockers
from .policy import POLICY
from .reconcile import reconcile
from .reports import yearly_source_report
from .verification import EvidenceError, verify_bundle


def run(root, evidence_path=None):
    root = Path(root).resolve()
    with (root / "configs/config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)["data"]
    start, end = pd.Timestamp(config["start_date"]), pd.Timestamp(config["end_date"])
    years = range(start.year, (end - pd.Timedelta(days=1)).year + 1)
    raw = read_raw(root / config["raw_csv_path"])
    kbs, missing_years = read_kbs_cache(root / "results/data_audit/kbs_cache", years)
    raw = raw[(raw.date >= start) & (raw.date < end)].reset_index(drop=True)
    kbs = kbs[(kbs.date >= start) & (kbs.date < end)].reset_index(drop=True)
    yahoo = read_yahoo_rejection_evidence(root / "results/data_audit/HPG_yahoo_rejections_full_history.csv")
    yahoo = yahoo[(yahoo.date >= start) & (yahoo.date < end)].reset_index(drop=True)
    alignment = diagnostic_2025(raw, kbs) if not kbs.empty else pd.DataFrame(columns=["date", "alignment"])
    sessions = reconcile(raw, kbs, yahoo, alignment)
    summary = yearly_source_report(sessions)
    coverage = [{"year": int(y), "first_cached_session": str(g.date.min().date()),
                 "last_cached_session": str(g.date.max().date()), "cached_rows": len(g),
                 "completeness": "unverified"} for y, g in kbs.groupby(kbs.date.dt.year)]
    input_paths = [root / config["raw_csv_path"], root / "results/data_audit/HPG_yahoo_rejections_full_history.csv"]
    input_paths.extend(root / f"results/data_audit/kbs_cache/HPG_KBS_{year}.parquet" for year in years)
    provenance = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in input_paths if path.exists()]
    metadata = {"mode": "audit", "interval": {"start_inclusive": str(start.date()), "end_exclusive": str(end.date())},
                "missing_cache_years": missing_years, "kbs_coverage": coverage, "policy": POLICY,
                "provenance": provenance,
                "yahoo_evidence_scope": "Snapshot lịch sử chỉ gồm các ngày thiếu được audit; không phải toàn bộ Yahoo",
                "unresolved": ["Độ đầy đủ lịch KBS", "Cơ sở price, volume và corporate actions",
                               "Date alignment 2025", "Volume outlier", "Phiên Yahoo bị thiếu"],
                "alignment_2025_counts": {str(k): int(v) for k, v in alignment.alignment.value_counts().items()}}
    reports = {"sessions": sessions, "yearly_source_quality": summary, "alignment_2025": alignment}
    if evidence_path is None:
        metadata["quality_gate_blockers"] = blockers(reports, metadata)
    else:
        try:
            candidate, decisions, provenance = verify_bundle(root, evidence_path)
            if provenance.get("candidate_builder") == "schema_2":
                reports["candidate_sessions"] = candidate.merge(
                    decisions.loc[decisions.decision.eq("accept"), ["date", "source", "reason", "reviewer", "evidence_reference"]],
                    on="date", validate="one_to_one")
            else:
                by_date = decisions.set_index("date")
                raw_mask = sessions.source.eq("raw_yahoo_archive")
                selected = sessions.loc[raw_mask, "date"].map(by_date["decision"])
                sessions.loc[raw_mask, "verification_status"] = selected.map({"accept": "verified", "exclude": "excluded_scope"}).values
                sessions.loc[raw_mask, "decision"] = selected.map({"accept": "accept", "exclude": "exclude"}).values
                sessions.loc[raw_mask, "reason"] = sessions.loc[raw_mask, "date"].map(by_date["reason"]).values
            reports["verification_decisions"] = decisions
            reports["yearly_source_quality"] = yearly_source_report(sessions)
            metadata["verified_provenance"] = provenance
            metadata["policy"] = {**POLICY, "dataset_export": "evidence_gate_passed",
                                  "verified_basis": provenance["basis"]}
            metadata["quality_gate_blockers"] = []
        except EvidenceError as exc:
            metadata["quality_gate_blockers"] = [f"Hồ sơ bằng chứng bị từ chối: {exc}"]
    return reports, metadata


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "build"], help="audit offline hoặc build qua Quality Gate")
    parser.add_argument("--root", default=".", help="thư mục gốc của dự án")
    parser.add_argument("--output-dir", help="thư mục mới để lưu báo cáo audit")
    parser.add_argument("--evidence", help="đường dẫn manifest bằng chứng được thẩm định trong project")
    parser.add_argument("--profile", choices=["validated", "research"], default="validated", help="chính sách dữ liệu khi build")
    args = parser.parse_args()
    if args.profile == "research":
        if args.command != "build" or args.evidence or args.output_dir:
            parser.error("Profile research chỉ dùng với build, không dùng --evidence hoặc --output-dir")
        from .research import export_research
        try:
            target, quality = export_research(args.root)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Research gate BLOCKED: {exc}")
            raise SystemExit(2) from exc
        print(json.dumps(quality, indent=2, ensure_ascii=False))
        print(f"Đã tạo Raw Research: {target}")
        return
    reports, metadata = run(args.root, args.evidence)
    print(reports["yearly_source_quality"].to_string(index=False))
    print("\n" + json.dumps(metadata, indent=2, ensure_ascii=False))
    if args.output_dir:
        write_reports(args.output_dir, args.root, reports, metadata)
        print(f"Đã ghi báo cáo audit tại {args.output_dir}.")
    else:
        print("Chưa ghi báo cáo; dùng --output-dir với thư mục mới để lưu.")
    if args.command == "build":
        try:
            target = export_dataset(reports, metadata, args.root, args.evidence)
        except RuntimeError as exc:
            print(str(exc))
            raise SystemExit(2) from exc
        print(f"Đã xuất dataset qua Quality Gate: {target}")


if __name__ == "__main__":
    main()
