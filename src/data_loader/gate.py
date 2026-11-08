"""Quality Gate: chỉ xuất dữ liệu khi đủ bằng chứng và không còn cảnh báo."""

from .policy import POLICY
from .verification import EvidenceError, verify_bundle


def blockers(reports, metadata, root=None, evidence_path=None):
    if evidence_path is not None:
        if root is None:
            return ["Thiếu project root để kiểm tra hồ sơ bằng chứng"]
        try:
            verify_bundle(root, evidence_path)
        except EvidenceError as exc:
            return [f"Hồ sơ bằng chứng bị từ chối: {exc}"]
        return []
    sessions = reports["sessions"]
    raw = sessions[sessions.source == "raw_yahoo_archive"]
    kbs = sessions[sessions.source == "kbs_cache"]
    issues = []
    if raw.empty:
        issues.append("Không có dữ liệu Raw để kiểm định")
    unverified = (raw.verification_status != "verified").sum()
    if unverified:
        issues.append(f"Raw chưa được xác minh đầy đủ price, volume và lịch: {int(unverified)}")
    if metadata["missing_cache_years"]:
        issues.append(f"Thiếu cache KBS các năm: {metadata['missing_cache_years']}")
    for description, count in (
        ("Raw chưa có KBS đối chiếu", (raw.verification_status == "unverified_no_kbs_session").sum()),
        ("Raw có cảnh báo cần kiểm tra", (raw.decision == "hold_review").sum()),
        ("Raw không đạt OHLCV validation", (raw.decision == "reject").sum()),
        ("KBS có phiên thiếu trong Raw", (kbs.verification_status == "kbs_only").sum()),
        ("KBS không đạt OHLCV validation", (kbs.decision == "reject").sum()),
    ):
        if count:
            issues.append(f"{description}: {int(count)}")
    if POLICY["price_adjustment"] == "unverified_source_specific" or POLICY["volume_units_and_adjustment"] == "unverified_source_specific":
        issues.append("Chưa xác minh đơn vị và cơ sở điều chỉnh price/volume, corporate actions")
    if any(item["completeness"] == "unverified" for item in metadata["kbs_coverage"]):
        issues.append("Độ đầy đủ của lịch KBS chưa được xác minh độc lập")
    if not metadata["kbs_coverage"]:
        issues.append("Không có lịch KBS để kiểm chứng")
    return issues
