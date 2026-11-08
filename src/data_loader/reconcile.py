"""Đối chiếu từng nguồn mà không tạo hoặc ghép dataset."""

import numpy as np
import pandas as pd

from .adapters import OHLCV
from .validation import validate


def volume_diagnostics(raw, kbs):
    """Tìm bất thường hai chiều bằng median 30 phiên trước, không biến đổi volume.

    Ngưỡng 10 lần chỉ để sàng lọc, không xác nhận đơn vị hay corporate action.
    Các phép tính chỉ dùng dữ liệu trước phiên đang xét.
    """
    values = raw.sort_values("date").set_index("date").volume.astype(float)
    kbs_volume = kbs.set_index("date").volume.astype(float)
    ratios = values.div(kbs_volume.reindex(values.index)).replace([np.inf, -np.inf], np.nan)
    results = {}
    for name, series in (("volume_source", values), ("volume_cross_source_ratio", ratios)):
        baseline = series.where(series > 0).rolling(30, min_periods=10).median().shift(1)
        for label, condition in (("high", series > baseline * 10), ("low", (series > 0) & (series < baseline / 10))):
            for date in series.index[condition.fillna(False)]:
                results.setdefault(date, []).append(f"{name}_{label}_exploratory")
    return results


def reconcile(raw, kbs, yahoo_evidence, alignment):
    records = []
    raw_dates, kbs_dates = set(raw.date), set(kbs.date)
    suspects = set(alignment.loc[alignment.alignment == "PREVIOUS_SESSION", "date"])
    volume_outliers = volume_diagnostics(raw, kbs)
    for source, frame, other_dates in (("raw_yahoo_archive", raw, kbs_dates), ("kbs_cache", kbs, raw_dates)):
        flags = validate(frame)
        for index, row in frame.iterrows():
            problems = list(flags.columns[flags.loc[index]])
            other_present = row.date in other_dates
            if source == "raw_yahoo_archive" and row.date in suspects:
                problems.append("possible_previous_session_alignment_exploratory")
            if source == "raw_yahoo_archive":
                problems.extend(volume_outliers.get(row.date, []))
            invalid = bool(flags.loc[index].any())
            if not invalid and source == "raw_yahoo_archive" and len({float(row[col]) for col in ("open", "high", "low", "close")}) == 1:
                problems.append("flat_ohlc_possible_placeholder")
            if invalid:
                status, decision, reason = "invalid_ohlcv", "reject", "Không đạt OHLCV validation"
            elif source == "raw_yahoo_archive" and not other_present:
                status, decision, reason = "unverified_no_kbs_session", "retain_unverified", "Không có phiên KBS đối chiếu; độ bao phủ chưa được xác minh"
                problems.append("no_matching_kbs_session")
            elif source == "raw_yahoo_archive" and problems:
                status, decision, reason = "requires_review", "hold_review", "Có cảnh báo OHLC, alignment hoặc volume; chưa được chấp nhận"
            elif source == "raw_yahoo_archive":
                status, decision, reason = "calendar_match_only", "retain_provisional", "Chỉ khớp ngày KBS; price, volume và alignment chưa được xác minh"
            elif other_present:
                status, decision, reason = "calendar_match_only", "reference_only", "Khớp ngày Raw; cơ sở OHLCV chưa được xác minh"
            else:
                status, decision, reason = "kbs_only", "reference_only", "KBS có phiên thiếu trong Raw; không tự động bổ sung"
                problems.append("missing_from_raw_archive")
            records.append({"date": row.date, "source": source, "verification_status": status,
                            "decision": decision, "anomaly_types": "|".join(problems), "reason": reason,
                            **{col: row[col] for col in OHLCV}})
    for _, row in yahoo_evidence.iterrows():
        present = row.present_in_yahoo
        if not isinstance(present, (bool, np.bool_)):
            raise ValueError("Giá trị present_in_yahoo không hợp lệ trong bằng chứng lịch sử")
        if present:
            one = pd.DataFrame([row[OHLCV]])
            failures = validate(one).iloc[0]
            problems = list(failures.index[failures])
            decision = "reject" if problems else "hold_unverified"
            reason = "Quan sát Yahoo lịch sử; " + ("không đạt OHLCV validation" if problems else "không tự động sử dụng")
        else:
            problems, decision, reason = ["absent_from_yahoo_snapshot"], "not_available", "Không có dữ liệu Yahoo trong snapshot lịch sử"
        records.append({"date": row.date, "source": "yahoo_rejection_snapshot", "verification_status": "historical_snapshot",
                        "decision": decision, "anomaly_types": "|".join(problems), "reason": reason,
                        **{col: row[col] for col in OHLCV}})
    return pd.DataFrame(records).sort_values(["date", "source"]).reset_index(drop=True)
