"""Tổng hợp chất lượng theo quan sát của từng nguồn."""


def yearly_source_report(sessions):
    df = sessions.copy()
    df["year"] = df.date.dt.year
    df["has_anomaly"] = df.anomaly_types.ne("")
    return (df.groupby(["year", "source", "verification_status", "decision"], dropna=False)
              .agg(records=("date", "size"), anomalous=("has_anomaly", "sum"))
              .reset_index())
