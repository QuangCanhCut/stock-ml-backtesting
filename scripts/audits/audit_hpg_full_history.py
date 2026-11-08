"""Compare the raw HPG trading calendar with yearly KBS reference data."""

import time
from pathlib import Path

import pandas as pd
from vnstock import Quote

START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-11-01")  # Exclusive
RAW_PATH = Path("data/raw/HPG_10y_daily.csv")
OUTPUT = Path("results/data_audit")
CACHE_DIR = OUTPUT / "kbs_cache"


def normalize_dates(values):
    """Keep calendar dates while removing any timezone and time of day."""
    return pd.to_datetime(values).dt.tz_localize(None).dt.normalize()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_PATH)
    raw["date"] = normalize_dates(raw["date"])
    raw = raw[(raw["date"] >= START) & (raw["date"] < END)].copy()
    if raw["date"].duplicated().any():
        raise ValueError("Duplicate dates found in raw dataset")

    quote = Quote(symbol="HPG", source="kbs")
    summaries = []
    missing_frames = []
    raw_only_frames = []
    last_api_call = None

    expected_years = list(range(START.year, (END - pd.Timedelta(days=1)).year + 1))
    for year in expected_years:
        begin = max(START, pd.Timestamp(f"{year}-01-01"))
        finish = min(END, pd.Timestamp(f"{year + 1}-01-01"))
        print(f"\nAuditing year {year}...", flush=True)
        cache_file = CACHE_DIR / f"HPG_KBS_{year}.parquet"

        if cache_file.exists():
            print(f"Loading KBS {year} from local cache...", flush=True)
            kbs = pd.read_parquet(cache_file)
        else:
            for attempt in range(3):
                if last_api_call is not None:
                    time.sleep(max(0, 8 - (time.monotonic() - last_api_call)))
                last_api_call = time.monotonic()
                try:
                    kbs = quote.history(
                        start=begin.strftime("%Y-%m-%d"),
                        end=finish.strftime("%Y-%m-%d"),
                        interval="1D",
                    )
                    break
                except Exception as exc:
                    is_rate_limit = (
                        "EntitlementError" in type(exc).__name__
                        and "hạn mức theo phút" in str(exc)
                    )
                    if not is_rate_limit or attempt == 2:
                        raise
                    print(f"KBS minute quota reached for {year}; retrying after cooldown...", flush=True)
                    time.sleep(70)
            if kbs is None or kbs.empty:
                raise RuntimeError(f"KBS returned no data for {year}; audit incomplete")
            kbs.to_parquet(cache_file, index=False)

        if kbs is None or kbs.empty:
            raise RuntimeError(f"KBS returned no data for {year}; audit incomplete")
        kbs = kbs.copy()
        kbs["date"] = normalize_dates(kbs["time"])
        kbs = kbs[(kbs["date"] >= begin) & (kbs["date"] < finish)]
        kbs = kbs[kbs["volume"] > 0].copy()
        if kbs.empty:
            raise RuntimeError(f"KBS has no positive-volume sessions for {year}; audit incomplete")
        if kbs["date"].duplicated().any():
            raise ValueError(f"Duplicate KBS dates: {year}")

        raw_year = raw[(raw["date"] >= begin) & (raw["date"] < finish)]
        raw_dates = set(raw_year["date"])
        kbs_dates = set(kbs["date"])
        common = raw_dates & kbs_dates
        missing = kbs_dates - raw_dates
        raw_only = raw_dates - kbs_dates
        row = {
            "year": year,
            "raw_sessions": len(raw_dates),
            "kbs_sessions": len(kbs_dates),
            "common": len(common),
            "missing_in_raw": len(missing),
            "raw_only": len(raw_only),
        }
        summaries.append(row)
        missing_frames.append(kbs[kbs["date"].isin(missing)].copy())
        raw_only_frames.append(raw_year[raw_year["date"].isin(raw_only)].copy())
        print(
            ", ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )

        if year == 2021:
            expected = {"raw_sessions": 224, "kbs_sessions": 250, "missing_in_raw": 26}
            differences = [f"{key}: expected {value}, got {row[key]}"
                           for key, value in expected.items() if row[key] != value]
            reference = OUTPUT / "HPG_missing_2021_KBS_reference.csv"
            if reference.exists():
                reference_dates = set(normalize_dates(pd.read_csv(reference)["date"]))
                if missing != reference_dates:
                    differences.append(
                        f"missing dates differ from independent audit: "
                        f"{len(reference_dates - missing)} absent and {len(missing - reference_dates)} extra"
                    )
            if differences:
                print("WARNING: 2021 differs from independent audit: " + "; ".join(differences), flush=True)
            else:
                print("2021 matches independent audit counts and available reference dates.", flush=True)

    summary = pd.DataFrame(summaries)
    actual_years = summary["year"].tolist() if "year" in summary else []
    if len(summary) != len(expected_years) or actual_years != expected_years:
        raise RuntimeError(f"Audit incomplete: expected years {expected_years}, got {actual_years}")
    missing_df = pd.concat(missing_frames, ignore_index=True)
    raw_only_df = pd.concat(raw_only_frames, ignore_index=True)
    summary.to_csv(OUTPUT / "HPG_full_history_summary.csv", index=False)
    missing_df.to_csv(OUTPUT / "HPG_full_history_missing.csv", index=False)
    raw_only_df.to_csv(OUTPUT / "HPG_full_history_raw_only.csv", index=False)
    print("\n=== FULL HISTORY AUDIT ===")
    print(summary.to_string(index=False))
    print("\n=== TOTALS ===")
    print("Total missing:", summary["missing_in_raw"].sum())
    print("Total raw-only:", summary["raw_only"].sum())
    print("Audit completed successfully.")


if __name__ == "__main__":
    main()
