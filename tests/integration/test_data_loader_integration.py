import json
import subprocess
import sys

import pandas as pd
import yaml


def test_offline_dry_run_writes_only_new_audit_reports(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    (root / "results/data_audit/kbs_cache").mkdir(parents=True)
    config = {"data": {"start_date": "2025-01-01", "end_date": "2026-01-01", "raw_csv_path": "data/raw/input.csv"}}
    (root / "configs/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    raw = pd.DataFrame({"date": ["2025-05-05", "2025-05-07"], "open": [10, 10], "high": [11, 11],
                        "low": [9, 9], "close": [10, 10], "volume": [100, 100]})
    raw_path = root / "data/raw/input.csv"
    raw.to_csv(raw_path, index=False)
    before = raw_path.read_bytes()
    kbs = raw.rename(columns={"date": "time"})
    kbs.to_parquet(root / "results/data_audit/kbs_cache/HPG_KBS_2025.parquet", index=False)
    output = root / "results/data_audit/new_report"
    result = subprocess.run([sys.executable, "-m", "src.data_loader", "audit", "--root", str(root),
                             "--output-dir", str(output)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert raw_path.read_bytes() == before
    sessions = pd.read_csv(output / "sessions.csv")
    assert len(sessions) == 4
    assert set(sessions.source) == {"raw_yahoo_archive", "kbs_cache"}
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["policy"]["dataset_export"] == "blocked_pending_policy_and_verification"
    assert not metadata["missing_cache_years"]
    second = subprocess.run([sys.executable, "-m", "src.data_loader", "audit", "--root", str(root),
                             "--output-dir", str(output)], capture_output=True, text=True, encoding="utf-8")
    assert second.returncode != 0
    assert raw_path.read_bytes() == before


def test_missing_cache_is_reported_without_network_or_raw_loss(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    config = {"data": {"start_date": "2025-01-01", "end_date": "2026-01-01", "raw_csv_path": "data/raw/input.csv"}}
    (root / "configs/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    raw = pd.DataFrame({"date": ["2025-01-02"], "open": [10], "high": [11], "low": [9], "close": [10], "volume": [100]})
    raw.to_csv(root / "data/raw/input.csv", index=False)
    from src.data_loader.__main__ import run
    reports, metadata = run(root)
    assert metadata["missing_cache_years"] == [2025]
    assert reports["sessions"].iloc[0].decision == "retain_unverified"
    from src.data_loader.export import export_dataset
    import pytest
    with pytest.raises(RuntimeError, match="Quality Gate"):
        export_dataset(reports, metadata, root)
    assert not (root / "data/raw/HPG_10y_daily_validated.parquet").exists()
    build = subprocess.run([sys.executable, "-m", "src.data_loader", "build", "--root", str(root)], capture_output=True, text=True, encoding="utf-8")
    assert build.returncode == 2, build.stderr
    assert "Quality Gate" in build.stdout
    assert not (root / "data/raw/HPG_10y_daily_validated.parquet").exists()
