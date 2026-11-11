"""Xuất manifest và báo cáo chia thời gian WP3.2 trên dataset Research."""

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, load_research_dataset, make_modeling_view
from src.models.temporal_split import (
    DEFAULT_PERIODS, PRE_VALIDATION_GAP_SESSIONS, TEST_START_YEAR,
    TemporalSplit, build_temporal_splits,
)

OUTPUT = Path("results/modeling/splits")
TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/unit", "tests/integration", "-q"]


def _fold_record(split: TemporalSplit) -> dict:
    """Chuyển fold thành bản ghi có thể đọc lại, với index của ModelingView."""
    return {
        "rq": split.rq,
        "fold_name": split.fold_name,
        "role": split.role,
        "train_start": str(split.train_start_date.date()),
        "train_end": str(split.train_end_date.date()),
        "validation_start": str(split.validation_start_date.date()),
        "validation_end": str(split.validation_end_date.date()),
        "evaluation_boundary": str(split.evaluation_boundary_date.date()),
        "train_samples": len(split.train_indices),
        "validation_samples": len(split.validation_indices),
        "purged_samples": split.purged_count,
        "gap_samples": split.gap_count,
        "overlap_exclusions": split.overlap_exclusions,
        "total_train_exclusions": split.excluded_count,
        "boundary_exclusions": split.boundary_excluded_count,
        "train_class_counts": split.train_class_counts,
        "validation_class_counts": split.validation_class_counts,
        "train_one_class": min(split.train_class_counts.values()) == 0,
        "validation_one_class": (min(split.validation_class_counts.values()) == 0
                                 if split.validation_class_counts is not None else None),
        "status": "PASS",
    }


def _manifest_fold(split: TemporalSplit) -> dict:
    """Lưu chỉ số vị trí view và master để phục dựng, kiểm tra gap và purging."""
    return {
        **_fold_record(split),
        "train_indices": split.train_indices,
        "validation_indices": split.validation_indices,
        "train_master_positions": split.train_master_positions,
        "validation_master_positions": split.validation_master_positions,
    }


def run(root: Path = ROOT, test_result: dict | None = None) -> dict:
    """Tạo split manifest cho hai RQ và ghi kết quả vào thư mục WP3.2.

    Args:
        root: Thư mục gốc repository.
        test_result: Kết quả pytest thực tế nếu đã chạy kiểm thử.

    Returns:
        Manifest gồm cấu hình, dữ liệu nguồn và toàn bộ chỉ số fold.

    Raises:
        ValueError: Khi dataset hoặc một bất biến split không đạt.
    """
    root = Path(root).resolve()
    config = load_project_config(root / "configs/config.yaml")
    data = load_research_dataset(root, config)
    source = root / RESEARCH_PATH
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    manifest = {
        "schema_version": 1,
        "dataset": {"path": RESEARCH_PATH.as_posix(),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "config_sha256": hashlib.sha256((root / "configs/config.yaml").read_bytes()).hexdigest(),
                    "feature_schema": data.attrs["feature_columns"],
                    "master_sessions": len(data),
                    "first": str(data.index[0].date()),
                    "last": str(data.index[-1].date())},
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "index_reference": "Chỉ số 0-based trong ModelingView.X/y; vị trí master lưu riêng.",
        "split_configuration": {
            "pre_validation_gap_sessions": PRE_VALIDATION_GAP_SESSIONS,
            "test_start_year": TEST_START_YEAR,
            "fixed_holding_sessions": config["labeling"]["fixed_holding_sessions"],
            "periods": [vars(period) for period in DEFAULT_PERIODS],
        },
        "rq_views": {},
        "folds": [],
        "validation": {"status": "PASS", "checks": [
            "X/y/metadata cùng index và đúng horizon theo phiên master",
            "Train trước vùng đánh giá, không giao mẫu hoặc khoảng nhãn",
            "Gap neo vào năm phiên master ngay trước vùng đánh giá",
            "Nhãn Validation phát triển kết thúc trước kỳ kế tiếp",
            "Development không sử dụng tín hiệu hoặc nhãn vùng Test",
        ]},
        "test_result": test_result,
        "status": "COMPLETED_WITH_LIMITATIONS",
    }
    rows = []
    for rq, target in RQ_TARGETS.items():
        view = make_modeling_view(data, config, rq)
        splits = build_temporal_splits(data.index, view, rq, config)
        manifest["rq_views"][rq] = {"target": target, "eligible_samples": len(view.X),
                                    "first": str(view.X.index[0].date()),
                                    "last": str(view.X.index[-1].date())}
        manifest["folds"].extend(_manifest_fold(split) for split in splits)
        rows.extend(_fold_record(split) for split in splits)

    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "fold_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "wp32_validation_report.md").write_text(
        render_report(manifest, rows), encoding="utf-8")
    return manifest


def render_report(manifest: dict, rows: list[dict]) -> str:
    """Trình bày bằng chứng split, bảo vệ Test và kết quả kiểm thử bằng tiếng Việt."""
    config = manifest["split_configuration"]
    lines = [
        "# Báo cáo WP3.2: Chia thời gian và purging theo nhãn\n",
        "## 1. Tổng quan WP3.2\n",
        "Tạo ba development folds mở rộng và một vùng Test dự trữ cho RQ1/RQ2. "
        "Không huấn luyện hay đánh giá mô hình.\n",
        "## 2. Kiến trúc triển khai\n",
        "Tái sử dụng `ModelingView`, schema và Parquet Research của WP3.1. "
        "`src/models/temporal_split.py` xác minh ngày nhãn và tạo chỉ số vị trí trong view; "
        "manifest lưu thêm vị trí trên master calendar.\n",
        "## 3. Thiết kế Temporal Split\n",
        "| Fold | Train theo năm | Vùng đánh giá | Vai trò |\n|---|---|---|---|",
    ]
    for period in config["periods"]:
        evaluation = (str(period["evaluation_start_year"])
                      if period["evaluation_end_year"] == period["evaluation_start_year"]
                      else f"{period['evaluation_start_year']}–hết dữ liệu")
        role = "Test dự trữ" if period["role"] == "reserved_test" else "Phát triển"
        lines.append(f"| `{period['name']}` | đến {period['train_end_year']} | {evaluation} | {role} |")
    lines += [
        "\n## 4. Purging và Gap\n",
        "Giữ mẫu Train khi `label_end_date < evaluation_boundary_date` và "
        f"vị trí tín hiệu master < `boundary_position - {config['pre_validation_gap_sessions']}`. "
        "Khoảng nhãn chạm ranh giới bị purge. Gap là đúng năm phiên master ngay trước "
        "ranh giới, kể cả khi một phiên không có mẫu eligible. "
        "Hai nguyên nhân loại được đếm riêng, phần giao trừ đúng một lần.\n",
    ]
    for rq in RQ_TARGETS:
        lines += [f"## {5 if rq == 'RQ1' else 6}. Kết quả {rq}\n",
                  "| Fold | Train | Đánh giá | Purged | Gap | Cả hai | Loại thực | Nhãn vượt kỳ |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["rq"] == rq:
                lines.append(f"| `{row['fold_name']}` | {row['train_samples']} | "
                             f"{row['validation_samples']} | {row['purged_samples']} | "
                             f"{row['gap_samples']} | {row['overlap_exclusions']} | "
                             f"{row['total_train_exclusions']} | {row['boundary_exclusions']} |")
        lines.append("\nBảng CSV và manifest ghi ngày thực tế cùng phân bố lớp của Train/Validation phát triển. "
                     "Không ghi phân bố nhãn Test.\n")
    lines += [
        "## 7. Leakage Validation\n",
        "Các bất biến đã được kiểm tra trực tiếp khi tạo từng fold: index tăng dần, "
        "không giao mẫu, Train trước Validation, nhãn Train kết thúc trước ranh giới, "
        "gap đủ năm phiên master và nhãn Development kết thúc trước kỳ tiếp theo. "
        "Ranh giới Test là phiên master đầu tiên từ năm 2022. "
        "Test được giữ cho đánh giá cuối cùng; không dùng để chọn mô hình, threshold hay feature.\n",
        "## 8. Unit/Integration Tests\n",
    ]
    result = manifest["test_result"]
    if result is None:
        lines.append("Chưa chạy kiểm thử trong lần xuất báo cáo này; cần chạy suite trước khi nghiệm thu.\n")
    else:
        lines.append(f"Lệnh: `{result['command']}`. Kết quả: {result['passed']} PASS, "
                     f"{result['failed']} FAIL.\n")
    lines += [
        "## 9. Giới hạn nghiên cứu\n",
        "Giới hạn giá điều chỉnh, corporate actions, lịch giao dịch chính thức và "
        "khả năng khớp lệnh của WP3.1 vẫn áp dụng. Trường `embargo_days` của Phase 2 "
        "không được diễn giải thành embargo sau Test; WP3.2 dùng `pre_validation_gap_sessions` "
        "được ghi rõ trong manifest.\n",
        "## 10. Kết luận WP3.2\n",
        f"`{manifest['status']}`: split và kiểm tra rò rỉ đã thực thi; "
        "các chỉ số sẵn sàng cho WP3.3 trong phạm vi giới hạn dữ liệu nguồn.\n",
    ]
    return "\n".join(lines)


def _run_tests(root: Path) -> dict:
    """Chạy suite liên quan và chỉ ghi số PASS khi pytest thực sự kết thúc thành công."""
    result = subprocess.run(TEST_COMMAND, cwd=root, capture_output=True,
                            text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Kiểm thử WP3.2 thất bại:\n{result.stdout}\n{result.stderr}")
    counts = re.search(r"(\d+) passed", result.stdout)
    if counts is None:
        raise RuntimeError("Không đọc được số kiểm thử đã đạt từ pytest")
    return {"command": ".venv\\Scripts\\python.exe -m pytest tests/unit tests/integration -q",
            "passed": int(counts.group(1)), "failed": 0}


if __name__ == "__main__":
    run(ROOT, _run_tests(ROOT))
