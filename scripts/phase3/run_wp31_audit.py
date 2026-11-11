"""Chạy kiểm toán WP3.1 trên Parquet Research và xuất báo cáo có thể tái lập."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow

from src.dataset_builder import expected_feature_columns, load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset
from src.models.data_validation import audit_dataset

OUTPUT = Path("results/modeling")
LIMITATIONS = [
    "Giá điều chỉnh và sự kiện doanh nghiệp chưa được xác minh tại từng thời điểm.",
    "Lịch giao dịch chính thức, nhất là 94 phiên đầu năm 2016, chưa được đối chiếu độc lập.",
    "Chưa chứng minh thời điểm công bố dữ liệu và khả năng giao dịch tại giá Open.",
    "Chi phí và slippage chỉ là giả định; chưa phản ánh khớp lệnh thực tế.",
]


def run(root: Path = ROOT) -> dict:
    """Đọc nguồn, kiểm toán và tạo JSON/Markdown mà không sửa dataset.

    Args:
        root: Thư mục gốc repository.

    Returns:
        Nội dung tóm tắt kiểm toán đã ghi ra đĩa.

    Raises:
        FileNotFoundError: Khi thiếu dataset hoặc cấu hình.
        ValueError: Khi một phép kiểm tra dữ liệu thất bại.
    """
    root = Path(root).resolve()
    config = load_project_config(root / "configs/config.yaml")
    dataset = load_research_dataset(root, config)
    audit = audit_dataset(dataset, config)
    source = root / RESEARCH_PATH
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    names = expected_feature_columns(config)
    summary = {
        "dataset": {"path": RESEARCH_PATH.as_posix(),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "rows": len(dataset), "columns": len(dataset.columns),
                    "first": str(dataset.index.min().date()),
                    "last": str(dataset.index.max().date()),
                    "feature_count": len(names), "feature_names": names},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "pyarrow": pyarrow.__version__,
                        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None},
        **audit,
        "limitations": LIMITATIONS,
        "status": "COMPLETED_WITH_LIMITATIONS",
    }
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "wp31_audit_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: dict) -> str:
    """Chuyển kết quả thực thi thành báo cáo tiếng Việt có bảng bằng chứng."""
    dataset = summary["dataset"]
    checks = summary["checks"]
    lines = [
        "# Báo cáo WP3.1: Chuẩn bị dữ liệu và kiểm toán rò rỉ\n",
        "## 1. Tổng quan WP3.1\n",
        "Kiểm tra dataset Research cho RQ1 và RQ2 trước khi chia tập theo thời gian.\n",
        "## 2. Đánh giá repository hiện tại\n",
        "Tái sử dụng `expected_feature_columns`, `validate_dataset`, `get_training_view` của Phase 2; "
        "đối chiếu độc lập công thức nhãn và tính lại feature trên các tiền tố.\n",
        "## 3. Kiểm toán dataset\n",
        "| Chỉ số | Giá trị thực tế |\n|---|---|",
        f"| Nguồn | `{dataset['path']}` |",
        f"| SHA-256 | `{dataset['sha256']}` |",
        f"| Kích thước | {dataset['rows']} phiên × {dataset['columns']} cột |",
        f"| Khoảng ngày | {dataset['first']} đến {dataset['last']} |",
        f"| Feature | {dataset['feature_count']} |",
        f"| Môi trường | Python {summary['environment']['python']}; pandas {summary['environment']['pandas']}; "
        f"NumPy {summary['environment']['numpy']}; PyArrow {summary['environment']['pyarrow']} |",
        f"| Git commit | `{summary['environment']['git_commit']}` |\n",
        "## 4. Feature và target\n",
        "Thứ tự feature lấy từ cấu hình Phase 2: " + ", ".join(f"`{n}`" for n in dataset["feature_names"]) + ".\n",
        "| RQ | Target | Đủ điều kiện | Loại | Class 0 | Class 1 | Khoảng ngày đủ điều kiện |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for rq, view in summary["views"].items():
        lines.append(f"| {rq} | `{view['target']}` | {view['eligible']} | {view['excluded']} | "
                     f"{view['class_counts']['0']} ({view['class_proportions']['0']:.2%}) | "
                     f"{view['class_counts']['1']} ({view['class_proportions']['1']:.2%}) | "
                     f"{view['first']} đến {view['last']} |")
    lines += [
        "\nCác hàng bị loại gồm feature chưa đầy đủ và nhãn tương lai thiếu; "
        "hai nguyên nhân được thống kê không chồng lặp trong JSON.\n",
        "## 5. Kiểm toán rò rỉ dữ liệu\n",
        "| Kiểm tra | Kết quả |\n|---|---|",
    ]
    lines += [f"| `{name}` | {result} |" for name, result in checks.items()]
    lines += [
        f"\nRQ1 đối chiếu `Close(t+1)/Close(t)` và nhãn hướng; RQ2 đối chiếu "
        "`Open(t+1)` đến `Open(t+4)` và công thức phí hai chiều. "
        f"Tổng phí gần đúng theo cấu hình là {summary['cost_proxy_approximation']:.2%}; "
        "return ròng dùng công thức nhân tỷ lệ, không trừ trực tiếp ngưỡng này.\n",
        f"Feature thiếu ở {summary['feature_incomplete_rows']} hàng; "
        f"warm-up đầu chuỗi {summary['warmup_rows']} hàng. "
        f"Nhãn RQ1 thiếu {summary['target_missing']['target_direction']} hàng, "
        f"nhãn RQ2 thiếu {summary['target_missing']['target_fixed_hold_net_proxy']} hàng. "
        f"Số giá trị Inf trong feature: {summary['nonfinite_values']}. "
        f"Biến động Close trên 12%: {summary['outliers']['absolute_close_return_above_12pct']} hàng; "
        f"volume ratio ngoài [0,1; 10]: {summary['outliers']['volume_ratio_outside_0_1_to_10']} hàng.\n",
        "## 6. Kiểm thử phần mềm\n",
        "Chạy `.venv\\Scripts\\python.exe -m pytest tests/unit tests/integration -q`. "
        "Kết quả của lần chạy được ghi riêng trong phản hồi thực thi để tránh "
        "một bản báo cáo tái tạo sau này giữ số kiểm thử đã cũ.\n",
        "## 7. File được thêm hoặc chỉnh sửa\n",
        "| File | Mục đích |\n|---|---|",
        "| `src/models/data.py` | Đọc Research và tạo view kèm mốc kết thúc nhãn |",
        "| `src/models/data_validation.py` | Kiểm toán schema, nhãn và feature |",
        "| `scripts/phase3/run_wp31_audit.py` | Xuất báo cáo JSON và Markdown |",
        "| `tests/unit/test_wp31.py` | Kiểm thử dữ liệu tổng hợp |\n",
        "## 8. Giới hạn nghiên cứu\n",
    ]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += [
        "\nUnit test đạt không chứng minh đã loại bỏ mọi rò rỉ dữ liệu.\n",
        "## 9. Đánh giá hoàn thành WP3.1\n",
        f"`{summary['status']}`: các phép kiểm tra thực thi đạt, còn giới hạn nguồn và thực thi giao dịch.\n",
        "## 10. Chuẩn bị cho WP3.2\n",
        "`ModelingView.metadata` cung cấp `signal_date` và `label_end_date` "
        "để purge các khoảng nhãn khi chia walk-forward. Chưa tạo fold hay fit mô hình.\n",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
