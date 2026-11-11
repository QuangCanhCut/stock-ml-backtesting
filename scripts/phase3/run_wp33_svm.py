"""Chạy SVM baseline trên development folds và lưu bằng chứng thực nghiệm."""

import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow
import sklearn

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, load_research_dataset, make_modeling_view
from src.models.modeling_safety import MANIFEST_PATH, get_development_splits
from src.models.svm_baseline import (
    MODEL_CONFIGS, aggregate_results, evaluate_fold, select_configurations,
)

LOGGER = logging.getLogger(__name__)
OUTPUT = Path("results/modeling/svm")
SELECTION_RULE = (
    "Mean Balanced Accuracy cao nhất; nếu bằng nhau chọn độ lệch chuẩn nhỏ hơn, "
    "sau đó Linear trước RBF và thứ tự cấu hình đã công bố."
)


def run(root: Path = ROOT, test_result: dict | None = None) -> dict:
    """Thực hiện protocol cố định và ghi artifacts vào thư mục WP3.3 mới.

    Args:
        root: Thư mục gốc repository.
        test_result: Kết quả pytest thực tế nếu suite đã được chạy.

    Returns:
        Metadata và cấu hình được chọn theo development folds.

    Raises:
        FileExistsError: Khi thư mục kết quả đã tồn tại để tránh ghi đè nghiên cứu.
        ValueError: Khi split, dữ liệu hoặc metrics không hợp lệ.
    """
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Thư mục kết quả đã tồn tại, không ghi đè: {output}")
    config_path = root / "configs/config.yaml"
    config = load_project_config(config_path)
    dataset = load_research_dataset(root, config)
    metrics, predictions = [], []
    for rq in RQ_TARGETS:
        folds = get_development_splits(root, rq)
        view = make_modeling_view(dataset, config, rq)
        for model_config in MODEL_CONFIGS:
            LOGGER.info("Đang đánh giá %s, cấu hình %s trên development folds", rq, model_config.config_id)
            for fold in folds:
                record, prediction = evaluate_fold(view, rq, fold, model_config)
                metrics.append(record)
                predictions.append(prediction)

    fold_metrics = pd.DataFrame(metrics)
    prediction_data = pd.concat(predictions, ignore_index=True)
    for metric in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision"):
        dummy = fold_metrics[fold_metrics.config_id == "dummy_prior"].set_index(["rq", "fold_name"])[metric]
        baseline = pd.MultiIndex.from_frame(fold_metrics[["rq", "fold_name"]]).map(dummy)
        fold_metrics[f"delta_{metric}_vs_dummy"] = fold_metrics[metric] - np.asarray(baseline, dtype=float)
    comparison = aggregate_results(fold_metrics)
    selected = select_configurations(comparison)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    metadata = {
        "status": "COMPLETED_WITH_LIMITATIONS",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": RESEARCH_PATH.as_posix(),
        "dataset_sha256": hashlib.sha256((root / RESEARCH_PATH).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256((root / MANIFEST_PATH).read_bytes()).hexdigest(),
        "feature_schema": dataset.attrs["feature_columns"],
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
        "development_folds": ["dev_1", "dev_2", "dev_3"],
        "random_seed": None,
        "model_configurations": [asdict(model_config) for model_config in MODEL_CONFIGS],
        "model_selection_rule": SELECTION_RULE,
        "selected": selected,
        "test_result": test_result,
    }
    output.mkdir(parents=True, exist_ok=False)
    fold_metrics.to_csv(output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    prediction_data.to_parquet(output / "validation_predictions.parquet", index=False)
    (output / "selected_configs.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "wp33_svm_report.md").write_text(
        render_report(metadata, fold_metrics, comparison), encoding="utf-8")
    return metadata


def render_report(metadata: dict, metrics: pd.DataFrame, comparison: pd.DataFrame) -> str:
    """Tạo báo cáo tiếng Việt từ kết quả development đã thực thi."""
    lines = [
        "# Báo cáo WP3.3: SVM baseline\n",
        "## 1. Tổng quan WP3.3\n",
        "Đánh giá Dummy, Linear SVM và RBF SVM cho RQ1 và RQ2 trên ba development folds. "
        "RQ2 dự đoán nhãn kinh tế, không phải lợi nhuận giao dịch thực tế.\n",
        "## 2. Kiến trúc Modeling Pipeline\n",
        "Dataset và modeling view tái sử dụng WP3.1; indices lấy từ API development "
        "đã kiểm tra fingerprint và split của WP3.2.1. Mỗi fold có estimator mới.\n",
        "## 3. Thiết lập thực nghiệm\n",
        f"Dataset SHA-256: `{metadata['dataset_sha256']}`; manifest SHA-256: `{metadata['split_manifest_sha256']}`. "
        f"Feature: {len(metadata['feature_schema'])}; scikit-learn: {metadata['versions']['scikit_learn']}. "
        "Dummy dùng `strategy=prior`. SVM dùng `StandardScaler` trong Pipeline, "
        "C ∈ {0,1; 1; 10}, class_weight ∈ {None, balanced}; RBF dùng gamma=scale. "
        "Mọi SVC dùng probability=False; không có calibration.\n",
    ]
    for rq, section in (("RQ1", 4), ("RQ2", 5)):
        lines += [f"## {section}. Kết quả {rq}\n",
                  "Development Walk-Forward CV Performance: Balanced Accuracy theo từng fold.\n",
                  "| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Trung bình | Độ lệch chuẩn | So với Dummy |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in comparison[comparison.rq == rq].itertuples():
            frame = metrics[(metrics.rq == rq) & (metrics.config_id == row.config_id)].set_index("fold_name")
            values = [frame.loc[name, "balanced_accuracy"] for name in metadata["development_folds"]]
            lines.append(f"| `{row.config_id}` | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | "
                         f"{row.mean_balanced_accuracy:.4f} | {row.std_balanced_accuracy:.4f} | "
                         f"{row.delta_mean_balanced_accuracy_vs_dummy:+.4f} |")
        lines.append("\nCác metrics còn lại, confusion matrix và chênh lệch theo fold có trong CSV.\n")
    lines += [
        "## 6. So sánh với Dummy Baseline\n",
        "Chênh lệch được tính trên cùng mẫu Validation của từng RQ và fold. "
        "Dummy dự báo lớp phổ biến theo phân bố Train; điểm ranking là tỷ lệ lớp 1 của Train.\n",
        "## 7. Hyperparameter Selection\n",
        metadata["model_selection_rule"] + "\n",
        "| RQ | Cấu hình SVM được chọn | Mean Balanced Accuracy | So với Dummy |",
        "|---|---|---:|---:|",
    ]
    for rq, result in metadata["selected"].items():
        lines.append(f"| {rq} | `{result['config_id']}` | {result['mean_balanced_accuracy']:.4f} | "
                     f"{result['delta_mean_balanced_accuracy_vs_dummy']:+.4f} |")
    lines += [
        "\n## 8. Phân tích kết quả\n",
        "Mức chênh lệch giữa các năm cần được xem cùng độ lệch chuẩn; ba fold có thể "
        "chịu phụ thuộc thời gian và các nhãn RQ2 có khoảng nắm giữ chồng lấn. "
        "Hiệu suất của cấu hình thắng đã được dùng để chọn tham số, do đó có thiên lệch "
        "lựa chọn và không phải ước lượng Test độc lập. Không suy ra trading alpha.\n",
        "## 9. Data Leakage và Test Isolation\n",
        "Scaler chỉ fit qua Pipeline trên X_train của từng fold. Prediction artifacts "
        "chỉ chứa Validation Development trước 2022 và nhãn phải kết thúc trước vùng Test. "
        "SVM decision score không phải xác suất; chưa thực hiện calibration.\n",
        "## 10. Automated Tests\n",
    ]
    result = metadata["test_result"]
    lines.append((f"Lệnh: `{result['command']}`. Kết quả: {result['passed']} PASS, {result['failed']} FAIL.\n"
                  if result else "Chưa có kết quả kiểm thử được cung cấp cho lần chạy này.\n"))
    lines += [
        "## 11. Files và Artifacts\n",
        "`metrics_by_fold.csv`, `model_comparison.csv`, `selected_configs.json` và "
        "`validation_predictions.parquet` nằm cùng thư mục với báo cáo. Metadata ghi "
        "tham số, phiên bản, fingerprint và quy tắc lựa chọn.\n",
        "## 12. Giới hạn nghiên cứu\n",
        "Các giới hạn nguồn WP3.1 về adjusted prices, corporate actions, lịch phiên "
        "và khả năng thực thi vẫn áp dụng. Reserved Test chưa được đánh giá; "
        "không diễn giải decision score thành xác suất hoặc chỉ số phân loại thành lợi nhuận. "
        "scikit-learn 1.9.1 cảnh báo tham số `SVC(probability=...)` sẽ bị loại trong phiên bản tương lai; "
        "cần kiểm tra tính tương thích trước khi nâng cấp.\n",
        "## 13. Kết luận WP3.3\n",
        "`COMPLETED_WITH_LIMITATIONS`: kết quả dùng để lựa chọn cấu hình trên "
        "Development Walk-Forward CV, chưa phải Final Out-of-Sample Test Performance.\n",
        "## 14. Khuyến nghị WP3.4\n",
        "Dùng cùng dataset, eligibility, development folds và metrics để so sánh LightGBM; "
        "tiếp tục giữ Reserved Test và thiết kế calibration riêng nếu cần xác suất.\n",
    ]
    return "\n".join(lines)


def _run_tests(root: Path) -> dict:
    """Chạy regression suite và chỉ báo PASS theo output thực tế của pytest."""
    command = [sys.executable, "-m", "pytest", "tests/unit", "tests/integration", "-q"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Kiểm thử thất bại:\n{result.stdout}\n{result.stderr}")
    counts = re.search(r"(\d+) passed", result.stdout)
    if counts is None:
        raise RuntimeError("Không xác định được số test đã đạt")
    return {"command": ".venv\\Scripts\\python.exe -m pytest tests/unit tests/integration -q",
            "passed": int(counts.group(1)), "failed": 0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(ROOT, _run_tests(ROOT))
