"""Chạy thực nghiệm WP3.4: LightGBM modeling, Feature Importance và lưu trữ Final Development Models."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, load_research_dataset, make_modeling_view
from src.models.modeling_safety import MANIFEST_PATH, get_development_splits
from src.models.svm_baseline import fit_and_export_final_development_models as export_svm_final_dev
from src.models.train_lightgbm import (
    EARLY_STOPPING_ROUNDS, MODEL_CONFIGS, aggregate_results, evaluate_fold,
    fit_and_export_final_development_models as export_lgbm_final_dev,
    select_configurations,
)

LOGGER = logging.getLogger(__name__)
OUTPUT = Path("results/modeling/lightgbm")
SELECTION_RULE = (
    "Mean Balanced Accuracy cao nhất qua 3 development folds; nếu bằng nhau chọn độ lệch chuẩn nhỏ hơn, "
    "sau đó cấu trúc cây đơn giản hơn (max_depth -> num_leaves -> min_child_samples -> reg_lambda) "
    "và thứ tự cấu hình đã công bố trước khi chạy."
)


def load_svm_baseline_results(root: Path) -> dict[str, Any]:
    """Đọc kết quả cấu hình SVM thắng cuộc từ WP3.3 để đối chiếu công bằng."""
    svm_selected_path = root / "results/modeling/svm/selected_configs.json"
    svm_comparison_path = root / "results/modeling/svm/model_comparison.csv"
    svm_metrics_path = root / "results/modeling/svm/metrics_by_fold.csv"

    if not svm_selected_path.is_file() or not svm_comparison_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy artifacts WP3.3 tại {svm_selected_path} hoặc {svm_comparison_path}."
        )

    selected_data = json.loads(svm_selected_path.read_text(encoding="utf-8"))
    comparison_df = pd.read_csv(svm_comparison_path)
    metrics_df = pd.read_csv(svm_metrics_path) if svm_metrics_path.is_file() else None

    results: dict[str, Any] = {}
    for rq in ("RQ1", "RQ2"):
        winner_info = selected_data["selected"][rq]
        cfg_id = winner_info["config_id"]
        row = comparison_df[(comparison_df.rq == rq) & (comparison_df.config_id == cfg_id)].iloc[0]

        fold_accs: dict[str, float] = {}
        if metrics_df is not None:
            m_sub = metrics_df[(metrics_df.rq == rq) & (metrics_df.config_id == cfg_id)]
            for _, r in m_sub.iterrows():
                fold_accs[r["fold_name"]] = float(r["balanced_accuracy"])

        results[rq] = {
            "config_id": cfg_id,
            "model_name": winner_info["model_name"],
            "mean_balanced_accuracy": float(row["mean_balanced_accuracy"]),
            "std_balanced_accuracy": float(row["std_balanced_accuracy"]),
            "delta_mean_balanced_accuracy_vs_dummy": float(row["delta_mean_balanced_accuracy_vs_dummy"]),
            "fold_balanced_accuracies": fold_accs,
        }
    return results


def run(root: Path = ROOT, test_result: dict | None = None) -> dict[str, Any]:
    """Thực hiện protocol cố định WP3.4 và xuất toàn bộ artifacts ra đĩa.

    Args:
        root: Thư mục gốc repository.
        test_result: Kết quả pytest thực tế nếu đã chạy suite.

    Returns:
        Dict metadata tổng hợp của toàn bộ thực nghiệm WP3.4.

    Raises:
        FileExistsError: Khi thư mục kết quả đã tồn tại (chống ghi đè).
        ValueError: Khi tính toán chỉ số hoặc dữ liệu không nhất quán.
    """
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Thư mục kết quả WP3.4 đã tồn tại, không ghi đè: {output}")

    config_path = root / "configs/config.yaml"
    config = load_project_config(config_path)
    dataset = load_research_dataset(root, config)

    # 1. Tải kết quả SVM baseline từ WP3.3 để đối chiếu
    svm_baselines = load_svm_baseline_results(root)

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    training_logs: list[dict[str, Any]] = []
    fold_best_iters: dict[str, dict[str, list[int]]] = {"RQ1": {}, "RQ2": {}}

    # 2. Đánh giá tất cả cấu hình trên 3 development folds
    for rq in RQ_TARGETS:
        folds = get_development_splits(root, rq)
        view = make_modeling_view(dataset, config, rq)

        for model_config in MODEL_CONFIGS:
            LOGGER.info("Đang đánh giá %s, cấu hình %s trên development folds...", rq, model_config.config_id)
            cfg_iters: list[int] = []
            for fold in folds:
                record, pred, logs, _ = evaluate_fold(
                    view, rq, fold, model_config, compute_perm_importance=False
                )
                metrics.append(record)
                predictions.append(pred)
                training_logs.extend(logs)
                if record["best_iteration"] is not None:
                    cfg_iters.append(int(record["best_iteration"]))

            if cfg_iters:
                fold_best_iters[rq][model_config.config_id] = cfg_iters

    fold_metrics = pd.DataFrame(metrics)
    prediction_data = pd.concat(predictions, ignore_index=True)
    training_log_df = pd.DataFrame(training_logs)

    # 3. Tính toán chênh lệch so với Dummy baseline theo từng fold
    for metric in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision"):
        dummy_series = (
            fold_metrics[fold_metrics.config_id == "dummy_prior"]
            .set_index(["rq", "fold_name"])[metric]
        )
        baseline = pd.MultiIndex.from_frame(fold_metrics[["rq", "fold_name"]]).map(dummy_series)
        fold_metrics[f"delta_{metric}_vs_dummy"] = fold_metrics[metric] - np.asarray(baseline, dtype=float)

    # 4. Tổng hợp so sánh mô hình và lựa chọn cấu hình thắng cuộc
    comparison = aggregate_results(fold_metrics)

    # Bổ sung chênh lệch so với SVM baseline của WP3.3 vào comparison
    for rq in RQ_TARGETS:
        svm_mean_bacc = svm_baselines[rq]["mean_balanced_accuracy"]
        mask = comparison.rq == rq
        comparison.loc[mask, "delta_mean_balanced_accuracy_vs_svm"] = (
            comparison.loc[mask, "mean_balanced_accuracy"] - svm_mean_bacc
        )

    selected = select_configurations(comparison)

    # 5. Phân tích Feature Importance (Gain, Split & Permutation) cho cấu hình thắng cuộc
    feature_importance_records: list[dict[str, Any]] = []
    winning_iters: dict[str, list[int]] = {}

    for rq in RQ_TARGETS:
        winner_id = selected[rq]["config_id"]
        winning_cfg = next(cfg for cfg in MODEL_CONFIGS if cfg.config_id == winner_id)
        folds = get_development_splits(root, rq)
        view = make_modeling_view(dataset, config, rq)
        winning_iters[rq] = fold_best_iters[rq][winner_id]

        LOGGER.info("Tính toán Permutation Importance cho %s (cấu hình %s)...", rq, winner_id)
        for fold in folds:
            _, _, _, imp_records = evaluate_fold(
                view, rq, fold, winning_cfg, compute_perm_importance=True
            )
            feature_importance_records.extend(imp_records)

    feature_importance_df = pd.DataFrame(feature_importance_records)

    # Thêm dòng tổng hợp trung bình 3 folds cho từng feature
    avg_imp_rows: list[dict[str, Any]] = []
    for (rq, feature_name), grp in feature_importance_df.groupby(["rq", "feature_name"], sort=False):
        avg_imp_rows.append({
            "rq": rq,
            "fold_name": "mean_3_folds",
            "config_id": grp.config_id.iloc[0],
            "feature_name": feature_name,
            "split_importance": float(grp.split_importance.mean()),
            "gain_importance": float(grp.gain_importance.mean()),
            "perm_importance_mean": float(grp.perm_importance_mean.mean()),
            "perm_importance_std": float(grp.perm_importance_mean.std(ddof=0)),
        })
    full_feature_importance_df = pd.concat(
        [feature_importance_df, pd.DataFrame(avg_imp_rows)], ignore_index=True
    )

    # 6. Fit và xuất Final Development Models (2016–2021) vào models/
    LOGGER.info("Huấn luyện và lưu Final Development Models vào models/svm/ và models/lightgbm/...")
    svm_final_metadata = export_svm_final_dev(root)
    lgbm_final_metadata = export_lgbm_final_dev(
        root, selected_configs=selected, fold_best_iterations=winning_iters
    )

    # Đảm bảo placeholder models/lstm/.gitkeep tồn tại
    lstm_dir = root / "models/lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    (lstm_dir / ".gitkeep").touch()

    # 7. Tạo metadata tổng thể của WP3.4
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    metadata = {
        "status": "COMPLETED_WITH_LIMITATIONS",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": RESEARCH_PATH.as_posix(),
        "dataset_sha256": hashlib.sha256((root / RESEARCH_PATH).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256((root / MANIFEST_PATH).read_bytes()).hexdigest(),
        "feature_schema": dataset.attrs["feature_columns"],
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "development_folds": ["dev_1", "dev_2", "dev_3"],
        "random_seed": 42,
        "model_selection_rule": SELECTION_RULE,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "model_configurations": [asdict(cfg) for cfg in MODEL_CONFIGS],
        "svm_baselines": svm_baselines,
        "selected": selected,
        "final_development_models": {
            "svm": svm_final_metadata["models"],
            "lightgbm": lgbm_final_metadata["models"],
        },
        "test_result": test_result,
    }

    # 8. Ghi artifacts ra results/modeling/lightgbm/
    output.mkdir(parents=True, exist_ok=False)
    fold_metrics.to_csv(output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    training_log_df.to_csv(output / "training_log.csv", index=False, encoding="utf-8-sig")
    full_feature_importance_df.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    prediction_data.to_parquet(output / "validation_predictions.parquet", index=False)
    (output / "selected_configs.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # 9. Soạn báo cáo kỹ thuật tiếng Việt Markdown
    report_content = render_report(
        metadata, fold_metrics, comparison, full_feature_importance_df, svm_baselines
    )
    (output / "wp34_lightgbm_report.md").write_text(report_content, encoding="utf-8")

    return metadata


def render_report(
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    svm_baselines: dict[str, Any],
) -> str:
    """Tạo báo cáo kỹ thuật Markdown đầy đủ bằng tiếng Việt theo chuẩn Quantitative Research."""
    lines = [
        "# Báo cáo WP3.4: LightGBM Modeling, Feature Importance & Final Development Models\n",
        "## 1. Tổng quan WP3.4\n",
        "Nghiệm thu mô hình **LightGBM** (Gradient Boosted Decision Trees) cho bài toán RQ1 (dự báo hướng lợi suất) "
        "và RQ2 (dự báo lợi nhuận kinh tế sau toàn bộ ma sát thị trường) trên ba development folds (`dev_1`, `dev_2`, `dev_3`). "
        "Đánh giá và so sánh 1-1 với **Dummy baseline** và **SVM baseline (WP3.3)**. "
        "Trích xuất Feature Importance và lưu trữ Final Development Models (2016–2021) vào thư mục `models/`.\n",
        "## 2. Thiết lập thực nghiệm và Kiến trúc an toàn\n",
        f"- **Dataset Parquet:** `{metadata['dataset_path']}` (SHA-256: `{metadata['dataset_sha256']}`).\n"
        f"- **Split Manifest:** `{MANIFEST_PATH.as_posix()}` (SHA-256: `{metadata['split_manifest_sha256']}`).\n"
        f"- **Feature Schema:** {len(metadata['feature_schema'])} causal features không chứa look-ahead bias.\n"
        f"- **Môi trường:** Python {metadata['versions']['python']}, LightGBM {metadata['versions']['lightgbm']}, "
        f"scikit-learn {metadata['versions']['scikit_learn']}, joblib {metadata['versions']['joblib']}.\n"
        "- **Kiểm soát rò rỉ:** Dữ liệu đầu vào cho LightGBM là raw features (không dùng scaler toàn cục). "
        "Chỉ sử dụng API `get_development_splits()` cho 3 dev folds. "
        "Tập Reserved Test (2022–2026) được bảo vệ tuyệt đối, không có bất kỳ dự báo hay đánh giá nào.\n"
        f"- **Early Stopping:** Dừng sau {metadata['early_stopping_rounds']} rounds nếu `valid_binary_logloss` không cải thiện.\n",
    ]

        # Kết quả của từng câu hỏi nghiên cứu
    for rq, title, sec_num in (("RQ1", "Dự báo hướng giá (Direction)", 3),
                              ("RQ2", "Nhãn kinh tế sau chi phí (Fixed-Hold Net Proxy)", 4)):
        winner = metadata["selected"][rq]
        svm_w = svm_baselines[rq]
        lines += [
            f"## {sec_num}. Kết quả {rq} ({title})\n",
            "### Bảng so sánh hiệu suất Walk-Forward CV (Balanced Accuracy theo từng fold)\n",
            "| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Mean BAcc | Std BAcc | vs Dummy | vs SVM Baseline |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        # Mô hình đối chứng Dummy
        dummy_row = comparison[(comparison.rq == rq) & (comparison.config_id == "dummy_prior")].iloc[0]
        d_m = metrics[(metrics.rq == rq) & (metrics.config_id == "dummy_prior")].set_index("fold_name")
        lines.append(
            f"| `dummy_prior` (Baseline) | {d_m.loc['dev_1', 'balanced_accuracy']:.4f} | "
            f"{d_m.loc['dev_2', 'balanced_accuracy']:.4f} | {d_m.loc['dev_3', 'balanced_accuracy']:.4f} | "
            f"{dummy_row.mean_balanced_accuracy:.4f} | {dummy_row.std_balanced_accuracy:.4f} | "
            f"{dummy_row.delta_mean_balanced_accuracy_vs_dummy:+.4f} | "
            f"{dummy_row.mean_balanced_accuracy - svm_w['mean_balanced_accuracy']:+.4f} |"
        )

        # Cấu hình SVM thắng cuộc từ WP3.3
        s_f = svm_w["fold_balanced_accuracies"]
        lines.append(
            f"| `{svm_w['config_id']}` (SVM WP3.3) | {s_f.get('dev_1', np.nan):.4f} | "
            f"{s_f.get('dev_2', np.nan):.4f} | {s_f.get('dev_3', np.nan):.4f} | "
            f"{svm_w['mean_balanced_accuracy']:.4f} | {svm_w['std_balanced_accuracy']:.4f} | "
            f"{svm_w['delta_mean_balanced_accuracy_vs_dummy']:+.4f} | +0.0000 |"
        )

        # Các cấu hình LightGBM
        for row in comparison[(comparison.rq == rq) & (comparison.model_name != "dummy")].itertuples():
            f_m = metrics[(metrics.rq == rq) & (metrics.config_id == row.config_id)].set_index("fold_name")
            is_win = " **(Winner)**" if row.config_id == winner["config_id"] else ""
            lines.append(
                f"| `{row.config_id}`{is_win} | {f_m.loc['dev_1', 'balanced_accuracy']:.4f} | "
                f"{f_m.loc['dev_2', 'balanced_accuracy']:.4f} | {f_m.loc['dev_3', 'balanced_accuracy']:.4f} | "
                f"{row.mean_balanced_accuracy:.4f} | {row.std_balanced_accuracy:.4f} | "
                f"{row.delta_mean_balanced_accuracy_vs_dummy:+.4f} | "
                f"{row.delta_mean_balanced_accuracy_vs_svm:+.4f} |"
            )

        lines += [
            f"\n**Cấu hình LightGBM thắng cuộc {rq}:** `{winner['config_id']}`\n",
            f"- Mean Balanced Accuracy: **{winner['mean_balanced_accuracy']:.4f}** "
            f"(Std: {winner['std_balanced_accuracy']:.4f}).\n",
            f"- Chênh lệch so với Dummy baseline: **{winner['delta_mean_balanced_accuracy_vs_dummy']:+.4f}**.\n",
            f"- Chênh lệch so với SVM baseline (WP3.3): **{winner['mean_balanced_accuracy'] - svm_w['mean_balanced_accuracy']:+.4f}**.\n",
            f"- Mean ROC-AUC: {winner['mean_roc_auc']:.4f}; Mean Generalization Gap (Train - Valid): {winner['mean_generalization_gap']:.4f}.\n",
            f"- Best iterations qua 3 folds: {metadata['final_development_models']['lightgbm'][rq]['fold_best_iterations']} "
            f"(Trung bình làm tròn chọn `final_n_estimators = {metadata['final_development_models']['lightgbm'][rq]['final_n_estimators']}`).\n",
        ]

    # Phân tích Feature Importance
    lines += [
        "## 5. Phân tích Feature Importance (Gain, Split & Validation Permutation)\n",
        "So sánh giữa In-sample Tree Importance (khớp tập Train) và Out-of-sample Permutation Importance "
        "(xáo trộn đặc trưng trên tập Validation độc lập của từng fold với `scoring='balanced_accuracy'`):\n",
        "| RQ | Đặc trưng (Feature) | Mean Split | Mean Gain | Mean Permutation Importance | Std Permutation |\n"
        "|---|---|---:|---:|---:|---:|",
    ]

    for rq in RQ_TARGETS:
        winner_id = metadata["selected"][rq]["config_id"]
        sub_imp = feature_importance[
            (feature_importance.rq == rq)
            & (feature_importance.fold_name == "mean_3_folds")
            & (feature_importance.config_id == winner_id)
        ].copy()
        sub_imp = sub_imp.sort_values("perm_importance_mean", ascending=False)
        for row in sub_imp.head(7).itertuples():
            lines.append(
                f"| {rq} | `{row.feature_name}` | {row.split_importance:.1f} | "
                f"{row.gain_importance:.2f} | **{row.perm_importance_mean:+.4f}** | {row.perm_importance_std:.4f} |"
            )

    lines += [
        "\n*Ghi chú quan trọng:* Đặc trưng có Gain cao trên tập Train không nhất thiết mang lại giá trị "
        "dự báo ngoài mẫu nếu Permutation Importance trên Validation là âm hoặc xấp xỉ 0. "
        "Bảng đầy đủ 26 đặc trưng được lưu tại `feature_importance.csv`.\n",
        "## 6. Lưu trữ Final Development Models (2016–2021) vào `models/`\n",
        "Tuân thủ triệt để nguyên tắc không lưu rời rạc từng fold; chỉ fit lại đúng 1 model cuối cùng "
        "cho mỗi RQ trên toàn bộ dữ liệu phát triển 2016–2021 (1.297 mẫu hợp lệ sau purging và 5 phiên gap):\n",
        "| Họ mô hình | File Model Joblib | Metadata JSON | SHA-256 Joblib (10 ký tự đầu) | Train Samples |\n"
        "|---|---|---|---|---:|",
    ]

    for model_family in ("svm", "lightgbm"):
        for rq in ("RQ1", "RQ2"):
            m_info = metadata["final_development_models"][model_family][rq]
            lines.append(
                f"| {model_family.upper()} ({rq}) | `{m_info['model_file']}` | "
                f"`{model_family}_final_dev_metadata.json` | `{m_info['model_sha256'][:10]}` | {m_info['train_samples']} |"
            )

    lines += [
        "\nĐã tạo sẵn thư mục giữ chỗ `models/lstm/.gitkeep` cho WP3.5.\n",
        "## 7. Cảnh báo Phương pháp luận và Giới hạn\n",
        "- **Validation Selection Bias:** Điểm số Validation phản ánh thiên lệch lựa chọn do quá trình tinh chỉnh "
        "siêu tham số và dừng sớm (`early_stopping`). Tuyệt đối không coi đây là lợi nhuận thực tế hay alpha giao dịch.\n"
        "- **Bảo vệ Reserved Test:** Vùng Test 2022–2026 chưa được mở khóa và chưa thực hiện backtest.\n"
        "- **Độ nhạy mẫu nhỏ:** Chuỗi thời gian tài chính HPG có tỷ lệ SNR thấp; các cây quyết định có thể "
        "gặp hiện tượng regime shift qua các chu kỳ kinh tế 2019, 2020 và 2021.\n",
        "## 8. Kết quả Kiểm thử Tự động\n",
    ]
    res = metadata["test_result"]
    lines.append(
        f"Lệnh: `{res['command']}`. Kết quả: {res['passed']} PASS, {res['failed']} FAIL.\n"
        if res else "Chưa có kết quả kiểm thử được cung cấp trong lần gọi này.\n"
    )
    lines += [
        "## 9. Kết luận WP3.4\n",
        "`COMPLETED_WITH_LIMITATIONS`: Quá trình huấn luyện, đánh giá LightGBM, trích xuất Feature Importance "
        "và lưu trữ Final Development Model (2016–2021) hoàn thành đạt chuẩn.\n",
    ]
    return "\n".join(lines)


def _run_tests(root: Path) -> dict[str, Any]:
    """Chạy toàn bộ test suite để đảm bảo không có bất kỳ regression nào."""
    command = [sys.executable, "-m", "pytest", "tests/unit", "tests/integration", "-q", "--tb=short"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Kiểm thử thất bại:\n{result.stdout}\n{result.stderr}")
    counts = re.search(r"(\d+) passed", result.stdout)
    if counts is None:
        raise RuntimeError("Không xác định được số test đã đạt từ output của pytest")
    return {
        "command": ".venv\\Scripts\\python.exe -m pytest tests/unit tests/integration -q --tb=short",
        "passed": int(counts.group(1)),
        "failed": 0,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(ROOT, _run_tests(ROOT))
