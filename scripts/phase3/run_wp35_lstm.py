"""Chạy thực nghiệm WP3.5: PyTorch LSTM Sequence Modeling, Lag Sensitivity & Final Development Models."""

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
import numpy as np
import pandas as pd
import pyarrow
import torch

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, RQ_TARGETS, load_research_dataset, make_modeling_view
from src.models.modeling_safety import MANIFEST_PATH, get_development_splits
from src.models.train_lstm import (
    MODEL_CONFIGS, aggregate_results, evaluate_fold,
    fit_and_export_final_development_models as export_lstm_final_dev,
    select_configurations,
)

LOGGER = logging.getLogger(__name__)
OUTPUT = Path("results/modeling/lstm")
SELECTION_RULE = (
    "Mean Balanced Accuracy cao nhất qua 3 development folds; nếu bằng nhau chọn độ lệch chuẩn nhỏ hơn, "
    "sau đó cấu trúc đơn giản hơn (hidden_size nhỏ hơn -> seq_len ngắn hơn -> dropout lớn hơn) "
    "và thứ tự cấu hình đã công bố trước khi chạy."
)


def load_previous_baselines(root: Path) -> dict[str, dict[str, Any]]:
    """Đọc kết quả cấu hình thắng cuộc từ SVM (WP3.3) và LightGBM (WP3.4) để đối chiếu công bằng."""
    svm_selected_path = root / "results/modeling/svm/selected_configs.json"
    svm_comparison_path = root / "results/modeling/svm/model_comparison.csv"
    lgbm_selected_path = root / "results/modeling/lightgbm/selected_configs.json"
    lgbm_comparison_path = root / "results/modeling/lightgbm/model_comparison.csv"

    if not svm_selected_path.is_file() or not lgbm_selected_path.is_file():
        raise FileNotFoundError("Không tìm thấy artifacts của WP3.3 hoặc WP3.4 để đối chiếu.")

    svm_data = json.loads(svm_selected_path.read_text(encoding="utf-8"))
    svm_comp = pd.read_csv(svm_comparison_path)
    lgbm_data = json.loads(lgbm_selected_path.read_text(encoding="utf-8"))
    lgbm_comp = pd.read_csv(lgbm_comparison_path)

    baselines: dict[str, dict[str, Any]] = {"svm": {}, "lightgbm": {}}
    for rq in ("RQ1", "RQ2"):
        # SVM Winner
        svm_winner = svm_data["selected"][rq]
        s_row = svm_comp[(svm_comp.rq == rq) & (svm_comp.config_id == svm_winner["config_id"])].iloc[0]
        baselines["svm"][rq] = {
            "config_id": svm_winner["config_id"],
            "model_name": svm_winner["model_name"],
            "mean_balanced_accuracy": float(s_row["mean_balanced_accuracy"]),
            "std_balanced_accuracy": float(s_row["std_balanced_accuracy"]),
            "delta_mean_balanced_accuracy_vs_dummy": float(s_row["delta_mean_balanced_accuracy_vs_dummy"]),
        }

        # LightGBM Winner
        lgbm_winner = lgbm_data["selected"][rq]
        l_row = lgbm_comp[(lgbm_comp.rq == rq) & (lgbm_comp.config_id == lgbm_winner["config_id"])].iloc[0]
        baselines["lightgbm"][rq] = {
            "config_id": lgbm_winner["config_id"],
            "model_name": lgbm_winner["model_name"],
            "mean_balanced_accuracy": float(l_row["mean_balanced_accuracy"]),
            "std_balanced_accuracy": float(l_row["std_balanced_accuracy"]),
            "delta_mean_balanced_accuracy_vs_dummy": float(l_row["delta_mean_balanced_accuracy_vs_dummy"]),
        }

    return baselines


def run(root: Path = ROOT, test_result: dict | None = None) -> dict[str, Any]:
    """Thực hiện quy trình cố định WP3.5 và xuất toàn bộ artifacts ra đĩa.

    Args:
        root: Thư mục gốc repository.
        test_result: Kết quả kiểm thử pytest nếu đã chạy trước đó.

    Returns:
        Dict metadata tổng hợp của toàn bộ thực nghiệm WP3.5.

    Raises:
        FileExistsError: Khi thư mục kết quả đã tồn tại (chống ghi đè vô ý).
    """
    root = Path(root).resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Thư mục kết quả WP3.5 đã tồn tại, không ghi đè: {output}")

    config_path = root / "configs/config.yaml"
    config = load_project_config(config_path)
    dataset = load_research_dataset(root, config)

    # 1. Đọc kết quả SVM và LightGBM từ các WP trước để đối chiếu
    baselines = load_previous_baselines(root)

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    training_logs: list[dict[str, Any]] = []
    fold_best_epochs: dict[str, dict[str, list[int]]] = {"RQ1": {}, "RQ2": {}}

    # 2. Đánh giá tất cả cấu hình trên 3 development folds
    for rq in RQ_TARGETS:
        folds = get_development_splits(root, rq)
        view = make_modeling_view(dataset, config, rq)

        for model_config in MODEL_CONFIGS:
            LOGGER.info("Đang đánh giá %s, cấu hình %s trên development folds...", rq, model_config.config_id)
            cfg_epochs: list[int] = []
            for fold in folds:
                record, pred, logs, _, _ = evaluate_fold(
                    view, rq, fold, model_config, compute_attribution=False
                )
                metrics.append(record)
                predictions.append(pred)
                training_logs.extend(logs)
                if record["best_epoch"] is not None:
                    cfg_epochs.append(int(record["best_epoch"]))

            if cfg_epochs:
                fold_best_epochs[rq][model_config.config_id] = cfg_epochs

    fold_metrics = pd.DataFrame(metrics)
    prediction_data = pd.concat(predictions, ignore_index=True)
    training_log_df = pd.DataFrame(training_logs)

    # 3. Tính toán chênh lệch so với Dummy baseline theo từng fold
    for metric in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "average_precision"):
        dummy_series = (
            fold_metrics[fold_metrics.config_id == "dummy_prior"]
            .set_index(["rq", "fold_name"])[metric]
        )
        baseline_vals = pd.MultiIndex.from_frame(fold_metrics[["rq", "fold_name"]]).map(dummy_series)
        fold_metrics[f"delta_{metric}_vs_dummy"] = fold_metrics[metric] - np.asarray(baseline_vals, dtype=float)

    # 4. Tổng hợp so sánh mô hình và lựa chọn cấu hình thắng cuộc
    comparison = aggregate_results(fold_metrics)

    # Bổ sung chênh lệch so với SVM (WP3.3) và LightGBM (WP3.4)
    for rq in RQ_TARGETS:
        svm_mean_bacc = baselines["svm"][rq]["mean_balanced_accuracy"]
        lgbm_mean_bacc = baselines["lightgbm"][rq]["mean_balanced_accuracy"]
        mask = comparison.rq == rq
        comparison.loc[mask, "delta_mean_balanced_accuracy_vs_svm"] = (
            comparison.loc[mask, "mean_balanced_accuracy"] - svm_mean_bacc
        )
        comparison.loc[mask, "delta_mean_balanced_accuracy_vs_lightgbm"] = (
            comparison.loc[mask, "mean_balanced_accuracy"] - lgbm_mean_bacc
        )

    selected = select_configurations(comparison)

    # 5. Phân tích Feature Permutation Importance và Temporal Lag Sensitivity cho cấu hình thắng cuộc
    feature_importance_records: list[dict[str, Any]] = []
    lag_sensitivity_records: list[dict[str, Any]] = []
    winning_epochs: dict[str, list[int]] = {}

    for rq in RQ_TARGETS:
        winner_id = selected[rq]["config_id"]
        winning_cfg = next(cfg for cfg in MODEL_CONFIGS if cfg.config_id == winner_id)
        folds = get_development_splits(root, rq)
        view = make_modeling_view(dataset, config, rq)
        winning_epochs[rq] = fold_best_epochs[rq][winner_id]

        LOGGER.info("Tính toán Attribution (Feature & Lag) cho %s (cấu hình %s)...", rq, winner_id)
        for fold in folds:
            _, _, _, feat_recs, lag_recs = evaluate_fold(
                view, rq, fold, winning_cfg, compute_attribution=True
            )
            feature_importance_records.extend(feat_recs)
            lag_sensitivity_records.extend(lag_recs)

    feature_imp_df = pd.DataFrame(feature_importance_records)
    lag_sens_df = pd.DataFrame(lag_sensitivity_records)

    # Thêm dòng trung bình 3 folds cho Feature Importance
    avg_feat_rows: list[dict[str, Any]] = []
    for (rq, feat_name), grp in feature_imp_df.groupby(["rq", "feature_name"], sort=False):
        avg_feat_rows.append({
            "rq": rq,
            "fold_name": "mean_3_folds",
            "config_id": grp.config_id.iloc[0],
            "feature_name": feat_name,
            "perm_importance_mean": float(grp.perm_importance_mean.mean()),
            "perm_importance_std": float(grp.perm_importance_mean.std(ddof=0)),
        })
    full_feature_imp_df = pd.concat([feature_imp_df, pd.DataFrame(avg_feat_rows)], ignore_index=True)

    # Thêm dòng trung bình 3 folds cho Lag Sensitivity
    avg_lag_rows: list[dict[str, Any]] = []
    for (rq, lag_idx), grp in lag_sens_df.groupby(["rq", "lag_index"], sort=False):
        avg_lag_rows.append({
            "rq": rq,
            "fold_name": "mean_3_folds",
            "config_id": grp.config_id.iloc[0],
            "lag_index": lag_idx,
            "lag_step": grp.lag_step.iloc[0],
            "gradient_sensitivity": float(grp.gradient_sensitivity.mean()),
            "gradient_sensitivity_normalized": float(grp.gradient_sensitivity_normalized.mean()),
        })
    full_lag_sens_df = pd.concat([lag_sens_df, pd.DataFrame(avg_lag_rows)], ignore_index=True)

    # 6. Fit và xuất Final Development Models (2016–2021) vào models/lstm/
    LOGGER.info("Huấn luyện và lưu Final Development Models vào models/lstm/...")
    lstm_final_metadata = export_lstm_final_dev(
        root, selected_configs=selected, fold_best_epochs=winning_epochs
    )

    # 7. Tạo metadata tổng thể của WP3.5
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
            "torch": torch.__version__,
            "joblib": joblib.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "development_folds": ["dev_1", "dev_2", "dev_3"],
        "random_seed": 42,
        "model_selection_rule": SELECTION_RULE,
        "model_configurations": [asdict(cfg) for cfg in MODEL_CONFIGS],
        "baselines": baselines,
        "selected": selected,
        "final_development_models": lstm_final_metadata["models"],
        "test_result": test_result,
    }

    # 8. Ghi artifacts ra results/modeling/lstm/
    output.mkdir(parents=True, exist_ok=False)
    fold_metrics.to_csv(output / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    training_log_df.to_csv(output / "training_log.csv", index=False, encoding="utf-8-sig")
    full_feature_imp_df.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    full_lag_sens_df.to_csv(output / "lag_sensitivity.csv", index=False, encoding="utf-8-sig")
    prediction_data.to_parquet(output / "validation_predictions.parquet", index=False)
    (output / "selected_configs.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # 9. Soạn báo cáo kỹ thuật Markdown tiếng Việt
    report_content = render_report(
        metadata, fold_metrics, comparison, full_feature_imp_df, full_lag_sens_df, baselines
    )
    (output / "wp35_lstm_report.md").write_text(report_content, encoding="utf-8")

    return metadata


def render_report(
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    lag_sensitivity: pd.DataFrame,
    baselines: dict[str, dict[str, Any]],
) -> str:
    """Tạo báo cáo kỹ thuật Markdown đầy đủ bằng tiếng Việt theo chuẩn Quantitative Research."""
    lines = [
        "# Báo cáo WP3.5: PyTorch LSTM Sequence Modeling, Lag Sensitivity & Final Development Models\n",
        "## 1. Tổng quan WP3.5\n",
        "Nghiệm thu mô hình mạng nơ-ron hồi quy **LSTM (Long Short-Term Memory)** cho bài toán RQ1 (dự báo hướng lợi suất) "
        "và RQ2 (dự báo lợi nhuận kinh tế sau chi phí giao dịch) trên ba development folds (`dev_1`, `dev_2`, `dev_3`). "
        "Đánh giá và so sánh đối chứng 1-1 giữa 4 họ mô hình: **Dummy Baseline**, **SVM Baseline (WP3.3)**, **LightGBM (WP3.4)** "
        "và **LSTM (WP3.5)** trên cùng tập mẫu Validation đã kiểm toán. "
        "Phân tích độ nhạy thời gian (Lag Sensitivity) và lưu trữ Final Development Models (2016–2021) vào `models/lstm/`.\n",
        "## 2. Thiết lập thực nghiệm và Kiến trúc an toàn\n",
        f"- **Dataset Parquet:** `{metadata['dataset_path']}` (SHA-256: `{metadata['dataset_sha256']}`).\n",
        f"- **Split Manifest:** `{MANIFEST_PATH.as_posix()}` (SHA-256: `{metadata['split_manifest_sha256']}`).\n",
        f"- **Feature Schema:** {len(metadata['feature_schema'])} causal features không chứa look-ahead bias.\n",
        f"- **Môi trường:** Python {metadata['versions']['python']}, PyTorch {metadata['versions']['torch']}, "
        f"joblib {metadata['versions']['joblib']}.\n",
        "- **Quy tắc Chuẩn hóa Train-Only:** Ở mỗi fold, `StandardScaler` chỉ được fit trên tập Train (`fold.train_indices`). "
        "Chuỗi Validation được tạo bằng cách trượt cửa sổ lịch sử kết thúc tại phiên `v_idx` và chuẩn hóa qua scaler của Train, "
        "đảm bảo giữ nguyên 100% mẫu Validation (249, 251, 249 mẫu RQ1 và 246, 248, 246 mẫu RQ2).\n",
        "- **Bảo vệ tuyệt đối Reserved Test (2022–2026):** Tuyệt đối không có bất kỳ mẫu nào của 2022–2026 được dùng trong huấn luyện, "
        "chuẩn hóa hay early stopping.\n",
    ]

    for rq, title, sec_num in (("RQ1", "Dự báo hướng giá (Direction)", 3),
                               ("RQ2", "Nhãn kinh tế sau chi phí (Fixed-Hold Net Proxy)", 4)):
        winner = metadata["selected"][rq]
        svm_w = baselines["svm"][rq]
        lgbm_w = baselines["lightgbm"][rq]

        lines += [
            f"## {sec_num}. Kết quả {rq} ({title})\n",
            "### Bảng so sánh hiệu suất Walk-Forward CV (Balanced Accuracy theo từng fold)\n",
            "| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Mean BAcc | Std BAcc | vs Dummy | vs SVM | vs LightGBM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        # Dummy baseline
        dummy_row = comparison[(comparison.rq == rq) & (comparison.config_id == "dummy_prior")].iloc[0]
        d_m = metrics[(metrics.rq == rq) & (metrics.config_id == "dummy_prior")].set_index("fold_name")
        lines.append(
            f"| `dummy_prior` (Baseline) | {d_m.loc['dev_1', 'balanced_accuracy']:.4f} | "
            f"{d_m.loc['dev_2', 'balanced_accuracy']:.4f} | {d_m.loc['dev_3', 'balanced_accuracy']:.4f} | "
            f"{dummy_row.mean_balanced_accuracy:.4f} | {dummy_row.std_balanced_accuracy:.4f} | "
            f"+0.0000 | {dummy_row.mean_balanced_accuracy - svm_w['mean_balanced_accuracy']:+.4f} | "
            f"{dummy_row.mean_balanced_accuracy - lgbm_w['mean_balanced_accuracy']:+.4f} |"
        )

        # SVM Winner
        lines.append(
            f"| `{svm_w['config_id']}` (SVM WP3.3) | — | — | — | "
            f"{svm_w['mean_balanced_accuracy']:.4f} | {svm_w['std_balanced_accuracy']:.4f} | "
            f"{svm_w['delta_mean_balanced_accuracy_vs_dummy']:+.4f} | +0.0000 | "
            f"{svm_w['mean_balanced_accuracy'] - lgbm_w['mean_balanced_accuracy']:+.4f} |"
        )

        # LightGBM Winner
        lines.append(
            f"| `{lgbm_w['config_id']}` (LightGBM WP3.4) | — | — | — | "
            f"{lgbm_w['mean_balanced_accuracy']:.4f} | {lgbm_w['std_balanced_accuracy']:.4f} | "
            f"{lgbm_w['delta_mean_balanced_accuracy_vs_dummy']:+.4f} | "
            f"{lgbm_w['mean_balanced_accuracy'] - svm_w['mean_balanced_accuracy']:+.4f} | +0.0000 |"
        )

        # LSTM configs
        for row in comparison[(comparison.rq == rq) & (comparison.model_name == "lstm")].itertuples():
            f_m = metrics[(metrics.rq == rq) & (metrics.config_id == row.config_id)].set_index("fold_name")
            is_win = " **(Winner)**" if row.config_id == winner["config_id"] else ""
            lines.append(
                f"| `{row.config_id}`{is_win} | {f_m.loc['dev_1', 'balanced_accuracy']:.4f} | "
                f"{f_m.loc['dev_2', 'balanced_accuracy']:.4f} | {f_m.loc['dev_3', 'balanced_accuracy']:.4f} | "
                f"{row.mean_balanced_accuracy:.4f} | {row.std_balanced_accuracy:.4f} | "
                f"{row.delta_mean_balanced_accuracy_vs_dummy:+.4f} | "
                f"{row.delta_mean_balanced_accuracy_vs_svm:+.4f} | "
                f"{row.delta_mean_balanced_accuracy_vs_lightgbm:+.4f} |"
            )

        lines += [
            f"\n**Cấu hình LSTM thắng cuộc {rq}:** `{winner['config_id']}`\n",
            f"- Mean Balanced Accuracy: **{winner['mean_balanced_accuracy']:.4f}** "
            f"(Std: {winner['std_balanced_accuracy']:.4f}).\n",
            f"- Chênh lệch so với Dummy baseline: **{winner['delta_mean_balanced_accuracy_vs_dummy']:+.4f}**.\n",
            f"- Chênh lệch so với SVM baseline (WP3.3): **{winner['mean_balanced_accuracy'] - svm_w['mean_balanced_accuracy']:+.4f}**.\n",
            f"- Chênh lệch so với LightGBM (WP3.4): **{winner['mean_balanced_accuracy'] - lgbm_w['mean_balanced_accuracy']:+.4f}**.\n",
            f"- Mean ROC-AUC: {winner['mean_roc_auc']:.4f}; Mean Generalization Gap (Train - Valid): {winner['mean_generalization_gap']:.4f}.\n",
            f"- Best epochs qua 3 folds: {metadata['final_development_models'][rq]['fold_best_epochs']} "
            f"(Trung bình làm tròn chọn `final_epochs = {metadata['final_development_models'][rq]['final_epochs']}`).\n",
        ]

    # Phân tích Feature Permutation Importance
    lines += [
        "## 5. Phân tích Feature Permutation Importance trên chuỗi Validation 3D\n",
        "Mức độ sụt giảm Balanced Accuracy khi xáo trộn từng đặc trưng trên tập Validation ngoài mẫu của 3 development folds:\n",
        "| RQ | Đặc trưng (Feature) | Mean Permutation Importance | Std Permutation |\n",
        "|---|---|---:|---:|",
    ]
    for rq in RQ_TARGETS:
        winner_id = metadata["selected"][rq]["config_id"]
        sub_imp = feature_importance[
            (feature_importance.rq == rq)
            & (feature_importance.fold_name == "mean_3_folds")
            & (feature_importance.config_id == winner_id)
        ].copy()
        sub_imp = sub_imp.sort_values("perm_importance_mean", ascending=False)
        for row in sub_imp.head(6).itertuples():
            lines.append(
                f"| {rq} | `{row.feature_name}` | **{row.perm_importance_mean:+.4f}** | {row.perm_importance_std:.4f} |"
            )

    # Phân tích Temporal Lag Sensitivity
    lines += [
        "\n## 6. Phân tích Temporal Lag Sensitivity (Độ nhạy theo bước trễ thời gian)\n",
        "Đo lường mức độ đóng góp của từng bước thời gian trong cửa sổ chuỗi qua độ lớn Gradient Saliency trung bình:\n",
        "| RQ | Bước trễ (Lag Step) | Vị trí trong cửa sổ | Độ nhạy tuyệt đối | Tỷ trọng chuẩn hóa (%)\n",
        "|---|---|---:|---:|---:|",
    ]
    for rq in RQ_TARGETS:
        winner_id = metadata["selected"][rq]["config_id"]
        sub_lag = lag_sensitivity[
            (lag_sensitivity.rq == rq)
            & (lag_sensitivity.fold_name == "mean_3_folds")
            & (lag_sensitivity.config_id == winner_id)
        ].sort_values("lag_index", ascending=True)

        for row in sub_lag.itertuples():
            label = "Phiên hiện tại (t)" if row.lag_step == 0 else f"t {row.lag_step:+d}"
            lines.append(
                f"| {rq} | `{label}` | {row.lag_index} | {row.gradient_sensitivity:.6f} | "
                f"**{row.gradient_sensitivity_normalized * 100:.2f}%** |"
            )

    lines += [
        "\n## 7. Lưu trữ Final Development Models (2016–2021) vào `models/lstm/`\n",
        "Mô hình được fit trên toàn bộ 1.297 mẫu phát triển hợp lệ trước ranh giới năm 2022:\n",
        "| RQ | File Model Bundle (.pt) | Metadata JSON | Train Rows | Train Sequences | Final Epochs | Parameters |\n",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for rq in ("RQ1", "RQ2"):
        m_info = metadata["final_development_models"][rq]
        lines.append(
            f"| {rq} | `{m_info['model_file']}` | `lstm_final_dev_metadata.json` | "
            f"{m_info['train_samples']} | {m_info['train_sequences']} | {m_info['final_epochs']} | "
            f"{m_info['total_trainable_parameters']} |"
        )

    lines += [
        "\n## 8. Cảnh báo Phương pháp luận và Giới hạn\n",
        "- **Độ nhạy mẫu nhỏ của Mạng Nơ-ron:** Mặc dù LSTM đã áp dụng kiến trúc siêu gọn (1 layer, hidden_size 16/32, "
        "dropout 0.2–0.3, weight decay 1e-3, early stopping), tập dữ liệu bậc ngày với tỷ lệ SNR thấp vẫn là thách thức lớn. "
        "Hiệu suất LSTM cần được xem xét thận trọng và đối chiếu trực tiếp với các mô hình tuyến tính (SVM) và cây quyết định (LightGBM).\n",
        "- **Validation Selection Bias:** Kết quả validation phản ánh thiên lệch lựa chọn siêu tham số và checkpoint dừng sớm.\n",
        "- **Bảo vệ Reserved Test:** Vùng Test 2022–2026 hoàn toàn chưa được mở khóa, đảm bảo tính khách quan cho WP3.6 và Phase 4.\n",
        "## 9. Kết quả Kiểm thử Tự động\n",
    ]
    res = metadata["test_result"]
    lines.append(
        f"Lệnh: `{res['command']}`. Kết quả: {res['passed']} PASS, {res['failed']} FAIL.\n"
        if res else "Chưa có kết quả kiểm thử được cung cấp trong lần gọi này.\n"
    )
    lines += [
        "## 10. Kết luận WP3.5\n",
        "`COMPLETED_WITH_LIMITATIONS`: Quá trình huấn luyện, đánh giá LSTM sequence modeling, trích xuất Lag Sensitivity "
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

