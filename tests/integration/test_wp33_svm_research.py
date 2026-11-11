"""Kiểm tra pipeline WP3.3 trên một development fold của HPG Research."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.dataset_builder import load_project_config
from src.models.data import RESEARCH_PATH, load_research_dataset, make_modeling_view
from src.models.evaluation import binary_metrics
from src.models.modeling_safety import MANIFEST_PATH, get_development_splits
from src.models.svm_baseline import MODEL_CONFIGS, evaluate_fold

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (ROOT / RESEARCH_PATH).is_file(),
                    reason="Chưa có dataset Research trong môi trường kiểm thử")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_real_development_predictions_and_integrity(rq):
    paths = [ROOT / RESEARCH_PATH, ROOT / "configs/config.yaml", ROOT / MANIFEST_PATH]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    config = load_project_config(paths[1])
    view = make_modeling_view(load_research_dataset(ROOT, config), config, rq)
    folds = get_development_splits(ROOT, rq)
    record, predictions = evaluate_fold(view, rq, folds[0], MODEL_CONFIGS[1])
    assert record["rq"] == rq and record["fold_name"] == "dev_1"
    assert predictions.signal_date.tolist() == view.X.index[list(folds[0].validation_indices)].tolist()
    assert predictions.signal_date.dt.year.max() < 2022
    assert predictions.label_end_date.dt.year.max() < 2022
    assert predictions.ranking_score.notna().all()
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]


@pytest.mark.skipif(not (ROOT / "results/modeling/svm/validation_predictions.parquet").is_file(),
                    reason="Chưa có prediction artifacts của WP3.3")
@pytest.mark.parametrize("rq", ["RQ1", "RQ2"])
def test_saved_predictions_match_verified_indices(rq):
    """Đối chiếu ngày và nhãn từng nhóm prediction với view và fold đã xác minh."""
    config = load_project_config(ROOT / "configs/config.yaml")
    view = make_modeling_view(load_research_dataset(ROOT, config), config, rq)
    folds = {fold.fold_name: fold for fold in get_development_splits(ROOT, rq)}
    saved = pd.read_parquet(ROOT / "results/modeling/svm/validation_predictions.parquet")
    metrics = pd.read_csv(ROOT / "results/modeling/svm/metrics_by_fold.csv")
    groups = saved[saved.rq == rq].groupby(["config_id", "fold_name"])
    assert len(groups) == len(MODEL_CONFIGS) * len(folds)
    for (config_id, fold_name), group in groups:
        indices = list(folds[fold_name].validation_indices)
        assert group.signal_date.tolist() == view.X.index[indices].tolist()
        assert group.y_true.tolist() == view.y.iloc[indices].astype(int).tolist()
        assert group.label_end_date.dt.year.max() < 2022
        actual = binary_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy(),
                                group.ranking_score.to_numpy())
        row = metrics[(metrics.rq == rq) & (metrics.config_id == config_id)
                      & (metrics.fold_name == fold_name)].iloc[0]
        for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                       "roc_auc", "average_precision"):
            assert row[metric] == pytest.approx(actual[metric])
