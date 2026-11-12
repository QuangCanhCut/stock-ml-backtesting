"""Kiểm thử đơn vị cho bộ sinh tín hiệu nhân quả và kiểm thử tính bất biến tiền tố (WP4.2)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.signals import (
    apply_hysteresis_thresholds,
    compute_causal_expanding_zscore,
    compute_signal_diagnostics,
    generate_causal_quantile_signals,
    generate_fixed_signals,
)


def test_hysteresis_state_machine() -> None:
    """Kiểm tra máy trạng thái trễ chuyển đổi đúng các trạng thái ENTRY, BUFFER, EXIT."""
    # Ngưỡng cố định 0.53 / 0.48
    entry_th = np.full(6, 0.53)
    exit_th = np.full(6, 0.48)

    # Chuỗi xác suất mô phỏng:
    # 0: 0.50 (buffer -> giữ 0)
    # 1: 0.55 (>= 0.53 -> ENTRY, pos=1)
    # 2: 0.51 (buffer -> giữ 1)
    # 3: 0.49 (buffer -> giữ 1)
    # 4: 0.45 (<= 0.48 -> EXIT, pos=0)
    # 5: 0.50 (buffer -> giữ 0)
    probs = np.array([0.50, 0.55, 0.51, 0.49, 0.45, 0.50])

    triggers, positions = apply_hysteresis_thresholds(
        scores=probs,
        entry_thresholds=entry_th,
        exit_thresholds=exit_th,
        initial_position=0,
    )

    expected_triggers = ["BUFFER", "ENTRY", "BUFFER", "BUFFER", "EXIT", "BUFFER"]
    expected_positions = [0, 1, 1, 1, 0, 0]

    assert list(triggers) == expected_triggers
    assert list(positions) == expected_positions


def test_fixed_signals_generation() -> None:
    """Kiểm tra hàm sinh tín hiệu ngưỡng cố định."""
    probs = np.array([0.40, 0.54, 0.50, 0.47])
    res = generate_fixed_signals(probs, entry_thresh=0.53, exit_thresh=0.48)

    assert list(res["trigger"]) == ["EXIT", "ENTRY", "BUFFER", "EXIT"]
    assert list(res["desired_position"]) == [0, 1, 1, 0]
    assert np.all(res["entry_threshold_val"] == 0.53)
    assert np.all(res["exit_threshold_val"] == 0.48)


def test_causal_quantile_prefix_invariance() -> None:
    """Kiểm thử tính bất biến tiền tố (Prefix Invariance) — Bảo chứng Zero Look-Ahead Bias.

    Nguyên lý: Tín hiệu và ngưỡng phân vị tại ngày t (t <= 50) TUYỆT ĐỐI KHÔNG ĐƯỢC PHÉP
    thay đổi khi dữ liệu tương lai (t > 50) bị sửa đổi hoặc biến động mạnh.
    """
    np.random.seed(42)
    seed_buf = np.random.normal(loc=0.0, scale=1.0, size=200)

    # Chuỗi điểm số gốc gồm 100 phiên
    scores_original = np.random.normal(loc=0.1, scale=1.0, size=100)

    # Chạy lần 1 trên chuỗi gốc
    res_orig = generate_causal_quantile_signals(
        test_scores=scores_original,
        seed_history=seed_buf,
        q_entry=80.0,
        q_exit=30.0,
    )

    # Tạo chuỗi bị sửa đổi dữ liệu tương lai (từ phiên 50 đến 99 bị nhân 10 hoặc đổi dấu)
    scores_mutated = scores_original.copy()
    scores_mutated[50:] = scores_mutated[50:] * -5.0 + 10.0

    # Chạy lần 2 trên chuỗi bị đột biến tương lai
    res_mut = generate_causal_quantile_signals(
        test_scores=scores_mutated,
        seed_history=seed_buf,
        q_entry=80.0,
        q_exit=30.0,
    )

    # Khẳng định 100%: 50 phiên đầu tiên (t = 0..49) hoàn toàn bất biến (chênh lệch = 0)
    for i in range(50):
        assert pytest.approx(res_orig["entry_threshold_val"][i], abs=1e-12) == res_mut["entry_threshold_val"][i]
        assert pytest.approx(res_orig["exit_threshold_val"][i], abs=1e-12) == res_mut["exit_threshold_val"][i]
        assert res_orig["trigger"][i] == res_mut["trigger"][i]
        assert res_orig["desired_position"][i] == res_mut["desired_position"][i]

    # Khẳng định: Các phiên từ 50 trở đi đã bị thay đổi (chứng minh test phân biệt được đột biến)
    diff_count = np.sum(res_orig["entry_threshold_val"][50:] != res_mut["entry_threshold_val"][50:])
    assert diff_count > 0


def test_causal_expanding_zscore_prefix_invariance() -> None:
    """Kiểm thử tính bất biến tiền tố của hàm tính Causal Expanding Z-Score."""
    np.random.seed(123)
    seed_buf = np.random.normal(size=100)
    test_scores = np.random.normal(size=60)

    z_orig = compute_causal_expanding_zscore(test_scores, seed_buf)

    mutated_scores = test_scores.copy()
    mutated_scores[30:] += 100.0  # Thay đổi mạnh tương lai

    z_mut = compute_causal_expanding_zscore(mutated_scores, seed_buf)

    # 30 phiên đầu tiên phải bằng nhau tuyệt đối
    np.testing.assert_allclose(z_orig[:30], z_mut[:30], atol=1e-12)
    # Từ phiên 30 trở đi phải khác nhau
    assert not np.allclose(z_orig[30:], z_mut[30:], atol=1e-4)


def test_diagnostics_calculation() -> None:
    """Kiểm tra hàm tính chẩn đoán hành vi tín hiệu."""
    mock_signals = pd.DataFrame({
        "protocol": ["test_proto"] * 6,
        "rq": ["RQ1"] * 6,
        "model_name": ["svm"] * 6,
        "signal_mode": ["fixed"] * 6,
        "trigger": ["ENTRY", "BUFFER", "EXIT", "ENTRY", "BUFFER", "BUFFER"],
        "desired_position": [1, 1, 0, 1, 1, 1],
        "is_ceiling_locked": [False, False, False, True, False, False],
        "is_floor_locked": [False, False, True, False, False, False],
    })

    diag = compute_signal_diagnostics(mock_signals)
    assert len(diag) == 1
    row = diag.iloc[0]

    assert row["total_sessions"] == 6
    assert row["entry_trigger_count"] == 2
    assert row["exit_trigger_count"] == 1
    # Vị thế 1 ở 5 phiên trên tổng 6: 5/6 = 83.33%
    assert pytest.approx(row["desired_time_in_market_pct"], abs=0.1) == 83.33
    # Đợt nắm giữ: đợt 1 (2 phiên), đợt 2 (3 phiên) -> TB (2+3)/2 = 2.5
    assert row["avg_desired_holding_sessions"] == 2.5
    assert row["ceiling_locked_entries"] == 1
    assert row["floor_locked_exits"] == 1

