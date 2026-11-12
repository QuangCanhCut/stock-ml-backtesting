"""Bộ sinh tín hiệu nhân quả, lọc ngưỡng và kết hợp đa mô hình (WP4.2).

Module này hiện thực hóa:
- Mode 1: Ngưỡng xác suất cố định (fixed_053_048) với máy trạng thái trễ (Hysteresis State Machine).
- Mode 2: Ngưỡng phân vị nhân quả mở rộng (Causal Expanding Quantile Thresholds)
  với bộ đệm khởi tạo OOF Dev để tránh hiện tượng Đói Tín Hiệu (Signal Starvation).
  - Sub-mode 2A: quantile_q80_q30 (vào top 20%, ra bottom 30%).
  - Sub-mode 2B: quantile_q70_q40 (vào top 30%, ra bottom 40%).
- Mode 3: Kết hợp đa mô hình (Equal-weight Causal Z-Score Ensemble: SVM + LightGBM + LSTM).
- Bộ phân tích chẩn đoán hành vi tín hiệu (Signal Diagnostics).
"""

from typing import Any

import numpy as np
import pandas as pd


def apply_hysteresis_thresholds(
    scores: np.ndarray,
    entry_thresholds: np.ndarray,
    exit_thresholds: np.ndarray,
    initial_position: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Áp dụng máy trạng thái trễ (Hysteresis State Machine) để sinh tín hiệu và vị thế mong muốn.

    Quy tắc trạng thái:
    - Nếu score >= entry_thresh: trigger = 'ENTRY', desired_position = 1
    - Nếu score <= exit_thresh: trigger = 'EXIT', desired_position = 0
    - Nếu exit_thresh < score < entry_thresh (vùng đệm buffer):
        trigger = 'BUFFER', desired_position = desired_position_{t-1}

    Args:
        scores: Mảng điểm số 1D (xác suất hoặc ranking score) độ dài N.
        entry_thresholds: Mảng ngưỡng vào vị thế độ dài N.
        exit_thresholds: Mảng ngưỡng thoát vị thế độ dài N.
        initial_position: Trạng thái vị thế ban đầu trước khi bắt đầu chuỗi (mặc định 0 - tiền mặt).

    Returns:
        tuple (triggers, desired_positions) gồm mảng chuỗi trigger và mảng vị thế nhị phân (0 hoặc 1).
    """
    n = len(scores)
    triggers = np.empty(n, dtype=object)
    positions = np.zeros(n, dtype=int)

    current_pos = int(initial_position)
    for i in range(n):
        s = float(scores[i])
        e_th = float(entry_thresholds[i])
        x_th = float(exit_thresholds[i])

        if s >= e_th:
            triggers[i] = "ENTRY"
            current_pos = 1
        elif s <= x_th:
            triggers[i] = "EXIT"
            current_pos = 0
        else:
            triggers[i] = "BUFFER"

        positions[i] = current_pos

    return triggers, positions


def generate_fixed_signals(
    calibrated_probs: np.ndarray,
    entry_thresh: float = 0.53,
    exit_thresh: float = 0.48,
    initial_position: int = 0,
) -> dict[str, np.ndarray]:
    """Sinh tín hiệu theo chế độ ngưỡng xác suất cố định (Mode 1: fixed_053_048).

    Args:
        calibrated_probs: Mảng xác suất hiệu chuẩn 1D trong đoạn [0, 1].
        entry_thresh: Ngưỡng xác suất để kích hoạt Mua (mặc định 0.53).
        exit_thresh: Ngưỡng xác suất để kích hoạt Bán (mặc định 0.48).
        initial_position: Vị thế khởi đầu (mặc định 0).

    Returns:
        Từ điển chứa mảng trigger, desired_position, entry_threshold_val, exit_threshold_val.
    """
    n = len(calibrated_probs)
    entry_arr = np.full(n, fill_value=float(entry_thresh), dtype=float)
    exit_arr = np.full(n, fill_value=float(exit_thresh), dtype=float)

    triggers, positions = apply_hysteresis_thresholds(
        scores=calibrated_probs,
        entry_thresholds=entry_arr,
        exit_thresholds=exit_arr,
        initial_position=initial_position,
    )

    return {
        "trigger": triggers,
        "desired_position": positions,
        "entry_threshold_val": entry_arr,
        "exit_threshold_val": exit_arr,
    }


def generate_causal_quantile_signals(
    test_scores: np.ndarray,
    seed_history: np.ndarray,
    q_entry: float = 80.0,
    q_exit: float = 30.0,
    initial_position: int = 0,
) -> dict[str, np.ndarray]:
    """Sinh tín hiệu theo ngưỡng phân vị nhân quả mở rộng (Mode 2: Causal Expanding Quantiles).

    Quy tắc nhân quả bất biến (Strictly Causal):
    - Tại mỗi phiên Test t, tập lịch sử chỉ bao gồm:
        history_t = [seed_history, test_scores[0..t]]
    - Ngưỡng phân vị Q_entry và Q_exit được tính trên history_t.
    - Điểm số của tương lai (t+1 .. N-1) tuyệt đối không được phép xuất hiện.

    Args:
        test_scores: Mảng ranking_score của tập Test (độ dài N).
        seed_history: Mảng điểm số Out-of-Fold của tập Development (độ dài M) làm bộ đệm khởi tạo.
        q_entry: Phân vị vào lệnh (ví dụ 80.0 cho Top 20%).
        q_exit: Phân vị thoát lệnh (ví dụ 30.0 cho Bottom 30%).
        initial_position: Vị thế khởi đầu (mặc định 0).

    Returns:
        Từ điển chứa mảng trigger, desired_position, entry_threshold_val, exit_threshold_val.
    """
    n = len(test_scores)
    entry_arr = np.empty(n, dtype=float)
    exit_arr = np.empty(n, dtype=float)

    # Sử dụng danh sách động để tăng tốc nối mảng
    current_history = list(seed_history)

    for i in range(n):
        s_i = float(test_scores[i])
        current_history.append(s_i)

        entry_arr[i] = float(np.percentile(current_history, q_entry))
        exit_arr[i] = float(np.percentile(current_history, q_exit))

    triggers, positions = apply_hysteresis_thresholds(
        scores=test_scores,
        entry_thresholds=entry_arr,
        exit_thresholds=exit_arr,
        initial_position=initial_position,
    )

    return {
        "trigger": triggers,
        "desired_position": positions,
        "entry_threshold_val": entry_arr,
        "exit_threshold_val": exit_arr,
    }


def compute_causal_expanding_zscore(
    test_scores: np.ndarray,
    seed_history: np.ndarray,
) -> np.ndarray:
    """Tính chuỗi Z-Score nhân quả mở rộng cho một mô hình đơn lẻ.

    Tại phiên Test t:
        history_t = [seed_history, test_scores[0..t]]
        z_t = (test_scores[t] - mean(history_t)) / std(history_t)

    Args:
        test_scores: Mảng điểm số Test (độ dài N).
        seed_history: Mảng điểm số OOF Dev (độ dài M).

    Returns:
        Mảng 1D các giá trị Z-score nhân quả (độ dài N).
    """
    n = len(test_scores)
    z_scores = np.empty(n, dtype=float)
    current_history = list(seed_history)

    for i in range(n):
        s_i = float(test_scores[i])
        current_history.append(s_i)

        m = float(np.mean(current_history))
        s = float(np.std(current_history))
        if s < 1e-12:
            s = 1.0

        z_scores[i] = (s_i - m) / s

    return z_scores


def build_all_backtest_signals(
    aligned_df: pd.DataFrame,
    dev_oof_buffers: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Tạo bảng dữ liệu tín hiệu đầy đủ cho tất cả các tổ hợp giao thức, RQ, mô hình và signal_mode.

    Hỗ trợ:
    - 2 giao thức: frozen_2016_2021 và annual_expanding_retrain
    - 2 bài toán: RQ1 và RQ2
    - 4 mô hình đơn lẻ (dummy, svm, lightgbm, lstm) x 3 modes (fixed_053_048, quantile_q80_q30, quantile_q70_q40)
    - 1 mô hình kết hợp (ensemble) x 2 modes (quantile_q80_q30, quantile_q70_q40)

    Args:
        aligned_df: DataFrame đã căn chỉnh từ align_predictions_with_market.
        dev_oof_buffers: Bộ đệm điểm số Dev OOF từ load_dev_oof_seed_buffers.

    Returns:
        DataFrame tín hiệu chuẩn hóa sẵn sàng lưu file Parquet.
    """
    records: list[dict[str, Any]] = []

    # Danh sách các nhóm giao thức và RQ
    protocols = list(aligned_df["protocol"].unique())
    rqs = list(aligned_df["rq"].unique())

    for protocol in protocols:
        for rq in rqs:
            sub_slice = aligned_df[
                (aligned_df["protocol"] == protocol) & (aligned_df["rq"] == rq)
            ]
            if sub_slice.empty:
                continue

            models = ["dummy", "svm", "lightgbm", "lstm"]
            model_subsets: dict[str, pd.DataFrame] = {}
            for m in models:
                m_df = sub_slice[sub_slice["model_name"] == m].sort_values("signal_date")
                if not m_df.empty:
                    model_subsets[m] = m_df

            if not model_subsets:
                continue

            # -------------------------------------------------------------
            # 1. Sinh tín hiệu cho các mô hình đơn lẻ
            # -------------------------------------------------------------
            for m in models:
                if m not in model_subsets:
                    continue

                m_df = model_subsets[m]
                cal_prob = m_df["calibrated_probability"].to_numpy(dtype=float)
                rank_score = m_df["ranking_score"].to_numpy(dtype=float)
                seed_buf = dev_oof_buffers[rq][m]

                # Mode 1: fixed_053_048
                sig_fixed = generate_fixed_signals(cal_prob, entry_thresh=0.53, exit_thresh=0.48)

                # Mode 2A: quantile_q80_q30
                sig_q80 = generate_causal_quantile_signals(rank_score, seed_buf, q_entry=80.0, q_exit=30.0)

                # Mode 2B: quantile_q70_q40
                sig_q70 = generate_causal_quantile_signals(rank_score, seed_buf, q_entry=70.0, q_exit=40.0)

                mode_map = {
                    "fixed_053_048": sig_fixed,
                    "quantile_q80_q30": sig_q80,
                    "quantile_q70_q40": sig_q70,
                }

                for mode_name, sig_dict in mode_map.items():
                    for idx, (_, row) in enumerate(m_df.iterrows()):
                        records.append({
                            "protocol": protocol,
                            "rq": rq,
                            "model_name": m,
                            "signal_mode": mode_name,
                            "signal_date": row["signal_date"],
                            "execution_date": row["execution_date"],
                            "earliest_sell_date": row["earliest_sell_date"],
                            "prev_close": float(row["prev_close"]),
                            "exec_open": float(row["exec_open"]),
                            "exec_close": float(row["exec_close"]),
                            "earliest_sell_open": float(row["earliest_sell_open"]) if not pd.isna(row["earliest_sell_open"]) else np.nan,
                            "volatility_21": float(row["volatility_21"]),
                            "atr_ratio_14": float(row["atr_ratio_14"]),
                            "ranking_score": float(rank_score[idx]),
                            "calibrated_probability": float(cal_prob[idx]),
                            "entry_threshold_val": float(sig_dict["entry_threshold_val"][idx]),
                            "exit_threshold_val": float(sig_dict["exit_threshold_val"][idx]),
                            "trigger": str(sig_dict["trigger"][idx]),
                            "desired_position": int(sig_dict["desired_position"][idx]),
                            "is_ceiling_locked": bool(row["is_ceiling_locked"]),
                            "is_floor_locked": bool(row["is_floor_locked"]),
                            "forward_log_return_1": float(row["forward_log_return_1"]),
                            "forward_trade_gross_return": float(row["forward_trade_gross_return"]),
                            "forward_trade_net_return_proxy": float(row["forward_trade_net_return_proxy"]),
                            "y_true": int(row["y_true"]),
                        })

            # -------------------------------------------------------------
            # 2. Sinh tín hiệu cho mô hình kết hợp ENSEMBLE (SVM + LGBM + LSTM)
            # -------------------------------------------------------------
            if all(m in model_subsets for m in ("svm", "lightgbm", "lstm")):
                ref_df = model_subsets["svm"]
                n_sessions = len(ref_df)

                # Tính Z-Score nhân quả mở rộng cho từng mô hình thành phần
                z_test_list = []
                prob_test_list = []

                for m in ("svm", "lightgbm", "lstm"):
                    m_scores = model_subsets[m]["ranking_score"].to_numpy(dtype=float)
                    m_seed = dev_oof_buffers[rq][m]
                    z_m = compute_causal_expanding_zscore(m_scores, m_seed)
                    z_test_list.append(z_m)
                    prob_test_list.append(model_subsets[m]["calibrated_probability"].to_numpy(dtype=float))

                # Điểm Z-Score trung bình trọng số đều (1/3)
                z_ens = np.mean(z_test_list, axis=0)
                prob_ens = np.mean(prob_test_list, axis=0)
                seed_ens = dev_oof_buffers[rq]["ensemble"]

                # Sinh tín hiệu phân vị trên chuỗi Z-Score Ensemble
                sig_ens_q80 = generate_causal_quantile_signals(z_ens, seed_ens, q_entry=80.0, q_exit=30.0)
                sig_ens_q70 = generate_causal_quantile_signals(z_ens, seed_ens, q_entry=70.0, q_exit=40.0)

                ens_modes = {
                    "quantile_q80_q30": sig_ens_q80,
                    "quantile_q70_q40": sig_ens_q70,
                }

                for mode_name, sig_dict in ens_modes.items():
                    for idx, (_, row) in enumerate(ref_df.iterrows()):
                        records.append({
                            "protocol": protocol,
                            "rq": rq,
                            "model_name": "ensemble",
                            "signal_mode": mode_name,
                            "signal_date": row["signal_date"],
                            "execution_date": row["execution_date"],
                            "earliest_sell_date": row["earliest_sell_date"],
                            "prev_close": float(row["prev_close"]),
                            "exec_open": float(row["exec_open"]),
                            "exec_close": float(row["exec_close"]),
                            "earliest_sell_open": float(row["earliest_sell_open"]) if not pd.isna(row["earliest_sell_open"]) else np.nan,
                            "volatility_21": float(row["volatility_21"]),
                            "atr_ratio_14": float(row["atr_ratio_14"]),
                            "ranking_score": float(z_ens[idx]),
                            "calibrated_probability": float(prob_ens[idx]),
                            "entry_threshold_val": float(sig_dict["entry_threshold_val"][idx]),
                            "exit_threshold_val": float(sig_dict["exit_threshold_val"][idx]),
                            "trigger": str(sig_dict["trigger"][idx]),
                            "desired_position": int(sig_dict["desired_position"][idx]),
                            "is_ceiling_locked": bool(row["is_ceiling_locked"]),
                            "is_floor_locked": bool(row["is_floor_locked"]),
                            "forward_log_return_1": float(row["forward_log_return_1"]),
                            "forward_trade_gross_return": float(row["forward_trade_gross_return"]),
                            "forward_trade_net_return_proxy": float(row["forward_trade_net_return_proxy"]),
                            "y_true": int(row["y_true"]),
                        })

    signals_df = pd.DataFrame(records)
    return signals_df


def compute_signal_diagnostics(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê chi tiết hành vi tín hiệu và mức độ tham gia thị trường.

    Args:
        signals_df: DataFrame toàn bộ tín hiệu từ build_all_backtest_signals.

    Returns:
        DataFrame chứa các chỉ số chẩn đoán cho từng tổ hợp (protocol, rq, model_name, signal_mode).
    """
    rows: list[dict[str, Any]] = []

    group_cols = ["protocol", "rq", "model_name", "signal_mode"]
    for (protocol, rq, model_name, signal_mode), grp in signals_df.groupby(group_cols, sort=False):
        total_sessions = len(grp)
        entry_count = int((grp["trigger"] == "ENTRY").sum())
        exit_count = int((grp["trigger"] == "EXIT").sum())
        time_in_market_pct = float((grp["desired_position"] == 1).mean() * 100.0)

        # Đếm số lần chuyển đổi trạng thái (switches)
        positions = grp["desired_position"].to_numpy(dtype=int)
        prev_pos = 0
        switches = 0
        holding_lengths: list[int] = []
        current_hold = 0

        for p in positions:
            if p != prev_pos:
                switches += 1
                if prev_pos == 1 and current_hold > 0:
                    holding_lengths.append(current_hold)
                    current_hold = 0
            if p == 1:
                current_hold += 1
            prev_pos = p

        # Ghi nhận đợt nắm giữ cuối cùng nếu kết thúc chuỗi vẫn đang giữ vị thế
        if current_hold > 0:
            holding_lengths.append(current_hold)

        avg_holding = float(np.mean(holding_lengths)) if holding_lengths else 0.0

        # Kẹt thanh khoản trần / sàn
        ceil_entries = int(((grp["trigger"] == "ENTRY") & grp["is_ceiling_locked"]).sum())
        floor_exits = int(((grp["trigger"] == "EXIT") & grp["is_floor_locked"]).sum())

        rows.append({
            "protocol": protocol,
            "rq": rq,
            "model_name": model_name,
            "signal_mode": signal_mode,
            "total_sessions": total_sessions,
            "entry_trigger_count": entry_count,
            "exit_trigger_count": exit_count,
            "desired_time_in_market_pct": round(time_in_market_pct, 2),
            "desired_state_switches": switches,
            "avg_desired_holding_sessions": round(avg_holding, 2),
            "ceiling_locked_entries": ceil_entries,
            "floor_locked_exits": floor_exits,
        })

    return pd.DataFrame(rows)

