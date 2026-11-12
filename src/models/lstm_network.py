"""Kiến trúc mạng PyTorch LSTMClassifier và bộ tạo chuỗi (Sequence Builder) chống rò rỉ dữ liệu.

Module này hiện thực hóa các thành phần cốt lõi của WP3.5:
- LSTMClassifier: Mạng nơ-ron hồi quy LSTM gọn nhẹ, chống overfitting cho dữ liệu tài chính nhỏ.
- build_fold_sequences: Xây dựng chuỗi 3D (N, seq_len, 26) cho tập Train và Validation của từng fold,
  đảm bảo chuẩn hóa Train-Only (StandardScaler) và giữ nguyên 100% số mẫu Validation để so sánh
  đối chứng 1-1 với SVM (WP3.3) và LightGBM (WP3.4).
- build_full_development_sequences: Xây dựng chuỗi trên toàn bộ 1.297 mẫu phát triển 2016–2021
  để refit bản model cuối trước ranh giới năm 2022.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from src.models.data import ModelingView
from src.models.temporal_split import TemporalSplit


class LSTMClassifier(nn.Module):
    """Mạng LSTM gọn nhẹ cho phân loại chuỗi thời gian tài chính (26 đặc trưng HPG).

    Kiến trúc mạng:
    1. Lớp nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    2. Lấy vector trạng thái ẩn của bước thời gian cuối cùng: last_hidden = lstm_out[:, -1, :]
    3. Lớp nn.Dropout(p=dropout) chống học vẹt (overfitting)
    4. Lớp tuyến tính nn.Linear(hidden_size, 1) xuất ra logits 1D cho hàm mất mát BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_size: int = 26,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = float(dropout)

        # Trong PyTorch, nếu num_layers=1 thì nn.LSTM cảnh báo nếu dropout > 0.
        # Do đó đặt dropout=0 trong nn.LSTM và áp dụng nn.Dropout riêng biệt ở đầu ra.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass).

        Args:
            x: Tensor đầu vào 3D shape (batch_size, seq_len, input_size).

        Returns:
            Tensor logits 1D shape (batch_size,).
        """
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # Bước thời gian cuối cùng t: (batch_size, hidden_size)
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped).squeeze(-1)  # Shape: (batch_size,)
        return logits

    def count_trainable_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện của mạng."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_fold_sequences(
    view: ModelingView,
    fold: TemporalSplit,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Xây dựng ma trận chuỗi 3D cho Train và Validation của một development fold.

    Quy tắc an toàn chống rò rỉ thông tin (No Data Leakage):
    1. StandardScaler chỉ được fit trên tập Train (view.X.iloc[list(fold.train_indices)]).
    2. Chuỗi Train: Được cắt từ các mẫu nằm hoàn toàn trong fold.train_indices (gồm N_train - seq_len + 1 chuỗi).
    3. Chuỗi Validation: Đối với mỗi mẫu v_idx trong fold.validation_indices, lấy cửa sổ ngữ cảnh lịch sử
       độ dài seq_len kết thúc tại v_idx: [v_idx - seq_len + 1, ..., v_idx] từ view.X và chuẩn hóa bằng
       StandardScaler của tập Train. Điều này đảm bảo 100% mẫu Validation được giữ nguyên (không làm mất
       seq_len - 1 mẫu đầu năm của Validation) và hoàn toàn mang tính nhân quả (causal, không nhìn vào tương lai).

    Args:
        view: ModelingView chuẩn hóa của RQ tương ứng.
        fold: Đối tượng TemporalSplit hợp lệ từ get_development_splits().
        seq_len: Độ dài cửa sổ trượt (số phiên lịch sử L).

    Returns:
        Tuple gồm (X_train_seq, y_train_seq, X_valid_seq, y_valid_seq, fitted_scaler).

    Raises:
        ValueError: Khi seq_len không hợp lệ hoặc dữ liệu không đủ phiên để tạo chuỗi.
    """
    if seq_len < 1:
        raise ValueError(f"Độ dài chuỗi seq_len phải >= 1, nhận được {seq_len}")

    train_indices = list(fold.train_indices)
    val_indices = list(fold.validation_indices)

    if len(train_indices) < seq_len:
        raise ValueError(
            f"Tập Train có {len(train_indices)} mẫu, không đủ để tạo chuỗi độ dài {seq_len}"
        )

    # 1. Fit StandardScaler chỉ trên các quan sát thuộc tập Train của fold này
    scaler = StandardScaler()
    X_train_raw = view.X.iloc[train_indices].to_numpy(dtype=np.float32)
    scaler.fit(X_train_raw)

    # Chuẩn hóa toàn bộ ma trận view.X bằng scaler đã fit chỉ trên Train
    X_scaled_all = scaler.transform(view.X.to_numpy(dtype=np.float32)).astype(np.float32)
    y_all = view.y.to_numpy(dtype=np.float32)

    # 2. Tạo chuỗi 3D cho tập Train: (N_train - seq_len + 1, seq_len, n_features)
    n_train_seq = len(train_indices) - seq_len + 1
    n_features = view.X.shape[1]

    X_train_seq = np.empty((n_train_seq, seq_len, n_features), dtype=np.float32)
    y_train_seq = np.empty(n_train_seq, dtype=np.float32)

    for i in range(n_train_seq):
        start_idx = train_indices[i]
        end_idx = train_indices[i + seq_len - 1]
        X_train_seq[i] = X_scaled_all[start_idx : end_idx + 1]
        y_train_seq[i] = y_all[end_idx]

    # 3. Tạo chuỗi 3D cho tập Validation: đúng bằng len(val_indices) mẫu
    n_val_seq = len(val_indices)
    X_valid_seq = np.empty((n_val_seq, seq_len, n_features), dtype=np.float32)
    y_valid_seq = np.empty(n_val_seq, dtype=np.float32)

    for k, v_idx in enumerate(val_indices):
        start_idx = v_idx - seq_len + 1
        if start_idx < 0:
            raise ValueError(
                f"Chỉ số validation {v_idx} không có đủ {seq_len} phiên lịch sử phía trước (start={start_idx})"
            )
        X_valid_seq[k] = X_scaled_all[start_idx : v_idx + 1]
        y_valid_seq[k] = y_all[v_idx]

    return X_train_seq, y_train_seq, X_valid_seq, y_valid_seq, scaler


def build_full_development_sequences(
    view: ModelingView,
    train_indices: tuple[int, ...],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Xây dựng chuỗi 3D trên toàn bộ giai đoạn phát triển hợp lệ (2016–2021) để refit bản model cuối.

    Tuyệt đối không truy cập hay chuẩn hóa trên bất kỳ quan sát nào ngoài train_indices (bảo vệ Reserved Test).

    Args:
        view: ModelingView của RQ.
        train_indices: 1.297 chỉ số hợp lệ giai đoạn 2016–2021 trước ranh giới năm 2022.
        seq_len: Độ dài cửa sổ trượt.

    Returns:
        Tuple gồm (X_train_seq, y_train_seq, fitted_scaler).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len phải >= 1, nhận được {seq_len}")

    indices = list(train_indices)
    if len(indices) < seq_len:
        raise ValueError(
            f"Dữ liệu huấn luyện có {len(indices)} mẫu, không đủ tạo chuỗi độ dài {seq_len}"
        )

    # Fit scaler chỉ trên các mẫu thuộc train_indices
    scaler = StandardScaler()
    X_train_raw = view.X.iloc[indices].to_numpy(dtype=np.float32)
    scaler.fit(X_train_raw)

    # Chuẩn hóa tập huấn luyện phát triển
    X_scaled = scaler.transform(X_train_raw).astype(np.float32)
    y_train = view.y.iloc[indices].to_numpy(dtype=np.float32)

    n_seq = len(indices) - seq_len + 1
    n_features = view.X.shape[1]

    X_train_seq = np.empty((n_seq, seq_len, n_features), dtype=np.float32)
    y_train_seq = np.empty(n_seq, dtype=np.float32)

    for i in range(n_seq):
        X_train_seq[i] = X_scaled[i : i + seq_len]
        y_train_seq[i] = y_train[i + seq_len - 1]

    return X_train_seq, y_train_seq, scaler

