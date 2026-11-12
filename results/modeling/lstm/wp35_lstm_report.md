# Báo cáo WP3.5: PyTorch LSTM Sequence Modeling, Lag Sensitivity & Final Development Models

## 1. Tổng quan WP3.5

Nghiệm thu mô hình mạng nơ-ron hồi quy **LSTM (Long Short-Term Memory)** cho bài toán RQ1 (dự báo hướng lợi suất) và RQ2 (dự báo lợi nhuận kinh tế sau chi phí giao dịch) trên ba development folds (`dev_1`, `dev_2`, `dev_3`). Đánh giá và so sánh đối chứng 1-1 giữa 4 họ mô hình: **Dummy Baseline**, **SVM Baseline (WP3.3)**, **LightGBM (WP3.4)** và **LSTM (WP3.5)** trên cùng tập mẫu Validation đã kiểm toán. Phân tích độ nhạy thời gian (Lag Sensitivity) và lưu trữ Final Development Models (2016–2021) vào `models/lstm/`.

## 2. Thiết lập thực nghiệm và Kiến trúc an toàn

- **Dataset Parquet:** `data/processed/HPG_features_labeled_research.parquet` (SHA-256: `4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e`).

- **Split Manifest:** `results/modeling/splits/split_manifest.json` (SHA-256: `f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42`).

- **Feature Schema:** 26 causal features không chứa look-ahead bias.

- **Môi trường:** Python 3.13.2, PyTorch 2.15.0+cpu, joblib 1.6.0.

- **Quy tắc Chuẩn hóa Train-Only:** Ở mỗi fold, `StandardScaler` chỉ được fit trên tập Train (`fold.train_indices`). Chuỗi Validation được tạo bằng cách trượt cửa sổ lịch sử kết thúc tại phiên `v_idx` và chuẩn hóa qua scaler của Train, đảm bảo giữ nguyên 100% mẫu Validation (249, 251, 249 mẫu RQ1 và 246, 248, 246 mẫu RQ2).

- **Bảo vệ tuyệt đối Reserved Test (2022–2026):** Tuyệt đối không có bất kỳ mẫu nào của 2022–2026 được dùng trong huấn luyện, chuẩn hóa hay early stopping.

## 3. Kết quả RQ1 (Dự báo hướng giá (Direction))

### Bảng so sánh hiệu suất Walk-Forward CV (Balanced Accuracy theo từng fold)

| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Mean BAcc | Std BAcc | vs Dummy | vs SVM | vs LightGBM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dummy_prior` (Baseline) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0260 | -0.0248 |
| `linear_c0.1_balanced` (SVM WP3.3) | — | — | — | 0.5260 | 0.0078 | +0.0260 | +0.0000 | +0.0012 |
| `lgbm_d3_lr0.03_mc80_l5_none` (LightGBM WP3.4) | — | — | — | 0.5248 | 0.0351 | +0.0248 | -0.0012 | +0.0000 |
| `lstm_seq10_h16_lr0.001_none` | 0.5024 | 0.5306 | 0.5261 | 0.5197 | 0.0124 | +0.0197 | -0.0063 | -0.0051 |
| `lstm_seq10_h16_lr0.001_balanced` | 0.4974 | 0.5182 | 0.5214 | 0.5124 | 0.0106 | +0.0124 | -0.0136 | -0.0124 |
| `lstm_seq10_h32_lr0.0005_none` **(Winner)** | 0.5120 | 0.5235 | 0.5342 | 0.5232 | 0.0091 | +0.0232 | -0.0028 | -0.0016 |
| `lstm_seq10_h32_lr0.0005_balanced` | 0.5082 | 0.5129 | 0.5410 | 0.5207 | 0.0145 | +0.0207 | -0.0053 | -0.0041 |
| `lstm_seq20_h16_lr0.001_none` | 0.4964 | 0.5149 | 0.5414 | 0.5175 | 0.0185 | +0.0175 | -0.0084 | -0.0073 |
| `lstm_seq20_h16_lr0.001_balanced` | 0.4971 | 0.5090 | 0.5287 | 0.5116 | 0.0130 | +0.0116 | -0.0144 | -0.0132 |
| `lstm_seq20_h32_lr0.0005_none` | 0.5131 | 0.5044 | 0.5410 | 0.5195 | 0.0156 | +0.0195 | -0.0065 | -0.0053 |
| `lstm_seq20_h32_lr0.0005_balanced` | 0.5161 | 0.4820 | 0.5236 | 0.5072 | 0.0181 | +0.0072 | -0.0188 | -0.0176 |

**Cấu hình LSTM thắng cuộc RQ1:** `lstm_seq10_h32_lr0.0005_none`

- Mean Balanced Accuracy: **0.5232** (Std: 0.0091).

- Chênh lệch so với Dummy baseline: **+0.0232**.

- Chênh lệch so với SVM baseline (WP3.3): **-0.0028**.

- Chênh lệch so với LightGBM (WP3.4): **-0.0016**.

- Mean ROC-AUC: 0.5281; Mean Generalization Gap (Train - Valid): 0.0505.

- Best epochs qua 3 folds: [22, 1, 9] (Trung bình làm tròn chọn `final_epochs = 11`).

## 4. Kết quả RQ2 (Nhãn kinh tế sau chi phí (Fixed-Hold Net Proxy))

### Bảng so sánh hiệu suất Walk-Forward CV (Balanced Accuracy theo từng fold)

| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Mean BAcc | Std BAcc | vs Dummy | vs SVM | vs LightGBM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dummy_prior` (Baseline) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0272 | -0.0231 |
| `linear_c1_balanced` (SVM WP3.3) | — | — | — | 0.5272 | 0.0229 | +0.0272 | +0.0000 | +0.0042 |
| `lgbm_d3_lr0.05_mc40_l1_balanced` (LightGBM WP3.4) | — | — | — | 0.5231 | 0.0306 | +0.0231 | -0.0042 | +0.0000 |
| `lstm_seq10_h16_lr0.001_none` | 0.5386 | 0.4992 | 0.5340 | 0.5240 | 0.0176 | +0.0240 | -0.0033 | +0.0009 |
| `lstm_seq10_h16_lr0.001_balanced` | 0.5308 | 0.5113 | 0.5193 | 0.5205 | 0.0080 | +0.0205 | -0.0068 | -0.0026 |
| `lstm_seq10_h32_lr0.0005_none` | 0.5063 | 0.5074 | 0.4584 | 0.4907 | 0.0229 | -0.0093 | -0.0366 | -0.0324 |
| `lstm_seq10_h32_lr0.0005_balanced` | 0.4813 | 0.4795 | 0.4603 | 0.4737 | 0.0095 | -0.0263 | -0.0535 | -0.0494 |
| `lstm_seq20_h16_lr0.001_none` | 0.5263 | 0.4984 | 0.5223 | 0.5157 | 0.0123 | +0.0157 | -0.0116 | -0.0074 |
| `lstm_seq20_h16_lr0.001_balanced` **(Winner)** | 0.5373 | 0.5197 | 0.5295 | 0.5288 | 0.0072 | +0.0288 | +0.0016 | +0.0057 |
| `lstm_seq20_h32_lr0.0005_none` | 0.4955 | 0.5026 | 0.4599 | 0.4860 | 0.0187 | -0.0140 | -0.0412 | -0.0371 |
| `lstm_seq20_h32_lr0.0005_balanced` | 0.4951 | 0.5024 | 0.4748 | 0.4907 | 0.0117 | -0.0093 | -0.0365 | -0.0323 |

**Cấu hình LSTM thắng cuộc RQ2:** `lstm_seq20_h16_lr0.001_balanced`

- Mean Balanced Accuracy: **0.5288** (Std: 0.0072).

- Chênh lệch so với Dummy baseline: **+0.0288**.

- Chênh lệch so với SVM baseline (WP3.3): **+0.0016**.

- Chênh lệch so với LightGBM (WP3.4): **+0.0057**.

- Mean ROC-AUC: 0.5080; Mean Generalization Gap (Train - Valid): 0.0174.

- Best epochs qua 3 folds: [1, 7, 4] (Trung bình làm tròn chọn `final_epochs = 4`).

## 5. Phân tích Feature Permutation Importance trên chuỗi Validation 3D

Mức độ sụt giảm Balanced Accuracy khi xáo trộn từng đặc trưng trên tập Validation ngoài mẫu của 3 development folds:

| RQ | Đặc trưng (Feature) | Mean Permutation Importance | Std Permutation |

|---|---|---:|---:|
| RQ1 | `momentum_21` | **+0.0122** | 0.0120 |
| RQ1 | `macd_ratio` | **+0.0122** | 0.0153 |
| RQ1 | `volume_zscore_20` | **+0.0117** | 0.0178 |
| RQ1 | `rsi_14` | **+0.0087** | 0.0069 |
| RQ1 | `volume_ratio_20` | **+0.0082** | 0.0104 |
| RQ1 | `sma_ratio_20` | **+0.0072** | 0.0158 |
| RQ2 | `macd_ratio` | **+0.0365** | 0.0069 |
| RQ2 | `rsi_14` | **+0.0342** | 0.0165 |
| RQ2 | `bollinger_width_20` | **+0.0240** | 0.0211 |
| RQ2 | `atr_ratio_14` | **+0.0171** | 0.0138 |
| RQ2 | `momentum_21` | **+0.0160** | 0.0028 |
| RQ2 | `ema_ratio_26` | **+0.0146** | 0.0038 |

## 6. Phân tích Temporal Lag Sensitivity (Độ nhạy theo bước trễ thời gian)

Đo lường mức độ đóng góp của từng bước thời gian trong cửa sổ chuỗi qua độ lớn Gradient Saliency trung bình:

| RQ | Bước trễ (Lag Step) | Vị trí trong cửa sổ | Độ nhạy tuyệt đối | Tỷ trọng chuẩn hóa (%)

|---|---|---:|---:|---:|
| RQ1 | `t -9` | 0 | 0.001039 | **1.96%** |
| RQ1 | `t -8` | 1 | 0.001289 | **2.47%** |
| RQ1 | `t -7` | 2 | 0.001590 | **3.11%** |
| RQ1 | `t -6` | 3 | 0.001983 | **3.97%** |
| RQ1 | `t -5` | 4 | 0.002510 | **5.15%** |
| RQ1 | `t -4` | 5 | 0.003221 | **6.78%** |
| RQ1 | `t -3` | 6 | 0.004227 | **9.15%** |
| RQ1 | `t -2` | 7 | 0.005718 | **12.72%** |
| RQ1 | `t -1` | 8 | 0.008201 | **18.79%** |
| RQ1 | `Phiên hiện tại (t)` | 9 | 0.014868 | **35.91%** |
| RQ2 | `t -19` | 0 | 0.000028 | **0.07%** |
| RQ2 | `t -18` | 1 | 0.000035 | **0.09%** |
| RQ2 | `t -17` | 2 | 0.000043 | **0.11%** |
| RQ2 | `t -16` | 3 | 0.000055 | **0.14%** |
| RQ2 | `t -15` | 4 | 0.000072 | **0.19%** |
| RQ2 | `t -14` | 5 | 0.000093 | **0.24%** |
| RQ2 | `t -13` | 6 | 0.000121 | **0.32%** |
| RQ2 | `t -12` | 7 | 0.000158 | **0.42%** |
| RQ2 | `t -11` | 8 | 0.000207 | **0.55%** |
| RQ2 | `t -10` | 9 | 0.000275 | **0.74%** |
| RQ2 | `t -9` | 10 | 0.000374 | **1.01%** |
| RQ2 | `t -8` | 11 | 0.000512 | **1.39%** |
| RQ2 | `t -7` | 12 | 0.000705 | **1.92%** |
| RQ2 | `t -6` | 13 | 0.000989 | **2.70%** |
| RQ2 | `t -5` | 14 | 0.001396 | **3.84%** |
| RQ2 | `t -4` | 15 | 0.001994 | **5.50%** |
| RQ2 | `t -3` | 16 | 0.002905 | **8.04%** |
| RQ2 | `t -2` | 17 | 0.004310 | **11.99%** |
| RQ2 | `t -1` | 18 | 0.006913 | **19.32%** |
| RQ2 | `Phiên hiện tại (t)` | 19 | 0.014815 | **41.42%** |

## 7. Lưu trữ Final Development Models (2016–2021) vào `models/lstm/`

Mô hình được fit trên toàn bộ 1.297 mẫu phát triển hợp lệ trước ranh giới năm 2022:

| RQ | File Model Bundle (.pt) | Metadata JSON | Train Rows | Train Sequences | Final Epochs | Parameters |

|---|---|---|---:|---:|---:|---:|
| RQ1 | `rq1_lstm_final_dev_2016_2021.pt` | `lstm_final_dev_metadata.json` | 1297 | 1288 | 11 | 7713 |
| RQ2 | `rq2_lstm_final_dev_2016_2021.pt` | `lstm_final_dev_metadata.json` | 1297 | 1278 | 4 | 2833 |

## 8. Cảnh báo Phương pháp luận và Giới hạn

- **Độ nhạy mẫu nhỏ của Mạng Nơ-ron:** Mặc dù LSTM đã áp dụng kiến trúc siêu gọn (1 layer, hidden_size 16/32, dropout 0.2–0.3, weight decay 1e-3, early stopping), tập dữ liệu bậc ngày với tỷ lệ SNR thấp vẫn là thách thức lớn. Hiệu suất LSTM cần được xem xét thận trọng và đối chiếu trực tiếp với các mô hình tuyến tính (SVM) và cây quyết định (LightGBM).

- **Validation Selection Bias:** Kết quả validation phản ánh thiên lệch lựa chọn siêu tham số và checkpoint dừng sớm.

- **Bảo vệ Reserved Test:** Vùng Test 2022–2026 hoàn toàn chưa được mở khóa, đảm bảo tính khách quan cho WP3.6 và Phase 4.

## 9. Kết quả Kiểm thử Tự động

Lệnh: `.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q --tb=short`. Kết quả: 214 PASS, 0 FAIL.

## 10. Kết luận WP3.5

`COMPLETED_WITH_LIMITATIONS`: Quá trình huấn luyện, đánh giá LSTM sequence modeling, trích xuất Lag Sensitivity và lưu trữ Final Development Model (2016–2021) hoàn thành đạt chuẩn.
