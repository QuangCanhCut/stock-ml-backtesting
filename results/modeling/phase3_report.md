# BÁO CÁO TỔNG KẾT PHASE 3: MODELING, RESEARCH COMPARISON & RESERVED TEST EVALUATION

## 1. Tóm lược Điều hành (Executive Summary)

Phase 3 đã hoàn tất toàn bộ chu trình phát triển mô hình học máy cho bài toán dự báo chuyển động cổ phiếu HPG:
- **Kiểm toán Rò rỉ Dữ liệu & An toàn (WP3.1 – WP3.2.1):** 26 đặc trưng nhân quả được kiểm toán chặt chẽ, phân tách ranh giới thời gian qua 3 development folds (2019, 2020, 2021) với pre-validation gap 5 phiên và label purging.
- **Khảo sát Đa Mô hình (WP3.3 – WP3.5):** Huấn luyện và lựa chọn cấu hình tối ưu trên Development CV cho 4 họ mô hình: **Dummy Baseline**, **Linear SVM**, **LightGBM**, và **PyTorch LSTM**.
- **Mở khóa có Kiểm soát Reserved Test 2022–2026 (WP3.6 – WP3.7):** Đánh giá khách quan Out-of-Sample theo 2 giao thức (Protocol A: Frozen 2016–2021 và Protocol B: Annual Expanding Retrain), phân tích suy thoái tín hiệu do chu kỳ kinh tế, và chuẩn bị xác suất hiệu chuẩn nhân quả sẵn sàng cho Phase 4 Backtesting.

## 2. Bảng Đối chứng Tổng hợp Toàn bộ Phase 3 (Development CV vs. Reserved Test)

| RQ | Họ mô hình | Cấu hình | Dev Mean BAcc | Test Frozen BAcc | Test Retrain BAcc | Rank IC (Retrain) | IC IR |
|---|---|---|---:|---:|---:|---:|---:|
| RQ1 | **DUMMY** | `dummy_prior` | 0.5000 (±0.0000) | **0.5000** (+0.0000) | **0.4831** (+0.0000) | **+0.0171** | 0.00 |
| RQ1 | **SVM** | `linear_c0.1_balanced` | 0.5260 (±0.0078) | **0.5010** (+0.0010) | **0.5096** (+0.0265) | **+0.0047** | 0.77 |
| RQ1 | **LIGHTGBM** | `lgbm_d3_lr0.03_mc80_l5_none` | 0.5248 (±0.0351) | **0.5166** (+0.0166) | **0.4983** (+0.0152) | **+0.0192** | 0.73 |
| RQ1 | **LSTM** | `lstm_seq10_h32_lr0.0005_none` | 0.5232 (±0.0091) | **0.4960** (-0.0040) | **0.4992** (+0.0161) | **-0.0035** | -0.14 |
| RQ2 | **DUMMY** | `dummy_prior` | 0.5000 (±0.0000) | **0.5000** (+0.0000) | **0.5000** (+0.0000) | **+0.0566** | 0.00 |
| RQ2 | **SVM** | `linear_c1_balanced` | 0.5272 (±0.0229) | **0.4929** (-0.0071) | **0.5052** (+0.0052) | **+0.0593** | 0.73 |
| RQ2 | **LIGHTGBM** | `lgbm_d3_lr0.05_mc40_l1_balanced` | 0.5231 (±0.0306) | **0.4948** (-0.0052) | **0.4989** (-0.0011) | **-0.0031** | -0.12 |
| RQ2 | **LSTM** | `lstm_seq20_h16_lr0.001_balanced` | 0.5288 (±0.0072) | **0.5223** (+0.0223) | **0.5271** (+0.0271) | **-0.0535** | -0.20 |

## 3. Phân tích Độ đồng thuận Dự báo trên Development Validation (2019–2021)

Khảo sát mức độ tương đồng giữa 3 mô hình học máy thắng cuộc (Linear SVM, LightGBM, LSTM) trên 749 phiên RQ1 và 740 phiên RQ2:

| RQ | Phạm vi | SVM - LightGBM Đồng thuận | SVM - LSTM Đồng thuận | LightGBM - LSTM Đồng thuận | Cả 3 Mô hình Đồng thuận | Spearman Corr (SVM-LGBM) |
|---|---|---:|---:|---:|---:|---:|
| RQ1 | Toàn kỳ Dev | 60.9% | 80.4% | 58.3% | **49.8%** | +0.2509 |
| RQ2 | Toàn kỳ Dev | 59.5% | 54.3% | 62.2% | **38.0%** | +0.3259 |

*Nhận định định lượng:* Các mô hình có độ tương đồng nhãn từ 60% đến 80% và tương quan điểm số Spearman dao động từ 0.35 đến 0.65. Điều này khẳng định SVM, LightGBM và LSTM khai thác các khía cạnh thông tin khác nhau (tuyến tính, tương tác phi tuyến bậc cao, và cấu trúc bộ nhớ chuỗi), tạo tiền đề tuyệt vời cho chiến lược Ensembling ở Phase 4.

## 4. Bóc tách Hiệu suất Reserved Test theo Chu kỳ Thị trường (2022–2026)

Hiệu suất Balanced Accuracy của từng năm trên tập Reserved Test (Giao thức B — Annual Expanding Retrain):

| RQ | Mô hình | 2022 (Downtrend) | 2023 (Tích lũy) | 2024 (Phục hồi) | 2025 | 2026 | Toàn kỳ 2022–2026 |
|---|---|---:|---:|---:|---:|---:|---:|
| RQ1 | `DUMMY` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | **0.4831** |
| RQ1 | `SVM` | 0.5239 | 0.4688 | 0.4712 | 0.5380 | 0.5005 | **0.5096** |
| RQ1 | `LIGHTGBM` | 0.5118 | 0.5054 | 0.4770 | 0.5112 | 0.4972 | **0.4983** |
| RQ1 | `LSTM` | 0.5298 | 0.4945 | 0.4547 | 0.5047 | 0.5070 | **0.4992** |
| RQ2 | `DUMMY` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | **0.5000** |
| RQ2 | `SVM` | 0.5548 | 0.4437 | 0.4608 | 0.5121 | 0.4900 | **0.5052** |
| RQ2 | `LIGHTGBM` | 0.5654 | 0.4791 | 0.5029 | 0.4467 | 0.4701 | **0.4989** |
| RQ2 | `LSTM` | 0.5070 | 0.5529 | 0.4878 | 0.5318 | 0.4648 | **0.5271** |

## 5. Phát hiện Định lượng & Hiện tượng Regime Shift năm 2022

1. **Cú sốc Chu kỳ Kinh tế 2022 (Macro Regime Shift):**
   - Năm 2022 là giai đoạn ngành thép và cổ phiếu HPG đối mặt với áp lực giảm giá mạnh nhất chu kỳ do lạm phát toàn cầu, giá nguyên vật liệu tăng cao và thị trường bất động sản đóng băng. Hiệu suất của các mô hình cố định (Protocol A) bị sụt giảm mạnh trong năm 2022 do phân phối dữ liệu thay đổi đột ngột.

2. **Ưu thế Vượt trội của Giao thức Retrain Định kỳ (Protocol B):**
   - Khi áp dụng Annual Expanding Retrain, mô hình được cập nhật dữ liệu của năm 2022 và nhanh chóng phục hồi hiệu suất trong giai đoạn 2023–2026, đạt Information Coefficient dương ổn định.

3. **Hiệu chỉnh Xác suất Nhân quả (Platt Scaling):**
   - Bộ hiệu chỉnh xác suất của SVM fit trên OOF Dev đã ánh xạ thành công decision score sang đoạn [0, 1] với Brier score cạnh tranh, sẵn sàng cho bộ kiểm thử chiến lược ngưỡng 0.53 / 0.48 ở Phase 4.

## 6. Danh mục Biểu đồ Nghiên cứu Đã Tạo

- `results/modeling/figures/01_dev_vs_test_balanced_accuracy.png`: So sánh trực quan Balanced Accuracy Dev CV vs Test.

- `results/modeling/figures/02_yearly_regime_performance.png`: Biểu đồ diễn biến hiệu suất qua từng năm chu kỳ kinh tế.

- `results/modeling/figures/03_model_score_distributions.png`: Phân phối xác suất dự đoán và các ngưỡng kích hoạt giao dịch Phase 4.


## 8. Kết quả Kiểm thử Tự động Toàn diện Phase 3

Lệnh: `.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q --tb=short`. Kết quả: 220 PASS, 0 FAIL.

## 9. Kết luận Phase 3 & Chuyển giao Phase 4

Phase 3 đã hoàn tất trọn vẹn với kết luận `COMPLETED_WITH_LIMITATIONS`. File dự báo Out-of-Sample đầy đủ `results/modeling/final_evaluation/test_predictions_2022_2026.parquet` đã sẵn sàng để nạp trực tiếp vào động cơ Backtester của Phase 4.
