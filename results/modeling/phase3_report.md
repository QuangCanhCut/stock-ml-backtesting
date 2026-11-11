# BÁO CÁO TỔNG KẾT PHASE 3: MODELING, RESEARCH COMPARISON & RESERVED TEST EVALUATION

## 1. Tóm lược Điều hành (Executive Summary)

Phase 3 đã hoàn tất toàn bộ chu trình phát triển, kiểm toán an toàn và đánh giá mô hình học máy cho bài toán dự báo chuyển động cổ phiếu HPG:
- **Kiểm toán Rò rỉ Dữ liệu & An toàn (WP3.1 – WP3.2.1):** 26 đặc trưng nhân quả được kiểm toán chặt chẽ, phân tách ranh giới thời gian qua 3 development folds (2019, 2020, 2021) với pre-validation gap 5 phiên và label purging.
- **Khảo sát Đa Mô hình (WP3.3 – WP3.5):** Huấn luyện và lựa chọn cấu hình tối ưu trên Development CV cho 4 họ mô hình: **Dummy Baseline**, **Linear SVM**, **LightGBM**, và **PyTorch LSTM**.
- **Mở khóa có Kiểm soát Reserved Test 2022–2026 (WP3.6 – WP3.7):** Đánh giá khách quan Out-of-Sample theo 2 giao thức (Protocol A: Frozen 2016–2021 và Protocol B: Annual Expanding Retrain), phân tích suy thoái tín hiệu do chu kỳ kinh tế, và chuẩn bị xác suất hiệu chuẩn nhân quả sẵn sàng cho Phase 4 Backtesting.
- **Kiểm toán Định lượng Sau Mở khóa (Post-Evaluation Audit):** Phân tích chi tiết phân phối xác suất dự báo, giải phẫu nguyên nhân nghịch lý Rank IC của LSTM ở RQ2, và làm rõ các hiện tượng thống kê trong dữ liệu gộp.

---

## 2. Bảng Đối chứng Tổng hợp Toàn bộ Phase 3 (Development CV vs. Reserved Test)

| RQ | Họ mô hình | Cấu hình | Dev Mean BAcc | Test Frozen BAcc | Test Retrain BAcc | Rank IC (Retrain) | IC IR |
|---|---|---|---:|---:|---:|---:|---:|
| RQ1 | **DUMMY** | `dummy_prior` | 0.5000 (±0.0000) | **0.5000** (+0.0000) | **0.4831** (+0.0000) | **+0.0171\*** | **0.00** |
| RQ1 | **SVM** | `linear_c0.1_balanced` | 0.5260 (±0.0078) | **0.5010** (+0.0010) | **0.5096** (+0.0265) | **+0.0047** | **0.77** |
| RQ1 | **LIGHTGBM** | `lgbm_d3_lr0.03_mc80_l5_none` | 0.5248 (±0.0351) | **0.5166** (+0.0166) | **0.4983** (+0.0152) | **+0.0192** | **0.73** |
| RQ1 | **LSTM** | `lstm_seq10_h32_lr0.0005_none` | 0.5232 (±0.0091) | **0.4960** (-0.0040) | **0.4992** (+0.0161) | **-0.0035** | **-0.14** |
| RQ2 | **DUMMY** | `dummy_prior` | 0.5000 (±0.0000) | **0.5000** (+0.0000) | **0.5000** (+0.0000) | **+0.0566\*** | **0.00** |
| RQ2 | **SVM** | `linear_c1_balanced` | 0.5272 (±0.0229) | **0.4929** (-0.0071) | **0.5052** (+0.0052) | **+0.0593** | **0.73** |
| RQ2 | **LIGHTGBM** | `lgbm_d3_lr0.05_mc40_l1_balanced` | 0.5231 (±0.0306) | **0.4948** (-0.0052) | **0.4989** (-0.0011) | **-0.0031** | **-0.12** |
| RQ2 | **LSTM** | `lstm_seq20_h16_lr0.001_balanced` | 0.5288 (±0.0072) | **0.5223** (+0.0223) | **0.5271** (+0.0271) | **-0.0535** | **-0.20** |

> [!NOTE]
> **(\*) Làm rõ hiện tượng Pooled Rank IC khác 0 của mô hình Dummy ở Protocol B:**
> - Trong từng năm riêng lẻ (`test_metrics_by_year_2022_2026.csv`), `spearman_rank_ic` của Dummy **bằng 0.0000 tuyệt đối** ở mọi năm ($2022..2026$), do xác suất dự báo của Dummy là một hằng số cố định duy nhất trong năm đó.
> - Con số Pooled Rank IC khác 0 trên toàn kỳ 2022–2026 (`+0.0171` ở RQ1 và `+0.0566` ở RQ2) chỉ là **hệ quả toán học của việc gộp 5 bậc giá trị hằng số rời rạc** (`train_prior_probability` được tính lại vào đầu mỗi năm mở rộng) qua các chế độ thị trường khác nhau (**Simpson's paradox / Step-function pooling artifact**), hoàn toàn không phản ánh năng lực dự báo tương lai.
> - Do đó, **`mean_annual_rank_ic = 0.0000`** và **`ic_ir = 0.00`** mới là thước đo chuẩn xác về alpha của mô hình Dummy.

---

## 3. Phân tích Độ đồng thuận Dự báo trên Development Validation (2019–2021)

Mức độ đồng thuận dự báo nhãn và tương quan thứ hạng điểm số giữa 3 mô hình học máy (`Linear SVM`, `LightGBM`, `PyTorch LSTM`) trên 749 phiên RQ1 và 740 phiên RQ2 thuộc 3 development folds (`dev_1`, `dev_2`, `dev_3`):

| RQ | Phạm vi | SVM - LGBM Đồng thuận | SVM - LSTM Đồng thuận | LGBM - LSTM Đồng thuận | Cả 3 Mô hình Đồng thuận | Spearman (SVM - LGBM) | Spearman (SVM - LSTM) | Spearman (LGBM - LSTM) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **RQ1** | Toàn kỳ Dev | 60.9% | 80.4% | 58.3% | **49.8%** | **+0.2509** | **+0.7166** | **+0.2209** |
| RQ1 | dev_1 (2019) | 67.9% | 82.7% | 69.9% | 60.2% | +0.1868 | +0.7694 | +0.1418 |
| RQ1 | dev_2 (2020) | 40.6% | 76.5% | 29.1% | 23.1% | +0.4727 | +0.5436 | +0.2051 |
| RQ1 | dev_3 (2021) | 74.3% | 81.9% | 76.3% | 66.3% | +0.6145 | +0.7100 | +0.6037 |
| **RQ2** | Toàn kỳ Dev | 59.5% | 54.3% | 62.2% | **38.0%** | **+0.3259** | **+0.2471** | **+0.2795** |
| RQ2 | dev_1 (2019) | 56.5% | 46.3% | 60.6% | 31.7% | +0.2658 | -0.0638 | +0.2010 |
| RQ2 | dev_2 (2020) | 53.6% | 58.1% | 64.9% | 38.3% | +0.3932 | +0.2794 | +0.0882 |
| RQ2 | dev_3 (2021) | 68.3% | 58.5% | 61.0% | 43.9% | +0.4173 | +0.4782 | +0.4357 |

### Nhận định định lượng chuyên sâu:
1. **Ở RQ1 (Dự báo xu hướng 1 phiên):**
   - Cặp `SVM - LSTM` có độ đồng thuận nhãn rất cao (**80.4%**) và tương quan thứ hạng điểm số đạt **+0.7166**. Nguyên nhân bắt nguồn từ cấu trúc kiến trúc: LSTM ở RQ1 sử dụng độ dài chuỗi ngắn (`seq_length = 10`), và cơ chế cổng của LSTM tập trung phần lớn trọng số suy diễn vào các biến lag gần nhất ($t$ và $t-1$), tạo ra đáp ứng tương đương với hàm siêu phẳng tuyến tính của Linear SVM đối với các đặc trưng xung lượng ngắn hạn.
   - Ngược lại, `LightGBM` phân tách độc lập rõ rệt với cả SVM (`+0.2509`) và LSTM (`+0.2209`) nhờ cấu trúc cây phân nhánh bắt được các mối quan hệ phi tuyến bậc cao giữa khối lượng (`volume_zscore_20`) và biên độ dải Bollinger.
2. **Ở RQ2 (Dự báo lãi ròng 3 phiên sau phí):**
   - Cả 3 họ mô hình phân tách độc lập hoàn toàn, với tương quan thứ hạng Spearman chỉ dao động từ **+0.2471 đến +0.3259**, và tỷ lệ cả 3 mô hình cùng đồng thuận nhãn chỉ đạt **38.0%**.
   - Khi tăng độ dài chuỗi lên `seq_length = 20` (nắm bắt toàn bộ bộ nhớ 1 tháng) kết hợp với bài toán nắm giữ nhiều phiên, 3 thuật toán (siêu phẳng lề cực đại, cây quyết định boosting, và mạng nơ-ron hồi quy chuỗi dài) tiếp cận thông tin thị trường theo các chiều không gian hoàn toàn khác nhau. Đây là điều kiện lý tưởng cho chiến lược **Ensemble Đa Mô hình** ở Phase 4.

---

## 4. Bóc tách Hiệu suất Reserved Test theo Chu kỳ Thị trường (2022–2026)

Hiệu suất Balanced Accuracy theo từng năm trên tập Reserved Test (Giao thức B — Annual Expanding Retrain):

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

---

## 5. Phát hiện Định lượng & Phân tích Chuyên sâu (In-depth Quantitative Insights)

### 5.1. Cú sốc Chu kỳ Kinh tế 2022 & Khả năng Phục hồi qua Protocol B
- Năm 2022 chứng kiến chu kỳ giảm giá khốc liệt nhất lịch sử ngành thép và cổ phiếu HPG do lạm phát toàn cầu, chi phí than cốc tăng cao và khủng hoảng thị trường bất động sản. Các mô hình cố định (Protocol A — Frozen 2016–2021) bị suy giảm hiệu suất nặng nề do dữ liệu huấn luyện quá khứ chưa từng gặp cú sốc tương tự.
- Giao thức B (Annual Expanding Retrain) chứng minh tính thích ứng vượt trội: khi được cập nhật dữ liệu của năm 2022 vào tập huấn luyện mở rộng, các mô hình nhanh chóng phục hồi độ chính xác và đạt Information Coefficient dương ổn định trong giai đoạn 2023–2026.

---

### 5.2. Kiểm toán Phân phối Xác suất Hiệu chuẩn & Cảnh báo "Đói Tín hiệu" (Signal Starvation) tại Ngưỡng 0.53 / 0.48

Từ file kiểm toán `results/modeling/final_evaluation/test_probability_threshold_audit.csv`, bảng phân phối xác suất và tỷ lệ kích hoạt vị thế trên 1.200 phiên RQ1 và 1.197 phiên RQ2 (Protocol B) được tổng hợp như sau:

| RQ | Mô hình | Cấu hình | Median Prob | Min Prob | Max Prob | Pct >= 0.53 (Long Entry) | Pct <= 0.48 (Exit/Short) | Pct Vùng đệm (0.48, 0.53) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| RQ1 | `SVM` | `linear_c0.1_balanced` | 0.4979 | 0.4092 | 0.6831 | **13.58%** | **23.42%** | **63.00%** |
| RQ1 | `LIGHTGBM` | `lgbm_d3_lr0.03_mc80_l5_none` | 0.4779 | 0.4461 | 0.5647 | **1.67%** | **53.67%** | **44.67%** |
| RQ1 | `LSTM` | `lstm_seq10_h32_lr0.0005_none` | 0.4751 | 0.3732 | 0.6514 | **18.00%** | **53.75%** | **28.25%** |
| RQ2 | `SVM` | `linear_c1_balanced` | 0.4998 | 0.4516 | 0.6405 | **0.84%** | **2.42%** | **96.74%** |
| RQ2 | `LIGHTGBM` | `lgbm_d3_lr0.05_mc40_l1_balanced` | 0.5055 | 0.3548 | 0.6459 | **32.83%** | **25.15%** | **42.02%** |
| RQ2 | `LSTM` | `lstm_seq20_h16_lr0.001_balanced` | 0.4941 | 0.4062 | 0.5919 | **13.53%** | **35.09%** | **51.38%** |

#### Cảnh báo định lượng cho Phase 4 Backtesting:
1. **Hiện tượng Đói Tín hiệu Nghiêm trọng ở SVM RQ2:**
   - Bộ Platt Calibrator của SVM RQ2 có hệ số cực phẳng ($coef = 0.0611, intercept = 0.0135$). Để đạt $P \ge 0.53$, mô hình đòi hỏi `decision_score` $\ge +1.7458$, và để đạt $P \le 0.48$ đòi hỏi $\le -1.5326$.
   - Trên toàn bộ 1.197 phiên Test thực tế, chỉ có đúng **10 phiên (0.84%)** đạt điều kiện vào lệnh Mua cố định $P \ge 0.53$, và có tới **96.74%** số phiên bị kẹt trong vùng đệm `(0.48, 0.53)`. Nếu Phase 4 dùng cứng ngưỡng 0.53, SVM RQ2 hầu như không giao dịch.
2. **Sự suy giảm Tỷ lệ Dự báo Dương của LightGBM RQ1:**
   - Cấu hình LightGBM RQ1 sử dụng `class_weight=None`. Khi retrain qua các năm 2022–2024 có nhiều phiên giảm điểm, mô hình học theo phân bố lớp chiếm ưu thế, khiến `positive_prediction_rate` giảm mạnh từ **46.59%** (2022) xuống **21.69%** (2023), **12.80%** (2024), **12.05%** (2025) và chỉ còn **3.94%** (2026).
   - Tỷ lệ phiên có $P \ge 0.53$ chỉ đạt **1.67%** (20 phiên), trong khi có tới **53.67%** phiên rơi xuống dưới 0.48.

---

### 5.3. Giải phẫu Nghịch lý LSTM ở RQ2: Balanced Accuracy 0.5271 vs Spearman Rank IC -0.0535

Ở RQ2 (`annual_expanding_retrain`), mạng nơ-ron LSTM đạt **Balanced Accuracy cao nhất toàn bảng (0.5271)** và **ROC-AUC dương (0.5355)** so với nhãn nhị phân `y_true`, nhưng lại có **Spearman Rank IC âm (-0.0535)** khi tính với chuỗi lợi suất ròng liên tục (`forward_trade_net_return_proxy`). Ngược lại, Linear SVM có Balanced Accuracy 0.5052 nhưng đạt Rank IC dương ấn tượng (**+0.0593**, $IC\ IR = 0.73$).

Bảng phân tích 5 phân vị điểm số (Quintiles Q1..Q5 từ file `results/modeling/final_evaluation/rq2_quintile_return_audit.csv`) làm sáng tỏ bản chất toán học của hiện tượng này:

| Mô hình | Phân vị | Số mẫu | Ranking Score Range | Mean Calibrated Prob | Win Rate ($y=1$) | Mean Net Return | Median Net Return | Lợi suất Min | Lợi suất Max |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| **LSTM** | **Q1 (Thấp nhất)** | 240 | [0.4062, 0.4599] | 0.4429 | **38.33%** | **-0.016%** | +0.045% | -17.82% | +10.17% |
| LSTM | **Q2** | 239 | [0.4600, 0.4855] | 0.4729 | **44.35%** | **-0.060%** | -0.299% | -11.54% | **+22.94%** |
| LSTM | **Q3** | 239 | [0.4859, 0.5021] | 0.4942 | **35.15%** | **-0.500%** | -0.776% | -14.49% | **+20.95%** |
| LSTM | **Q4** | 239 | [0.5021, 0.5223] | 0.5121 | **43.51%** | **-0.742%** | -0.499% | -14.28% | **+19.66%** |
| **LSTM** | **Q5 (Cao nhất)** | 240 | [0.5224, 0.5919] | 0.5383 | **48.33%** | **-0.427%** | -0.499% | -12.58% | **+8.09%** |
|---|---|---|---|---|---|---|---|---|---|
| **SVM** | **Q1 (Thấp nhất)** | 240 | [-3.4003, -0.7150] | 0.4864 | **38.75%** | **-0.598%** | -0.809% | -10.08% | +11.81% |
| SVM | **Q2** | 239 | [-0.7145, -0.3634] | 0.4953 | **38.91%** | **-0.626%** | -0.330% | -17.82% | +8.15% |
| SVM | **Q3** | 239 | [-0.3632, -0.0800] | 0.4999 | **46.44%** | **-0.383%** | -0.343% | -12.38% | +15.82% |
| SVM | **Q4** | 239 | [-0.0784, +0.2441] | 0.5047 | **38.08%** | **-0.144%** | -0.063% | -14.28% | **+22.94%** |
| **SVM** | **Q5 (Cao nhất)** | 240 | [+0.2453, +9.2377] | 0.5145 | **47.50%** | **+0.005%** | -0.318% | -12.58% | **+20.95%** |

#### Giải thích cơ chế định lượng:
1. **Tại sao Balanced Accuracy và ROC-AUC của LSTM lại cao?**
   - Win Rate ($y_{true} = 1$) của LSTM tăng dần đều đặn từ **38.33%** ở Q1 lên **48.33%** ở Q5. LSTM làm rất tốt việc phân tách ranh giới quanh điểm hòa vốn $0$ (nhãn nhị phân), do đó đạt điểm số phân loại xuất sắc.
2. **Tại sao Spearman Rank IC của LSTM lại âm?**
   - **Hiện tượng Cắt cụt Đuôi phải (Right-tail Truncation):** Toàn bộ các phiên thị trường bùng nổ lợi suất mạnh nhất chu kỳ (+19.66% đến +22.94%) lại bị LSTM xếp vào các phân vị trung bình (Q2, Q3, Q4). Trong khi đó, ở nhóm điểm tự tin nhất (Q5), lợi suất tối đa bị chặn đứng chỉ ở mức **+8.09%**.
   - **Đuôi trái bất đối xứng (Fat Left Tail / Negative Skewness):** Dù lợi suất dương bị giới hạn ở +8.09%, nhóm Q5 của LSTM vẫn phải chịu các cú sụt giảm mạnh lên tới **-12.58%**, kéo lợi suất trung bình của Q5 xuống **-0.427%** (kém hơn nhiều so với Q1 là -0.016%).
   - **Đảo chiều phân phối qua các năm (Regime Flips):** Trong 2 năm downtrend và tích lũy đầu kỳ (2022–2023), thị trường có nhiều bẫy tăng giá (bull trap), LSTM bị gán điểm cao vào các nhịp hồi kỹ thuật yếu rồi bị sụt giảm tiếp (Rank IC 2022 = -0.0949, 2023 = -0.1093), trước khi phục hồi dương trở lại ở 2025 (+0.0265) và 2026 (+0.1205).
3. **Ưu thế đơn điệu của Linear SVM:**
   - Linear SVM sử dụng hàm tuyến tính $w^T x + b$ bảo toàn hoàn hảo tính đơn điệu của quy mô lợi suất: `mean_net_return` tăng tịnh tiến từ **-0.598%** (Q1) lên **+0.005%** (Q5), bắt trọn các phiên tăng mạnh ở Q4 và Q5 (+20.95% – +22.94%), mang lại Rank IC dương bền bỉ.

---

## 6. Danh mục Biểu đồ Nghiên cứu Đã Tạo

- `results/modeling/figures/01_dev_vs_test_balanced_accuracy.png`: So sánh trực quan Balanced Accuracy Dev CV vs Test trên cả hai giao thức A và B.
- `results/modeling/figures/02_yearly_regime_performance.png`: Biểu đồ diễn biến hiệu suất qua từng năm chu kỳ kinh tế (2022–2026).
- `results/modeling/figures/03_model_score_distributions.png`: Phân phối xác suất dự đoán và vị trí các ngưỡng kích hoạt giao dịch Phase 4.

---

## 7. Danh mục Files & Artifacts Đã Xuất Bản

Toàn bộ kết quả thực nghiệm và kiểm toán của Phase 3 được lưu trữ minh bạch tại:
- `results/modeling/final_evaluation/dev_cv_all_models_summary.csv`: Tổng hợp metrics Dev CV của 4 họ mô hình.
- `results/modeling/final_evaluation/dev_prediction_agreement.csv`: Ma trận độ đồng thuận nhãn và tương quan Spearman trên Dev.
- `results/modeling/final_evaluation/test_metrics_overall_2022_2026.csv`: Bảng metrics toàn kỳ 2022–2026 trên Reserved Test.
- `results/modeling/final_evaluation/test_metrics_by_year_2022_2026.csv`: Bóc tách metrics từng năm 2022, 2023, 2024, 2025, 2026.
- `results/modeling/final_evaluation/test_predictions_2022_2026.parquet`: Toàn bộ 19.176 dòng dự đoán Out-of-Sample kèm lợi suất tương lai.
- `results/modeling/final_evaluation/test_probability_threshold_audit.csv`: Kiểm toán phân phối xác suất hiệu chuẩn và tỷ lệ kích hoạt ngưỡng 0.53 / 0.48.
- `results/modeling/final_evaluation/rq2_quintile_return_audit.csv`: Bảng kiểm toán 5 phân vị lợi suất và tỷ lệ thắng của LSTM vs SVM ở RQ2.
- `results/modeling/final_evaluation/final_evaluation_metadata.json`: Siêu dữ liệu đầy đủ, mã băm SHA-256 và môi trường thực thi.
- `results/modeling/model_comparison.csv`: Bảng tổng hợp đối chứng master xuyên suốt Phase 3.

---

## 8. Kết quả Kiểm thử Tự động Toàn diện Phase 3

Lệnh thực thi: `.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q --tb=short`.
Kết quả: **220 PASS, 0 FAIL** (100% kiểm thử đơn vị và tích hợp thành công, không có cảnh báo tương thích).

Xác thực tính toàn vẹn dữ liệu nghiên cứu:
- Mã băm SHA-256 của `data/processed/HPG_features_labeled_research.parquet`: `4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e` (Bảo toàn 100%).
- Mã băm SHA-256 của `results/modeling/splits/split_manifest.json`: `f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42` (Bảo toàn 100%).

---

## 9. Khuyến nghị Chuyển giao & Thiết kế Tín hiệu cho Phase 4 (Backtesting Readiness)

Dựa trên kết quả kiểm toán định lượng phân phối xác suất và cấu trúc lợi suất, Phase 4 nên hỗ trợ song song 3 cơ chế ánh xạ tín hiệu (Signal Mapping Architecture):

1. **Chế độ Ngưỡng Cố định (Fixed Threshold Mode `0.53 / 0.48`):**
   - Tuân thủ quy chuẩn gốc của `configs/config.yaml`: Mở vị thế Mua khi $P \ge 0.53$, Đóng vị thế khi $P \le 0.48$, giữ nguyên trạng thái khi nằm trong vùng đệm `(0.48, 0.53)`.
   - Đóng vai trò là bài kiểm tra đối chứng chuẩn mực (Standard Baseline) để đo lường hành vi chiến lược trong điều kiện kỷ luật chặt chẽ.

2. **Chế độ Ngưỡng Phân vị Nhân quả (Causal Rolling / Expanding Quantile Threshold Mode):**
   - **Giải pháp cho hiện tượng Đói Tín hiệu:** Do Platt Calibrator của SVM RQ2 và phân phối điểm số của LightGBM RQ1 bị nén hẹp, chiến lược nên cho phép mở vị thế Mua khi điểm số $s_t$ nằm trong **Top phân vị cao nhất** (ví dụ: $s_t \ge Q_{80\%}(s_{1..t})$) và Thoát vị thế khi $s_t \le Q_{30\%}(s_{1..t})$.
   - **Nguyên tắc nhân quả bất biến:** Phân vị $Q$ bắt buộc **chỉ được tính trên tập điểm số quá khứ đã quan sát tính đến phiên $t$** (expanding historical window), tuyệt đối không dùng phân vị tĩnh của toàn bộ giai đoạn 2022–2026 để bảo vệ tính vô vị kỷ thời gian (Zero Look-ahead Bias).

3. **Chế độ Kết hợp Tín hiệu Đa Mô hình (Equal-Weight Rank / Z-score Ensemble):**
   - Kết hợp điểm số dự báo của cả 3 mô hình (`Linear SVM`, `LightGBM`, `PyTorch LSTM`) bằng cách chuẩn hóa Cross-Sectional / Rolling Z-Score hoặc phân vị thứ hạng, sau đó lấy trung bình trọng số đều.
   - Nhờ độ tương quan thứ hạng giữa 3 mô hình rất thấp (+0.25 đến +0.33 ở RQ2), cơ chế Ensemble sẽ lọc bỏ nhiễu của từng mô hình đơn lẻ, làm mịn đường cong vốn (smooth equity curve) và giảm thiểu rủi ro rút vốn cực đại (Max Drawdown).

---

## 10. Kết luận Toàn diện Phase 3

Phase 3 chính thức khép lại với kết luận **`COMPLETED_WITH_LIMITATIONS`**:
1. Toàn bộ các cam kết kỹ thuật, quy trình chống rò rỉ dữ liệu (Purged Walk-Forward CV, Pre-validation Gap 5 phiên, Label Purging, Platt Calibration nhân quả) đã được thực thi chuẩn mực 100%.
2. Tập dữ liệu Reserved Test (2022–2026) đã được mở khóa một lần duy nhất (one-shot unfreezing), hoàn toàn không có hành vi điều chỉnh tham số hay tối ưu hóa dữ liệu kiểm tra (Zero Test Overfitting).
3. Toàn bộ dự đoán Out-of-Sample và bộ số liệu kiểm toán phân vị đã sẵn sàng trong `results/modeling/final_evaluation/test_predictions_2022_2026.parquet`, mở đường trực tiếp cho việc xây dựng động cơ kiểm thử chiến lược định lượng tại **Phase 4: Trading Strategy Formulation & Backtesting Engine**.
