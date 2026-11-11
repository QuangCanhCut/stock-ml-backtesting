# Báo cáo WP3.4: LightGBM Modeling, Feature Importance & Final Development Models

## 1. Tổng quan WP3.4

Nghiệm thu mô hình **LightGBM** (Gradient Boosted Decision Trees) cho bài toán RQ1 (dự báo hướng lợi suất) và RQ2 (dự báo lợi nhuận kinh tế sau toàn bộ ma sát thị trường) trên ba development folds (`dev_1`, `dev_2`, `dev_3`). Đánh giá và so sánh 1-1 với **Dummy baseline** và **SVM baseline (WP3.3)**. Trích xuất Feature Importance và lưu trữ Final Development Models (2016–2021) vào thư mục `models/`.

## 2. Thiết lập thực nghiệm và Kiến trúc an toàn

- **Dataset Parquet:** `data/processed/HPG_features_labeled_research.parquet` (SHA-256: `4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e`).
- **Split Manifest:** `results/modeling/splits/split_manifest.json` (SHA-256: `f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42`).
- **Feature Schema:** 26 causal features không chứa look-ahead bias.
- **Môi trường:** Python 3.13.2, LightGBM 4.7.0, scikit-learn 1.9.1, joblib 1.6.0.
- **Kiểm soát rò rỉ:** Dữ liệu đầu vào cho LightGBM là raw features (không dùng scaler toàn cục). Chỉ sử dụng API `get_development_splits()` cho 3 dev folds. Tập Reserved Test (2022–2026) được bảo vệ tuyệt đối, không có bất kỳ dự báo hay đánh giá nào.
- **Early Stopping:** Dừng sau 30 rounds nếu `valid_binary_logloss` không cải thiện.

## 3. Kết quả RQ1 (Dự báo hướng giá (Direction))

### Bảng so sánh hiệu suất Walk-Forward CV (Balanced Accuracy theo từng fold)

| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Mean BAcc | Std BAcc | vs Dummy | vs SVM Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dummy_prior` (Baseline) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0260 |
| `linear_c0.1_balanced` (SVM WP3.3) | 0.5188 | 0.5368 | 0.5223 | 0.5260 | 0.0078 | +0.0260 | +0.0000 |
| `lgbm_d3_lr0.03_mc40_l1_none` | 0.4837 | 0.5000 | 0.5510 | 0.5116 | 0.0287 | +0.0116 | -0.0144 |
| `lgbm_d3_lr0.03_mc40_l1_balanced` | 0.4937 | 0.4938 | 0.5431 | 0.5102 | 0.0232 | +0.0102 | -0.0158 |
| `lgbm_d3_lr0.03_mc80_l5_none` **(Winner)** | 0.5000 | 0.5000 | 0.5744 | 0.5248 | 0.0351 | +0.0248 | -0.0012 |
| `lgbm_d3_lr0.03_mc80_l5_balanced` | 0.4800 | 0.4947 | 0.5600 | 0.5116 | 0.0348 | +0.0116 | -0.0144 |
| `lgbm_d3_lr0.05_mc40_l1_none` | 0.4688 | 0.5000 | 0.5502 | 0.5063 | 0.0335 | +0.0063 | -0.0197 |
| `lgbm_d3_lr0.05_mc40_l1_balanced` | 0.4658 | 0.4938 | 0.5553 | 0.5050 | 0.0374 | +0.0050 | -0.0210 |
| `lgbm_d3_lr0.05_mc80_l5_none` | 0.5000 | 0.5000 | 0.5672 | 0.5224 | 0.0317 | +0.0224 | -0.0036 |
| `lgbm_d3_lr0.05_mc80_l5_balanced` | 0.4800 | 0.4947 | 0.5681 | 0.5143 | 0.0385 | +0.0143 | -0.0117 |
| `lgbm_d5_lr0.03_mc40_l1_none` | 0.5031 | 0.5057 | 0.5477 | 0.5188 | 0.0204 | +0.0188 | -0.0072 |
| `lgbm_d5_lr0.03_mc40_l1_balanced` | 0.4909 | 0.4973 | 0.5460 | 0.5114 | 0.0246 | +0.0114 | -0.0146 |
| `lgbm_d5_lr0.03_mc80_l5_none` | 0.5000 | 0.5000 | 0.5371 | 0.5124 | 0.0175 | +0.0124 | -0.0136 |
| `lgbm_d5_lr0.03_mc80_l5_balanced` | 0.4800 | 0.5097 | 0.5350 | 0.5082 | 0.0225 | +0.0082 | -0.0177 |
| `lgbm_d5_lr0.05_mc40_l1_none` | 0.4981 | 0.5000 | 0.5608 | 0.5196 | 0.0291 | +0.0196 | -0.0063 |
| `lgbm_d5_lr0.05_mc40_l1_balanced` | 0.4724 | 0.4973 | 0.5490 | 0.5062 | 0.0319 | +0.0062 | -0.0198 |
| `lgbm_d5_lr0.05_mc80_l5_none` | 0.5000 | 0.5000 | 0.5532 | 0.5177 | 0.0251 | +0.0177 | -0.0082 |
| `lgbm_d5_lr0.05_mc80_l5_balanced` | 0.4800 | 0.5097 | 0.5380 | 0.5092 | 0.0237 | +0.0092 | -0.0167 |

**Cấu hình LightGBM thắng cuộc RQ1:** `lgbm_d3_lr0.03_mc80_l5_none`

- Mean Balanced Accuracy: **0.5248** (Std: 0.0351).

- Chênh lệch so với Dummy baseline: **+0.0248**.

- Chênh lệch so với SVM baseline (WP3.3): **-0.0012**.

- Mean ROC-AUC: 0.5110; Mean Generalization Gap (Train - Valid): 0.0208.

- Best iterations qua 3 folds: [1, 1, 39] (Trung bình làm tròn chọn `final_n_estimators = 14`).

## 4. Kết quả RQ2 (Nhãn kinh tế sau chi phí (Fixed-Hold Net Proxy))

### Bảng so sánh hiệu suất Walk-Forward CV (Balanced Accuracy theo từng fold)

| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Mean BAcc | Std BAcc | vs Dummy | vs SVM Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dummy_prior` (Baseline) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0272 |
| `linear_c1_balanced` (SVM WP3.3) | 0.5016 | 0.5572 | 0.5229 | 0.5272 | 0.0229 | +0.0272 | +0.0000 |
| `lgbm_d3_lr0.03_mc40_l1_none` | 0.5715 | 0.5000 | 0.4894 | 0.5203 | 0.0364 | +0.0203 | -0.0069 |
| `lgbm_d3_lr0.03_mc40_l1_balanced` | 0.5682 | 0.4798 | 0.5136 | 0.5205 | 0.0364 | +0.0205 | -0.0067 |
| `lgbm_d3_lr0.03_mc80_l5_none` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0272 |
| `lgbm_d3_lr0.03_mc80_l5_balanced` | 0.5265 | 0.4669 | 0.4809 | 0.4914 | 0.0254 | -0.0086 | -0.0358 |
| `lgbm_d3_lr0.05_mc40_l1_none` | 0.5635 | 0.5000 | 0.4922 | 0.5186 | 0.0319 | +0.0186 | -0.0087 |
| `lgbm_d3_lr0.05_mc40_l1_balanced` **(Winner)** | 0.5450 | 0.4798 | 0.5445 | 0.5231 | 0.0306 | +0.0231 | -0.0042 |
| `lgbm_d3_lr0.05_mc80_l5_none` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0272 |
| `lgbm_d3_lr0.05_mc80_l5_balanced` | 0.5265 | 0.4669 | 0.4809 | 0.4914 | 0.0254 | -0.0086 | -0.0358 |
| `lgbm_d5_lr0.03_mc40_l1_none` | 0.5619 | 0.5000 | 0.5000 | 0.5206 | 0.0292 | +0.0206 | -0.0066 |
| `lgbm_d5_lr0.03_mc40_l1_balanced` | 0.5380 | 0.4905 | 0.4632 | 0.4972 | 0.0309 | -0.0028 | -0.0300 |
| `lgbm_d5_lr0.03_mc80_l5_none` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0272 |
| `lgbm_d5_lr0.03_mc80_l5_balanced` | 0.5242 | 0.4551 | 0.4353 | 0.4715 | 0.0381 | -0.0285 | -0.0557 |
| `lgbm_d5_lr0.05_mc40_l1_none` | 0.5688 | 0.4992 | 0.5000 | 0.5227 | 0.0326 | +0.0227 | -0.0046 |
| `lgbm_d5_lr0.05_mc40_l1_balanced` | 0.5365 | 0.4905 | 0.4632 | 0.4967 | 0.0302 | -0.0033 | -0.0305 |
| `lgbm_d5_lr0.05_mc80_l5_none` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 | -0.0272 |
| `lgbm_d5_lr0.05_mc80_l5_balanced` | 0.5242 | 0.4551 | 0.4353 | 0.4715 | 0.0381 | -0.0285 | -0.0557 |

**Cấu hình LightGBM thắng cuộc RQ2:** `lgbm_d3_lr0.05_mc40_l1_balanced`

- Mean Balanced Accuracy: **0.5231** (Std: 0.0306).

- Chênh lệch so với Dummy baseline: **+0.0231**.

- Chênh lệch so với SVM baseline (WP3.3): **-0.0042**.

- Mean ROC-AUC: 0.5302; Mean Generalization Gap (Train - Valid): 0.1518.

- Best iterations qua 3 folds: [29, 1, 29] (Trung bình làm tròn chọn `final_n_estimators = 20`).

## 5. Phân tích Feature Importance (Gain, Split & Validation Permutation)

So sánh giữa In-sample Tree Importance (khớp tập Train) và Out-of-sample Permutation Importance (xáo trộn đặc trưng trên tập Validation độc lập của từng fold với `scoring='balanced_accuracy'`):

| RQ | Đặc trưng (Feature) | Mean Split | Mean Gain | Mean Permutation Importance | Std Permutation |
|---|---|---:|---:|---:|---:|
| RQ1 | `momentum_21` | 11.0 | 63.68 | **+0.0190** | 0.0269 |
| RQ1 | `atr_ratio_14` | 6.7 | 34.73 | **+0.0086** | 0.0122 |
| RQ1 | `bollinger_percent_b_20` | 4.7 | 21.36 | **+0.0046** | 0.0065 |
| RQ1 | `sma_ratio_10` | 2.3 | 13.68 | **+0.0044** | 0.0062 |
| RQ1 | `return_lag_21` | 3.7 | 20.16 | **+0.0043** | 0.0061 |
| RQ1 | `return_lag_10` | 2.0 | 11.33 | **+0.0036** | 0.0050 |
| RQ1 | `return_lag_2` | 8.0 | 39.76 | **+0.0034** | 0.0048 |
| RQ2 | `momentum_10` | 6.7 | 42.98 | **+0.0158** | 0.0152 |
| RQ2 | `atr_ratio_14` | 7.0 | 49.45 | **+0.0139** | 0.0128 |
| RQ2 | `return_lag_2` | 4.7 | 23.69 | **+0.0132** | 0.0120 |
| RQ2 | `log_return` | 5.0 | 29.43 | **+0.0092** | 0.0155 |
| RQ2 | `return_lag_10` | 9.7 | 49.82 | **+0.0044** | 0.0125 |
| RQ2 | `sma_ratio_10` | 2.0 | 11.92 | **+0.0040** | 0.0093 |
| RQ2 | `volume_change_1` | 3.0 | 12.37 | **+0.0036** | 0.0026 |

*Ghi chú quan trọng:* Đặc trưng có Gain cao trên tập Train không nhất thiết mang lại giá trị dự báo ngoài mẫu nếu Permutation Importance trên Validation là âm hoặc xấp xỉ 0. Bảng đầy đủ 26 đặc trưng được lưu tại `feature_importance.csv`.

## 6. Lưu trữ Final Development Models (2016–2021) vào `models/`

Tuân thủ triệt để nguyên tắc không lưu rời rạc từng fold; chỉ fit lại đúng 1 model cuối cùng cho mỗi RQ trên toàn bộ dữ liệu phát triển 2016–2021 (1.297 mẫu hợp lệ sau purging và 5 phiên gap):

| Họ mô hình | File Model Joblib | Metadata JSON | SHA-256 Joblib (10 ký tự đầu) | Train Samples |
|---|---|---|---|---:|
| SVM (RQ1) | `rq1_svm_final_dev_2016_2021.joblib` | `svm_final_dev_metadata.json` | `f383af4b54` | 1297 |
| SVM (RQ2) | `rq2_svm_final_dev_2016_2021.joblib` | `svm_final_dev_metadata.json` | `16b65ac9df` | 1297 |
| LIGHTGBM (RQ1) | `rq1_lgbm_final_dev_2016_2021.joblib` | `lightgbm_final_dev_metadata.json` | `01bce23c14` | 1297 |
| LIGHTGBM (RQ2) | `rq2_lgbm_final_dev_2016_2021.joblib` | `lightgbm_final_dev_metadata.json` | `a1855d7924` | 1297 |

Đã tạo sẵn thư mục giữ chỗ `models/lstm/.gitkeep` cho WP3.5.

## 7. Cảnh báo Phương pháp luận và Giới hạn

- **Validation Selection Bias:** Điểm số Validation phản ánh thiên lệch lựa chọn do quá trình tinh chỉnh siêu tham số và dừng sớm (`early_stopping`). Tuyệt đối không coi đây là lợi nhuận thực tế hay alpha giao dịch.
- **Bảo vệ Reserved Test:** Vùng Test 2022–2026 chưa được mở khóa và chưa thực hiện backtest.
- **Độ nhạy mẫu nhỏ:** Chuỗi thời gian tài chính HPG có tỷ lệ SNR thấp; các cây quyết định có thể gặp hiện tượng regime shift qua các chu kỳ kinh tế 2019, 2020 và 2021.

## 8. Kết quả Kiểm thử Tự động

Chưa có kết quả kiểm thử được cung cấp trong lần gọi này.

## 9. Kết luận WP3.4

`COMPLETED_WITH_LIMITATIONS`: Quá trình huấn luyện, đánh giá LightGBM, trích xuất Feature Importance và lưu trữ Final Development Model (2016–2021) hoàn thành đạt chuẩn.
