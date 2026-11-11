# Báo cáo WP3.3: SVM baseline

## 1. Tổng quan WP3.3

Đánh giá Dummy, Linear SVM và RBF SVM cho RQ1 và RQ2 trên ba development folds. RQ2 dự đoán nhãn kinh tế, không phải lợi nhuận giao dịch thực tế.

## 2. Kiến trúc Modeling Pipeline

Dataset và modeling view tái sử dụng WP3.1; indices lấy từ API development đã kiểm tra fingerprint và split của WP3.2.1. Mỗi fold có estimator mới.

## 3. Thiết lập thực nghiệm

Dataset SHA-256: `4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e`; manifest SHA-256: `f35a62524648488d84e52f7579a72e40e6d1b2c595814cb20f02535d3ecd5b42`. Feature: 26; scikit-learn: 1.9.1. Dummy dùng `strategy=prior`. SVM dùng `StandardScaler` trong Pipeline, C ∈ {0,1; 1; 10}, class_weight ∈ {None, balanced}; RBF dùng gamma=scale. Mọi SVC dùng probability=False; không có calibration.

## 4. Kết quả RQ1

Development Walk-Forward CV Performance: Balanced Accuracy theo từng fold.

| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Trung bình | Độ lệch chuẩn | So với Dummy |
|---|---:|---:|---:|---:|---:|---:|
| `dummy_prior` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 |
| `linear_c0.1_none` | 0.5191 | 0.4993 | 0.5130 | 0.5104 | 0.0083 | +0.0104 |
| `linear_c0.1_balanced` | 0.5188 | 0.5368 | 0.5223 | 0.5260 | 0.0078 | +0.0260 |
| `linear_c1_none` | 0.5088 | 0.4919 | 0.5083 | 0.5030 | 0.0078 | +0.0030 |
| `linear_c1_balanced` | 0.5025 | 0.5389 | 0.5071 | 0.5162 | 0.0162 | +0.0162 |
| `linear_c10_none` | 0.5262 | 0.5011 | 0.5032 | 0.5102 | 0.0114 | +0.0102 |
| `linear_c10_balanced` | 0.4823 | 0.5379 | 0.4939 | 0.5047 | 0.0239 | +0.0047 |
| `rbf_c0.1_none` | 0.4997 | 0.5000 | 0.5049 | 0.5016 | 0.0024 | +0.0016 |
| `rbf_c0.1_balanced` | 0.4967 | 0.5221 | 0.5101 | 0.5096 | 0.0104 | +0.0096 |
| `rbf_c1_none` | 0.4892 | 0.4779 | 0.5003 | 0.4891 | 0.0091 | -0.0109 |
| `rbf_c1_balanced` | 0.5099 | 0.4982 | 0.4909 | 0.4997 | 0.0078 | -0.0003 |
| `rbf_c10_none` | 0.5191 | 0.4743 | 0.5210 | 0.5048 | 0.0216 | +0.0048 |
| `rbf_c10_balanced` | 0.5248 | 0.5009 | 0.5163 | 0.5140 | 0.0099 | +0.0140 |

Các metrics còn lại, confusion matrix và chênh lệch theo fold có trong CSV.

## 5. Kết quả RQ2

Development Walk-Forward CV Performance: Balanced Accuracy theo từng fold.

| Cấu hình | Dev 1 | Dev 2 | Dev 3 | Trung bình | Độ lệch chuẩn | So với Dummy |
|---|---:|---:|---:|---:|---:|---:|
| `dummy_prior` | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 | +0.0000 |
| `linear_c0.1_none` | 0.4607 | 0.5000 | 0.5441 | 0.5016 | 0.0341 | +0.0016 |
| `linear_c0.1_balanced` | 0.4479 | 0.5341 | 0.5711 | 0.5177 | 0.0516 | +0.0177 |
| `linear_c1_none` | 0.5217 | 0.5160 | 0.5279 | 0.5219 | 0.0048 | +0.0219 |
| `linear_c1_balanced` | 0.5016 | 0.5572 | 0.5229 | 0.5272 | 0.0229 | +0.0272 |
| `linear_c10_none` | 0.5211 | 0.5591 | 0.5004 | 0.5269 | 0.0243 | +0.0269 |
| `linear_c10_balanced` | 0.5193 | 0.5544 | 0.5036 | 0.5258 | 0.0212 | +0.0258 |
| `rbf_c0.1_none` | 0.5000 | 0.5000 | 0.5248 | 0.5083 | 0.0117 | +0.0083 |
| `rbf_c0.1_balanced` | 0.4950 | 0.5032 | 0.5045 | 0.5009 | 0.0042 | +0.0009 |
| `rbf_c1_none` | 0.4527 | 0.5386 | 0.4904 | 0.4939 | 0.0351 | -0.0061 |
| `rbf_c1_balanced` | 0.4682 | 0.5491 | 0.4889 | 0.5021 | 0.0343 | +0.0021 |
| `rbf_c10_none` | 0.4764 | 0.4895 | 0.5690 | 0.5116 | 0.0409 | +0.0116 |
| `rbf_c10_balanced` | 0.4661 | 0.5018 | 0.5718 | 0.5132 | 0.0439 | +0.0132 |

Các metrics còn lại, confusion matrix và chênh lệch theo fold có trong CSV.

## 6. So sánh với Dummy Baseline

Chênh lệch được tính trên cùng mẫu Validation của từng RQ và fold. Dummy dự báo lớp phổ biến theo phân bố Train; điểm ranking là tỷ lệ lớp 1 của Train.

## 7. Hyperparameter Selection

Mean Balanced Accuracy cao nhất; nếu bằng nhau chọn độ lệch chuẩn nhỏ hơn, sau đó Linear trước RBF và thứ tự cấu hình đã công bố.

| RQ | Cấu hình SVM được chọn | Mean Balanced Accuracy | So với Dummy |
|---|---|---:|---:|
| RQ1 | `linear_c0.1_balanced` | 0.5260 | +0.0260 |
| RQ2 | `linear_c1_balanced` | 0.5272 | +0.0272 |

## 8. Phân tích kết quả

Mức chênh lệch giữa các năm cần được xem cùng độ lệch chuẩn; ba fold có thể chịu phụ thuộc thời gian và các nhãn RQ2 có khoảng nắm giữ chồng lấn. Hiệu suất của cấu hình thắng đã được dùng để chọn tham số, do đó có thiên lệch lựa chọn và không phải ước lượng Test độc lập. Không suy ra trading alpha.

## 9. Data Leakage và Test Isolation

Scaler chỉ fit qua Pipeline trên X_train của từng fold. Prediction artifacts chỉ chứa Validation Development trước 2022 và nhãn phải kết thúc trước vùng Test. SVM decision score không phải xác suất; chưa thực hiện calibration.

## 10. Automated Tests

Lệnh: `.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q --tb=short`. Kết quả: 191 PASS, 0 FAIL; 8 cảnh báo tương thích từ scikit-learn.

## 11. Files và Artifacts

`metrics_by_fold.csv`, `model_comparison.csv`, `selected_configs.json` và `validation_predictions.parquet` nằm cùng thư mục với báo cáo. Metadata ghi tham số, phiên bản, fingerprint và quy tắc lựa chọn.

## 12. Giới hạn nghiên cứu

Các giới hạn nguồn WP3.1 về adjusted prices, corporate actions, lịch phiên và khả năng thực thi vẫn áp dụng. Reserved Test chưa được đánh giá; không diễn giải decision score thành xác suất hoặc chỉ số phân loại thành lợi nhuận. scikit-learn 1.9.1 cảnh báo tham số `SVC(probability=...)` sẽ bị loại trong phiên bản tương lai; cần kiểm tra tính tương thích trước khi nâng cấp.

## 13. Kết luận WP3.3

`COMPLETED_WITH_LIMITATIONS`: kết quả dùng để lựa chọn cấu hình trên Development Walk-Forward CV, chưa phải Final Out-of-Sample Test Performance.

## 14. Khuyến nghị WP3.4

Dùng cùng dataset, eligibility, development folds và metrics để so sánh LightGBM; tiếp tục giữ Reserved Test và thiết kế calibration riêng nếu cần xác suất.
