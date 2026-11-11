# Báo cáo WP3.1: Chuẩn bị dữ liệu và kiểm toán rò rỉ

## 1. Tổng quan WP3.1

Kiểm tra dataset Research cho RQ1 và RQ2 trước khi chia tập theo thời gian.

## 2. Đánh giá repository hiện tại

Tái sử dụng `expected_feature_columns`, `validate_dataset`, `get_training_view` của Phase 2; đối chiếu độc lập công thức nhãn và tính lại feature trên các tiền tố.

## 3. Kiểm toán dataset

| Chỉ số | Giá trị thực tế |
|---|---|
| Nguồn | `data/processed/HPG_features_labeled_research.parquet` |
| SHA-256 | `4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e` |
| Kích thước | 2702 phiên × 37 cột |
| Khoảng ngày | 2016-01-04 đến 2026-10-30 |
| Feature | 26 |
| Môi trường | Python 3.13.2; pandas 3.1.0; NumPy 2.5.4; PyArrow 26.0.1 |
| Git commit | `d907bac88710b0a20274569054551459f274f280` |

## 4. Feature và target

Thứ tự feature lấy từ cấu hình Phase 2: `log_return`, `return_lag_1`, `return_lag_2`, `return_lag_3`, `return_lag_5`, `return_lag_10`, `return_lag_21`, `sma_ratio_10`, `sma_ratio_20`, `sma_ratio_50`, `sma_ratio_200`, `ema_ratio_12`, `ema_ratio_26`, `macd_ratio`, `momentum_5`, `momentum_10`, `momentum_21`, `rsi_14`, `volatility_21`, `volatility_63`, `atr_ratio_14`, `bollinger_width_20`, `bollinger_percent_b_20`, `volume_zscore_20`, `volume_change_1`, `volume_ratio_20`.

| RQ | Target | Đủ điều kiện | Loại | Class 0 | Class 1 | Khoảng ngày đủ điều kiện |
|---|---|---:|---:|---:|---:|---|
| RQ1 | `target_direction` | 2502 | 200 | 1320 (52.76%) | 1182 (47.24%) | 2016-10-20 đến 2026-10-29 |
| RQ2 | `target_fixed_hold_net_proxy` | 2499 | 203 | 1347 (53.90%) | 1152 (46.10%) | 2016-10-20 đến 2026-10-26 |

Các hàng bị loại gồm feature chưa đầy đủ và nhãn tương lai thiếu; hai nguyên nhân được thống kê không chồng lặp trong JSON.

## 5. Kiểm toán rò rỉ dữ liệu

| Kiểm tra | Kết quả |
|---|---|
| `schema` | PASS |
| `feature_leakage` | PASS |
| `temporal_alignment_rq1` | PASS |
| `temporal_alignment_rq2` | PASS |
| `feature_prefix_invariance` | PASS |
| `view_eligibility` | PASS |

RQ1 đối chiếu `Close(t+1)/Close(t)` và nhãn hướng; RQ2 đối chiếu `Open(t+1)` đến `Open(t+4)` và công thức phí hai chiều. Tổng phí gần đúng theo cấu hình là 0.50%; return ròng dùng công thức nhân tỷ lệ, không trừ trực tiếp ngưỡng này.

Feature thiếu ở 199 hàng; warm-up đầu chuỗi 199 hàng. Nhãn RQ1 thiếu 1 hàng, nhãn RQ2 thiếu 4 hàng. Số giá trị Inf trong feature: 0. Biến động Close trên 12%: 0 hàng; volume ratio ngoài [0,1; 10]: 0 hàng.

## 6. Kiểm thử phần mềm

Chạy `.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q`. Kết quả của lần chạy được ghi riêng trong phản hồi thực thi để tránh một bản báo cáo tái tạo sau này giữ số kiểm thử đã cũ.

## 7. File được thêm hoặc chỉnh sửa

| File | Mục đích |
|---|---|
| `src/models/data.py` | Đọc Research và tạo view kèm mốc kết thúc nhãn |
| `src/models/data_validation.py` | Kiểm toán schema, nhãn và feature |
| `scripts/phase3/run_wp31_audit.py` | Xuất báo cáo JSON và Markdown |
| `tests/unit/test_wp31.py` | Kiểm thử dữ liệu tổng hợp |

## 8. Giới hạn nghiên cứu

- Giá điều chỉnh và sự kiện doanh nghiệp chưa được xác minh tại từng thời điểm.
- Lịch giao dịch chính thức, nhất là 94 phiên đầu năm 2016, chưa được đối chiếu độc lập.
- Chưa chứng minh thời điểm công bố dữ liệu và khả năng giao dịch tại giá Open.
- Chi phí và slippage chỉ là giả định; chưa phản ánh khớp lệnh thực tế.

Unit test đạt không chứng minh đã loại bỏ mọi rò rỉ dữ liệu.

## 9. Đánh giá hoàn thành WP3.1

`COMPLETED_WITH_LIMITATIONS`: các phép kiểm tra thực thi đạt, còn giới hạn nguồn và thực thi giao dịch.

## 10. Chuẩn bị cho WP3.2

`ModelingView.metadata` cung cấp `signal_date` và `label_end_date` để purge các khoảng nhãn khi chia walk-forward. Chưa tạo fold hay fit mô hình.
