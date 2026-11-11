# Báo cáo WP3.2: Chia thời gian và purging theo nhãn

## 1. Tổng quan WP3.2

Tạo ba development folds mở rộng và một vùng Test dự trữ cho RQ1/RQ2. Không huấn luyện hay đánh giá mô hình.

## 2. Kiến trúc triển khai

Tái sử dụng `ModelingView`, schema và Parquet Research của WP3.1. `src/models/temporal_split.py` xác minh ngày nhãn và tạo chỉ số vị trí trong view; manifest lưu thêm vị trí trên master calendar.

## 3. Thiết kế Temporal Split

| Fold | Train theo năm | Vùng đánh giá | Vai trò |
|---|---|---|---|
| `dev_1` | đến 2018 | 2019 | Phát triển |
| `dev_2` | đến 2019 | 2020 | Phát triển |
| `dev_3` | đến 2020 | 2021 | Phát triển |
| `reserved_test` | đến 2021 | 2022–hết dữ liệu | Test dự trữ |

## 4. Purging và Gap

Giữ mẫu Train khi `label_end_date < evaluation_boundary_date` và vị trí tín hiệu master < `boundary_position - 5`. Khoảng nhãn chạm ranh giới bị purge. Gap là đúng năm phiên master ngay trước ranh giới, kể cả khi một phiên không có mẫu eligible. Hai nguyên nhân loại được đếm riêng, phần giao trừ đúng một lần.

## 5. Kết quả RQ1

| Fold | Train | Đánh giá | Purged | Gap | Cả hai | Loại thực | Nhãn vượt kỳ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_1` | 545 | 249 | 1 | 5 | 1 | 5 | 1 |
| `dev_2` | 795 | 251 | 1 | 5 | 1 | 5 | 1 |
| `dev_3` | 1047 | 249 | 1 | 5 | 1 | 5 | 1 |
| `reserved_test` | 1297 | 1200 | 1 | 5 | 1 | 5 | 0 |

Bảng CSV và manifest ghi ngày thực tế cùng phân bố lớp của Train/Validation phát triển. Không ghi phân bố nhãn Test.

## 6. Kết quả RQ2

| Fold | Train | Đánh giá | Purged | Gap | Cả hai | Loại thực | Nhãn vượt kỳ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dev_1` | 545 | 246 | 4 | 5 | 4 | 5 | 4 |
| `dev_2` | 795 | 248 | 4 | 5 | 4 | 5 | 4 |
| `dev_3` | 1047 | 246 | 4 | 5 | 4 | 5 | 4 |
| `reserved_test` | 1297 | 1197 | 4 | 5 | 4 | 5 | 0 |

Bảng CSV và manifest ghi ngày thực tế cùng phân bố lớp của Train/Validation phát triển. Không ghi phân bố nhãn Test.

## 7. Leakage Validation

Các bất biến đã được kiểm tra trực tiếp khi tạo từng fold: index tăng dần, không giao mẫu, Train trước Validation, nhãn Train kết thúc trước ranh giới, gap đủ năm phiên master và nhãn Development kết thúc trước kỳ tiếp theo. Ranh giới Test là phiên master đầu tiên từ năm 2022. Test được giữ cho đánh giá cuối cùng; không dùng để chọn mô hình, threshold hay feature.

## 8. Unit/Integration Tests

Lệnh: `.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q`. Kết quả: 154 PASS, 0 FAIL.

## 9. Giới hạn nghiên cứu

Giới hạn giá điều chỉnh, corporate actions, lịch giao dịch chính thức và khả năng khớp lệnh của WP3.1 vẫn áp dụng. Trường `embargo_days` của Phase 2 không được diễn giải thành embargo sau Test; WP3.2 dùng `pre_validation_gap_sessions` được ghi rõ trong manifest.

## 10. Kết luận WP3.2

`COMPLETED_WITH_LIMITATIONS`: split và kiểm tra rò rỉ đã thực thi; các chỉ số sẵn sàng cho WP3.3 trong phạm vi giới hạn dữ liệu nguồn.
