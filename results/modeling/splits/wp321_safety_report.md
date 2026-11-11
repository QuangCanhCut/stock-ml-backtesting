# Báo cáo WP3.2.1: An toàn modeling và khả năng tái lập

## 1. Tổng quan

WP3.2.1 bổ sung bước xác minh bắt buộc trước khi dùng chỉ số split của WP3.2. Mục tiêu là phát hiện manifest cũ hoặc bị sửa và tránh trao nhầm chỉ số Reserved Test cho quy trình phát triển mô hình. Không huấn luyện hay đánh giá mô hình trong bước này.

## 2. Phân tích repository

| Thành phần hiện có | Cách tái sử dụng |
|---|---|
| `load_project_config`, `load_research_dataset` | Đọc config và Parquet Research, kiểm tra schema master |
| `expected_feature_columns`, `make_modeling_view` | Xác minh thứ tự feature và eligibility riêng cho RQ1/RQ2 |
| `build_temporal_splits` | Tính lại fold để đối chiếu indices và ranh giới với manifest |
| `split_manifest.json` | Nguồn fingerprint, split specification và vị trí trong view/master |

Manifest hiện tại ghi 26 feature theo thứ tự, 8 fold cho hai RQ và gap 5 phiên master. Các thay đổi cũ của WP3.1/WP3.2 trong working tree được giữ nguyên.

## 3. Bảo vệ Reserved Test

`get_development_splits(root, rq)` chỉ trả ba fold có `role="development"`. `get_development_split` từ chối tên `reserved_test`; `require_development_split` từ chối fold mang role Test. `get_final_evaluation_split` chỉ trả fold Test khi người gọi ghi rõ `final_evaluation=True`. Các hàm lấy fold từ repository xác minh manifest hiện tại trước khi trao chỉ số; hàm guard riêng chỉ kiểm tra role của fold được truyền vào.

Đây là hàng rào ở cấp API và quy trình. Code có quyền đọc trực tiếp Parquet hoặc JSON vẫn có thể truy cập Test nếu cố ý bỏ qua API.

## 4. Xác minh fingerprint

| Kiểm tra | Kết quả thực thi |
|---|---|
| SHA-256 Parquet Research | Khớp manifest: `4cc485687c1bc13aae7f3989be77a0d1e4da8abb1f2e3da6a470ed3e6c5e8f4e` |
| SHA-256 `configs/config.yaml` | Khớp manifest: `03dc301d0be4a6d31f94589fb34c1d8a026c01dcb49d218091899fa8d436f6b3` |
| Đường dẫn, số phiên, ngày đầu/cuối | Khớp dataset Research hiện tại: 2.702 phiên, 2016-01-04 đến 2026-10-30 |
| Feature schema | 26 tên và thứ tự khớp config, metadata dataset và manifest |
| Split specification | RQ1/RQ2, tên fold, ranh giới năm, gap 5, thời hạn giữ 3 phiên và Test từ 2022 khớp code WP3.2 |

Hash được tính từ bytes của tệp ở mỗi lần gọi API, không hardcode trong mã nguồn. Thay đổi nội dung YAML, kể cả thay đổi trình bày không ảnh hưởng ngữ nghĩa, cũng yêu cầu chạy lại quy trình kiểm định WP3.2 để tạo manifest tương ứng.

## 5. Kiểm tra toàn vẹn indices

Trước khi trả fold, validator kiểm tra indices là số nguyên không âm, trong phạm vi `ModelingView`, tăng dần, duy nhất, không giao Train/Validation và khớp số mẫu đã lưu. Ngày tương ứng phải khớp vị trí master. Sau đó toàn bộ fold của cả RQ1 và RQ2 được tính lại bằng `build_temporal_splits` và đối chiếu với manifest, gồm role, ngày, số mẫu, purging, gap, phần giao và chỉ số vị trí. Manifest không khớp gây `ValueError`; API không tự ghi hoặc tái tạo manifest.

## 6. Unit và integration tests

Lệnh đã chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q --tb=short
```

Kết quả: **179 PASS, 0 FAIL**. Trong đó 25 test mới kiểm tra API development/final, fingerprint đúng và sai, schema đổi thứ tự hoặc thiếu feature, split specification sai, indices âm/vượt phạm vi/trùng/giao nhau, ngày sai, dataset cũ và tính bất biến của ba tệp nguồn sau integration test. Không có test nào tính chỉ số hiệu suất mô hình trên Reserved Test.

## 7. File thay đổi

| File | Chức năng và lý do |
|---|---|
| `src/models/modeling_safety.py` | API truy cập fold an toàn và xác minh manifest trước modeling |
| `tests/unit/test_modeling_safety.py` | Kiểm tra các tình huống lỗi và quyền truy cập bằng dữ liệu hiện hành trong bộ nhớ |
| `tests/integration/test_modeling_safety_research.py` | Xác minh API trên HPG Research và hash nguồn không đổi |
| `results/modeling/splits/wp321_safety_report.md` | Ghi bằng chứng và giới hạn của WP3.2.1 |

## 8. Giới hạn

- Hàng rào API không thay thế kiểm soát quyền truy cập tệp hoặc quy trình phê duyệt Final Evaluation.
- Các giới hạn về giá điều chỉnh, corporate actions, lịch giao dịch chính thức và khả năng khớp lệnh từ WP3.1 vẫn áp dụng.
- Manifest WP3.2 hiện tại vẫn là nguồn chỉ số; nếu dataset hoặc config thay đổi, cần chạy lại WP3.2 và kiểm định trước khi modeling.

## 9. Kết luận

`COMPLETED_WITH_LIMITATIONS`. API development đã được xác minh trên dataset và manifest hiện tại; đủ điều kiện bắt đầu WP3.3 khi các script modeling dùng API này để lấy fold và giữ Reserved Test cho Final Evaluation tường minh.
