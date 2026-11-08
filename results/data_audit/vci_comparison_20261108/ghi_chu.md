# Đối chiếu VCI với Raw và KBS

Snapshot gồm 11 khoảng theo năm từ 2016 đến 2026, thu qua `vnstock` 5.2.2; metadata từng file ghi thời điểm, request và SHA-256. Đây là DataFrame đã được vnstock xử lý và chuẩn hóa, không phải HTTP response nguyên bản. VCI và KBS là hai nguồn được công cụ ghi nhận, nhưng tính độc lập của toàn bộ chuỗi cung cấp cần xác minh thêm. Cần kiểm tra quyền sử dụng trước khi phân phối snapshot.

- VCI trả 2.704 dòng từ 04/01/2016 đến 30/10/2026. Trong đó 63/63 ngày có trong báo cáo KBS nhưng thiếu Raw xuất hiện theo ngày trong VCI; 94/94 ngày Raw đầu 2016 trước khi KBS bắt đầu cũng xuất hiện theo ngày trong VCI. Điều này chưa xác nhận OHLCV hay lịch chính thức.
- VCI có 2 dòng volume bằng 0 và OHLC phẳng vào 23/01 và 24/01/2018, không đạt validation. Chưa đủ căn cứ xác nhận hai ngày này là phiên giao dịch hay placeholder.
- Hai dòng KBS có hình học OHLC lỗi năm 2019 có bản ghi VCI hợp lệ về hình học, nhưng giá không hoàn toàn trùng. Cần bằng chứng basis và đối chiếu nguồn có thẩm quyền trước khi thay thế.
- Ngày 04/10/2021, VCI và KBS đều ghi volume 45.357.500; Raw ghi 7.610. Chưa đủ căn cứ xác nhận phép điều chỉnh hoặc thay đổi Raw.
- Trong cửa sổ 06/05–31/10/2025 có 127 ngày trong báo cáo; volume VCI không bằng volume KBS trên cả 127 ngày. Đây chưa phải bằng chứng định lượng cho phép dịch ngày hoặc chọn một nguồn; cần xác minh định nghĩa volume và corporate actions.

`vci_kbs_raw.csv` giữ các giá trị theo nguồn và ratio thăm dò; không áp dụng ratio để biến đổi dữ liệu. Price basis, volume basis, calendar chính thức và tính point-in-time chưa được xác minh. Chưa tạo candidate được duyệt, Raw validated hoặc Processed Dataset mới.
