# Review bằng chứng Data Loader

Phạm vi: kiểm tra cache KBS và báo cáo audit hiện có; không gọi API, không xác nhận độc lập nguồn hoặc tạo dataset validated.

- Raw có 2.639 phiên, từ 04/01/2016 đến 30/10/2026. Các phiên này chưa có hồ sơ chứng minh đầy đủ lịch, price, volume và adjustment.
- So với cache KBS, 63 phiên không có trong Raw. Báo cáo Yahoo lịch sử ghi 58 phiên `INVALID_VOLUME` và 5 phiên `ABSENT_FROM_YAHOO`; snapshot này không phải toàn bộ lịch sử Yahoo. Cần lịch độc lập và OHLCV có provenance, basis thống nhất trước khi quyết định có thể dùng những phiên này.
- 94 phiên Raw từ 04/01 đến 24/05/2016 không có đối chiếu trong cache KBS; cache bắt đầu 25/05/2016. Chưa có căn cứ kết luận 94 phiên sai hoặc không tồn tại.
- Audit alignment 2025 đối chiếu lịch KBS cache trước khi ghép với Raw cho 101 `PREVIOUS_SESSION`, 26 `SAME_DAY`, trong cửa sổ 06/05–31/10/2025. Tỷ lệ sử dụng trong audit là giả định, không xác nhận date shift; cần nguồn độc lập có ngày và basis đã xác minh.
- Ngày 04/10/2021, Raw có volume 7.610, KBS có 45.357.500. Ratio không xác nhận đơn vị hay phép điều chỉnh; phiên này được giữ ở `hold_review` và cần chứng cứ độc lập.
- KBS 07/08/2019 có open 6,63 thấp hơn low 6,70; KBS 04/09/2019 có open 6,83 cao hơn high 6,70. Không sửa nến hoặc mặc định Raw đúng; cần đối chiếu bản ghi từ nguồn độc lập và nguồn có thẩm quyền.
- Cơ sở điều chỉnh giá, volume, corporate actions và tính point-in-time chưa có tài liệu được thẩm định. Không xác nhận tỷ lệ Yahoo/KBS là hệ số điều chỉnh.

Chi tiết quan sát theo nguồn có trong `audit_loader_final_20261108/sessions.csv`; danh sách alignment có trong `audit_loader_final_20261108/alignment_2025.csv`. Đây là bằng chứng chẩn đoán, không phải xác nhận độc lập. Quality Gate chưa đạt; không tạo file validated hay chạy lại Phase 2.
