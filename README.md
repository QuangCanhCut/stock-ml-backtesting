# Stock ML Backtesting

## Data Loader và kiểm định dữ liệu

Data Loader thống nhất nằm tại `src/data_loader/`. Chạy audit offline từ thư mục gốc dự án:

```powershell
python -m src.data_loader audit
python -m src.data_loader build
```

`build` chỉ được xuất `data/raw/HPG_10y_daily_validated.parquet` khi Quality Gate đạt. Hiện có các blocker về calendar, volume, date alignment và chính sách điều chỉnh; chưa sử dụng dữ liệu này để chạy lại Phase 2. Xem [hướng dẫn Data Loader](src/data_loader/README.md) để biết cách lưu báo cáo và giới hạn của audit.

Chế độ nghiên cứu riêng, với các giới hạn chất lượng được ghi rõ:

```powershell
python -m src.data_loader build --profile research
python -m scripts.run_research_phase2
```

Các lệnh tạo file Research mới và không ghi đè dataset cũ. EDA mô tả sử dụng cả giai đoạn test; giá và corporate actions chưa được xác minh point-in-time nên chưa đủ cơ sở cho backtest thực thi.

## Giả định chi phí giao dịch

Các thành phần phí trong `market_microstructure` của `configs/config.yaml` là nguồn cấu hình chung. Ngưỡng cho nhãn Close-to-Close và EDA là tổng gần đúng phí mua + phí bán + thuế bán + hai lần slippage (hiện là 0,50%). Nhãn fixed-hold dùng từng thành phần trong công thức return ròng, nên không hoàn toàn bằng phép trừ tổng gần đúng. Ngưỡng Close-to-Close chỉ là heuristic, không phải PnL giao dịch. Không thêm `round_trip_cost` riêng vào config.

Các dataset và báo cáo đã lưu trước thay đổi này không tự cập nhật; cần tạo lại riêng trước khi dùng nhãn hoặc thống kê mới. Không so sánh trực tiếp kết quả cũ và mới mà không ghi nhận thay đổi giả định.
