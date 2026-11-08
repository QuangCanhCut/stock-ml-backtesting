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
