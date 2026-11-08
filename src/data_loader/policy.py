"""Chính sách corporate action và đơn vị trong giai đoạn audit."""

# Cơ sở dữ liệu chưa được xác minh độc lập cho toàn bộ thời gian.
POLICY = {
    "price_adjustment": "unverified_source_specific",
    "volume_units_and_adjustment": "unverified_source_specific",
    "automatic_transforms": False,
    "automatic_date_shift": False,
    "cross_source_ohlcv_merge": False,
    "dataset_export": "blocked_pending_policy_and_verification",
    "features": "Chỉ dùng basis đã xác minh point-in-time; không tự áp dụng adjustment từ tương lai",
    "backtesting": "Cần giá giao dịch và corporate actions có ngày hiệu lực; không mặc định giá adjusted là giá khớp lệnh",
}
