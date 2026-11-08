"""
MODULE: src/data_loader/legacy.py
MỤC ĐÍCH: 
    Thu thập, điều chỉnh giá (Adjusted Price), kiểm định chất lượng (Sanity Check) 
    và lưu trữ dữ liệu lịch sử 10 năm của cổ phiếu HPG chuẩn Quantitative Finance.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
import yfinance as yf

# ============================================================================
# 1. CẤU HÌNH LOGGING (NHẬT KÝ HỆ THỐNG)
# ============================================================================
# Thay vì dùng hàm print() đơn điệu, các hệ thống Production dùng logging
# để ghi lại chính xác thời gian (timestamp) và mức độ thông báo (INFO, WARNING, ERROR).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 2. HÀM ĐỌC FILE CẤU HÌNH (CONFIG LOADER)
# ============================================================================
def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Đọc các tham số cài đặt từ file configs/config.yaml.
    Giúp tách biệt hoàn toàn tham số (mã CK, ngày tháng, đường dẫn) khỏi code logic.
    """
    path = Path(config_path)
    
    # Kiểm tra xem file config có tồn tại không trước khi đọc
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path.resolve()}")
    
    # Mở file với bảng mã utf-8 (để đọc được tiếng Việt trong comment của file yaml)
    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    return config_data


# ============================================================================
# 3. TẢI DỮ LIỆU TỪ YAHOO FINANCE (CHUẨN HÓA GIÁ ĐIỀU CHỈNH)
# ============================================================================
def fetch_from_yfinance(yf_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Tải dữ liệu cổ phiếu từ Yahoo Finance (ví dụ: HPG.VN) và tự động tính toán
    Giá điều chỉnh (Adjusted OHLC) để loại bỏ biến động giả do chia cổ tức/tách cổ phiếu.
    """
    logger.info(f"Đang kết nối Yahoo Finance tải mã {yf_ticker} ({start_date} -> {end_date})...")
    
    # auto_adjust=False: Để lấy cả cột 'Close' (giá gốc trên bảng điện) 
    # và cột 'Adj Close' (giá đã điều chỉnh cổ tức/cổ phiếu thưởng).
    df = yf.download(
        yf_ticker, 
        start=start_date, 
        end=end_date, 
        auto_adjust=False, 
        progress=False
    )

    # Kiểm tra nếu tải về bị rỗng (do mất mạng hoặc sai mã cổ phiếu)
    if df is None or df.empty:
        raise ValueError(f"Không tải được dữ liệu cho mã {yf_ticker} từ yfinance.")

    # Xử lý cấu trúc cột nhiều tầng (MultiIndex) trên các phiên bản yfinance mới:
    # Ví dụ: chuyển từ ('Close', 'HPG.VN') về thành 'Close' đơn giản.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ------------------------------------------------------------------------
    # BƯỚC QUAN TRỌNG NHẤT TRONG QUANT: ĐIỀU CHỈNH GIÁ (PRICE ADJUSTMENT)
    # ------------------------------------------------------------------------
    # Giải thích: Khi HPG chia cổ tức bằng cổ phiếu (ví dụ tỷ lệ 30%), giá đóng cửa
    # trên bảng điện ('Close') ngày hôm sau sẽ bị sụt giảm tương ứng ~23%. 
    # Nếu dùng giá chưa điều chỉnh, mô hình ML sẽ tưởng HPG bị bán tháo sập sàn!
    #
    # Công thức: Hệ số điều chỉnh (adj_factor) = Giá đóng cửa điều chỉnh / Giá đóng cửa gốc.
    # Sau đó nhân hệ số này cho toàn bộ Open, High, Low để giữ nguyên tỷ lệ hình học của cây nến.
    if "Adj Close" in df.columns and "Close" in df.columns:
        adj_factor = df["Adj Close"] / df["Close"]
        df["open"] = df["Open"] * adj_factor
        df["high"] = df["High"] * adj_factor
        df["low"] = df["Low"] * adj_factor
        df["close"] = df["Adj Close"]
    else:
        # Dự phòng nếu dữ liệu đã được điều chỉnh sẵn từ trước
        df["open"] = df["Open"]
        df["high"] = df["High"]
        df["low"] = df["Low"]
        df["close"] = df["Close"]

    # Lấy cột khối lượng giao dịch (Volume)
    df["volume"] = df["Volume"]

    # Chuẩn hóa cột thời gian (Index):
    # - Chuyển về kiểu datetime chuẩn của Pandas
    # - tz_localize(None): Xóa thông tin múi giờ (timezone) để dễ dàng ghép nối (join/merge) về sau
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    
    # Sắp xếp dữ liệu theo thứ tự thời gian từ quá khứ đến hiện tại (Bắt buộc để chống Look-ahead bias!)
    df = df.sort_index()

    # Chỉ giữ lại đúng 5 cột tiêu chuẩn (OHLCV) viết thường đồng nhất
    return df[["open", "high", "low", "close", "volume"]]


# ============================================================================
# 4. KIỂM ĐỊNH & LÀM SẠCH DỮ LIỆU (DATA SANITY CHECKS)
# ============================================================================
def clean_and_validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trạm kiểm soát chất lượng dữ liệu trước khi đưa vào phân tích EDA & Machine Learning.
    Thực hiện 4 bước lọc lỗi kinh điển của dữ liệu chứng khoán:
    """
    initial_rows = len(df)

    # Bước 4.1: Loại bỏ các ngày bị lặp (Duplicate timestamps) nếu API trả về thừa
    # keep="last": Giữ lại bản ghi cập nhật mới nhất của ngày hôm đó
    df = df[~df.index.duplicated(keep="last")].copy()

    # Bước 4.2: Loại bỏ các dòng bị khuyết thiếu dữ liệu (NaN / Null)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    
    # Bước 4.3: Loại bỏ các ngày nghỉ lễ Tết của Việt Nam
    # (Đôi khi Yahoo Finance vẫn tạo dòng vào ngày nghỉ lễ của VN nhưng để Volume = 0)
    df = df[(df["volume"] > 0) & (df["close"] > 0)].copy()

    # Bước 4.4: Đảm bảo quy tắc hình học bất biến của nến giá (Candle Consistency)
    # Sau khi nhân hệ số điều chỉnh cổ tức (số thực dấu phẩy động), đôi khi có sai số làm tròn cực nhỏ.
    # Quy tắc bắt buộc: Giá High phải là giá lớn nhất trong phiên, Giá Low phải là giá nhỏ nhất.
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    cleaned_rows = len(df)
    removed_rows = initial_rows - cleaned_rows
    
    logger.info(
        f"Làm sạch hoàn tất: Giữ lại {cleaned_rows}/{initial_rows} phiên giao dịch hợp lệ "
        f"(Đã lọc bỏ {removed_rows} phiên lỗi/nghỉ lễ)."
    )
    logger.info(
        f"Khoảng thời gian dữ liệu thực tế: Từ {df.index.min().strftime('%Y-%m-%d')} "
        f"đến {df.index.max().strftime('%Y-%m-%d')}."
    )
    return df


# ============================================================================
# 5. HÀM ĐIỀU PHỐI CHÍNH (MAIN PIPELINE ORCHESTRATOR)
# ============================================================================
def run_data_pipeline(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Ch?n pipeline c? v? t?ng ghi ?? Raw v? kh?ng c? Quality Gate."""
    raise RuntimeError("Pipeline cũ bị khóa; dùng python -m src.data_loader audit hoặc build.")
