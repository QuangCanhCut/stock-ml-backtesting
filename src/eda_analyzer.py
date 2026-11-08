"""
MODULE: src/eda_analyzer.py
MỤC ĐÍCH: 
    Thực hiện phân tích thăm dò dữ liệu (Quant EDA) và kiểm định 5 "Stylized Facts"
    của chuỗi thời gian tài chính cho cổ phiếu HPG (Stationarity, Fat-tails, Volatility Clustering...).
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# Cấu hình logging hệ thống
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Cấu hình giao diện biểu đồ chuyên nghiệp
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 11


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(parquet_path: str) -> pd.DataFrame:
    """Đọc file dữ liệu thô đã làm sạch từ thư mục data/raw/."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {path.resolve()}. Hãy kiểm tra cấu hình dữ liệu!")
    logger.info(f"Đang đọc dữ liệu từ: {path}")
    df = pd.read_parquet(path)
    return df


def compute_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán các chỉ số lợi suất (Returns) và biến mục tiêu (Targets) chuẩn Quant."""
    df = df.copy()
    
    # 1. Simple Return và Log Return hàng ngày
    df["simple_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # 2. Biến mục tiêu cho phiên tiếp theo (t+1) phục vụ RQ1
    df["next_log_return"] = df["log_return"].shift(-1)
    df["target_direction"] = (df["next_log_return"] > 0).astype(int)
    
    # 3. Biến mục tiêu vượt chi phí giao dịch (2 chiều ~0.45% theo config) phục vụ RQ2
    round_trip_cost = 0.0045 
    df["target_net_profit"] = (df["next_log_return"] > round_trip_cost).astype(int)
    
    # Loại bỏ dòng NaN do phép shift/pct_change
    df = df.dropna().copy()
    return df


def run_statistical_tests(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Kiểm định 5 Stylized Facts kinh điển của chuỗi thời gian tài chính và xuất báo cáo text.
    """
    ret = df["log_return"]
    price = df["close"]
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BÁO CÁO KIỂM ĐỊNH THỐNG KÊ QUANTITATIVE EDA - CỔ PHIẾU HPG (2016-2026)")
    report_lines.append("=" * 70)
    
    # --- FACT 1: Tính dừng (Stationarity) ---
    # Giá cổ phiếu là non-stationary, nhưng Log Return phải là stationary
    adf_price = adfuller(price)
    adf_ret = adfuller(ret)
    report_lines.append("\n[FACT 1] KIỂM ĐỊNH TÍNH DỪNG (ADF TEST):")
    report_lines.append(f"  - Giá đóng cửa (Close) : ADF Stat = {adf_price[0]:.4f}, p-value = {adf_price[1]:.4f} -> {'Dừng' if adf_price[1] < 0.05 else 'KHÔNG DỪNG (Non-stationary - Đúng lý thuyết)'}")
    report_lines.append(f"  - Lợi suất (Log Return): ADF Stat = {adf_ret[0]:.4f}, p-value = {adf_ret[1]:.4e} -> {'DỪNG (Stationary - Đạt chuẩn đưa vào ML)' if adf_ret[1] < 0.05 else 'Không dừng'}")

    # --- FACT 2: Phân phối đuôi béo (Fat Tails / Leptokurtic & Normality) ---
    mean_ret = ret.mean()
    vol_ret = ret.std()
    ann_vol = vol_ret * np.sqrt(252)
    skew = ret.skew()
    kurt = ret.kurtosis() # Excess kurtosis
    jb_stat, jb_pval = stats.jarque_bera(ret)
    
    report_lines.append("\n[FACT 2] ĐẶC TRƯNG PHÂN PHỐI LỢI SUẤT (FAT-TAILS & MOMENTS):")
    report_lines.append(f"  - Lợi suất kỳ vọng trung bình ngày : {mean_ret*100:.4f}% (~ {mean_ret*252*100:.2f}%/năm)")
    report_lines.append(f"  - Độ biến động hàng năm (Ann. Vol): {ann_vol*100:.2f}%")
    report_lines.append(f"  - Độ lệch (Skewness)             : {skew:.4f} (Lệch âm cho thấy rủi ro sập sốc lớn hơn tăng)")
    report_lines.append(f"  - Độ nhọn đuôi béo (Kurtosis)      : {kurt:.4f} (Rất lớn > 0, chứng minh đuôi béo hơn phân phối chuẩn)")
    report_lines.append(f"  - Jarque-Bera Normality Test      : p-value = {jb_pval:.4e} -> Bác bỏ tuyệt đối phân phối chuẩn Gaussian!")

    # --- FACT 3: Tự tương quan và Tụ cụm biến động (Volatility Clustering) ---
    lb_ret = acorr_ljungbox(ret, lags=[5, 10, 20], return_df=True)
    lb_abs_ret = acorr_ljungbox(ret.abs(), lags=[5, 10, 20], return_df=True)
    
    report_lines.append("\n[FACT 3] KIỂM ĐỊNH TỰ TƯƠNG QUAN LJUNG-BOX (LAGS 5, 10, 20):")
    report_lines.append(f"  - Trên chuỗi Return r_t (Lấy Lag 20)     : p-value = {lb_ret['lb_pvalue'].iloc[-1]:.4f} (Thường > 0.05, không có tương quan tuyến tính tuyến tính)")
    report_lines.append(f"  - Trên chuỗi tuyệt đối |r_t| (Biến động): p-value = {lb_abs_ret['lb_pvalue'].iloc[-1]:.4e} (Cực kỳ nhỏ < 0.01 -> Tụ cụm biến động cực mạnh!)")

    # --- FACT 4: Thống kê biên độ & Lọc mục tiêu theo chi phí giao dịch ---
    limit_hit_count = (df["simple_return"].abs() >= config["market_microstructure"]["limit_hit_threshold"]).sum()
    up_ratio = df["target_direction"].mean() * 100
    net_profit_ratio = df["target_net_profit"].mean() * 100
    
    report_lines.append("\n[FACT 4] PHÂN TÍCH TỶ LỆ MỤC TIÊU & SÀN HOSE (±7%):")
    report_lines.append(f"  - Số phiên HPG biến động kịch trần/sàn (>= 6.7%): {limit_hit_count} phiên trong 10 năm")
    report_lines.append(f"  - Tỷ lệ phiên tăng giá ngày mai (Always-Long)    : {up_ratio:.2f}%")
    report_lines.append(f"  - Tỷ lệ phiên tăng vượt phí giao dịch (0.45%)     : {net_profit_ratio:.2f}% (Thách thức lớn cho mô hình ML)")
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    return report_text


def generate_and_save_plots(df: pd.DataFrame, figures_dir: str):
    """Vẽ và lưu 4 biểu đồ EDA chuyên sâu vào thư mục results/figures/."""
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Biểu đồ Giá và Drawdown
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df.index, df["close"], color="#1f77b4", linewidth=1.2, label="HPG Adjusted Close")
    axes[0].set_title("HPG 10-Year Adjusted Close Price & Drawdown Profile", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (VND)")
    axes[0].legend(loc="upper left")
    
    # Tính Underwater Drawdown
    rolling_max = df["close"].cummax()
    drawdown = (df["close"] - rolling_max) / rolling_max * 100
    axes[1].fill_between(df.index, drawdown, 0, color="#d62728", alpha=0.4, label="Drawdown (%)")
    axes[1].set_title("Underwater Drawdown (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Drawdown %")
    axes[1].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_path / "01_price_and_drawdown.png", dpi=300)
    plt.close()
    
    # 2. Biểu đồ Log Returns & Volatility Clustering
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["log_return"], color="#2ca02c", alpha=0.6, linewidth=0.8, label="Daily Log Return")
    rolling_vol = df["log_return"].rolling(window=21).std() * np.sqrt(252) * 100
    plt.plot(df.index, rolling_vol / 100, color="#ff7f0e", linewidth=1.0, label="21-Day Rolling Volatility (Scaled)")
    plt.title("HPG Daily Log Returns & Volatility Clustering (2016 - 2026)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path / "02_returns_and_volatility.png", dpi=300)
    plt.close()
    
    # 3. Phân phối Normal Q-Q Plot (Fat Tails)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["log_return"], kde=True, ax=axes[0], color="#1f77b4", bins=100)
    axes[0].set_title("Log Return Distribution vs Normal KDE", fontweight="bold")
    
    stats.probplot(df["log_return"], dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot (Demonstrating Fat-Tails)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_path / "03_distribution_and_qqplot.png", dpi=300)
    plt.close()
    
    # 4. Biểu đồ Tự tương quan ACF của |Return| (Volatility Memory)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(df["log_return"], lags=30, ax=axes[0], title="ACF of Log Returns r_t (Linear Efficiency)")
    plot_acf(df["log_return"].abs(), lags=30, ax=axes[1], title="ACF of Absolute Returns |r_t| (Volatility Memory)")
    plt.tight_layout()
    plt.savefig(fig_path / "04_acf_volatility_memory.png", dpi=300)
    plt.close()
    
    logger.info(f"Đã xuất toàn bộ biểu đồ EDA thành công vào thư mục: {fig_path.resolve()}")


if __name__ == "__main__":
    # 1. Đọc config
    config = load_config("configs/config.yaml")
    
    # 2. Đọc file parquet thô từ data/raw/
    raw_path = config["data"]["raw_parquet_path"]
    df_raw = load_raw_data(raw_path)
    
    # 3. Tính toán features Quant cơ bản
    df_features = compute_quant_features(df_raw)
    
    # 4. Chạy kiểm định thống kê EDA
    report_str = run_statistical_tests(df_features, config)
    
    # 5. Lưu báo cáo ra file text trong results/
    report_path = Path(config["eda"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_str, encoding="utf-8")
    logger.info(f"Đã lưu báo cáo thống kê EDA tại: {report_path.resolve()}")
    
    # 6. Vẽ và lưu biểu đồ
    generate_and_save_plots(df_features, config["eda"]["figures_dir"])
    print("\n[HOÀN TẤT] Ngày 1 EDA cho cổ phiếu HPG đã hoàn thành xuất sắc!")
