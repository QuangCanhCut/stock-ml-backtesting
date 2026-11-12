"""Quy tắc vi cấu trúc thị trường sàn HOSE và biểu phí giao dịch thực tế.

Module này hiện thực hóa WP4.1:
- Biểu phí giao dịch một chiều thực tế (mua 0.20%, bán 0.30%, tổng vòng quay 0.50%).
- Biên độ dao động trần/sàn sàn HOSE (±7%) và ngưỡng nhận diện kẹt thanh khoản (±6.7%).
- Quy tắc chu kỳ thanh toán T+2.5 tại Việt Nam và ánh xạ các mốc thời gian thực thi lệnh.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HOSEMicrostructureRules:
    """Các tham số vi cấu trúc và quy định giao dịch sàn HOSE (Tập đoàn Hòa Phát - HPG).

    Attributes:
        exchange: Tên sàn giao dịch (mặc định 'HOSE').
        trading_days_per_year: Số phiên giao dịch chuẩn mỗi năm (252 phiên).
        daily_price_limit: Biên độ giá trần/sàn theo quy định HOSE (0.07 tương ứng ±7.0%).
        limit_hit_threshold: Ngưỡng nhận diện phiên kẹt trần/sàn (0.067 tương ứng ±6.7%).
        settlement_cycle: Chu kỳ thanh toán theo quy chế VSDC (2.5 ngày).
        commission_buy: Phí môi giới chiều mua (0.0015 tương ứng 0.15%).
        commission_sell: Phí môi giới chiều bán (0.0015 tương ứng 0.15%).
        tax_sell: Thuế TNCN khi bán cổ phiếu (0.0010 tương ứng 0.10%).
        slippage: Trượt giá ước tính mỗi chiều (0.0005 tương ứng 0.05% hay 5 bps).
        initial_capital: Vốn ban đầu mô phỏng (100.000.000 VNĐ).
        risk_free_rate: Lãi suất phi rủi ro tham chiếu năm (0.045 tương ứng 4.5%/năm).
    """

    exchange: str = "HOSE"
    trading_days_per_year: int = 252
    daily_price_limit: float = 0.07
    limit_hit_threshold: float = 0.067
    settlement_cycle: float = 2.5
    commission_buy: float = 0.0015
    commission_sell: float = 0.0015
    tax_sell: float = 0.0010
    slippage: float = 0.0005
    initial_capital: float = 100_000_000.0
    risk_free_rate: float = 0.045

    @property
    def buy_cost_rate(self) -> float:
        """Tỷ lệ chi phí một chiều khi Mua (phí môi giới + trượt giá).

        Returns:
            0.0015 + 0.0005 = 0.0020 (0.20% hay 20 bps).
        """
        return self.commission_buy + self.slippage

    @property
    def sell_cost_rate(self) -> float:
        """Tỷ lệ chi phí một chiều khi Bán (phí môi giới + thuế chuyển nhượng + trượt giá).

        Returns:
            0.0015 + 0.0010 + 0.0005 = 0.0030 (0.30% hay 30 bps).
        """
        return self.commission_sell + self.tax_sell + self.slippage

    @property
    def effective_round_trip_cost(self) -> float:
        """Tổng chi phí thực tế cho một vòng quay giao dịch (Mua và Bán đủ 2 chiều).

        Returns:
            buy_cost_rate + sell_cost_rate = 0.0050 (0.50% hay 50 bps).
        """
        return self.buy_cost_rate + self.sell_cost_rate

    def is_ceiling_locked(self, open_next: float, prev_close: float) -> bool:
        """Kiểm tra xem phiên khớp lệnh có bị kẹt cứng ở mức giá Trần (trắng bên bán) hay không.

        Khi giá mở cửa tăng kịch trần (>= +6.7%), bên mua chất lệnh trần hàng triệu cổ phiếu,
        bên bán không ai bán ra dẫn tới việc không thể khớp lệnh Mua trong thực tế.

        Args:
            open_next: Giá mở cửa tại phiên khớp lệnh (Open t+1).
            prev_close: Giá đóng cửa tại phiên phát tín hiệu (Close t).

        Returns:
            True nếu phiên bị kẹt giá trần, False nếu thanh khoản bình thường.
        """
        if prev_close <= 0.0:
            return False
        return_from_prev = (open_next / prev_close) - 1.0
        return bool(return_from_prev >= self.limit_hit_threshold)

    def is_floor_locked(self, open_next: float, prev_close: float) -> bool:
        """Kiểm tra xem phiên khớp lệnh có bị kẹt cứng ở mức giá Sàn (trắng bên mua) hay không.

        Khi giá mở cửa giảm kịch sàn (<= -6.7%), bên bán bán tháo giá sàn hàng triệu cổ phiếu,
        bên mua không ai mua dẫn tới việc không thể khớp lệnh Bán/Cắt lỗ trong thực tế.

        Args:
            open_next: Giá mở cửa tại phiên khớp lệnh (Open t+1).
            prev_close: Giá đóng cửa tại phiên phát tín hiệu (Close t).

        Returns:
            True nếu phiên bị kẹt giá sàn, False nếu thanh khoản bình thường.
        """
        if prev_close <= 0.0:
            return False
        return_from_prev = (open_next / prev_close) - 1.0
        return bool(return_from_prev <= -self.limit_hit_threshold)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HOSEMicrostructureRules":
        """Khởi tạo luật vi cấu trúc từ từ điển cấu hình dự án (configs/config.yaml).

        Args:
            config: Từ điển chứa cấu hình gốc của dự án.

        Returns:
            Đối tượng HOSEMicrostructureRules chuẩn hóa.
        """
        micro = config.get("market_microstructure", {})
        backtest_cfg = config.get("backtest", {})

        return cls(
            exchange=str(micro.get("exchange", "HOSE")),
            trading_days_per_year=int(micro.get("trading_days_per_year", 252)),
            daily_price_limit=float(micro.get("daily_price_limit", 0.07)),
            limit_hit_threshold=float(micro.get("limit_hit_threshold", 0.067)),
            settlement_cycle=float(micro.get("settlement_cycle", 2.5)),
            commission_buy=float(micro.get("commission_buy", 0.0015)),
            commission_sell=float(micro.get("commission_sell", 0.0015)),
            tax_sell=float(micro.get("tax_sell", 0.0010)),
            slippage=float(micro.get("slippage", 0.0005)),
            initial_capital=float(backtest_cfg.get("initial_capital", 100_000_000.0)),
            risk_free_rate=float(backtest_cfg.get("risk_free_rate", 0.045)),
        )

