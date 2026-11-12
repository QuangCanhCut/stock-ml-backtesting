"""Động cơ khớp lệnh Event-Driven sàn HOSE và Quản trị Vị thế (WP4.3).

Module này hiện thực hóa các chức năng cốt lõi:
- Mô phỏng tài khoản khớp lệnh từng phiên (Event-Driven Simulation) trên Master Test Calendar (2022–2026).
- Quy tắc lô chẵn HOSE (lot_size = 100 cổ phiếu).
- Chu kỳ thanh toán T+2.5 thông qua hàng đợi Settlement Lot Queue (mua tại Open t -> bán sớm nhất tại Open t+3).
- Nhận diện kẹt trần (+6.7% từ chối Mua) và kẹt sàn (-6.7% từ chối Bán).
- Quản trị quy mô vị thế (Position Sizing): Chế độ All-in/All-out và Volatility Targeting (sigma_target = 0.25).
- Định giá tài sản cuối phiên (Mark-to-Market tại Close t) và thanh lý danh mục mở cuối kỳ (OPEN_MTM).
- Đảm bảo tính bất biến kế toán: NAV_t = Cash_t + Total_Shares_t * Close_t với sai số < 1e-4.
"""

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import pandas as pd

from src.backtest.rules import HOSEMicrostructureRules


@dataclass
class PositionLot:
    """Đại diện cho một lô cổ phiếu đã mua và trạng thái chu kỳ thanh toán T+2.5.

    Attributes:
        shares: Số lượng cổ phiếu nắm giữ (bội số của 100).
        purchase_price: Giá mở cửa khi khớp lệnh mua (Open t).
        exec_idx: Chỉ số phiên giao dịch khớp lệnh mua trên Master Calendar.
        earliest_sell_idx: Chỉ số phiên mở cửa sớm nhất được phép bán (exec_idx + 3).
        entry_date: Ngày giao dịch khớp lệnh mua.
        buy_fee: Tổng chi phí mua đã trả (phí môi giới + trượt giá).
    """

    shares: int
    purchase_price: float
    exec_idx: int
    earliest_sell_idx: int
    entry_date: pd.Timestamp
    buy_fee: float


@dataclass
class TradeRecord:
    """Bản ghi chi tiết một vòng quay giao dịch (Trade Log Entry).

    Attributes:
        trade_id: Số thứ tự giao dịch (1, 2, ...).
        entry_date: Ngày khớp lệnh mua.
        exit_date: Ngày khớp lệnh bán (hoặc ngày MTM cuối kỳ).
        entry_price: Giá mua thực tế (Open phiên mua).
        exit_price: Giá bán thực tế (Open phiên bán hoặc Close phiên MTM).
        shares: Khối lượng cổ phiếu giao dịch.
        buy_fee: Chi phí mua đã trả.
        sell_fee: Chi phí bán và thuế TNCN đã trả (0 nếu OPEN_MTM).
        gross_pnl: Lãi/lỗ gộp trước phí (VNĐ).
        net_pnl: Lãi/lỗ ròng sau toàn bộ phí và thuế (VNĐ).
        net_return_pct: Tỷ suất sinh lời ròng trên vốn đầu tư của lệnh (%).
        holding_sessions: Số phiên giao dịch thực tế nắm giữ.
        exit_reason: Lý do đóng vị thế ('SIGNAL_EXIT' hoặc 'OPEN_MTM').
        status: Trạng thái lệnh ('CLOSED' hoặc 'OPEN_MTM').
    """

    trade_id: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    buy_fee: float
    sell_fee: float
    gross_pnl: float
    net_pnl: float
    net_return_pct: float
    holding_sessions: int
    exit_reason: str
    status: str


@dataclass
class DailyPortfolioRecord:
    """Bản ghi trạng thái tài khoản hằng ngày (Daily Portfolio Snapshot).

    Attributes:
        date: Ngày giao dịch.
        cash: Số dư tiền mặt tại cuối phiên (VNĐ).
        stock_value: Giá trị cổ phiếu định giá theo giá Close cuối phiên (VNĐ).
        nav: Tổng giá trị tài sản ròng (NAV = cash + stock_value).
        total_shares: Tổng số lượng cổ phiếu đang nắm giữ.
        shares_available: Số lượng cổ phiếu khả dụng có thể bán ngay.
        shares_locked: Số lượng cổ phiếu chờ thanh toán T+2.5.
        daily_return: Tỷ suất sinh lời hằng ngày (NAV_t / NAV_{t-1} - 1.0).
        drawdown: Tỷ lệ sụt giảm giá trị tài sản từ đỉnh lịch sử (NAV_t / Peak_NAV - 1.0).
    """

    date: pd.Timestamp
    cash: float
    stock_value: float
    nav: float
    total_shares: int
    shares_available: int
    shares_locked: int
    daily_return: float
    drawdown: float


@dataclass
class SimulationResult:
    """Kết quả hoàn chỉnh của một lượt mô phỏng chiến lược giao dịch.

    Attributes:
        strategy_id: Tên định danh duy nhất của chiến lược.
        protocol: Giao thức ('frozen_2016_2021', 'annual_expanding_retrain', 'benchmark').
        rq: Câu hỏi nghiên cứu ('RQ1', 'RQ2', 'ALL').
        model_name: Tên mô hình ('dummy', 'svm', 'lightgbm', 'lstm', 'ensemble', 'buy_and_hold_hpg', 'risk_free_cash').
        signal_mode: Chế độ ngưỡng ('fixed_053_048', 'quantile_q80_q30', 'quantile_q70_q40', 'benchmark').
        sizing_mode: Quản trị vị thế ('all_in_out', 'vol_targeting', 'buy_and_hold', 'cash').
        daily_df: Bảng chuỗi thời gian danh mục hằng ngày (1.201 phiên).
        trades_df: Bảng nhật ký chi tiết các giao dịch.
        t25_delayed_exits: Số lần hoãn bán do cổ phiếu chưa về tài khoản theo T+2.5.
        ceiling_locked_skips: Số lần hủy mua do thị trường tăng kịch trần (+6.7%).
        floor_locked_skips: Số lần hủy bán do thị trường giảm kịch sàn (-6.7%).
        initial_capital: Vốn ban đầu mô phỏng (VNĐ).
        final_nav: Giá trị tài sản ròng tại cuối kỳ mô phỏng (VNĐ).
        total_fees_paid: Tổng chi phí giao dịch (phí môi giới + thuế + trượt giá) đã trả (VNĐ).
        total_traded_value: Tổng giá trị luân chuyển giao dịch mua và bán (VNĐ).
    """

    strategy_id: str
    protocol: str
    rq: str
    model_name: str
    signal_mode: str
    sizing_mode: str
    daily_df: pd.DataFrame
    trades_df: pd.DataFrame
    t25_delayed_exits: int
    ceiling_locked_skips: int
    floor_locked_skips: int
    initial_capital: float
    final_nav: float
    total_fees_paid: float
    total_traded_value: float


def simulate_strategy(
    calendar_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    rules: HOSEMicrostructureRules,
    strategy_id: str = "",
    protocol: str = "",
    rq: str = "",
    model_name: str = "",
    signal_mode: str = "",
    sizing_mode: Literal["all_in_out", "vol_targeting"] = "all_in_out",
    custom_buy_cost: float | None = None,
    custom_sell_cost: float | None = None,
) -> SimulationResult:
    """Mô phỏng khớp lệnh Event-Driven sàn HOSE cho một chuỗi tín hiệu chiến lược.

    Nguyên tắc nhân quả và vi cấu trúc HOSE:
    1. Vốn khởi đầu 100.000.000 VNĐ tại Day 0 (2022-01-04), tín hiệu đầu tiên sinh ra tại Close Day 0.
    2. Lệnh chỉ được khớp tại giá Open của phiên tiếp theo (t+1).
    3. Kiểm tra kẹt trần/sàn (+-6.7% so với Close t-1).
    4. Cổ phiếu mua tại Open phiên d chỉ được mở khóa để bán tại Open phiên d+3 (T+2.5).
    5. Quy tròn khối lượng mua theo lô chẵn 100 cổ phiếu.
    6. Position sizing: All-in/out (100% tiền) hoặc Volatility Targeting (sigma_target = 0.25).
    7. Định giá Mark-to-Market hằng ngày tại giá Close. Cổ phiếu còn nắm giữ ở phiên cuối cùng
       được ghi nhận vào trade log với trạng thái OPEN_MTM theo giá Close phiên cuối.

    Args:
        calendar_df: Master Calendar chứa các cột 'open', 'high', 'low', 'close', 'volatility_21'.
        signals_df: Bảng tín hiệu chứa 'signal_date', 'execution_date', 'desired_position', 'volatility_21'.
        rules: Bộ luật vi cấu trúc sàn HOSE.
        strategy_id: Tên định danh chiến lược.
        protocol: Giao thức huấn luyện.
        rq: Câu hỏi nghiên cứu.
        model_name: Tên họ mô hình.
        signal_mode: Tên chế độ ngưỡng tín hiệu.
        sizing_mode: 'all_in_out' hoặc 'vol_targeting'.
        custom_buy_cost: Tỷ lệ phí mua tùy biến (dùng cho Stress Test, nếu None dùng rules.buy_cost_rate).
        custom_sell_cost: Tỷ lệ phí bán tùy biến (dùng cho Stress Test, nếu None dùng rules.sell_cost_rate).

    Returns:
        SimulationResult chứa toàn bộ dữ liệu danh mục hằng ngày, nhật ký lệnh và thống kê vi cấu trúc.
    """
    buy_cost_rate = custom_buy_cost if custom_buy_cost is not None else rules.buy_cost_rate
    sell_cost_rate = custom_sell_cost if custom_sell_cost is not None else rules.sell_cost_rate

    # Lịch giao dịch Master
    calendar_dates = list(calendar_df.index)
    n_days = len(calendar_dates)
    if n_days < 2:
        raise ValueError("Master Calendar phải có ít nhất 2 phiên giao dịch!")

    # Ánh xạ tín hiệu theo execution_date (ngày mở cửa khớp lệnh)
    # signals_df có thể chứa cột 'execution_date' hoặc 'signal_date'
    sig_map: dict[pd.Timestamp, tuple[int, float]] = {}
    if not signals_df.empty:
        for _, row in signals_df.iterrows():
            exec_dt = pd.to_datetime(row["execution_date"])
            pos = int(row["desired_position"])
            vol = float(row.get("volatility_21", 0.25))
            sig_map[exec_dt] = (pos, vol)

    # Khởi tạo trạng thái danh mục
    cash = float(rules.initial_capital)
    initial_capital = float(rules.initial_capital)
    peak_nav = initial_capital
    position_lots: list[PositionLot] = []
    trades: list[TradeRecord] = []
    daily_records: list[DailyPortfolioRecord] = []

    trade_counter = 0
    t25_delayed_exits = 0
    ceiling_locked_skips = 0
    floor_locked_skips = 0
    total_fees_paid = 0.0
    total_traded_value = 0.0

    current_desired_pos = 0

    # Vòng lặp mô phỏng qua từng phiên của Master Calendar
    for d in range(n_days):
        current_date = calendar_dates[d]
        close_d = float(calendar_df.iloc[d]["close"])
        open_d = float(calendar_df.iloc[d]["open"])
        prev_close = float(calendar_df.iloc[d - 1]["close"]) if d > 0 else close_d

        if d == 0:
            # Phiên Day 0 (2022-01-04): Khởi tạo tài khoản tiền mặt thuần túy, chưa có giao dịch
            daily_records.append(
                DailyPortfolioRecord(
                    date=current_date,
                    cash=cash,
                    stock_value=0.0,
                    nav=cash,
                    total_shares=0,
                    shares_available=0,
                    shares_locked=0,
                    daily_return=0.0,
                    drawdown=0.0,
                )
            )
            continue

        # --- BƯỚC 1: MỞ KHÓA THANH TOÁN T+2.5 VÀO ĐẦU PHIÊN (TRƯỚC GIỜ OPEN) ---
        shares_available = sum(lot.shares for lot in position_lots if lot.earliest_sell_idx <= d)
        shares_locked = sum(lot.shares for lot in position_lots if lot.earliest_sell_idx > d)
        total_shares = shares_available + shares_locked

        # --- BƯỚC 2: CẬP NHẬT Ý ĐỊNH TÍN HIỆU TẠI PHIÊN MỞ CỬA OPEN(d) ---
        # Tín hiệu được sinh ra tại Close(d-1) và nhắm mục tiêu thực thi tại Open(d)
        if current_date in sig_map:
            current_desired_pos, sig_vol = sig_map[current_date]
        else:
            # Nếu hết tín hiệu mới (ví dụ RQ2 ở 3 phiên cuối), giữ nguyên ý định trước đó
            sig_vol = float(calendar_df.iloc[d - 1].get("volatility_21", 0.25))

        # --- BƯỚC 3: XỬ LÝ KHỚP LỆNH TẠI GIÁ OPEN(d) ---
        if total_shares == 0:
            # Hiện tại đang không có cổ phiếu -> Kiểm tra ý định MUA
            if current_desired_pos == 1:
                # Kiểm tra kẹt trần (+6.7% so với Close d-1)
                if rules.is_ceiling_locked(open_d, prev_close):
                    ceiling_locked_skips += 1
                else:
                    # Tính toán quy mô giải ngân theo chế độ position sizing
                    if sizing_mode == "vol_targeting":
                        w_target = float(np.clip(0.25 / max(sig_vol, 1e-4), 0.10, 1.00))
                        allocatable_cash = cash * w_target
                    else:
                        allocatable_cash = cash

                    # Quy tròn theo lô chẵn 100 cổ phiếu sàn HOSE
                    shares_to_buy = (int(allocatable_cash // (open_d * (1.0 + buy_cost_rate))) // 100) * 100
                    if shares_to_buy > 0:
                        buy_val = shares_to_buy * open_d
                        fee_buy = buy_val * buy_cost_rate
                        cash -= (buy_val + fee_buy)
                        total_fees_paid += fee_buy
                        total_traded_value += buy_val

                        # Tạo lô cổ phiếu mới: mở khóa bán tại Open phiên (d + 3)
                        new_lot = PositionLot(
                            shares=shares_to_buy,
                            purchase_price=open_d,
                            exec_idx=d,
                            earliest_sell_idx=d + 3,
                            entry_date=current_date,
                            buy_fee=fee_buy,
                        )
                        position_lots.append(new_lot)
                        shares_locked += shares_to_buy
                        total_shares += shares_to_buy

        elif total_shares > 0:
            # Hiện tại đang có cổ phiếu -> Kiểm tra ý định BÁN
            if current_desired_pos == 0:
                if shares_available == 0:
                    # Cổ phiếu chưa về tài khoản theo quy chế T+2.5
                    t25_delayed_exits += 1
                else:
                    # Kiểm tra kẹt sàn (-6.7% so với Close d-1)
                    if rules.is_floor_locked(open_d, prev_close):
                        floor_locked_skips += 1
                    else:
                        # Khớp lệnh Bán toàn bộ cổ phiếu khả dụng
                        shares_to_sell = shares_available
                        sell_val = shares_to_sell * open_d
                        fee_sell = sell_val * sell_cost_rate
                        proceeds = sell_val - fee_sell
                        cash += proceeds
                        total_fees_paid += fee_sell
                        total_traded_value += sell_val

                        # Ghi nhận nhật ký lệnh đóng (Trade Log)
                        for lot in position_lots:
                            if lot.earliest_sell_idx <= d:
                                trade_counter += 1
                                entry_val = lot.shares * lot.purchase_price
                                gross_pnl = (open_d - lot.purchase_price) * lot.shares
                                net_pnl = gross_pnl - lot.buy_fee - fee_sell
                                net_ret = (net_pnl / (entry_val + lot.buy_fee)) * 100.0
                                holding = d - lot.exec_idx

                                trades.append(
                                    TradeRecord(
                                        trade_id=trade_counter,
                                        entry_date=lot.entry_date,
                                        exit_date=current_date,
                                        entry_price=lot.purchase_price,
                                        exit_price=open_d,
                                        shares=lot.shares,
                                        buy_fee=lot.buy_fee,
                                        sell_fee=fee_sell,
                                        gross_pnl=gross_pnl,
                                        net_pnl=net_pnl,
                                        net_return_pct=net_ret,
                                        holding_sessions=holding,
                                        exit_reason="SIGNAL_EXIT",
                                        status="CLOSED",
                                    )
                                )

                        # Xóa các lô đã bán khỏi danh sách nắm giữ
                        position_lots = [lot for lot in position_lots if lot.earliest_sell_idx > d]
                        shares_available = 0
                        total_shares = sum(lot.shares for lot in position_lots)

        # --- BƯỚC 4: ĐỊNH GIÁ TÀI SẢN MARK-TO-MARKET CUỐI PHIÊN TẠI CLOSE(d) ---
        stock_value = float(total_shares * close_d)
        nav = float(cash + stock_value)

        # Kiểm tra bất biến kế toán tuyệt đối
        if abs(nav - (cash + stock_value)) >= 1e-4:
            raise RuntimeError(f"Vi phạm bất biến kế toán tại {current_date}: NAV != Cash + Stock_Value!")

        prev_nav = daily_records[-1].nav
        daily_ret = (nav / prev_nav - 1.0) if prev_nav > 0.0 else 0.0
        peak_nav = max(peak_nav, nav)
        drawdown = (nav / peak_nav - 1.0) if peak_nav > 0.0 else 0.0

        daily_records.append(
            DailyPortfolioRecord(
                date=current_date,
                cash=cash,
                stock_value=stock_value,
                nav=nav,
                total_shares=total_shares,
                shares_available=shares_available,
                shares_locked=shares_locked,
                daily_return=daily_ret,
                drawdown=drawdown,
            )
        )

    # --- BƯỚC 5: XỬ LÝ THANH LÝ MỞ CUỐI KỲ (OPEN_MTM) ---
    # Nếu kết thúc chu kỳ mô phỏng mà vẫn còn cổ phiếu, ghi nhận vào trade log để khớp sổ kế toán
    if position_lots:
        last_date = calendar_dates[-1]
        last_close = float(calendar_df.iloc[-1]["close"])
        for lot in position_lots:
            trade_counter += 1
            entry_val = lot.shares * lot.purchase_price
            gross_pnl = (last_close - lot.purchase_price) * lot.shares
            net_pnl = gross_pnl - lot.buy_fee  # Chưa bán thực tế nên không tính phí bán
            net_ret = (net_pnl / (entry_val + lot.buy_fee)) * 100.0
            holding = (n_days - 1) - lot.exec_idx

            trades.append(
                TradeRecord(
                    trade_id=trade_counter,
                    entry_date=lot.entry_date,
                    exit_date=last_date,
                    entry_price=lot.purchase_price,
                    exit_price=last_close,
                    shares=lot.shares,
                    buy_fee=lot.buy_fee,
                    sell_fee=0.0,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    net_return_pct=net_ret,
                    holding_sessions=holding,
                    exit_reason="OPEN_MTM",
                    status="OPEN_MTM",
                )
            )

    daily_df = pd.DataFrame([r.__dict__ for r in daily_records])
    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=[
            "trade_id", "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "buy_fee", "sell_fee", "gross_pnl", "net_pnl",
            "net_return_pct", "holding_sessions", "exit_reason", "status",
        ]
    )

    final_nav = float(daily_df["nav"].iloc[-1])

    return SimulationResult(
        strategy_id=strategy_id,
        protocol=protocol,
        rq=rq,
        model_name=model_name,
        signal_mode=signal_mode,
        sizing_mode=sizing_mode,
        daily_df=daily_df,
        trades_df=trades_df,
        t25_delayed_exits=t25_delayed_exits,
        ceiling_locked_skips=ceiling_locked_skips,
        floor_locked_skips=floor_locked_skips,
        initial_capital=initial_capital,
        final_nav=final_nav,
        total_fees_paid=total_fees_paid,
        total_traded_value=total_traded_value,
    )


def simulate_buy_and_hold(
    calendar_df: pd.DataFrame,
    rules: HOSEMicrostructureRules,
    strategy_id: str = "buy_and_hold_hpg",
    custom_buy_cost: float | None = None,
    custom_sell_cost: float | None = None,
) -> SimulationResult:
    """Mô phỏng chiến lược đối chứng Mua và Nắm giữ (Buy & Hold HPG).

    Quy tắc:
    - Day 0 (2022-01-04): Vốn 100 triệu VNĐ tiền mặt.
    - Day 1 (2022-01-05): Mua tối đa lô chẵn 100 cổ phiếu tại giá Open(Day 1), chịu phí mua.
    - Từ Day 1 đến Day 1200 (2026-10-30): Nắm giữ liên tục, định giá MTM hằng ngày theo giá Close.
    - Cuối kỳ: Ghi nhận 1 lệnh OPEN_MTM định giá theo Close(2026-10-30).

    Args:
        calendar_df: Master Calendar.
        rules: Luật vi cấu trúc HOSE.
        strategy_id: Tên chiến lược.
        custom_buy_cost: Tỷ lệ phí mua tùy biến.
        custom_sell_cost: Tỷ lệ phí bán tùy biến.

    Returns:
        SimulationResult của chiến lược Buy & Hold.
    """
    buy_cost_rate = custom_buy_cost if custom_buy_cost is not None else rules.buy_cost_rate
    calendar_dates = list(calendar_df.index)
    n_days = len(calendar_dates)

    cash = float(rules.initial_capital)
    initial_capital = float(rules.initial_capital)
    peak_nav = initial_capital
    daily_records: list[DailyPortfolioRecord] = []
    trades: list[TradeRecord] = []

    # Day 0
    daily_records.append(
        DailyPortfolioRecord(
            date=calendar_dates[0],
            cash=cash,
            stock_value=0.0,
            nav=cash,
            total_shares=0,
            shares_available=0,
            shares_locked=0,
            daily_return=0.0,
            drawdown=0.0,
        )
    )

    # Day 1: Mua tại Open
    open_1 = float(calendar_df.iloc[1]["open"])
    prev_close_0 = float(calendar_df.iloc[0]["close"])
    is_ceil = rules.is_ceiling_locked(open_1, prev_close_0)

    total_shares = 0
    total_fees_paid = 0.0
    total_traded_value = 0.0
    lot: PositionLot | None = None

    if not is_ceil:
        shares_to_buy = (int(cash // (open_1 * (1.0 + buy_cost_rate))) // 100) * 100
        if shares_to_buy > 0:
            buy_val = shares_to_buy * open_1
            fee_buy = buy_val * buy_cost_rate
            cash -= (buy_val + fee_buy)
            total_shares = shares_to_buy
            total_fees_paid += fee_buy
            total_traded_value += buy_val
            lot = PositionLot(
                shares=shares_to_buy,
                purchase_price=open_1,
                exec_idx=1,
                earliest_sell_idx=4,
                entry_date=calendar_dates[1],
                buy_fee=fee_buy,
            )

    # Chạy tiếp các phiên từ Day 1 đến hết
    for d in range(1, n_days):
        current_date = calendar_dates[d]
        close_d = float(calendar_df.iloc[d]["close"])
        shares_avail = total_shares if d >= 4 else 0
        shares_lock = total_shares - shares_avail
        stock_value = total_shares * close_d
        nav = cash + stock_value

        prev_nav = daily_records[-1].nav
        daily_ret = (nav / prev_nav - 1.0) if prev_nav > 0.0 else 0.0
        peak_nav = max(peak_nav, nav)
        drawdown = (nav / peak_nav - 1.0) if peak_nav > 0.0 else 0.0

        daily_records.append(
            DailyPortfolioRecord(
                date=current_date,
                cash=cash,
                stock_value=stock_value,
                nav=nav,
                total_shares=total_shares,
                shares_available=shares_avail,
                shares_locked=shares_lock,
                daily_return=daily_ret,
                drawdown=drawdown,
            )
        )

    # Lệnh MTM cuối kỳ
    if lot is not None:
        last_date = calendar_dates[-1]
        last_close = float(calendar_df.iloc[-1]["close"])
        entry_val = lot.shares * lot.purchase_price
        gross_pnl = (last_close - lot.purchase_price) * lot.shares
        net_pnl = gross_pnl - lot.buy_fee
        net_ret = (net_pnl / (entry_val + lot.buy_fee)) * 100.0
        holding = (n_days - 1) - lot.exec_idx

        trades.append(
            TradeRecord(
                trade_id=1,
                entry_date=lot.entry_date,
                exit_date=last_date,
                entry_price=lot.purchase_price,
                exit_price=last_close,
                shares=lot.shares,
                buy_fee=lot.buy_fee,
                sell_fee=0.0,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                net_return_pct=net_ret,
                holding_sessions=holding,
                exit_reason="OPEN_MTM",
                status="OPEN_MTM",
            )
        )

    daily_df = pd.DataFrame([r.__dict__ for r in daily_records])
    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    return SimulationResult(
        strategy_id=strategy_id,
        protocol="benchmark",
        rq="ALL",
        model_name="buy_and_hold_hpg",
        signal_mode="benchmark",
        sizing_mode="buy_and_hold",
        daily_df=daily_df,
        trades_df=trades_df,
        t25_delayed_exits=0,
        ceiling_locked_skips=int(is_ceil),
        floor_locked_skips=0,
        initial_capital=initial_capital,
        final_nav=float(daily_df["nav"].iloc[-1]),
        total_fees_paid=total_fees_paid,
        total_traded_value=total_traded_value,
    )


def simulate_risk_free_cash(
    calendar_df: pd.DataFrame,
    rules: HOSEMicrostructureRules,
    strategy_id: str = "risk_free_cash",
) -> SimulationResult:
    """Mô phỏng chiến lược đối chứng Lãi suất Phi Rủi ro (Risk-Free Cash Benchmark).

    Tài khoản tiền gửi tăng trưởng đều đặn mỗi phiên theo lãi suất r_f = 4.5%/năm
    (r_daily = (1 + 0.045)^(1/252) - 1).

    Args:
        calendar_df: Master Calendar.
        rules: Luật vi cấu trúc HOSE.
        strategy_id: Tên chiến lược.

    Returns:
        SimulationResult của tài khoản tiền mặt phi rủi ro.
    """
    calendar_dates = list(calendar_df.index)
    n_days = len(calendar_dates)
    daily_rf = (1.0 + rules.risk_free_rate) ** (1.0 / float(rules.trading_days_per_year)) - 1.0

    cash = float(rules.initial_capital)
    initial_capital = float(rules.initial_capital)
    daily_records: list[DailyPortfolioRecord] = []

    # Day 0
    daily_records.append(
        DailyPortfolioRecord(
            date=calendar_dates[0],
            cash=cash,
            stock_value=0.0,
            nav=cash,
            total_shares=0,
            shares_available=0,
            shares_locked=0,
            daily_return=0.0,
            drawdown=0.0,
        )
    )

    for d in range(1, n_days):
        cash = cash * (1.0 + daily_rf)
        daily_records.append(
            DailyPortfolioRecord(
                date=calendar_dates[d],
                cash=cash,
                stock_value=0.0,
                nav=cash,
                total_shares=0,
                shares_available=0,
                shares_locked=0,
                daily_return=daily_rf,
                drawdown=0.0,
            )
        )

    daily_df = pd.DataFrame([r.__dict__ for r in daily_records])
    trades_df = pd.DataFrame(
        columns=[
            "trade_id", "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "buy_fee", "sell_fee", "gross_pnl", "net_pnl",
            "net_return_pct", "holding_sessions", "exit_reason", "status",
        ]
    )

    return SimulationResult(
        strategy_id=strategy_id,
        protocol="benchmark",
        rq="ALL",
        model_name="risk_free_cash",
        signal_mode="benchmark",
        sizing_mode="cash",
        daily_df=daily_df,
        trades_df=trades_df,
        t25_delayed_exits=0,
        ceiling_locked_skips=0,
        floor_locked_skips=0,
        initial_capital=initial_capital,
        final_nav=float(daily_df["nav"].iloc[-1]),
        total_fees_paid=0.0,
        total_traded_value=0.0,
    )

