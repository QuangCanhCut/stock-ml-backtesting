"""Kiểm thử đơn vị cho các quy tắc vi cấu trúc sàn HOSE (WP4.1)."""

import pytest

from src.backtest.rules import HOSEMicrostructureRules


def test_hose_transaction_cost_rates() -> None:
    """Kiểm tra biểu phí giao dịch một chiều và hai chiều theo quy định."""
    rules = HOSEMicrostructureRules(
        commission_buy=0.0015,
        commission_sell=0.0015,
        tax_sell=0.0010,
        slippage=0.0005,
    )

    # Phí mua: 0.15% + 0.05% = 0.20%
    assert pytest.approx(rules.buy_cost_rate, abs=1e-6) == 0.0020

    # Phí bán: 0.15% + 0.10% + 0.05% = 0.30%
    assert pytest.approx(rules.sell_cost_rate, abs=1e-6) == 0.0030

    # Tổng chi phí vòng quay: 0.20% + 0.30% = 0.50%
    assert pytest.approx(rules.effective_round_trip_cost, abs=1e-6) == 0.0050


def test_hose_limit_lock_detection() -> None:
    """Kiểm tra logic nhận diện kẹt trần (không mua được) và kẹt sàn (không bán được)."""
    rules = HOSEMicrostructureRules(limit_hit_threshold=0.067)
    prev_close = 20000.0

    # 1. Bình thường (+2.0% và -2.0%)
    assert not rules.is_ceiling_locked(open_next=20400.0, prev_close=prev_close)
    assert not rules.is_floor_locked(open_next=19600.0, prev_close=prev_close)

    # 2. Kẹt giá trần (>= +6.7%, ví dụ tăng trần +6.9% hay +7.0%)
    open_ceil = prev_close * 1.069
    assert rules.is_ceiling_locked(open_next=open_ceil, prev_close=prev_close)
    assert not rules.is_floor_locked(open_next=open_ceil, prev_close=prev_close)

    # 3. Kẹt giá sàn (<= -6.7%, ví dụ giảm sàn -6.8% hay -7.0%)
    open_floor = prev_close * (1.0 - 0.068)
    assert rules.is_floor_locked(open_next=open_floor, prev_close=prev_close)
    assert not rules.is_ceiling_locked(open_next=open_floor, prev_close=prev_close)


def test_hose_rules_from_config() -> None:
    """Kiểm tra khởi tạo HOSEMicrostructureRules từ từ điển config."""
    mock_config = {
        "market_microstructure": {
            "exchange": "HOSE",
            "trading_days_per_year": 252,
            "daily_price_limit": 0.07,
            "limit_hit_threshold": 0.067,
            "settlement_cycle": 2.5,
            "commission_buy": 0.0015,
            "commission_sell": 0.0015,
            "tax_sell": 0.0010,
            "slippage": 0.0005,
        },
        "backtest": {
            "initial_capital": 100_000_000,
            "risk_free_rate": 0.045,
        },
    }

    rules = HOSEMicrostructureRules.from_config(mock_config)
    assert rules.exchange == "HOSE"
    assert rules.initial_capital == 100_000_000.0
    assert rules.risk_free_rate == 0.045
    assert rules.daily_price_limit == 0.07
    assert rules.limit_hit_threshold == 0.067
    assert pytest.approx(rules.effective_round_trip_cost, abs=1e-6) == 0.0050

