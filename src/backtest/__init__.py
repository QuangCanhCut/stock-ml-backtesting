"""Gói module phục vụ Phase 4: Strategy Backtesting & Order Execution.

Module này cung cấp các thành phần:
- rules: Định nghĩa luật vi cấu trúc sàn HOSE, biểu phí giao dịch và chu kỳ thanh toán T+2.5.
- data: Data contract, kiểm toán tính toàn vẹn dữ liệu và căn chỉnh lịch giao dịch.
- signals: Bộ sinh tín hiệu nhân quả (Causal Signal Mapping) và kết hợp đa mô hình (Ensemble).
- engine: Động cơ khớp lệnh Event-Driven HOSE, hàng đợi T+2.5, lô chẵn 100 và position sizing.
- metrics: Thước đo định lượng (Sharpe, Sortino, Max Drawdown, Calmar, Turnover, Yearly Breakdown, Friction Stress Test).
"""

from src.backtest.data import (
    align_predictions_with_market,
    load_dev_oof_seed_buffers,
    load_market_calendar,
    verify_research_artifacts,
)
from src.backtest.engine import (
    DailyPortfolioRecord,
    PositionLot,
    SimulationResult,
    TradeRecord,
    simulate_buy_and_hold,
    simulate_risk_free_cash,
    simulate_strategy,
)
from src.backtest.metrics import (
    compute_strategy_metrics,
    compute_yearly_metrics,
    run_cost_sensitivity_stress_test,
)
from src.backtest.reporting import (
    export_trade_case_studies,
    generate_all_figures,
    plot_cost_sensitivity_and_turnover,
    plot_equity_curves,
    plot_underwater_drawdown,
    plot_yearly_regime_and_rq_comparison,
)
from src.backtest.rules import HOSEMicrostructureRules
from src.backtest.signals import (
    build_all_backtest_signals,
    compute_signal_diagnostics,
    generate_causal_quantile_signals,
    generate_fixed_signals,
)

__all__ = [
    "HOSEMicrostructureRules",
    "verify_research_artifacts",
    "load_market_calendar",
    "align_predictions_with_market",
    "load_dev_oof_seed_buffers",
    "generate_fixed_signals",
    "generate_causal_quantile_signals",
    "build_all_backtest_signals",
    "compute_signal_diagnostics",
    "PositionLot",
    "TradeRecord",
    "DailyPortfolioRecord",
    "SimulationResult",
    "simulate_strategy",
    "simulate_buy_and_hold",
    "simulate_risk_free_cash",
    "compute_strategy_metrics",
    "compute_yearly_metrics",
    "run_cost_sensitivity_stress_test",
    "export_trade_case_studies",
    "plot_equity_curves",
    "plot_underwater_drawdown",
    "plot_yearly_regime_and_rq_comparison",
    "plot_cost_sensitivity_and_turnover",
    "generate_all_figures",
]
