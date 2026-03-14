"""Mutable strategy surface for autonomous research.

Agents should modify only this file during offline experimentation.
The evaluator and live bot are separate on purpose.
"""

STRATEGY_NOTES = "Baseline trend-pullback candidate derived from the live bot."

PARAMS = {
    "fast_sma_period": 20,
    "slow_sma_period": 100,
    "rsi_period": 14,
    "rsi_entry_min": 35.0,
    "rsi_entry_max": 55.0,
    "rsi_exit_min": 68.0,
    "risk_per_trade": 0.01,
    "max_position_fraction": 0.20,
    "stop_atr_mult": 1.5,
    "target_r_multiple": 2.0,
    "entry_buffer_pct": 0.003,
    "min_stop_pct": 0.008,
}
