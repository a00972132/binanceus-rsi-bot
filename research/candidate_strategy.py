"""Mutable strategy surface for autonomous research.

Agents should modify only this file during offline experimentation.
The evaluator and live bot are separate on purpose.
"""

STRATEGY_NOTES = "Hypothesis 1: aggressive breakout requires volume expansion above recent average."

PARAMS = {
    "fast_sma_period": 20,
    "slow_sma_period": 100,
    "rsi_period": 14,
    "rsi_entry_min": 55.0,
    "rsi_entry_max": 80.0,
    "rsi_exit_min": 45.0,
    "risk_per_trade": 0.02,
    "max_position_fraction": 0.40,
    "stop_atr_mult": 2.5,
    "entry_buffer_pct": 0.001,
    "min_stop_pct": 0.012,
    "breakout_lookback": 20,
    "volume_window": 20,
    "min_volume_ratio": 1.25,
    "add_on_enabled": True,
    "max_add_ons": 1,
    "add_on_trigger_r": 1.0,
    "add_on_risk_fraction": 0.50,
}
