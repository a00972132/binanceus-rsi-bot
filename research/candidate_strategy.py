"""Mutable strategy surface for autonomous research.

Agents should modify only this file during offline experimentation.
The evaluator and live bot are separate on purpose.
"""

STRATEGY_NOTES = "Best current pattern: BTC 4h regime plus ETH/BTC 4h relative strength, with ETH 1h volatility-expansion entry."

PARAMS = {
    "entry_style": "volatility_expansion",
    "fast_sma_period": 20,
    "slow_sma_period": 100,
    "rsi_period": 14,
    "rsi_entry_min": 52.0,
    "rsi_entry_max": 75.0,
    "rsi_exit_min": 45.0,
    "risk_per_trade": 0.02,
    "max_position_fraction": 0.40,
    "stop_atr_mult": 2.5,
    "entry_buffer_pct": 0.001,
    "min_stop_pct": 0.012,
    "breakout_lookback": 15,
    "volume_window": 20,
    "min_volume_ratio": 1.15,
    "regime_symbol": "BTC/USDT",
    "regime_timeframe": "4h",
    "regime_fast_sma_period": 20,
    "regime_slow_sma_period": 50,
    "relative_strength_symbol": "BTC/USDT",
    "relative_strength_timeframe": "4h",
    "relative_strength_fast_sma_period": 20,
    "relative_strength_slow_sma_period": 50,
    "confirm_symbol": "",
    "confirm_timeframe": "",
    "confirm_fast_sma_period": 20,
    "confirm_slow_sma_period": 50,
    "confirm_breakout_lookback": 0,
    "confirm_volume_window": 0,
    "confirm_min_volume_ratio": 0.0,
    "confirm_min_atr_ratio": 0.0,
    "confirm_entry_buffer_pct": 0.0,
    "atr_contraction_window": 20,
    "max_atr_ratio": 99.0,
    "min_atr_ratio": 1.00,
    "max_pullback_from_fast_sma_pct": 0.03,
    "add_on_enabled": False,
    "max_add_ons": 0,
    "add_on_trigger_r": 1.0,
    "add_on_risk_fraction": 0.0,
}
