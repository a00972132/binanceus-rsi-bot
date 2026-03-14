"""Fixed offline evaluation harness for strategy research.

Do not modify this file during normal experiment loops.
It is the trading equivalent of autoresearch's immutable evaluator.
"""

from dataclasses import asdict
from typing import Any, Dict, List

from utils.backtest_strategy import Config, fetch_ohlcv, run_backtest


EVAL_WINDOWS = [
    {"symbol": "ETH/USDT", "timeframe": "1h", "days": 120},
    {"symbol": "ETH/USDT", "timeframe": "1h", "days": 240},
    {"symbol": "ETH/USDT", "timeframe": "4h", "days": 365},
]

DEFAULTS = {
    "initial_cash": 10_000.0,
    "fee_bps": 10.0,
    "slippage_bps": 12.0,
}


def build_config(params: Dict[str, Any], window: Dict[str, Any]) -> Config:
    merged = {}
    merged.update(DEFAULTS)
    merged.update(window)
    merged.update(params)
    return Config(**merged)


def score_result(result: Dict[str, float]) -> float:
    score = result["net_return_pct"]
    score -= max(0.0, abs(result["max_drawdown_pct"]) - 5.0) * 0.75
    if result["trades"] < 4:
        score -= (4 - result["trades"]) * 1.0
    if result["net_return_pct"] <= 0:
        score -= 2.0
    return score


def evaluate_candidate(params: Dict[str, Any]) -> Dict[str, Any]:
    windows: List[Dict[str, Any]] = []
    total_score = 0.0
    positive_windows = 0

    for window in EVAL_WINDOWS:
        cfg = build_config(params, window)
        df = fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.days)
        if len(df) < max(cfg.slow_sma_period + 10, 250):
            raise ValueError(f"Not enough candle history for {window}")
        result = run_backtest(df, cfg)
        total_score += score_result(result)
        if result["net_return_pct"] > 0:
            positive_windows += 1
        windows.append(
            {
                "window": window,
                "config": asdict(cfg),
                "result": result,
                "window_score": score_result(result),
            }
        )

    mean_return = sum(item["result"]["net_return_pct"] for item in windows) / len(windows)
    worst_drawdown = min(item["result"]["max_drawdown_pct"] for item in windows)
    mean_trades = sum(item["result"]["trades"] for item in windows) / len(windows)
    promote = positive_windows >= 2 and mean_return > 0 and worst_drawdown > -12.0 and mean_trades >= 4

    return {
        "summary": {
            "score": total_score,
            "positive_windows": positive_windows,
            "mean_return_pct": mean_return,
            "worst_drawdown_pct": worst_drawdown,
            "mean_trades": mean_trades,
            "promote_to_paper": promote,
        },
        "windows": windows,
    }
