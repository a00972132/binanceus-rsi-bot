"""Fixed offline evaluation harness for strategy research.

Do not modify this file during normal experiment loops.
It is the trading equivalent of autoresearch's immutable evaluator.
"""

from dataclasses import asdict
from typing import Any, Dict, List

from utils.backtest_strategy import Config, fetch_ohlcv, run_backtest


TUNING_WINDOWS = [
    {"symbol": "ETH/USDT", "timeframe": "1h", "days": 120, "end_offset_days": 360, "label": "tune_1h_old_1"},
    {"symbol": "ETH/USDT", "timeframe": "1h", "days": 120, "end_offset_days": 210, "label": "tune_1h_old_2"},
    {"symbol": "ETH/USDT", "timeframe": "4h", "days": 180, "end_offset_days": 180, "label": "tune_4h_old_1"},
]

HOLDOUT_WINDOWS = [
    {"symbol": "ETH/USDT", "timeframe": "1h", "days": 120, "end_offset_days": 0, "label": "holdout_1h_recent"},
    {"symbol": "ETH/USDT", "timeframe": "4h", "days": 120, "end_offset_days": 0, "label": "holdout_4h_recent"},
]

DEFAULTS = {
    "initial_cash": 10_000.0,
    "fee_bps": 10.0,
    "slippage_bps": 12.0,
}


def build_config(params: Dict[str, Any], window: Dict[str, Any]) -> Config:
    merged = {}
    merged.update(DEFAULTS)
    merged.update({key: value for key, value in window.items() if key != "label"})
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


def summarize_window_group(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not windows:
        return {
            "score": 0.0,
            "positive_windows": 0,
            "mean_return_pct": 0.0,
            "worst_drawdown_pct": 0.0,
            "mean_trades": 0.0,
            "mean_profit_factor": 0.0,
            "mean_expectancy_pct": 0.0,
            "mean_exposure_pct": 0.0,
        }

    return {
        "score": sum(item["window_score"] for item in windows),
        "positive_windows": sum(1 for item in windows if item["result"]["net_return_pct"] > 0),
        "mean_return_pct": sum(item["result"]["net_return_pct"] for item in windows) / len(windows),
        "worst_drawdown_pct": min(item["result"]["max_drawdown_pct"] for item in windows),
        "mean_trades": sum(item["result"]["trades"] for item in windows) / len(windows),
        "mean_profit_factor": sum(item["result"]["profit_factor"] for item in windows) / len(windows),
        "mean_expectancy_pct": sum(item["result"]["expectancy_pct"] for item in windows) / len(windows),
        "mean_exposure_pct": sum(item["result"]["exposure_pct"] for item in windows) / len(windows),
    }


def run_window_set(params: Dict[str, Any], windows: List[Dict[str, Any]], set_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for window in windows:
        cfg = build_config(params, window)
        df = fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.days, cfg.end_offset_days)
        if len(df) < max(cfg.slow_sma_period + 10, 250):
            raise ValueError(f"Not enough candle history for {window}")
        result = run_backtest(df, cfg)
        results.append(
            {
                "set": set_name,
                "label": window["label"],
                "window": window,
                "config": asdict(cfg),
                "result": result,
                "window_score": score_result(result),
            }
        )
    return results


def evaluate_candidate(params: Dict[str, Any]) -> Dict[str, Any]:
    windows: List[Dict[str, Any]] = []
    tuning_windows = run_window_set(params, TUNING_WINDOWS, "tuning")
    holdout_windows = run_window_set(params, HOLDOUT_WINDOWS, "holdout")
    windows.extend(tuning_windows)
    windows.extend(holdout_windows)
    tuning_summary = summarize_window_group(tuning_windows)
    holdout_summary = summarize_window_group(holdout_windows)
    overall_score = tuning_summary["score"] + (holdout_summary["score"] * 1.5)
    promote = (
        tuning_summary["positive_windows"] >= 2
        and tuning_summary["mean_return_pct"] > 0
        and tuning_summary["worst_drawdown_pct"] > -12.0
        and tuning_summary["mean_trades"] >= 4
        and holdout_summary["positive_windows"] >= 1
        and holdout_summary["mean_return_pct"] > 0
        and holdout_summary["worst_drawdown_pct"] > -10.0
        and holdout_summary["mean_profit_factor"] > 1.0
        and holdout_summary["mean_expectancy_pct"] > 0
    )

    return {
        "summary": {
            "score": overall_score,
            "tuning": tuning_summary,
            "holdout": holdout_summary,
            "promote_to_paper": promote,
        },
        "tuning_windows": tuning_windows,
        "holdout_windows": holdout_windows,
        "windows": windows,
    }
