import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


@dataclass
class Config:
    symbol: str
    timeframe: str
    days: int
    end_offset_days: int
    initial_cash: float
    fee_bps: float
    slippage_bps: float
    entry_style: str
    fast_sma_period: int
    slow_sma_period: int
    rsi_period: int
    rsi_entry_min: float
    rsi_entry_max: float
    rsi_exit_min: float
    risk_per_trade: float
    max_position_fraction: float
    stop_atr_mult: float
    entry_buffer_pct: float
    min_stop_pct: float
    breakout_lookback: int
    volume_window: int
    min_volume_ratio: float
    regime_symbol: str
    regime_timeframe: str
    regime_fast_sma_period: int
    regime_slow_sma_period: int
    relative_strength_symbol: str
    relative_strength_timeframe: str
    relative_strength_fast_sma_period: int
    relative_strength_slow_sma_period: int
    confirm_symbol: str
    confirm_timeframe: str
    confirm_fast_sma_period: int
    confirm_slow_sma_period: int
    confirm_breakout_lookback: int
    confirm_volume_window: int
    confirm_min_volume_ratio: float
    confirm_min_atr_ratio: float
    confirm_entry_buffer_pct: float
    atr_contraction_window: int
    max_atr_ratio: float
    min_atr_ratio: float
    max_pullback_from_fast_sma_pct: float
    add_on_enabled: bool
    max_add_ons: int
    add_on_trigger_r: float
    add_on_risk_fraction: float


def timeframe_to_ms(tf: str) -> int:
    unit = tf[-1].lower()
    value = int(tf[:-1])
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 60 * 60_000
    if unit == "d":
        return value * 24 * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def timeframe_to_hours(tf: str) -> float:
    return timeframe_to_ms(tf) / 3_600_000


def fetch_ohlcv(symbol: str, timeframe: str, days: int, end_offset_days: int = 0) -> pd.DataFrame:
    rows: List[list] = []
    end_time = int(pd.Timestamp.utcnow().timestamp() * 1000) - end_offset_days * 24 * 60 * 60 * 1000
    since = end_time - days * 24 * 60 * 60 * 1000
    step_ms = timeframe_to_ms(timeframe)
    compact_symbol = symbol.replace("/", "")

    while True:
        response = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={
                "symbol": compact_symbol,
                "interval": timeframe,
                "startTime": since,
                "endTime": end_time,
                "limit": 1000,
            },
            timeout=20,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_since = int(batch[-1][0]) + step_ms
        if next_since <= since or len(batch) < 1000 or next_since >= end_time:
            break
        since = next_since

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().set_index("timestamp")


def calculate_rsi(df: pd.DataFrame, period: int) -> float:
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_up = up.ewm(com=period - 1, adjust=False).mean()
    avg_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calculate_sma(df: pd.DataFrame, period: int) -> float:
    return float(df["close"].rolling(window=period).mean().iloc[-1])


def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    return float(calculate_atr_series(df, period).iloc[-1])


def calculate_macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    macd_line = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(
        span=26, adjust=False
    ).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def build_snapshot(df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    fast_sma = calculate_sma(df, cfg.fast_sma_period)
    slow_sma = calculate_sma(df, cfg.slow_sma_period)
    rsi = calculate_rsi(df, cfg.rsi_period)
    atr = calculate_atr(df)
    macd_line, signal_line = calculate_macd(df)
    macd_hist = float(macd_line.iloc[-1] - signal_line.iloc[-1])
    prev_macd_hist = float(macd_line.iloc[-2] - signal_line.iloc[-2])
    price = float(df["close"].iloc[-1])
    breakout_window = df["high"].iloc[:-1].tail(cfg.breakout_lookback)
    breakout_level = float(breakout_window.max()) if not breakout_window.empty else price
    volume_avg = float(df["volume"].rolling(window=cfg.volume_window).mean().iloc[-1])
    current_volume = float(df["volume"].iloc[-1])
    atr_avg = float(df["high"].sub(df["low"]).rolling(window=cfg.atr_contraction_window).mean().iloc[-1])
    return {
        "price": price,
        "fast_sma": fast_sma,
        "slow_sma": slow_sma,
        "rsi": rsi,
        "atr": atr,
        "macd_hist": macd_hist,
        "prev_macd_hist": prev_macd_hist,
        "breakout_level": breakout_level,
        "volume": current_volume,
        "volume_avg": volume_avg,
        "atr_avg_range": atr_avg,
    }


def build_regime_filter(base_df: pd.DataFrame, cfg: Config) -> Optional[pd.Series]:
    if not cfg.regime_timeframe:
        return None

    regime_symbol = cfg.regime_symbol or cfg.symbol
    same_market = regime_symbol == cfg.symbol and cfg.regime_timeframe == cfg.timeframe
    regime_df = base_df if same_market else fetch_ohlcv(
        regime_symbol, cfg.regime_timeframe, cfg.days, cfg.end_offset_days
    )
    if len(regime_df) < cfg.regime_slow_sma_period + 5:
        return None

    regime_close = regime_df["close"]
    regime_fast = regime_close.rolling(window=cfg.regime_fast_sma_period).mean()
    regime_slow = regime_close.rolling(window=cfg.regime_slow_sma_period).mean()
    regime_up = (regime_close > regime_fast) & (regime_fast > regime_slow)
    aligned = regime_up.reindex(base_df.index, method="ffill")
    return aligned.fillna(False)


def build_relative_strength_filter(base_df: pd.DataFrame, cfg: Config) -> Optional[pd.Series]:
    if not cfg.relative_strength_timeframe or not cfg.relative_strength_symbol:
        return None

    if cfg.relative_strength_timeframe == cfg.timeframe:
        base_ratio_df = base_df
    else:
        base_ratio_df = fetch_ohlcv(cfg.symbol, cfg.relative_strength_timeframe, cfg.days, cfg.end_offset_days)
    benchmark_df = fetch_ohlcv(
        cfg.relative_strength_symbol, cfg.relative_strength_timeframe, cfg.days, cfg.end_offset_days
    )
    min_len = cfg.relative_strength_slow_sma_period + 5
    if len(base_ratio_df) < min_len or len(benchmark_df) < min_len:
        return None

    joined = pd.concat(
        [
            base_ratio_df["close"].rename("base_close"),
            benchmark_df["close"].rename("benchmark_close"),
        ],
        axis=1,
    ).dropna()
    if joined.empty:
        return None

    ratio = joined["base_close"] / joined["benchmark_close"]
    fast = ratio.rolling(window=cfg.relative_strength_fast_sma_period).mean()
    slow = ratio.rolling(window=cfg.relative_strength_slow_sma_period).mean()
    rs_up = (ratio > fast) & (fast > slow)
    aligned = rs_up.reindex(base_df.index, method="ffill")
    return aligned.fillna(False)


def build_confirmation_filter(base_df: pd.DataFrame, cfg: Config) -> Optional[pd.Series]:
    if not cfg.confirm_timeframe:
        return None

    confirm_symbol = cfg.confirm_symbol or cfg.regime_symbol or cfg.symbol
    same_market = confirm_symbol == cfg.symbol and cfg.confirm_timeframe == cfg.timeframe
    confirm_df = base_df if same_market else fetch_ohlcv(
        confirm_symbol, cfg.confirm_timeframe, cfg.days, cfg.end_offset_days
    )
    if len(confirm_df) < max(cfg.confirm_slow_sma_period + 5, cfg.confirm_breakout_lookback + 5, 60):
        return None

    close = confirm_df["close"]
    high = confirm_df["high"]
    volume = confirm_df["volume"]
    fast_sma = close.rolling(window=cfg.confirm_fast_sma_period).mean()
    slow_sma = close.rolling(window=cfg.confirm_slow_sma_period).mean()
    macd_line, signal_line = calculate_macd(confirm_df)
    macd_hist = macd_line - signal_line
    breakout_ok = pd.Series(True, index=confirm_df.index)
    if cfg.confirm_breakout_lookback > 0:
        breakout_level = high.shift(1).rolling(window=cfg.confirm_breakout_lookback).max()
        breakout_ok = close >= breakout_level * (1 + cfg.confirm_entry_buffer_pct)

    volume_ok = pd.Series(True, index=confirm_df.index)
    if cfg.confirm_volume_window > 0 and cfg.confirm_min_volume_ratio > 0:
        volume_avg = volume.rolling(window=cfg.confirm_volume_window).mean()
        volume_ok = volume >= volume_avg * cfg.confirm_min_volume_ratio

    atr_ok = pd.Series(True, index=confirm_df.index)
    if cfg.confirm_min_atr_ratio > 0:
        atr = calculate_atr_series(confirm_df)
        atr_avg = (confirm_df["high"] - confirm_df["low"]).rolling(window=cfg.atr_contraction_window).mean()
        atr_ratio = atr / atr_avg
        atr_ok = atr_ratio >= cfg.confirm_min_atr_ratio

    confirm_up = (
        (close > fast_sma)
        & (fast_sma > slow_sma)
        & (macd_hist > 0)
        & (macd_hist >= macd_hist.shift(1))
        & breakout_ok
        & volume_ok
        & atr_ok
    )
    aligned = confirm_up.reindex(base_df.index, method="ffill")
    return aligned.fillna(False)


def should_enter_long(snapshot: Dict[str, float], cfg: Config) -> bool:
    regime_trend_up = snapshot.get("regime_trend_up", True)
    relative_strength_up = snapshot.get("relative_strength_up", True)
    confirm_momentum_up = snapshot.get("confirm_momentum_up", True)
    trend_up = snapshot["price"] > snapshot["fast_sma"] > snapshot["slow_sma"]
    volume_confirmed = snapshot["volume"] >= snapshot["volume_avg"] * cfg.min_volume_ratio
    pullback_pct = (snapshot["fast_sma"] - snapshot["price"]) / snapshot["fast_sma"] if snapshot["fast_sma"] > 0 else 0.0
    atr_ratio = snapshot["atr"] / snapshot["atr_avg_range"] if snapshot["atr_avg_range"] > 0 else 1.0
    if cfg.entry_style == "pullback_reversal":
        oversold = cfg.rsi_entry_min <= snapshot["rsi"] <= cfg.rsi_entry_max
        above_slow = snapshot["price"] > snapshot["slow_sma"]
        momentum_recovering = snapshot["macd_hist"] >= snapshot["prev_macd_hist"]
        pullback_ready = 0 <= pullback_pct <= cfg.max_pullback_from_fast_sma_pct
        return (
            regime_trend_up
            and relative_strength_up
            and confirm_momentum_up
            and above_slow
            and oversold
            and momentum_recovering
            and pullback_ready
            and volume_confirmed
        )
    if cfg.entry_style == "pullback_continuation":
        in_rsi_zone = cfg.rsi_entry_min <= snapshot["rsi"] <= cfg.rsi_entry_max
        above_slow = snapshot["price"] > snapshot["slow_sma"]
        fast_trend_up = snapshot["fast_sma"] > snapshot["slow_sma"]
        momentum_recovering = snapshot["macd_hist"] >= snapshot["prev_macd_hist"]
        pullback_ready = 0 <= pullback_pct <= cfg.max_pullback_from_fast_sma_pct
        return (
            regime_trend_up
            and relative_strength_up
            and confirm_momentum_up
            and fast_trend_up
            and above_slow
            and in_rsi_zone
            and momentum_recovering
            and pullback_ready
            and volume_confirmed
        )
    if cfg.entry_style == "volatility_expansion":
        in_rsi_zone = cfg.rsi_entry_min <= snapshot["rsi"] <= cfg.rsi_entry_max
        breakout = snapshot["price"] >= snapshot["breakout_level"] * (1 + cfg.entry_buffer_pct)
        momentum_turning = snapshot["macd_hist"] > 0 and snapshot["macd_hist"] >= snapshot["prev_macd_hist"]
        atr_expanding = atr_ratio >= cfg.min_atr_ratio
        return (
            regime_trend_up
            and relative_strength_up
            and confirm_momentum_up
            and trend_up
            and in_rsi_zone
            and breakout
            and momentum_turning
            and volume_confirmed
            and atr_expanding
        )

    in_rsi_zone = cfg.rsi_entry_min <= snapshot["rsi"] <= cfg.rsi_entry_max
    breakout = snapshot["price"] >= snapshot["breakout_level"] * (1 + cfg.entry_buffer_pct)
    momentum_turning = snapshot["macd_hist"] > 0 and snapshot["macd_hist"] >= snapshot["prev_macd_hist"]
    volatility_contracted = atr_ratio <= cfg.max_atr_ratio
    return (
        regime_trend_up
        and relative_strength_up
        and confirm_momentum_up
        and trend_up
        and in_rsi_zone
        and breakout
        and momentum_turning
        and volume_confirmed
        and volatility_contracted
    )


def should_exit_long(
    snapshot: Dict[str, float], entry_price: float, stop_price: float, peak_price: float, cfg: Config
) -> Tuple[bool, str]:
    trend_failed = (
        snapshot["price"] < snapshot["slow_sma"]
        and snapshot["macd_hist"] < 0
        and snapshot["rsi"] <= cfg.rsi_exit_min
    )
    stop_hit = snapshot["price"] <= stop_price
    trailing_stop = max(stop_price, peak_price - max(snapshot["atr"] * cfg.stop_atr_mult, entry_price * cfg.min_stop_pct))
    trail_hit = snapshot["price"] <= trailing_stop and snapshot["price"] > entry_price
    if stop_hit:
        return True, "stop_loss"
    if trail_hit:
        return True, "atr_trailing_stop"
    if trend_failed:
        return True, "trend_failure"
    return False, ""


def apply_cost(price: float, side: str, fee_bps: float, slippage_bps: float) -> float:
    total_bps = (fee_bps + slippage_bps) / 10_000
    return price * (1 + total_bps) if side == "buy" else price * (1 - total_bps)


def summarize_trade_metrics(
    trade_returns: List[float],
    trade_pnls: List[float],
    hold_bars: List[int],
    exposure_bars: int,
    total_bars: int,
    timeframe: str,
) -> Dict[str, Any]:
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
    profit_factor = 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 99.0
    avg_hold_bars = sum(hold_bars) / len(hold_bars) if hold_bars else 0.0
    avg_hold_hours = avg_hold_bars * timeframe_to_hours(timeframe)
    return {
        "profit_factor": profit_factor,
        "expectancy_pct": (sum(trade_returns) / len(trade_returns) * 100) if trade_returns else 0.0,
        "avg_trade_return_pct": (sum(trade_returns) / len(trade_returns) * 100) if trade_returns else 0.0,
        "avg_win_return_pct": (
            sum(trade for trade in trade_returns if trade > 0) / sum(1 for trade in trade_returns if trade > 0) * 100
        ) if any(trade > 0 for trade in trade_returns) else 0.0,
        "avg_loss_return_pct": (
            sum(trade for trade in trade_returns if trade < 0) / sum(1 for trade in trade_returns if trade < 0) * 100
        ) if any(trade < 0 for trade in trade_returns) else 0.0,
        "avg_hold_bars": avg_hold_bars,
        "avg_hold_hours": avg_hold_hours,
        "exposure_pct": (exposure_bars / total_bars * 100) if total_bars > 0 else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def run_backtest(df: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    warmup = max(cfg.slow_sma_period + 5, 220)
    regime_filter = build_regime_filter(df, cfg)
    relative_strength_filter = build_relative_strength_filter(df, cfg)
    confirmation_filter = build_confirmation_filter(df, cfg)
    cash = cfg.initial_cash
    quantity = 0.0
    entry_price = 0.0
    stop_price = 0.0
    peak_price = 0.0
    add_on_count = 0
    next_add_on_price = 0.0
    equity_curve: List[float] = []
    trade_returns: List[float] = []
    trade_pnls: List[float] = []
    trade_hold_bars: List[int] = []
    trades: List[Dict[str, Any]] = []
    entry_bar_index: Optional[int] = None

    for i in range(warmup, len(df) - 1):
        history = df.iloc[: i + 1]
        signal_df = history.iloc[:-1] if len(history) > 1 else history
        next_open = float(df["open"].iloc[i + 1])
        snapshot = build_snapshot(signal_df, cfg)
        if regime_filter is not None:
            snapshot["regime_trend_up"] = bool(regime_filter.iloc[i])
        if relative_strength_filter is not None:
            snapshot["relative_strength_up"] = bool(relative_strength_filter.iloc[i])
        if confirmation_filter is not None:
            snapshot["confirm_momentum_up"] = bool(confirmation_filter.iloc[i])
        equity = cash + (quantity * snapshot["price"])
        equity_curve.append(equity)

        if quantity > 0:
            peak_price = max(peak_price, snapshot["price"])
            exit_now, exit_reason = should_exit_long(snapshot, entry_price, stop_price, peak_price, cfg)
            if exit_now:
                exit_price = apply_cost(next_open, "sell", cfg.fee_bps, cfg.slippage_bps)
                closed_quantity = quantity
                cash += quantity * exit_price
                trade_return = (exit_price - entry_price) / entry_price
                hold_bars = max(1, (i + 1) - entry_bar_index) if entry_bar_index is not None else 1
                trade_pnl = closed_quantity * (exit_price - entry_price)
                trade_returns.append(trade_return)
                trade_pnls.append(trade_pnl)
                trade_hold_bars.append(hold_bars)
                trades.append(
                    {
                        "entry_time": df.index[entry_bar_index].isoformat() if entry_bar_index is not None else "",
                        "exit_time": df.index[i + 1].isoformat(),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": closed_quantity,
                        "return_pct": trade_return * 100,
                        "pnl": trade_pnl,
                        "hold_bars": hold_bars,
                        "hold_hours": hold_bars * timeframe_to_hours(cfg.timeframe),
                        "exit_reason": exit_reason,
                        "entry_style": cfg.entry_style,
                    }
                )
                quantity = 0.0
                entry_price = 0.0
                stop_price = 0.0
                peak_price = 0.0
                add_on_count = 0
                next_add_on_price = 0.0
                entry_bar_index = None
            elif cfg.add_on_enabled and add_on_count < cfg.max_add_ons:
                breakout_continuation = snapshot["price"] >= next_add_on_price
                momentum_ok = snapshot["macd_hist"] > 0 and snapshot["rsi"] >= cfg.rsi_entry_min
                if breakout_continuation and momentum_ok:
                    stop_distance = max(snapshot["atr"] * cfg.stop_atr_mult, snapshot["price"] * cfg.min_stop_pct)
                    risk_budget = equity * cfg.risk_per_trade * cfg.add_on_risk_fraction
                    risk_size = risk_budget / stop_distance
                    cap_size = (equity * cfg.max_position_fraction) / snapshot["price"]
                    cash_size = cash / snapshot["price"]
                    add_quantity = min(risk_size, max(cap_size - quantity, 0.0), cash_size)
                    if add_quantity > 0:
                        buy_price = apply_cost(next_open, "buy", cfg.fee_bps, cfg.slippage_bps)
                        cash -= add_quantity * buy_price
                        entry_price = ((entry_price * quantity) + (buy_price * add_quantity)) / (quantity + add_quantity)
                        quantity += add_quantity
                        stop_price = max(stop_price, buy_price - stop_distance)
                        peak_price = max(peak_price, buy_price)
                        add_on_count += 1
                        next_add_on_price = buy_price + (stop_distance * cfg.add_on_trigger_r)
            continue

        if should_enter_long(snapshot, cfg):
            stop_distance = max(snapshot["atr"] * cfg.stop_atr_mult, snapshot["price"] * cfg.min_stop_pct)
            risk_budget = equity * cfg.risk_per_trade
            risk_size = risk_budget / stop_distance
            cap_size = (equity * cfg.max_position_fraction) / snapshot["price"]
            cash_size = cash / snapshot["price"]
            quantity = min(risk_size, cap_size, cash_size)
            if quantity <= 0:
                quantity = 0.0
                continue
            buy_price = apply_cost(next_open, "buy", cfg.fee_bps, cfg.slippage_bps)
            cash -= quantity * buy_price
            entry_price = buy_price
            stop_price = buy_price - stop_distance
            peak_price = buy_price
            add_on_count = 0
            next_add_on_price = buy_price + (stop_distance * cfg.add_on_trigger_r)
            entry_bar_index = i + 1

    final_equity = cash + (quantity * float(df["close"].iloc[-1]))
    equity_series = pd.Series(equity_curve)
    rolling_peak = equity_series.cummax()
    max_drawdown = float(((equity_series / rolling_peak) - 1.0).min()) if not equity_series.empty else 0.0
    wins = sum(1 for trade in trade_returns if trade > 0)
    total_bars = max(len(df) - warmup - 1, 1)
    exposure_bars = sum(trade_hold_bars)
    if quantity > 0 and entry_bar_index is not None:
        exposure_bars += max(1, (len(df) - 1) - entry_bar_index)
    buy_hold = ((float(df["close"].iloc[-1]) / float(df["close"].iloc[warmup])) - 1.0) * 100
    metrics = summarize_trade_metrics(
        trade_returns,
        trade_pnls,
        trade_hold_bars,
        exposure_bars,
        total_bars,
        cfg.timeframe,
    )
    return {
        "trades": float(len(trade_returns)),
        "win_rate_pct": (wins / len(trade_returns) * 100) if trade_returns else 0.0,
        "net_return_pct": ((final_equity / cfg.initial_cash) - 1.0) * 100,
        "buy_hold_pct": buy_hold,
        "max_drawdown_pct": max_drawdown * 100,
        "final_equity": final_equity,
        "profit_factor": metrics["profit_factor"],
        "expectancy_pct": metrics["expectancy_pct"],
        "avg_trade_return_pct": metrics["avg_trade_return_pct"],
        "avg_win_return_pct": metrics["avg_win_return_pct"],
        "avg_loss_return_pct": metrics["avg_loss_return_pct"],
        "avg_hold_bars": metrics["avg_hold_bars"],
        "avg_hold_hours": metrics["avg_hold_hours"],
        "exposure_pct": metrics["exposure_pct"],
        "gross_profit": metrics["gross_profit"],
        "gross_loss": metrics["gross_loss"],
        "trade_log": trades,
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Backtest the simplified Binance US bot.")
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--end-offset-days", type=int, default=0)
    parser.add_argument("--initial-cash", type=float, default=10_000)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=12.0)
    parser.add_argument("--entry-style", default="breakout")
    parser.add_argument("--fast-sma-period", type=int, default=20)
    parser.add_argument("--slow-sma-period", type=int, default=100)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-entry-min", type=float, default=55.0)
    parser.add_argument("--rsi-entry-max", type=float, default=80.0)
    parser.add_argument("--rsi-exit-min", type=float, default=45.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.02)
    parser.add_argument("--max-position-fraction", type=float, default=0.40)
    parser.add_argument("--stop-atr-mult", type=float, default=2.5)
    parser.add_argument("--entry-buffer-pct", type=float, default=0.001)
    parser.add_argument("--min-stop-pct", type=float, default=0.012)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--min-volume-ratio", type=float, default=1.0)
    parser.add_argument("--regime-symbol", default="")
    parser.add_argument("--regime-timeframe", default="")
    parser.add_argument("--regime-fast-sma-period", type=int, default=20)
    parser.add_argument("--regime-slow-sma-period", type=int, default=50)
    parser.add_argument("--relative-strength-symbol", default="")
    parser.add_argument("--relative-strength-timeframe", default="")
    parser.add_argument("--relative-strength-fast-sma-period", type=int, default=20)
    parser.add_argument("--relative-strength-slow-sma-period", type=int, default=50)
    parser.add_argument("--confirm-symbol", default="")
    parser.add_argument("--confirm-timeframe", default="")
    parser.add_argument("--confirm-fast-sma-period", type=int, default=20)
    parser.add_argument("--confirm-slow-sma-period", type=int, default=50)
    parser.add_argument("--confirm-breakout-lookback", type=int, default=20)
    parser.add_argument("--confirm-volume-window", type=int, default=20)
    parser.add_argument("--confirm-min-volume-ratio", type=float, default=1.0)
    parser.add_argument("--confirm-min-atr-ratio", type=float, default=0.0)
    parser.add_argument("--confirm-entry-buffer-pct", type=float, default=0.0)
    parser.add_argument("--atr-contraction-window", type=int, default=20)
    parser.add_argument("--max-atr-ratio", type=float, default=1.0)
    parser.add_argument("--min-atr-ratio", type=float, default=0.0)
    parser.add_argument("--max-pullback-from-fast-sma-pct", type=float, default=0.03)
    parser.add_argument("--add-on-enabled", action="store_true", default=True)
    parser.add_argument("--no-add-on", dest="add_on_enabled", action="store_false")
    parser.add_argument("--max-add-ons", type=int, default=1)
    parser.add_argument("--add-on-trigger-r", type=float, default=1.0)
    parser.add_argument("--add-on-risk-fraction", type=float, default=0.50)
    args = parser.parse_args()
    return Config(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        end_offset_days=args.end_offset_days,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        entry_style=args.entry_style,
        fast_sma_period=args.fast_sma_period,
        slow_sma_period=args.slow_sma_period,
        rsi_period=args.rsi_period,
        rsi_entry_min=args.rsi_entry_min,
        rsi_entry_max=args.rsi_entry_max,
        rsi_exit_min=args.rsi_exit_min,
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position_fraction,
        stop_atr_mult=args.stop_atr_mult,
        entry_buffer_pct=args.entry_buffer_pct,
        min_stop_pct=args.min_stop_pct,
        breakout_lookback=args.breakout_lookback,
        volume_window=args.volume_window,
        min_volume_ratio=args.min_volume_ratio,
        regime_symbol=args.regime_symbol,
        regime_timeframe=args.regime_timeframe,
        regime_fast_sma_period=args.regime_fast_sma_period,
        regime_slow_sma_period=args.regime_slow_sma_period,
        relative_strength_symbol=args.relative_strength_symbol,
        relative_strength_timeframe=args.relative_strength_timeframe,
        relative_strength_fast_sma_period=args.relative_strength_fast_sma_period,
        relative_strength_slow_sma_period=args.relative_strength_slow_sma_period,
        confirm_symbol=args.confirm_symbol,
        confirm_timeframe=args.confirm_timeframe,
        confirm_fast_sma_period=args.confirm_fast_sma_period,
        confirm_slow_sma_period=args.confirm_slow_sma_period,
        confirm_breakout_lookback=args.confirm_breakout_lookback,
        confirm_volume_window=args.confirm_volume_window,
        confirm_min_volume_ratio=args.confirm_min_volume_ratio,
        confirm_min_atr_ratio=args.confirm_min_atr_ratio,
        confirm_entry_buffer_pct=args.confirm_entry_buffer_pct,
        atr_contraction_window=args.atr_contraction_window,
        max_atr_ratio=args.max_atr_ratio,
        min_atr_ratio=args.min_atr_ratio,
        max_pullback_from_fast_sma_pct=args.max_pullback_from_fast_sma_pct,
        add_on_enabled=args.add_on_enabled,
        max_add_ons=args.max_add_ons,
        add_on_trigger_r=args.add_on_trigger_r,
        add_on_risk_fraction=args.add_on_risk_fraction,
    )


def main() -> None:
    cfg = parse_args()
    df = fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.days, cfg.end_offset_days)
    if len(df) < max(cfg.slow_sma_period + 10, 250):
        raise SystemExit("Not enough candle history for this configuration.")
    result = run_backtest(df, cfg)
    print(f"Backtest: {cfg.symbol} {cfg.timeframe} over {cfg.days} days")
    print(f"Trades: {int(result['trades'])}")
    print(f"Win rate: {result['win_rate_pct']:.2f}%")
    print(f"Net return: {result['net_return_pct']:.2f}%")
    print(f"Buy and hold: {result['buy_hold_pct']:.2f}%")
    print(f"Max drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Final equity: ${result['final_equity']:.2f}")


if __name__ == "__main__":
    main()
