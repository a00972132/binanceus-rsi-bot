import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import requests


@dataclass
class Config:
    symbol: str
    timeframe: str
    days: int
    initial_cash: float
    fee_bps: float
    slippage_bps: float
    fast_sma_period: int
    slow_sma_period: int
    rsi_period: int
    rsi_entry_min: float
    rsi_entry_max: float
    rsi_exit_min: float
    risk_per_trade: float
    max_position_fraction: float
    stop_atr_mult: float
    target_r_multiple: float
    entry_buffer_pct: float
    min_stop_pct: float


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


def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    rows: List[list] = []
    since = int(pd.Timestamp.utcnow().timestamp() * 1000) - days * 24 * 60 * 60 * 1000
    step_ms = timeframe_to_ms(timeframe)
    compact_symbol = symbol.replace("/", "")

    while True:
        response = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={
                "symbol": compact_symbol,
                "interval": timeframe,
                "startTime": since,
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
        if next_since <= since or len(batch) < 1000:
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


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
    return {
        "price": price,
        "fast_sma": fast_sma,
        "slow_sma": slow_sma,
        "rsi": rsi,
        "atr": atr,
        "macd_hist": macd_hist,
        "prev_macd_hist": prev_macd_hist,
    }


def should_enter_long(snapshot: Dict[str, float], cfg: Config) -> bool:
    trend_up = snapshot["price"] > snapshot["fast_sma"] > snapshot["slow_sma"]
    in_rsi_zone = cfg.rsi_entry_min <= snapshot["rsi"] <= cfg.rsi_entry_max
    near_fast_sma = snapshot["price"] <= snapshot["fast_sma"] * (1 + cfg.entry_buffer_pct)
    momentum_turning = snapshot["macd_hist"] > 0 and snapshot["macd_hist"] >= snapshot["prev_macd_hist"]
    return trend_up and in_rsi_zone and near_fast_sma and momentum_turning


def should_exit_long(snapshot: Dict[str, float], entry_price: float, stop_price: float, target_price: float, peak_price: float, cfg: Config) -> bool:
    trend_failed = snapshot["price"] < snapshot["fast_sma"] and snapshot["macd_hist"] < 0
    target_hit = snapshot["price"] >= target_price and snapshot["rsi"] >= cfg.rsi_exit_min
    stop_hit = snapshot["price"] <= stop_price
    trailing_stop = max(stop_price, peak_price - max(snapshot["atr"], entry_price * cfg.min_stop_pct))
    trail_hit = snapshot["price"] <= trailing_stop and snapshot["price"] > entry_price
    return stop_hit or target_hit or trail_hit or (trend_failed and snapshot["price"] > entry_price)


def apply_cost(price: float, side: str, fee_bps: float, slippage_bps: float) -> float:
    total_bps = (fee_bps + slippage_bps) / 10_000
    return price * (1 + total_bps) if side == "buy" else price * (1 - total_bps)


def run_backtest(df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    warmup = max(cfg.slow_sma_period + 5, 220)
    cash = cfg.initial_cash
    quantity = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    peak_price = 0.0
    equity_curve: List[float] = []
    trade_returns: List[float] = []

    for i in range(warmup, len(df) - 1):
        history = df.iloc[: i + 1]
        signal_df = history.iloc[:-1] if len(history) > 1 else history
        next_open = float(df["open"].iloc[i + 1])
        snapshot = build_snapshot(signal_df, cfg)
        equity = cash + (quantity * snapshot["price"])
        equity_curve.append(equity)

        if quantity > 0:
            peak_price = max(peak_price, snapshot["price"])
            if should_exit_long(snapshot, entry_price, stop_price, target_price, peak_price, cfg):
                exit_price = apply_cost(next_open, "sell", cfg.fee_bps, cfg.slippage_bps)
                cash += quantity * exit_price
                trade_returns.append((exit_price - entry_price) / entry_price)
                quantity = 0.0
                entry_price = 0.0
                stop_price = 0.0
                target_price = 0.0
                peak_price = 0.0
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
            target_price = buy_price + (stop_distance * cfg.target_r_multiple)
            peak_price = buy_price

    final_equity = cash + (quantity * float(df["close"].iloc[-1]))
    equity_series = pd.Series(equity_curve)
    rolling_peak = equity_series.cummax()
    max_drawdown = float(((equity_series / rolling_peak) - 1.0).min()) if not equity_series.empty else 0.0
    wins = sum(1 for trade in trade_returns if trade > 0)
    buy_hold = ((float(df["close"].iloc[-1]) / float(df["close"].iloc[warmup])) - 1.0) * 100
    return {
        "trades": float(len(trade_returns)),
        "win_rate_pct": (wins / len(trade_returns) * 100) if trade_returns else 0.0,
        "net_return_pct": ((final_equity / cfg.initial_cash) - 1.0) * 100,
        "buy_hold_pct": buy_hold,
        "max_drawdown_pct": max_drawdown * 100,
        "final_equity": final_equity,
    }


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Backtest the simplified Binance US bot.")
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--initial-cash", type=float, default=10_000)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=12.0)
    parser.add_argument("--fast-sma-period", type=int, default=20)
    parser.add_argument("--slow-sma-period", type=int, default=100)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-entry-min", type=float, default=35.0)
    parser.add_argument("--rsi-entry-max", type=float, default=55.0)
    parser.add_argument("--rsi-exit-min", type=float, default=68.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-fraction", type=float, default=0.20)
    parser.add_argument("--stop-atr-mult", type=float, default=1.5)
    parser.add_argument("--target-r-multiple", type=float, default=2.0)
    parser.add_argument("--entry-buffer-pct", type=float, default=0.003)
    parser.add_argument("--min-stop-pct", type=float, default=0.008)
    args = parser.parse_args()
    return Config(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        initial_cash=args.initial_cash,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        fast_sma_period=args.fast_sma_period,
        slow_sma_period=args.slow_sma_period,
        rsi_period=args.rsi_period,
        rsi_entry_min=args.rsi_entry_min,
        rsi_entry_max=args.rsi_entry_max,
        rsi_exit_min=args.rsi_exit_min,
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position_fraction,
        stop_atr_mult=args.stop_atr_mult,
        target_r_multiple=args.target_r_multiple,
        entry_buffer_pct=args.entry_buffer_pct,
        min_stop_pct=args.min_stop_pct,
    )


def main() -> None:
    cfg = parse_args()
    df = fetch_ohlcv(cfg.symbol, cfg.timeframe, cfg.days)
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
