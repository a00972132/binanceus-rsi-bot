import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ccxt
import pandas as pd
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / "run"
LOG_DIR = ROOT_DIR / "logs"
STATE_PATH = RUN_DIR / "position_state.json"

RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    env_path = os.getenv("BOT_ENV_FILE")
    candidates = [Path(env_path)] if env_path else []
    candidates.append(ROOT_DIR / ".env")
    for candidate in candidates:
        try:
            if candidate.exists():
                load_dotenv(dotenv_path=str(candidate))
                return
        except Exception:
            continue


load_env()


def configure_logging() -> None:
    level_name = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(
        str(LOG_DIR / "trading_bot.log"), maxBytes=1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)


configure_logging()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    raise SystemExit("Missing BINANCE_API_KEY or BINANCE_API_SECRET")

SYMBOL = os.getenv("BOT_SYMBOL", "ETH/USDT")
TIMEFRAME = os.getenv("BOT_TIMEFRAME", "1h")
PAPER_TRADING = os.getenv("BOT_PAPER_TRADING", "false").lower() in {"1", "true", "yes", "y"}
DIAGNOSTICS = os.getenv("BOT_DIAGNOSTICS", "false").lower() in {"1", "true", "yes", "y"}

FAST_SMA_PERIOD = int(os.getenv("BOT_FAST_SMA_PERIOD", "20"))
SLOW_SMA_PERIOD = int(os.getenv("BOT_SLOW_SMA_PERIOD", "100"))
RSI_PERIOD = int(os.getenv("BOT_RSI_PERIOD", "14"))
RSI_ENTRY_MIN = float(os.getenv("BOT_RSI_ENTRY_MIN", "55"))
RSI_ENTRY_MAX = float(os.getenv("BOT_RSI_ENTRY_MAX", "80"))
RSI_EXIT_MIN = float(os.getenv("BOT_RSI_EXIT_MIN", "45"))
RISK_PER_TRADE = float(os.getenv("BOT_RISK_PER_TRADE", "0.02"))
MAX_POSITION_FRACTION = float(os.getenv("BOT_MAX_POSITION_FRACTION", "0.40"))
STOP_ATR_MULT = float(os.getenv("BOT_STOP_ATR_MULT", "2.5"))
ENTRY_BUFFER_PCT = float(os.getenv("BOT_ENTRY_BUFFER_PCT", "0.001"))
MAX_SPREAD_PERCENT = float(os.getenv("BOT_MAX_SPREAD_PERCENT", "0.20"))
MIN_STOP_PCT = float(os.getenv("BOT_MIN_STOP_PCT", "0.012"))
MIN_TRADE_INTERVAL = int(os.getenv("BOT_MIN_TRADE_INTERVAL", "3600"))
MAX_TRADES_PER_DAY = int(os.getenv("BOT_MAX_TRADES_PER_DAY", "3"))
BREAKOUT_LOOKBACK = int(os.getenv("BOT_BREAKOUT_LOOKBACK", "20"))
ADD_ON_ENABLED = os.getenv("BOT_ADD_ON_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
MAX_ADD_ONS = int(os.getenv("BOT_MAX_ADD_ONS", "1"))
ADD_ON_TRIGGER_R = float(os.getenv("BOT_ADD_ON_TRIGGER_R", "1.0"))
ADD_ON_RISK_FRACTION = float(os.getenv("BOT_ADD_ON_RISK_FRACTION", "0.50"))

BASE_ASSET, QUOTE_ASSET = SYMBOL.split("/")


exchange = ccxt.binanceus(
    {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    }
)
exchange.check_required_credentials()
exchange.load_markets()
exchange.load_time_difference()


@dataclass
class PositionState:
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    initial_stop_distance: float = 0.0
    peak_price: float = 0.0
    entered_at: float = 0.0
    add_on_count: int = 0
    next_add_on_price: float = 0.0

    def is_open(self) -> bool:
        return self.quantity > 0 and self.entry_price > 0


last_trade_time = 0.0
trade_day = ""
trade_count_today = 0


def retry_api_call(api_func, retries: int = 3, delay: int = 2):
    for attempt in range(retries):
        try:
            return api_func()
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            logging.warning("API error: %s", exc)
            time.sleep(delay * (2 ** attempt))
        except Exception as exc:
            logging.error("Unexpected API error: %s", exc)
            return None
    return None


def fetch_data(limit: int = 500) -> Optional[pd.DataFrame]:
    data = retry_api_call(
        lambda: exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    )
    if not data:
        return None
    df = pd.DataFrame(
        data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df.astype(float)


def fetch_balance() -> Optional[Dict[str, Any]]:
    return retry_api_call(lambda: exchange.fetch_balance())


def fetch_ticker_price() -> float:
    ticker = retry_api_call(lambda: exchange.fetch_ticker(SYMBOL))
    return float(ticker.get("last", 0.0)) if ticker else 0.0


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
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


def get_order_book_spread() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    order_book = retry_api_call(lambda: exchange.fetch_order_book(SYMBOL))
    if not order_book or not order_book.get("bids") or not order_book.get("asks"):
        return None, None, None
    bid = float(order_book["bids"][0][0])
    ask = float(order_book["asks"][0][0])
    return bid, ask, ask - bid


def load_position_state() -> PositionState:
    if not STATE_PATH.exists():
        return PositionState()
    try:
        with STATE_PATH.open("r") as file:
            data = json.load(file)
        return PositionState(**data)
    except Exception as exc:
        logging.warning("Could not load position state: %s", exc)
        return PositionState()


def save_position_state(state: PositionState) -> None:
    try:
        if not state.is_open():
            STATE_PATH.unlink(missing_ok=True)
            return
        with STATE_PATH.open("w") as file:
            json.dump(asdict(state), file)
    except Exception as exc:
        logging.warning("Could not persist position state: %s", exc)


def reset_daily_trade_counter() -> None:
    global trade_day, trade_count_today
    today = time.strftime("%Y-%m-%d")
    if trade_day != today:
        trade_day = today
        trade_count_today = 0


def can_trade_now() -> bool:
    global last_trade_time
    reset_daily_trade_counter()
    if trade_count_today >= MAX_TRADES_PER_DAY:
        return False
    if time.time() - last_trade_time < MIN_TRADE_INTERVAL:
        return False
    return True


def adjust_amount_for_market(amount: float, price: float) -> float:
    if amount <= 0 or price <= 0:
        return 0.0
    try:
        amount = float(exchange.amount_to_precision(SYMBOL, amount))
    except Exception:
        pass
    try:
        market = exchange.market(SYMBOL)
        min_amount = float((((market or {}).get("limits") or {}).get("amount") or {}).get("min") or 0.0)
        min_cost = float((((market or {}).get("limits") or {}).get("cost") or {}).get("min") or 0.0)
        if min_amount > 0:
            amount = max(amount, min_amount)
        if min_cost > 0 and price > 0:
            amount = max(amount, min_cost / price)
        amount = float(exchange.amount_to_precision(SYMBOL, amount))
    except Exception:
        pass
    return max(amount, 0.0)


def compute_equity(balance: Dict[str, Any], price: float) -> float:
    quote_total = float((balance.get("total") or {}).get(QUOTE_ASSET, 0.0))
    base_total = float((balance.get("total") or {}).get(BASE_ASSET, 0.0))
    return quote_total + (base_total * price)


def compute_trade_size(balance: Dict[str, Any], price: float, stop_distance: float) -> float:
    if price <= 0 or stop_distance <= 0:
        return 0.0
    quote_free = float((balance.get("free") or {}).get(QUOTE_ASSET, 0.0))
    equity = compute_equity(balance, price)
    risk_budget = equity * RISK_PER_TRADE
    risk_size = risk_budget / stop_distance
    cap_size = (equity * MAX_POSITION_FRACTION) / price
    cash_size = quote_free / price
    return adjust_amount_for_market(min(risk_size, cap_size, cash_size), price)


def create_position_state(quantity: float, fill_price: float, atr_value: float) -> PositionState:
    stop_distance = max(atr_value * STOP_ATR_MULT, fill_price * MIN_STOP_PCT)
    stop_price = fill_price - stop_distance
    return PositionState(
        quantity=quantity,
        entry_price=fill_price,
        stop_price=stop_price,
        initial_stop_distance=stop_distance,
        peak_price=fill_price,
        entered_at=time.time(),
        add_on_count=0,
        next_add_on_price=fill_price + (stop_distance * ADD_ON_TRIGGER_R),
    )


def update_position_state_on_add(
    state: PositionState, add_quantity: float, fill_price: float, atr_value: float
) -> PositionState:
    total_quantity = state.quantity + add_quantity
    if total_quantity <= 0:
        return state
    stop_distance = max(atr_value * STOP_ATR_MULT, fill_price * MIN_STOP_PCT)
    state.entry_price = (
        (state.entry_price * state.quantity) + (fill_price * add_quantity)
    ) / total_quantity
    state.quantity = total_quantity
    state.stop_price = max(state.stop_price, fill_price - stop_distance)
    state.initial_stop_distance = max(state.initial_stop_distance, stop_distance)
    state.peak_price = max(state.peak_price, fill_price)
    state.add_on_count += 1
    state.next_add_on_price = fill_price + (stop_distance * ADD_ON_TRIGGER_R)
    return state


def extract_fill_price(order: Optional[Dict[str, Any]], fallback: float) -> float:
    if not order:
        return fallback
    for key in ("average", "price"):
        value = order.get(key)
        try:
            if value:
                return float(value)
        except Exception:
            continue
    return fallback


def extract_fill_amount(order: Optional[Dict[str, Any]], fallback: float) -> float:
    if not order:
        return fallback
    for key in ("filled", "amount"):
        value = order.get(key)
        try:
            if value:
                return float(value)
        except Exception:
            continue
    return fallback


def place_order(side: str, amount: float, reference_price: float, reason: str) -> Optional[Dict[str, Any]]:
    global last_trade_time, trade_count_today
    if amount <= 0:
        return None
    bid, ask, spread = get_order_book_spread()
    if spread is None or ask is None or bid is None:
        logging.warning("No valid order book for %s", SYMBOL)
        return None
    spread_pct = (spread / ((bid + ask) / 2.0)) * 100
    if spread_pct > MAX_SPREAD_PERCENT:
        logging.info("Spread %.2f%% above cap %.2f%%, skipping %s", spread_pct, MAX_SPREAD_PERCENT, side)
        return None

    if PAPER_TRADING:
        last_trade_time = time.time()
        trade_count_today += 1
        logging.info("[PAPER] %s %.6f %s @ %.2f | %s", side.upper(), amount, BASE_ASSET, reference_price, reason)
        return {"side": side, "amount": amount, "price": reference_price, "average": reference_price}

    try:
        order = exchange.create_market_order(SYMBOL, side, amount)
        last_trade_time = time.time()
        trade_count_today += 1
        logging.info("%s %.6f %s | %s", side.upper(), amount, BASE_ASSET, reason)
        return order
    except Exception as exc:
        logging.error("Order failed: %s", exc)
        return None


def build_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    fast_sma = calculate_sma(df, FAST_SMA_PERIOD)
    slow_sma = calculate_sma(df, SLOW_SMA_PERIOD)
    rsi = calculate_rsi(df, RSI_PERIOD)
    atr = calculate_atr(df, 14)
    macd_line, signal_line = calculate_macd(df)
    macd_hist = float(macd_line.iloc[-1] - signal_line.iloc[-1])
    prev_macd_hist = float(macd_line.iloc[-2] - signal_line.iloc[-2])
    close = float(df["close"].iloc[-1])
    breakout_window = df["high"].iloc[:-1].tail(BREAKOUT_LOOKBACK)
    breakout_level = float(breakout_window.max()) if not breakout_window.empty else close
    return {
        "price": close,
        "fast_sma": fast_sma,
        "slow_sma": slow_sma,
        "rsi": rsi,
        "atr": atr,
        "macd_hist": macd_hist,
        "prev_macd_hist": prev_macd_hist,
        "breakout_level": breakout_level,
    }


def should_enter_long(snapshot: Dict[str, float]) -> Tuple[bool, str]:
    trend_up = snapshot["price"] > snapshot["fast_sma"] > snapshot["slow_sma"]
    in_rsi_zone = RSI_ENTRY_MIN <= snapshot["rsi"] <= RSI_ENTRY_MAX
    breakout = snapshot["price"] >= snapshot["breakout_level"] * (1 + ENTRY_BUFFER_PCT)
    momentum_turning = snapshot["macd_hist"] > 0 and snapshot["macd_hist"] >= snapshot["prev_macd_hist"]
    checks = {
        "trend_up": trend_up,
        "rsi_strength": in_rsi_zone,
        "breakout": breakout,
        "macd_turn": momentum_turning,
    }
    if all(checks.values()):
        return True, "breakout trend entry"
    if DIAGNOSTICS:
        logging.info("No entry: %s", checks)
    return False, ", ".join(key for key, value in checks.items() if not value)


def should_exit_long(snapshot: Dict[str, float], state: PositionState) -> Tuple[bool, str]:
    price = snapshot["price"]
    trend_failed = (
        price < snapshot["slow_sma"]
        and snapshot["macd_hist"] < 0
        and snapshot["rsi"] <= RSI_EXIT_MIN
    )
    stop_hit = price <= state.stop_price
    trailing_stop = max(
        state.stop_price,
        state.peak_price - max(snapshot["atr"] * STOP_ATR_MULT, state.entry_price * MIN_STOP_PCT),
    )
    trail_hit = price <= trailing_stop and price > state.entry_price
    if stop_hit:
        return True, "stop loss"
    if trail_hit:
        return True, "atr trailing stop"
    if trend_failed:
        return True, "trend failure"
    return False, ""


def should_add_on(snapshot: Dict[str, float], state: PositionState) -> Tuple[bool, str]:
    if not ADD_ON_ENABLED or not state.is_open():
        return False, ""
    if state.add_on_count >= MAX_ADD_ONS:
        return False, ""
    breakout_continuation = snapshot["price"] >= state.next_add_on_price
    momentum_ok = snapshot["macd_hist"] > 0 and snapshot["rsi"] >= RSI_ENTRY_MIN
    if breakout_continuation and momentum_ok:
        return True, "winner add-on"
    return False, ""


def sync_state_with_balance(state: PositionState, balance: Dict[str, Any]) -> PositionState:
    free_base = float((balance.get("free") or {}).get(BASE_ASSET, 0.0))
    if not state.is_open():
        return PositionState()
    if free_base <= 0:
        logging.warning("Tracked position cleared because free %s is zero", BASE_ASSET)
        return PositionState()
    if free_base < state.quantity:
        state.quantity = free_base
    return state


def run_bot() -> None:
    state = load_position_state()
    logging.info(
        "Bot config | symbol=%s timeframe=%s paper=%s fast=%s slow=%s rsi-entry=[%.1f, %.1f] breakout=%s stop_atr=%.1f add_on=%s",
        SYMBOL,
        TIMEFRAME,
        PAPER_TRADING,
        FAST_SMA_PERIOD,
        SLOW_SMA_PERIOD,
        RSI_ENTRY_MIN,
        RSI_ENTRY_MAX,
        BREAKOUT_LOOKBACK,
        STOP_ATR_MULT,
        ADD_ON_ENABLED,
    )

    while True:
        try:
            df = fetch_data()
            balance = fetch_balance()
            if df is None or balance is None or len(df) < max(SLOW_SMA_PERIOD + 5, 250):
                logging.warning("Insufficient data or balance, sleeping")
                time.sleep(60)
                continue

            df_sig = df.iloc[:-1] if len(df) > 1 else df
            snapshot = build_snapshot(df_sig)
            state = sync_state_with_balance(state, balance)

            if state.is_open():
                state.peak_price = max(state.peak_price, snapshot["price"])
                exit_now, reason = should_exit_long(snapshot, state)
                if exit_now and can_trade_now():
                    amount = adjust_amount_for_market(state.quantity, snapshot["price"])
                    order = place_order("sell", amount, snapshot["price"], reason)
                    if order:
                        fill_price = extract_fill_price(order, snapshot["price"])
                        fill_amount = extract_fill_amount(order, amount)
                        logging.info(
                            "Closed long %.6f %s @ %.2f | entry %.2f | %s",
                            fill_amount,
                            BASE_ASSET,
                            fill_price,
                            state.entry_price,
                            reason,
                        )
                        state = PositionState()
                        save_position_state(state)
                else:
                    add_on_now, add_reason = should_add_on(snapshot, state)
                    if add_on_now and can_trade_now():
                        stop_distance = max(snapshot["atr"] * STOP_ATR_MULT, snapshot["price"] * MIN_STOP_PCT)
                        add_amount = compute_trade_size(balance, snapshot["price"], stop_distance) * ADD_ON_RISK_FRACTION
                        add_amount = adjust_amount_for_market(add_amount, snapshot["price"])
                        order = place_order("buy", add_amount, snapshot["price"], add_reason)
                        if order:
                            fill_price = extract_fill_price(order, snapshot["price"])
                            fill_amount = extract_fill_amount(order, add_amount)
                            state = update_position_state_on_add(state, fill_amount, fill_price, snapshot["atr"])
                            save_position_state(state)
                            logging.info(
                                "Added long %.6f %s @ %.2f | avg %.2f | stop %.2f | add_count %s",
                                fill_amount,
                                BASE_ASSET,
                                fill_price,
                                state.entry_price,
                                state.stop_price,
                                state.add_on_count,
                            )
            else:
                enter_now, reason = should_enter_long(snapshot)
                if enter_now and can_trade_now():
                    stop_distance = max(snapshot["atr"] * STOP_ATR_MULT, snapshot["price"] * MIN_STOP_PCT)
                    amount = compute_trade_size(balance, snapshot["price"], stop_distance)
                    order = place_order("buy", amount, snapshot["price"], reason)
                    if order:
                        fill_price = extract_fill_price(order, snapshot["price"])
                        fill_amount = extract_fill_amount(order, amount)
                        state = create_position_state(fill_amount, fill_price, snapshot["atr"])
                        save_position_state(state)
                        logging.info(
                            "Opened long %.6f %s @ %.2f | stop %.2f | breakout %.2f",
                            fill_amount,
                            BASE_ASSET,
                            fill_price,
                            state.stop_price,
                            snapshot["breakout_level"],
                        )

            save_position_state(state)
            time.sleep(MIN_TRADE_INTERVAL)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
            break
        except Exception as exc:
            logging.error("Main loop error: %s", exc)
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
