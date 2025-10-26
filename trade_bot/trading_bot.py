import ccxt
import pandas as pd
import time
import logging
import os
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from collections import deque
from typing import Tuple, Dict, Any, List, Optional, Union

# Load environment variables (prefer BOT_ENV_FILE or project .env)
ROOT_DIR = Path(__file__).resolve().parents[1]
env_candidates = []
if os.getenv('BOT_ENV_FILE'):
    env_candidates.append(Path(os.getenv('BOT_ENV_FILE')))
env_candidates.append(ROOT_DIR / '.env')
env_candidates.append(Path("/Users/will/Desktop/Code/Tradingbot/binanceus_creds.env"))
loaded_env = False
for path in env_candidates:
    try:
        if path.exists():
            load_dotenv(dotenv_path=str(path))
            logging.info(f"Loaded environment from {path}")
            loaded_env = True
            break
    except Exception:
        pass
if not loaded_env:
    logging.warning("No environment file loaded; relying on process env variables.")

# Configure Binance API Keys
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    logging.error("API keys not found. Exiting.")
    raise SystemExit("Missing API keys")

# Configure logging (single root, no duplicates)
LOG_LEVEL = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
level_num = getattr(logging, LOG_LEVEL, logging.INFO)
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(level_num)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = RotatingFileHandler(str(LOG_DIR / 'trading_bot.log'), maxBytes=1024 * 1024, backupCount=5)
file_handler.setFormatter(fmt)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
root_logger.addHandler(console_handler)

logging.info("Trading bot started (updated version)")

# Exchange Configuration
exchange = ccxt.binanceus({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True},
})

# Validate credentials and load markets
exchange.check_required_credentials()
exchange.load_markets()
exchange.load_time_difference()
exchange.fetch_time()
logging.info("Exchange configuration complete")

# Trading Parameters
SYMBOL = os.getenv('BOT_SYMBOL', 'ETH/USDT')
TIMEFRAME = os.getenv('BOT_TIMEFRAME', '1m')  # 1‑minute candles
# Optional paper-trading (no live orders); set BOT_PAPER_TRADING=true to enable
PAPER_TRADING = os.getenv('BOT_PAPER_TRADING', 'false').lower() in ('1', 'true', 'yes', 'y')
RSI_PERIOD = 14
SMA_PERIOD = 50
TRAILING_STOP_PERCENT = 0.08  # 8% trailing stop
# Base defaults (may be overridden by env/aggressiveness)
MIN_TRADE_INTERVAL = 30  # seconds between trades
MAX_TRADES_PER_HOUR = 20  # trade frequency limit
RISK_PER_TRADE = 0.05  # 5% of portfolio per trade (base risk)
TAKE_PROFIT_LEVELS = [0.01, 0.02, 0.03]
POSITION_SCALE_OUT = [0.3, 0.3, 0.4]

# Tuning knobs
AGGR = os.getenv('BOT_AGGRESSIVENESS', 'balanced').lower()  # conservative|balanced|aggressive
RSI_BUY_MAX = float(os.getenv('BOT_RSI_BUY_MAX', '70'))
RSI_SELL_MIN = float(os.getenv('BOT_RSI_SELL_MIN', '30'))
# Model threshold (probability price goes up)
DEFAULT_THRESHOLDS = {'conservative': 0.70, 'balanced': 0.65, 'aggressive': 0.58}
PREDICTION_THRESHOLD = float(os.getenv('BOT_PREDICTION_THRESHOLD', str(DEFAULT_THRESHOLDS.get(AGGR, 0.65))))
# Confirmations required among [MACD direction, trend alignment, volume boost]
CONFIRMATIONS_REQUIRED_BUY = int(os.getenv('BOT_CONFIRMATIONS_REQUIRED_BUY', '2'))
CONFIRMATIONS_REQUIRED_SELL = int(os.getenv('BOT_CONFIRMATIONS_REQUIRED_SELL', '2'))
# Spread caps (% of ask)
DEFAULT_SPREAD_NORMAL = {'conservative': 0.10, 'balanced': 0.12, 'aggressive': 0.15}
DEFAULT_SPREAD_VOL = {'conservative': 0.18, 'balanced': 0.22, 'aggressive': 0.25}
MAX_SPREAD_PERCENT_NORMAL = float(os.getenv('BOT_MAX_SPREAD_PERCENT_NORMAL', str(DEFAULT_SPREAD_NORMAL.get(AGGR, 0.12))))
MAX_SPREAD_PERCENT_VOLATILE = float(os.getenv('BOT_MAX_SPREAD_PERCENT_VOLATILE', str(DEFAULT_SPREAD_VOL.get(AGGR, 0.22))))
# Interval and frequency overrides by aggressiveness
MIN_TRADE_INTERVAL = int(os.getenv('BOT_MIN_TRADE_INTERVAL', str({'conservative': 45, 'balanced': 30, 'aggressive': 20}.get(AGGR, 30))))
MAX_TRADES_PER_HOUR = int(os.getenv('BOT_MAX_TRADES_PER_HOUR', str({'conservative': 12, 'balanced': 20, 'aggressive': 30}.get(AGGR, 20))))

# Log selected symbol/timeframe for visibility
logging.info(f"Bot config => SYMBOL={SYMBOL}, TIMEFRAME={TIMEFRAME}")
logging.info(f"Paper trading: {'ON' if PAPER_TRADING else 'OFF'} | Log level: {LOG_LEVEL}")
logging.info(
    "Tuning => aggr=%s, pred_thresh=%.2f, conf_buy=%d, conf_sell=%d, "
    "spread(normal=%.2f%%, vol=%.2f%%), min_interval=%ss, max_trades/hr=%s",
    AGGR, PREDICTION_THRESHOLD, CONFIRMATIONS_REQUIRED_BUY, CONFIRMATIONS_REQUIRED_SELL,
    MAX_SPREAD_PERCENT_NORMAL, MAX_SPREAD_PERCENT_VOLATILE,
    MIN_TRADE_INTERVAL, MAX_TRADES_PER_HOUR,
)
diag_flag = os.getenv('BOT_DIAGNOSTICS', 'false').lower() in ('1','true','yes','y')
logging.info(f"Diagnostics: {'ON' if diag_flag else 'OFF'}")

# ML Model Parameters
FEATURE_WINDOW = 20
# Note: PREDICTION_THRESHOLD is set above from env/presets
MODEL_RETRAIN_INTERVAL_DEFAULT = 1000  # default retrain frequency

class MarketRegime:
    TRENDING = 'TRENDING'
    RANGING = 'RANGING'
    VOLATILE = 'VOLATILE'
    UNKNOWN = 'UNKNOWN'


class PredictiveTrader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.historical_data = deque(maxlen=5000)
        self.last_train_size = 0

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create feature matrix for the ML model."""
        features = []
        returns = df['close'].pct_change()
        features.extend([
            returns.rolling(window=5).mean(),
            returns.rolling(window=5).std(),
            (df['high'] - df['low']) / df['low'],
            (df['close'] - df['open']) / df['open'],
        ])
        bb = BollingerBands(df['close'])
        features.extend([
            (df['close'] - bb.bollinger_lband()) /
            (bb.bollinger_hband() - bb.bollinger_lband()),
            df['volume'].rolling(window=5).mean() /
            df['volume'].rolling(window=20).mean(),
        ])
        features.extend([
            df['close'].rolling(window=10).mean(),
            df['close'].rolling(window=50).mean(),
            df['volume'].rolling(window=10).std(),
            (df['close'] - df['close'].rolling(window=10).mean()) /
            df['close'].rolling(window=10).std(),
        ])
        
        # Combine features into a matrix
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values by replacing them with 0 or using an imputer
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return feature_matrix

    def prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create binary labels for price direction (1 for up, 0 for down)."""
        future_returns = df['close'].shift(-1).pct_change()
        return (future_returns > 0).astype(int)

    def train_model(self, df: pd.DataFrame) -> bool:
        """Train the ML model using available data."""
        if len(df) < FEATURE_WINDOW + 10:
            return False
        X = self.prepare_features(df)
        y = self.prepare_labels(df)
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        if len(X) < 100:
            return False
        X = self.scaler.fit_transform(X)
        # Exclude the last data point since it has no future label
        self.model.fit(X[:-1], y[:-1])
        return True

    def predict_direction(self, df: pd.DataFrame) -> float:
        """Return the probability of price moving up in the next candle."""
        try:
            X = self.prepare_features(df.tail(FEATURE_WINDOW))
            # Handle NaN values
            if np.isnan(X).any():
                # Replace NaN with 0 for features
                X = np.nan_to_num(X, nan=0.0)
            X = self.scaler.transform(X)
            # Double check for NaNs after transformation
            if np.isnan(X).any():
                logging.warning("NaN values detected after scaling. Using default probability.")
                return 0.5
            probabilities = self.model.predict_proba(X[-1:])
            return probabilities[0][1]
        except Exception as e:
            logging.error(f"Error in predict_direction: {str(e)}")
            return 0.5


class RiskManager:
    def __init__(self, max_drawdown: float = 0.15, max_position_size: float = 0.3):
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.last_trade_day = None

    def update_metrics(self, current_balance_usd: float):
        """
        Update peak balance and drawdown based on total portfolio value in USD.
        Also resets daily metrics when a new day starts.
        """
        if current_balance_usd > self.peak_balance:
            self.peak_balance = current_balance_usd
        if self.peak_balance > 0:
            self.current_drawdown = (
                self.peak_balance - current_balance_usd
            ) / self.peak_balance
        # Reset daily metrics on a new day
        current_day = time.strftime('%Y-%m-%d')
        if current_day != self.last_trade_day:
            self.daily_loss = 0.0
            self.daily_trades = 0
            self.last_trade_day = current_day

    def can_trade(self, current_balance_usd: float, proposed_position_size: float, price: float) -> bool:
        """
        Determine whether a new position should be opened based on drawdown,
        position sizing, and daily trade limits.
        """
        if self.current_drawdown >= self.max_drawdown:
            logging.warning(
                f"Max drawdown reached: {self.current_drawdown:.2%}")
            return False
        position_value = abs(proposed_position_size) * price if price else 0.0
        if current_balance_usd == 0:
            return False
        ratio = (position_value / current_balance_usd) if current_balance_usd > 0 else 1.0
        if ratio > self.max_position_size:
            diag = os.getenv('BOT_DIAGNOSTICS', 'false').lower() in ('1','true','yes','y')
            msg = (
                f"Position size too large: {ratio:.2%} "
                f"(allowed ≤ {self.max_position_size:.2%}, value=${position_value:,.2f}, equity=${current_balance_usd:,.2f})"
            )
            if diag:
                logging.warning(msg)
            else:
                logging.debug(msg)
            return False
        # Stop trading if daily losses exceed 5%
        if self.daily_loss < -0.05:
            logging.warning(
                f"Daily loss limit reached: {self.daily_loss:.2%}")
            return False
        if self.daily_trades >= MAX_TRADES_PER_HOUR * 24:
            logging.warning("Daily trade limit reached")
            return False
        return True

    def update_trade_result(self, profit_loss: float) -> None:
        self.daily_loss += profit_loss
        self.daily_trades += 1


# Initialize tracking variables
last_trade_time = 0.0
trade_count = 0
last_trade_hour = time.strftime('%H')
highest_balance = 0.0


def retry_api_call(api_func, retries: int = 3, delay: int = 2):
    for attempt in range(retries):
        try:
            return api_func()
        except ccxt.NetworkError as e:
            logging.warning(
                f"Network error: {e}. Retrying in {delay} s...")
        except ccxt.ExchangeError as e:
            logging.warning(
                f"Exchange error: {e}. Retrying in {delay} s...")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None
        time.sleep(delay * (2 ** attempt))
    return None


def fetch_data():
    """Fetch OHLCV data and return as a DataFrame."""
    try:
        data = retry_api_call(lambda: pd.DataFrame(
            exchange.fetch_ohlcv(
                SYMBOL, timeframe=TIMEFRAME, limit=1000),
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ))
        if data is not None and not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
        else:
            logging.warning("Fetched data is empty.")
        return data
    except ccxt.BaseError as e:
        logging.error(f"Exchange error while fetching data: {str(e)}")
        return None
    except FileNotFoundError as e:
        logging.error(f"File error: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching data: {str(e)}")
        return None


def fetch_balance():
    return retry_api_call(lambda: exchange.fetch_balance())


def fetch_ticker_price():
    price_data = retry_api_call(lambda: exchange.fetch_ticker(SYMBOL))
    return price_data.get('last', 0) if price_data else 0


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    try:
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=period - 1, adjust=False).mean()
        ema_down = down.ewm(com=period - 1, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating RSI: {str(e)}")
        return 50.0


def calculate_sma(df: pd.DataFrame, period: int = 200) -> float:
    try:
        return df['close'].rolling(window=period).mean().iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating SMA: {str(e)}")
        return df['close'].iloc[-1]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        df_tmp = df.copy()
        df_tmp['prev_close'] = df_tmp['close'].shift(1)
        tr1 = df_tmp['high'] - df_tmp['low']
        tr2 = (df_tmp['high'] - df_tmp['prev_close']).abs()
        tr3 = (df_tmp['low'] - df_tmp['prev_close']).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        return atr
    except Exception as e:
        logging.error(f"Error calculating ATR: {str(e)}")
        return max((df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.1, 1e-6)


def calculate_macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and signal line for the given DataFrame."""
    try:
        macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line, signal_line
    except Exception as e:
        logging.error(f"Error calculating MACD: {str(e)}")
        zeros = pd.Series([0] * len(df), index=df.index)
        return zeros, zeros


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> float:
    return df['volume'].rolling(window=period).mean().iloc[-1]


def check_trend(df: pd.DataFrame, sma_val: float, price: float) -> bool:
    return (price > sma_val and df['close'].iloc[-1] > df['close'].iloc[-2])


def check_volume(df: pd.DataFrame) -> bool:
    """Volume confirmation with tunable boost multiplier.
    Set env BOT_VOLUME_BOOST (e.g. 1.05 for +5%) to relax/tighten.
    """
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    try:
        vol_boost = float(os.getenv('BOT_VOLUME_BOOST', '1.10'))
    except Exception:
        vol_boost = 1.10
    # Guard against zero/NaN averages
    if not np.isfinite(avg_volume) or avg_volume <= 0:
        return False
    return current_volume > avg_volume * vol_boost


def get_trade_size(balance: dict, price: float, atr: float,
                   rsi_val: float = 50.0, momentum_score_val: float = 1.0,
                   risk_per_trade: Optional[float] = None,
                   atr_multiplier: Optional[float] = None) -> float:
    """
    Compute the position size in ETH based on risk and momentum.
    Uses the total account value in USD and adjusts position size based on
    momentum score. Returns 0 if price or atr is invalid.
    """
    if price <= 0 or atr <= 0:
        return 0.0
    # Extract free balances
    eth_free = balance['free'].get('ETH', 0.0)
    usdt_free = balance['free'].get('USDT', 0.0)
    eth_value_usd = eth_free * price
    total_account_value_usd = eth_value_usd + usdt_free
    rpt = RISK_PER_TRADE if risk_per_trade is None else float(risk_per_trade)
    risk_amount = total_account_value_usd * rpt
    if abs(rsi_val - 50) > 20:
        risk_amount *= 1.5
    # ATR is already in quote currency (USDT). Allow tuning via BOT_ATR_MULTIPLIER
    atr_mult = float(os.getenv('BOT_ATR_MULTIPLIER', str(atr_multiplier if atr_multiplier is not None else 1.0)))
    # Apply a minimum ATR floor to avoid oversized positions when ATR is extremely small
    min_atr_usd = float(os.getenv('BOT_MIN_ATR_USD', '0'))
    eff_atr = max(atr, min_atr_usd)
    stop_distance_usd = eff_atr * atr_mult
    if stop_distance_usd <= 0:
        return 0.0
    trade_size_eth = risk_amount / stop_distance_usd
    # Respect exchange limits (amount precision, min cost) when possible
    trade_size_eth = adjust_amount_for_market(trade_size_eth, price)
    # Use momentum score to scale size
    if momentum_score_val > 1.5:
        trade_size_eth *= 1.2
    return max(trade_size_eth, 0.0)


def adjust_amount_for_market(amount: float, price: float) -> float:
    """Round amount to market precision and enforce minimum notional/amount.
    Falls back to a $10 notional minimum when limits are unavailable.
    """
    try:
        if amount <= 0 or price <= 0:
            return 0.0
        # Round to exchange precision
        try:
            amount_precise = float(exchange.amount_to_precision(SYMBOL, amount))
        except Exception:
            amount_precise = amount
        # Enforce min cost/amount if available
        market = exchange.market(SYMBOL)
        min_cost = None
        min_amount = None
        if market and isinstance(market, dict):
            limits = market.get('limits') or {}
            cost_limits = limits.get('cost') or {}
            amt_limits = limits.get('amount') or {}
            min_cost = cost_limits.get('min')
            min_amount = amt_limits.get('min')
        if min_amount:
            amount_precise = max(amount_precise, float(min_amount))
        # If min_cost known, lift amount to satisfy it
        if min_cost and price > 0:
            amount_precise = max(amount_precise, float(min_cost) / price)
        # Fallback to $10 notional if nothing else available
        if (not min_cost) and price > 0 and amount_precise * price < 10.0:
            amount_precise = 10.0 / price
        # Final precision pass
        try:
            amount_precise = float(exchange.amount_to_precision(SYMBOL, amount_precise))
        except Exception:
            pass
        return max(amount_precise, 0.0)
    except Exception as e:
        logging.warning(f"Adjust amount failed; using raw amount. Err: {e}")
        return max(amount, 0.0)


def get_order_book_spread():
    order_book = retry_api_call(lambda: exchange.fetch_order_book(SYMBOL))
    if not order_book:
        return None, None, None
    bids = order_book['bids']
    asks = order_book['asks']
    if not bids or not asks:
        return None, None, None
    bid_price = bids[0][0]
    ask_price = asks[0][0]
    spread = ask_price - bid_price
    return bid_price, ask_price, spread


def log_trade_metrics(side: str, trade_size: float, price: float, indicators: dict) -> None:
    logging.info(f"""
====== TRADE EXECUTED ======
Side: {side.upper()}
Size: {trade_size:.6f} ETH
Price: ${price:.2f}
Notional: ${(trade_size * price):.2f}

Indicators:
- RSI: {indicators['rsi']:.2f}
- MACD: {indicators['macd']:.6f}
- Signal: {indicators['signal']:.6f}
- Volume MA Ratio: {indicators['volume_ratio']:.2f}
==========================
""")


def place_order(side: str, trade_size: float, current_price: float,
                df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Place a market order while respecting rate limits, spread thresholds, and
    daily trade limits. Requires the current price and latest data frame
    for logging indicators.
    """
    global last_trade_time, trade_count, last_trade_hour
    now = time.time()
    # Respect the minimum trade interval
    if (now - last_trade_time) < MIN_TRADE_INTERVAL:
        logging.warning("Trade interval too short, skipping.")
        return None
    current_hour = time.strftime('%H')
    if current_hour != last_trade_hour:
        trade_count = 0
        last_trade_hour = current_hour
    if trade_count >= MAX_TRADES_PER_HOUR:
        logging.warning("Reached max trades per hour, skipping trade.")
        return None
    bid_price, ask_price, spread = get_order_book_spread()
    if spread is None or ask_price is None or ask_price == 0:
        logging.warning("Could not fetch valid order book. Skipping trade.")
        return None
    mid_price = (bid_price + ask_price) / 2 if bid_price and ask_price else current_price
    base_price = mid_price if mid_price and mid_price > 0 else ask_price
    spread_percentage = (spread / base_price) * 100 if base_price else 100.0
    regime = detect_market_regime(df)
    max_spread_percentage = (
        MAX_SPREAD_PERCENT_VOLATILE if regime == MarketRegime.VOLATILE else MAX_SPREAD_PERCENT_NORMAL
    )
    if spread_percentage > max_spread_percentage:
        logging.warning(
            f"Spread too high ({spread_percentage:.2f}%). Skipping trade.")
        return None
    # Cap trade size by available balance (for buys) with small fee buffer
    try:
        if side == 'buy':
            bal = fetch_balance() or {}
            usdt_free = float((bal.get('free') or {}).get('USDT', 0.0))
            if current_price > 0 and usdt_free > 0:
                max_by_usdt = (usdt_free * 0.995) / current_price
                trade_size = min(trade_size, max_by_usdt)
    except Exception:
        pass
    # Normalize trade size to market constraints
    trade_size = adjust_amount_for_market(trade_size, current_price)
    # For sells, ensure we do not exceed free ETH after precision/min-cost bumps
    try:
        if side == 'sell':
            bal = fetch_balance() or {}
            eth_free = float((bal.get('free') or {}).get('ETH', 0.0))
            if eth_free > 0:
                trade_size = min(trade_size, eth_free * 0.999)
    except Exception:
        pass
    # Final sanity: skip orders below exchange minimums (prevents InvalidOrder spam)
    try:
        market = exchange.market(SYMBOL)
        limits = (market or {}).get('limits') or {}
        min_amount = float((limits.get('amount') or {}).get('min') or 0.0)
        min_cost = float((limits.get('cost') or {}).get('min') or 0.0)
        notional = (trade_size * current_price) if (trade_size and current_price) else 0.0
        below_amt = (min_amount > 0 and trade_size < min_amount)
        below_cost = (min_cost > 0 and notional < min_cost)
        if below_amt or below_cost:
            reason = []
            if below_amt:
                reason.append(f"amount {trade_size:.8f} < min {min_amount}")
            if below_cost:
                reason.append(f"notional ${notional:.2f} < min ${min_cost:.2f}")
            logging.info(f"Order skipped: below exchange minimums ({', '.join(reason)})")
            return None
    except Exception:
        # If we cannot read limits, continue; exchange will validate.
        pass
    if trade_size <= 0:
        logging.warning("Trade size <= 0 after adjustment; skipping.")
        return None
    try:
        # Paper mode: do not place live orders
        if PAPER_TRADING:
            last_trade_time = now  # type: ignore[assignment]
            trade_count += 1  # type: ignore[assignment]
            log_trade_metrics(side, trade_size, current_price, {
                'rsi': calculate_rsi(df, RSI_PERIOD),
                'macd': calculate_macd(df)[0].iloc[-1],
                'signal': calculate_macd(df)[1].iloc[-1],
                'volume_ratio': df['volume'].iloc[-1] / calculate_volume_ma(df)
            })
            fake_order = {
                'id': f'paper-{int(now)}',
                'symbol': SYMBOL,
                'side': side,
                'type': 'market',
                'amount': trade_size,
                'price': current_price,
                'info': {'paper': True}
            }
            logging.info(f"[PAPER] {side.upper()} {trade_size:.6f} {SYMBOL.split('/')[0]} @ ${current_price:.2f}")
            return fake_order
        # Live order
        if not exchange.has or not exchange.has.get('createMarketOrder', True):
            logging.error("Exchange does not support market orders via CCXT.")
            return None
        order = exchange.create_market_order(SYMBOL, side, abs(trade_size))
        last_trade_time = now
        trade_count += 1
        log_trade_metrics(side, trade_size, current_price, {
            'rsi': calculate_rsi(df, RSI_PERIOD),
            'macd': calculate_macd(df)[0].iloc[-1],
            'signal': calculate_macd(df)[1].iloc[-1],
            'volume_ratio': df['volume'].iloc[-1] / calculate_volume_ma(df)
        })
        return order
    except ccxt.InsufficientFunds as e:
        logging.error(f"Insufficient funds for {side} order: {e}")
    except ccxt.InvalidOrder as e:
        logging.error(f"Invalid order parameters: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in order placement: {e}")
    return None


def check_profit_loss(current_price: float) -> None:
    global highest_balance
    balance = fetch_balance()
    if not balance:
        logging.warning("Balance data unavailable. Skipping trailing stop check.")
        return
    eth_total = balance['total'].get('ETH', 0.0)
    usdt_total = balance['total'].get('USDT', 0.0)
    eth_value_usd = eth_total * current_price
    overall_portfolio_usd = usdt_total + eth_value_usd
    if overall_portfolio_usd > highest_balance:
        highest_balance = overall_portfolio_usd
    if overall_portfolio_usd <= highest_balance * (1 - TRAILING_STOP_PERCENT):
        logging.warning("Trailing stop triggered. Exiting bot.")
        raise SystemExit("Trailing stop reached")


def manage_take_profits(current_position: float, entry_price: float, current_price: float,
                        df: pd.DataFrame) -> float:
    """
    Determine how much of the current position to sell based on take‑profit
    levels and market regime. Returns the quantity of ETH to sell.
    """
    if current_position <= 0 or entry_price <= 0:
        return 0.0
    price_change = (current_price - entry_price) / entry_price
    # Local take‑profit levels based on market regime
    regime = detect_market_regime(df)
    if regime == MarketRegime.TRENDING:
        tp_levels = [level * 2 for level in TAKE_PROFIT_LEVELS]
    elif regime == MarketRegime.RANGING:
        tp_levels = TAKE_PROFIT_LEVELS
    elif regime == MarketRegime.VOLATILE:
        tp_levels = [level * 1.5 for level in TAKE_PROFIT_LEVELS]
    else:
        tp_levels = TAKE_PROFIT_LEVELS
    for level, scale_out in zip(tp_levels, POSITION_SCALE_OUT):
        if price_change >= level:
            amount_to_sell = current_position * scale_out
            logging.info(
                f"Take profit triggered at {level * 100:.2f}%, selling {scale_out * 100:.2f}% of position")
            return amount_to_sell
    return 0.0


def calculate_momentum_score(df: pd.DataFrame) -> float:
    rsi = calculate_rsi(df)
    rsi_score = abs(rsi - 50) / 50
    price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
    price_score = min(abs(price_change), 1.0)
    vol_avg = df['volume'].rolling(10).mean().iloc[-1]
    # Add check for zero volume average
    vol_score = 0.0
    if vol_avg > 0:
        vol_score = min(df['volume'].iloc[-1] / vol_avg, 2.0) - 1.0
    return max((rsi_score + price_score + max(0.0, vol_score)) / 1.5, 0.0)


def should_reenter_position(df: pd.DataFrame, last_exit_price: float, current_price: float,
                            side: str = 'buy') -> bool:
    if side == 'buy':
        return (current_price < last_exit_price * 0.995 and
                calculate_momentum_score(df) > 1.2)
    else:
        return (current_price > last_exit_price * 1.005 and
                calculate_momentum_score(df) > 1.2)


def detect_market_regime(df: pd.DataFrame, window: int = 20) -> str:
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    atr = calculate_atr(df)
    bb = BollingerBands(df['close'])
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    current_volatility = volatility.iloc[-1]
    current_bb_width = bb_width.iloc[-1]
    high_volatility = current_volatility > volatility.quantile(0.7)
    wide_bb = current_bb_width > bb_width.quantile(0.7)
    sma_short = df['close'].rolling(window=10).mean()
    sma_long = df['close'].rolling(window=30).mean()
    trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / max(atr, 1e-6)
    if high_volatility and wide_bb:
        return MarketRegime.VOLATILE
    elif trend_strength > 1.5:
        return MarketRegime.TRENDING
    elif trend_strength < 0.5:
        return MarketRegime.RANGING
    else:
        return MarketRegime.UNKNOWN


def adjust_parameters_for_regime(regime: str) -> dict:
    """
    Return adjusted risk parameters based on market regime without mutating
    global constants. The calling function can choose to apply these values.
    """
    risk = RISK_PER_TRADE
    trailing_stop = TRAILING_STOP_PERCENT
    tp_levels = TAKE_PROFIT_LEVELS.copy()
    if regime == MarketRegime.VOLATILE:
        return {
            'risk_per_trade': risk * 0.8,
            'trailing_stop': trailing_stop * 1.5,
            'take_profits': [level * 1.5 for level in tp_levels]
        }
    if regime == MarketRegime.TRENDING:
        return {
            'risk_per_trade': risk * 1.2,
            'trailing_stop': trailing_stop * 1.2,
            'take_profits': [level * 1.3 for level in tp_levels]
        }
    if regime == MarketRegime.RANGING:
        return {
            'risk_per_trade': risk * 0.9,
            'trailing_stop': trailing_stop * 0.8,
            'take_profits': [level * 0.7 for level in tp_levels]
        }
    return {
        'risk_per_trade': risk,
        'trailing_stop': trailing_stop,
        'take_profits': tp_levels
    }


# Initialize components
trader = PredictiveTrader()
risk_manager = RiskManager()


def run_bot():
    global highest_balance
    print("\n=== Trading Bot Starting ===")
    print("Press Ctrl+C to stop the bot\n")
    logging.info("=== Bot Session Started ===")
    
    # Initialize tracking variables
    candle_count = 0  # Add candle_count initialization here
    model_retrain_interval = MODEL_RETRAIN_INTERVAL_DEFAULT
    current_position = 0.0
    entry_price = 0.0
    peak_price_since_entry = 0.0  # for per-position trailing stop
    last_exit_price = 0.0
    last_trade_side = None
    last_entry_time = 0.0
    
    # Initial data fetch
    print("Fetching initial market data...")
    logging.info("Fetching initial market data...")
    df = fetch_data()
    if df is None or df.empty:
        logging.error("Could not fetch initial data. Exiting.")
        print("ERROR: Could not fetch initial data. Exiting.")
        return

    print(f"Successfully fetched {len(df)} candles")
    print("Training initial ML model...")
    
    # Train initial model with retries
    for attempt in range(3):
        if trader.train_model(df):
            logging.info("Initial ML model training successful.")
            break
        logging.warning(f"Initial training attempt {attempt + 1} failed. Retrying...")
        time.sleep(5)
    else:
        logging.error("Initial model training failed after multiple attempts. Exiting.")
        print("ERROR: Initial model training failed after multiple attempts. Exiting.")
        return

    print("\nBot is now running!")
    print("Monitoring market conditions...\n")
    
    iteration = 0
    while True:
        try:
            iteration += 1
            if iteration % 10 == 0:  # Every 10 iterations
                print(f"\nIteration {iteration} - Current Status:")
                price = fetch_ticker_price()
                print(f"Current ETH Price: ${price:.2f}")
                print(f"Last Analysis Time: {time.strftime('%H:%M:%S')}")

            # Reset indicators each iteration
            rsi_val = 50.0
            sma_val = 0.0
            macd_line = pd.Series([0])
            signal_line = pd.Series([0])
            atr_val = 0.0
            trend_bullish = False
            volume_confirmed = False
            momentum_score = 0.0
            up_probability = 0.5

            # Fetch data and current price
            df = fetch_data()
            if df is None or df.empty:
                logging.warning("No market data returned. Sleeping 60 s.")
                time.sleep(60)
                continue
            price = fetch_ticker_price()
            if price <= 0:
                logging.warning("Invalid price received. Sleeping 60 s.")
                time.sleep(60)
                continue

            # Calculate indicators
            try:
                # Use only closed candles for signal computation
                df_sig = df.iloc[:-1] if len(df) > 1 else df
                rsi_val = calculate_rsi(df_sig, RSI_PERIOD)
                sma_val = calculate_sma(df_sig, SMA_PERIOD)
                macd_line, signal_line = calculate_macd(df_sig)
                atr_val = calculate_atr(df_sig, 14)
                trend_bullish = check_trend(df_sig, sma_val, price)
                volume_confirmed = check_volume(df_sig)
                momentum_score = calculate_momentum_score(df_sig)
                if iteration % 10 == 0:  # Every 10 iterations
                    print(f"RSI: {rsi_val:.2f}")
                    print(f"Trend: {'Bullish' if trend_bullish else 'Bearish'}")
                    print(f"ML Up Probability: {up_probability:.2%}\n")
            except Exception as e:
                logging.error(f"Error calculating indicators: {e}")

            # Retrain model periodically
            candle_count += 1
            if candle_count % model_retrain_interval == 0:
                try:
                    logging.info("Retraining ML model...")
                    if trader.train_model(df.iloc[:-1] if len(df) > 1 else df):
                        logging.info("ML model retrained successfully.")
                    else:
                        logging.warning("Insufficient data for retraining ML model.")
                except Exception as e:
                    logging.error(f"Error during ML model retraining: {str(e)}")

            # Adjust retrain interval based on volatility
            if detect_market_regime(df) == MarketRegime.VOLATILE:
                model_retrain_interval = max(500, MODEL_RETRAIN_INTERVAL_DEFAULT // 2)
            else:
                model_retrain_interval = MODEL_RETRAIN_INTERVAL_DEFAULT

            # Predict direction
            try:
                up_probability = trader.predict_direction(df)
            except Exception as e:
                logging.error(f"Error in price prediction: {e}")
                up_probability = 0.5

            # Fetch balance and compute portfolio value
            try:
                current_balance = fetch_balance()
                if not current_balance:
                    logging.warning("Could not fetch balance. Sleeping 60 s.")
                    time.sleep(60)
                    continue
            except Exception as e:
                logging.error(f"Error fetching balance: {e}")
                time.sleep(60)
                continue

            eth_total = current_balance['total'].get('ETH', 0.0)
            usdt_total = current_balance['total'].get('USDT', 0.0)
            portfolio_value_usd = usdt_total + eth_total * price

            # Update risk metrics
            risk_manager.update_metrics(portfolio_value_usd)

            # Copy base risk per trade and adjust for drawdown (local)
            if risk_manager.current_drawdown < 0.05:
                current_risk_per_trade = RISK_PER_TRADE * 1.1
            elif risk_manager.current_drawdown > 0.1:
                current_risk_per_trade = RISK_PER_TRADE * 0.8
            else:
                current_risk_per_trade = RISK_PER_TRADE

            # Determine trade size
            trade_size = get_trade_size(
                current_balance, price, atr_val,
                rsi_val=rsi_val, momentum_score_val=momentum_score,
                risk_per_trade=current_risk_per_trade
            )

            # Clamp to the maximum allowed position size before risk check
            if price and price > 0 and portfolio_value_usd > 0:
                max_pos_value = float(getattr(risk_manager, 'max_position_size', 0.3)) * float(portfolio_value_usd)
                cap_size_eth = max_pos_value / float(price)
                if cap_size_eth > 0:
                    if trade_size > cap_size_eth:
                        logging.debug(
                            f"Clamping trade size from {trade_size:.6f} to {cap_size_eth:.6f} ETH to respect position cap"
                        )
                    trade_size = min(trade_size, cap_size_eth)

            # Check risk manager before trading
            if not risk_manager.can_trade(portfolio_value_usd, trade_size, price):
                msg = (
                    f"Trade skipped due to risk rules | "
                    f"Drawdown={risk_manager.current_drawdown:.2%}, "
                    f"DailyLoss={risk_manager.daily_loss:.2%}, "
                    f"Size={trade_size:.6f} ETH"
                )
                if os.getenv('BOT_DIAGNOSTICS', 'false').lower() in ('1','true','yes','y'):
                    logging.info(msg)
                else:
                    logging.debug(msg)
                trade_size = 0.0

            # Determine buy/sell signals (tunable + regime-adaptive)
            macd_up = macd_line.iloc[-1] > signal_line.iloc[-1]
            macd_down = macd_line.iloc[-1] < signal_line.iloc[-1]
            confs_buy = [macd_up, trend_bullish, volume_confirmed]
            confs_sell = [macd_down, (not trend_bullish), volume_confirmed]
            confirmations_met_buy = sum(1 for c in confs_buy if c)
            confirmations_met_sell = sum(1 for c in confs_sell if c)

            # Regime-aware confirmation requirements: relax in strong trends
            regime_now = detect_market_regime(df_sig)
            req_buy = max(0, CONFIRMATIONS_REQUIRED_BUY)
            if regime_now == MarketRegime.TRENDING:
                req_buy = max(1, req_buy - 1)
            confirmations_ok_buy = confirmations_met_buy >= req_buy
            confirmations_ok_sell = confirmations_met_sell >= max(0, CONFIRMATIONS_REQUIRED_SELL)

            # Primary buy: strict ML threshold
            buy_signal = (
                trade_size > 0 and
                up_probability > PREDICTION_THRESHOLD and
                rsi_val < RSI_BUY_MAX and
                confirmations_ok_buy
            )

            # Soft buy: slightly lower ML threshold with at least 1 confirmation
            try:
                soft_delta = float(os.getenv('BOT_SOFT_BUY_DELTA', '0.07'))
            except Exception:
                soft_delta = 0.07
            soft_threshold = max(0.50, PREDICTION_THRESHOLD - soft_delta)
            soft_buy_signal = (
                (not buy_signal) and
                trade_size > 0 and
                up_probability >= soft_threshold and
                rsi_val < min(RSI_BUY_MAX + 5, 80) and
                confirmations_met_buy >= 1
            )
            # Use available free ETH for sell gating instead of trade_size
            available_to_sell_gate = float(current_balance['free'].get('ETH', 0.0))
            # Add probability hysteresis to reduce flip-flops
            try:
                sell_hyst = float(os.getenv('BOT_SELL_HYSTERESIS', '0.05'))
            except Exception:
                sell_hyst = 0.05
            sell_cutoff = max(0.0, 1 - PREDICTION_THRESHOLD - max(0.0, min(sell_hyst, 0.2)))
            sell_signal = (
                available_to_sell_gate > 0 and
                up_probability < sell_cutoff and
                rsi_val > RSI_SELL_MIN and
                confirmations_ok_sell
            )

            # Soft sell: slightly higher ML threshold with at least 1 confirmation
            try:
                soft_sell_delta = float(os.getenv('BOT_SOFT_SELL_DELTA', '0.07'))
            except Exception:
                soft_sell_delta = 0.07
            soft_sell_threshold = max(0.0, min(1.0, 1.0 - PREDICTION_THRESHOLD + soft_sell_delta))
            soft_sell_signal = (
                (not sell_signal) and
                available_to_sell_gate > 0 and
                up_probability <= soft_sell_threshold and
                rsi_val > max(RSI_SELL_MIN - 5, 20) and
                confirmations_met_sell >= 1
            )

            # Strong bearish evaluation (for full-exit and hold override)
            try:
                strong_delta_eval = float(os.getenv('BOT_STRONG_SELL_DELTA', '0.10'))
            except Exception:
                strong_delta_eval = 0.10
            strong_threshold_eval = max(0.0, 1.0 - PREDICTION_THRESHOLD - strong_delta_eval)
            strong_bearish_now = (up_probability <= strong_threshold_eval and macd_down and (not trend_bullish))

            # Diagnostics when no trade
            if not buy_signal and not sell_signal:
                reasons = []
                if trade_size <= 0 and available_to_sell_gate <= 0:
                    reasons.append("size<=0")
                if up_probability <= PREDICTION_THRESHOLD and up_probability >= (1 - PREDICTION_THRESHOLD):
                    reasons.append(f"ml_prob={up_probability:.2f} near threshold")
                if rsi_val >= RSI_BUY_MAX:
                    reasons.append(f"rsi_buy_gate (RSI={rsi_val:.1f} >= {RSI_BUY_MAX})")
                if rsi_val <= RSI_SELL_MIN:
                    reasons.append(f"rsi_sell_gate (RSI={rsi_val:.1f} <= {RSI_SELL_MIN})")
                if not confirmations_ok_buy and not confirmations_ok_sell:
                    reasons.append(
                        f"confirmations unmet (buy={confirmations_met_buy}/{CONFIRMATIONS_REQUIRED_BUY}, "
                        f"sell={confirmations_met_sell}/{CONFIRMATIONS_REQUIRED_SELL})"
                    )
                if reasons:
                    msg = "No trade: " + "; ".join(reasons)
                    if os.getenv('BOT_DIAGNOSTICS', 'false').lower() in ('1','true','yes','y'):
                        logging.info(msg)
                    else:
                        logging.debug(msg)

            # Execute buy signal
            if buy_signal or soft_buy_signal:
                logging.info(
                    f"Buy signal detected (ML Prob: {up_probability:.2f}{' soft' if soft_buy_signal and not buy_signal else ''})")
                # Scale down size for soft buys
                if soft_buy_signal and not buy_signal:
                    trade_size *= float(os.getenv('BOT_SOFT_BUY_SIZE_FACTOR', '0.5'))
                    trade_size = max(trade_size, 0.0)
                # Place order and ensure no pre‑existing long is counted as multiple
                order = place_order('buy', trade_size, price, df_sig)
                if order:
                    current_position += trade_size
                    entry_price = price
                    peak_price_since_entry = price
                    last_trade_side = 'buy'
                    last_entry_time = time.time()

            # Execute sell signal (closing or scaling out)
            elif sell_signal or soft_sell_signal:
                logging.info(
                    f"Sell signal detected (ML Prob: {up_probability:.2f}{' soft' if soft_sell_signal and not sell_signal else ''})")
                # Only sell up to the size of current position
                available_to_sell = float(current_balance['free'].get('ETH', 0.0))
                # Minimum hold logic to avoid instant flip after buy
                try:
                    min_hold_sec = float(os.getenv('BOT_MIN_HOLD_SEC', '120'))
                except Exception:
                    min_hold_sec = 120.0
                try:
                    atr_within_hold_mult = float(os.getenv('BOT_SELL_ATR_WITHIN_HOLD', '0.25'))
                except Exception:
                    atr_within_hold_mult = 0.25
                now_ts = time.time()
                hold_active = (entry_price > 0 and current_position > 0 and (now_ts - (locals().get('last_entry_time', 0.0) or 0.0)) < min_hold_sec)
                allow_sell_during_hold = strong_bearish_now or (
                    entry_price > 0 and atr_val > 0 and price <= entry_price - atr_val * max(0.0, atr_within_hold_mult)
                )
                if hold_active and not allow_sell_during_hold:
                    logging.info(
                        f"Sell suppressed by min-hold ({int(min_hold_sec)}s). Awaiting strong/ATR/trailing exit.")
                    sell_signal = False
                    soft_sell_signal = False
                    # Skip to take-profit and trailing logic
                    pass
                # Scale down size for soft sells
                if soft_sell_signal and not sell_signal:
                    try:
                        trade_size *= float(os.getenv('BOT_SOFT_SELL_SIZE_FACTOR', '0.5'))
                    except Exception:
                        trade_size *= 0.5
                # Optional stronger exit if model is decisively bearish
                strong_sell = strong_bearish_now
                sell_full_flag = os.getenv('BOT_SELL_FULL_ON_STRONG_BEAR', 'true').lower() in ('1','true','yes','y')

                # Determine sell size; if computed size is zero, fallback to a fraction of holdings
                base_sell_size = trade_size
                if base_sell_size <= 0 and available_to_sell > 0:
                    try:
                        fb_soft = float(os.getenv('BOT_SOFT_SELL_FALLBACK_FRACTION', '0.10'))
                    except Exception:
                        fb_soft = 0.10
                    try:
                        fb_main = float(os.getenv('BOT_SELL_FALLBACK_FRACTION', '0.25'))
                    except Exception:
                        fb_main = 0.25
                    frac = fb_soft if (soft_sell_signal and not sell_signal) else fb_main
                    base_sell_size = max(available_to_sell * max(0.0, min(frac, 1.0)), 0.0)
                sell_size = min(base_sell_size, available_to_sell)
                if strong_sell and sell_full_flag and current_position > 0:
                    sell_size = min(current_position, available_to_sell)
                    logging.info("Strong bearish signal: selling full position")
                if sell_size > 0:
                    order = place_order('sell', sell_size, price, df_sig)
                    if order and entry_price > 0:
                        profit_loss = (price - entry_price) / entry_price
                        risk_manager.update_trade_result(profit_loss)
                        last_exit_price = price
                        last_trade_side = 'sell'
                    # Decrease current position accordingly
                    current_position = max(current_position - sell_size, 0.0)

            # Manage take profits on open long position
            if current_position > 0:
                tp_amount = manage_take_profits(
                    current_position, entry_price, price, df)
                if tp_amount > 0:
                    order = place_order('sell', tp_amount, price, df_sig)
                    if order:
                        current_position = max(current_position - tp_amount, 0.0)
                        logging.info(
                            f"Take profit executed, remaining position: {current_position:.6f} ETH")

            # Per-position trailing stop and ATR stop-loss (optional)
            if current_position > 0 and entry_price > 0:
                # Track peak price since entry for trailing
                peak_price_since_entry = max(peak_price_since_entry, price)
                try:
                    trail_pct = float(os.getenv('BOT_POS_TRAIL_PCT', str(TRAILING_STOP_PERCENT)))
                except Exception:
                    trail_pct = TRAILING_STOP_PERCENT
                # Regime-aware widening
                regime_now2 = detect_market_regime(df_sig)
                if regime_now2 == MarketRegime.TRENDING:
                    trail_pct *= 1.2
                elif regime_now2 == MarketRegime.VOLATILE:
                    trail_pct *= 1.5
                trail_stop_price = peak_price_since_entry * (1.0 - max(0.0, min(trail_pct, 0.95)))

                try:
                    stop_mult = float(os.getenv('BOT_STOP_ATR_MULT', '2.0'))
                except Exception:
                    stop_mult = 2.0
                hard_stop_price = entry_price - max(0.0, atr_val) * stop_mult

                exit_reason = None
                if price <= hard_stop_price:
                    exit_reason = f"ATR stop-loss hit (mult={stop_mult})"
                elif price <= trail_stop_price:
                    exit_reason = f"Trailing stop hit ({trail_pct*100:.2f}%)"

                if exit_reason:
                    available_to_sell = current_balance['free'].get('ETH', 0.0)
                    close_size = min(current_position, available_to_sell)
                    if close_size > 0:
                        logging.info(exit_reason + "; closing position")
                        order = place_order('sell', close_size, price, df_sig)
                        if order:
                            current_position = max(current_position - close_size, 0.0)
                            last_exit_price = price
                            last_trade_side = 'sell'
                            # reset peak since entry after closing
                            peak_price_since_entry = 0.0 if current_position == 0.0 else peak_price_since_entry

            # Check trailing stop
            check_profit_loss(price)
            time.sleep(MIN_TRADE_INTERVAL)
        except SystemExit as stop:
            logging.warning(str(stop))
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)


if __name__ == '__main__':
    run_bot()
