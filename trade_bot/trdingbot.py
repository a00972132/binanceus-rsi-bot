import ccxt
import pandas as pd
import time
import logging
import random
import os
from dotenv import load_dotenv

# Load environment variables
dotenv_path = "/Users/will/Desktop/Code/Tradingbot/binanceus_creds.env"
load_dotenv(dotenv_path=dotenv_path)

# Configure Binance API Keys
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    logging.error("API keys not found. Exiting.")
    exit()

# Configure logging
logging.basicConfig(
    filename="trading_bot.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("🔍 Trading bot started")

# ✅ Exchange Configuration
exchange = ccxt.binanceus({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True},
})

exchange.checkRequiredCredentials()
exchange.load_markets()
exchange.fetch_time()
logging.info("✅ Exchange configuration complete")

# 🔥 Trading Parameters
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
RSI_PERIOD = 14
SMA_PERIOD = 200
TRAILING_STOP_PERCENT = 0.05
MIN_TRADE_INTERVAL = 300           # Minimum 5 minutes between trades
MAX_TRADES_PER_HOUR = 3
RISK_PER_TRADE = 0.02  # 2% risk per trade

# Track Trading Data
last_trade_time = 0
trade_count = 0
last_trade_hour = time.strftime('%H')
initial_balance = None
highest_balance = 0

# ✅ Retry Logic for API Calls
def retry_api_call(api_func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return api_func()
        except ccxt.NetworkError as e:
            logging.warning(f"Network error: {e}. Retrying in {delay}s...")
        except ccxt.ExchangeError as e:
            logging.warning(f"Exchange error: {e}. Retrying in {delay}s...")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None
        time.sleep(delay * (2 ** attempt))
    return None

# ✅ Market Data Fetching
def fetch_data():
    """
    Fetch OHLCV data and return as a DataFrame.
    """
    return retry_api_call(lambda: pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=max(RSI_PERIOD, SMA_PERIOD) + 2),
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    ))

def fetch_balance():
    """
    Fetch account balance.
    """
    return retry_api_call(lambda: exchange.fetch_balance())

def fetch_ticker_price():
    """
    Fetch last ticker price for SYMBOL.
    """
    price_data = retry_api_call(lambda: exchange.fetch_ticker(SYMBOL))
    return price_data.get('last', 0) if price_data else 0

# 📊 Technical Indicators
def calculate_rsi(df, period=14):
    """
    Calculate RSI using Wilder's smoothing.
    """
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_sma(df, period=200):
    """
    Calculate Simple Moving Average.
    """
    return df['close'].rolling(window=period).mean().iloc[-1]

def calculate_atr(df, period=14):
    """
    Calculate Average True Range using True Range = max of:
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    and Wilder's smoothing for ATR.
    """
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    return df['atr'].iloc[-1]

def calculate_macd(df):
    """
    Calculate MACD line and Signal line.
    Returns (macd_line, signal_line).
    """
    macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    return macd_line, signal_line

def get_trade_size(price, balance, atr):
    """
    Position sizing based on RISK_PER_TRADE of free USDT, 
    divided by ATR as a volatility-based approximation.
    """
    usdt_free = balance['free'].get('USDT', 0)
    risk_amount = usdt_free * RISK_PER_TRADE
    if atr == 0 or price == 0:
        return 0
    # This is a simple method: trade_size = risk / atr
    # You could refine to incorporate an actual stop distance, etc.
    return (risk_amount / atr)

def get_order_book_spread():
    """
    Fetch order book and return the absolute spread.
    """
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

def place_order(side, trade_size):
    """
    Place a market order, respecting:
      - MIN_TRADE_INTERVAL
      - MAX_TRADES_PER_HOUR
      - Spread check (percentage-based)
    """
    global last_trade_time, trade_count, last_trade_hour

    now = time.time()
    # Check minimal interval
    if (now - last_trade_time) < MIN_TRADE_INTERVAL:
        logging.warning("Trade interval too short, skipping.")
        return None

    # Reset trade count on hour change
    current_hour = time.strftime('%H')
    if current_hour != last_trade_hour:
        trade_count = 0
        last_trade_hour = current_hour

    # Check trades per hour
    if trade_count >= MAX_TRADES_PER_HOUR:
        logging.warning("Reached max trades per hour, skipping trade.")
        return None

    bid_price, ask_price, spread = get_order_book_spread()
    if spread is None or ask_price is None or ask_price == 0:
        logging.warning("Could not fetch valid order book. Skipping trade.")
        return None

    # Percentage-based spread check
    spread_percentage = (spread / ask_price) * 100
    if spread_percentage > 0.1:  # e.g. skip if > 0.1% spread
        logging.warning(f"Spread too high ({spread_percentage:.2f}%). Skipping trade.")
        return None

    try:
        order = exchange.create_market_order(SYMBOL, side, abs(trade_size))
        last_trade_time = now
        trade_count += 1
        logging.info(f"✅ Market order placed: {side.upper()} {abs(trade_size)} {SYMBOL.split('/')[0]}")
        return order
    except Exception as e:
        logging.exception("❌ Order failed")
        return None

def check_profit_loss():
    """
    Simple trailing stop check based on total USDT balance.
    If total balance drops more than TRAILING_STOP_PERCENT from its peak, exit.
    """
    global initial_balance, highest_balance
    balance = fetch_balance()
    if not balance:
        return
    
    total_balance = balance['total'].get('USDT', 0)
    # Update highest balance
    if total_balance > highest_balance:
        highest_balance = total_balance
    
    # Check trailing stop
    if total_balance <= highest_balance * (1 - TRAILING_STOP_PERCENT):
        logging.info("🚨 Trailing Stop-Loss Triggered. Stopping bot.")
        # Depending on your preference:
        # Option A: exit immediately
        exit()
        # Option B: place a market sell of all positions, then exit
        # Option C: just go into a 'risk-off' mode (no new trades)
        
def run_bot():
    global initial_balance, highest_balance

    logging.info("🔍 Starting trading bot...")
    
    balance = fetch_balance()
    if balance:
        initial_balance = balance['total'].get('USDT', 0)
        highest_balance = initial_balance
        logging.info(f"✅ Initial balance: {initial_balance}")
    else:
        logging.error("❌ Failed to fetch initial balance. Exiting.")
        exit()
    
    while True:
        # Fetch Data
        df = fetch_data()
        if df is None or df.empty:
            logging.warning("No market data returned. Sleeping 60s.")
            time.sleep(60)
            continue
        
        # Calculate Indicators
        rsi_val = calculate_rsi(df, RSI_PERIOD)
        sma_val = calculate_sma(df, SMA_PERIOD)
        macd_line, signal_line = calculate_macd(df)
        price = fetch_ticker_price()
        atr_val = calculate_atr(df, 14)
        current_balance = fetch_balance()
        
        # Determine trade size
        trade_size = 0
        if current_balance:
            trade_size = get_trade_size(price, current_balance, atr_val)
        
        logging.info(
            f"📈 RSI: {rsi_val:.2f}, SMA: {sma_val:.2f}, "
            f"Price: {price:.2f}, ATR: {atr_val:.2f}"
        )
        
        # ---------------------------
        # Example unified signals
        # ---------------------------
        buy_signal = (rsi_val < 30) and (macd_line.iloc[-1] > signal_line.iloc[-1])
        sell_signal = (rsi_val > 70) and (macd_line.iloc[-1] < signal_line.iloc[-1])
        
        # Execute trade if signals
        if trade_size > 0:
            if buy_signal:
                place_order('buy', trade_size)
            elif sell_signal:
                place_order('sell', trade_size)
        
        # Check trailing stop logic
        check_profit_loss()
        
        # Sleep ~60s to avoid over-polling. 
        # You can adjust to match candle close times if you want.
        time.sleep(60)

# Run the bot
run_bot()
