import ccxt
import pandas as pd
import time
import logging
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
logging.info("Trading bot started")

# Exchange Configuration
exchange = ccxt.binanceus({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True},
})

exchange.checkRequiredCredentials()
exchange.load_markets()
exchange.load_time_difference()
exchange.fetch_time()
logging.info("Exchange configuration complete")

# Trading Parameters
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1m'
RSI_PERIOD = 14
SMA_PERIOD = 200
TRAILING_STOP_PERCENT = 0.05
MIN_TRADE_INTERVAL = 60  # Minimum 1 minute between trades
MAX_TRADES_PER_HOUR = 10
RISK_PER_TRADE = 0.02    # 2% risk per trade

# Track Trading Data
last_trade_time = 0
trade_count = 0
last_trade_hour = time.strftime('%H')
initial_balance = None
highest_balance = 0

# Retry Logic for API Calls
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

# Market Data Fetching
def fetch_data():
    """Fetch OHLCV data and return as a DataFrame."""
    return retry_api_call(lambda: pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=max(RSI_PERIOD, SMA_PERIOD) + 2),
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    ))

def fetch_balance():
    """Fetch account balance."""
    return retry_api_call(lambda: exchange.fetch_balance())

def fetch_ticker_price():
    """Fetch last ticker price for SYMBOL."""
    price_data = retry_api_call(lambda: exchange.fetch_ticker(SYMBOL))
    return price_data.get('last', 0) if price_data else 0

# Technical Indicators
def calculate_rsi(df, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period - 1, adjust=False).mean()
    ema_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_sma(df, period=200):
    """Calculate Simple Moving Average."""
    return df['close'].rolling(window=period).mean().iloc[-1]

def calculate_atr(df, period=14):
    """
    Calculate Average True Range using:
      1) true range = max of 
         (high - low), 
         abs(high - prev_close), 
         abs(low - prev_close)
      2) ATR via Wilder's smoothing
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

# UPDATED get_trade_size to risk 2% of total account (ETH + USDT)
def get_trade_size(balance, price, atr):
    """
    Calculate how many ETH to buy/sell, risking 2% of total portfolio
    (free ETH + free USDT).
    The "stop distance" is assumed to be 1× ATR in dollar terms = (atr * price).
    """
    if price == 0 or atr == 0:
        return 0
    
    # 1) Get free ETH and USDT
    eth_free = balance['free'].get('ETH', 0)
    usdt_free = balance['free'].get('USDT', 0)

    # 2) Convert ETH to USD + add free USDT => total USD
    eth_value_usd = eth_free * price
    total_account_value_usd = eth_value_usd + usdt_free

    # 3) Risk 2% of that total
    risk_amount = total_account_value_usd * RISK_PER_TRADE

    # 4) Stop distance in dollar terms (1× ATR)
    stop_distance_usd = atr * price
    if stop_distance_usd <= 0:
        return 0

    # 5) Final position size in ETH
    trade_size_eth = risk_amount / stop_distance_usd

    # OPTIONAL: Enforce a $10 notional min to avoid exchange rejections
    min_notional = 10
    if trade_size_eth * price < min_notional:
        # Option A: skip trade
        # return 0
        
        # Option B: force to $10
        trade_size_eth = min_notional / price

    return trade_size_eth

def get_order_book_spread():
    """Fetch order book and return the absolute spread (bid, ask, spread)."""
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

    spread_percentage = (spread / ask_price) * 100
    # If the spread is too large, skip
    if spread_percentage > 0.1:
        logging.warning(f"Spread too high ({spread_percentage:.2f}%). Skipping trade.")
        return None

    try:
        order = exchange.create_market_order(SYMBOL, side, abs(trade_size))
        last_trade_time = now
        trade_count += 1
        logging.info(f"Order placed: {side.upper()} {abs(trade_size):.6f} {SYMBOL.split('/')[0]}")
        return order
    except Exception as e:
        logging.error(f"Order failed: {e}")
        return None

def check_profit_loss():
    """
    Simple trailing stop check based on total holdings (ETH and USDT).
    If total balance drops more than TRAILING_STOP_PERCENT from its peak, exit.
    """
    global initial_balance, highest_balance
    balance = fetch_balance()
    if not balance:
        return
    
    # 1) Get the total ETH and USDT balances
    eth_total = balance['total'].get('ETH', 0)
    usdt_total = balance['total'].get('USDT', 0)
    
    # 2) Fetch current ETH price (in USDT) to convert ETH to USD
    current_price = fetch_ticker_price() 
    
    # 3) Convert ETH holdings into USDT value
    eth_value_usd = eth_total * current_price
    
    # 4) Combine both for overall portfolio value
    overall_portfolio_usd = usdt_total + eth_value_usd
    
    # 5) Update highest balance if we have a new peak
    if overall_portfolio_usd > highest_balance:
        highest_balance = overall_portfolio_usd
    
    # 6) If we drop below our highest balance by TRAILING_STOP_PERCENT, exit
    if overall_portfolio_usd <= highest_balance * (1 - TRAILING_STOP_PERCENT):
        logging.warning("Trailing stop triggered. Exiting bot.")
        exit()

def run_bot():
    global initial_balance, highest_balance

    logging.info("Starting trading bot...")
    
    balance = fetch_balance()
    if balance:
        initial_balance = balance['total'].get('USDT', 0)
        highest_balance = initial_balance
        logging.info(f"Initial USDT balance: {initial_balance:.2f}")
    else:
        logging.error("Failed to fetch initial balance. Exiting.")
        exit()
    
    while True:
        # -- Start of cycle logging --
        logging.info("----- New cycle -----")

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
            trade_size = get_trade_size(current_balance, price, atr_val)
        
        # Log indicator values + potential trade size
        logging.info(
            f"Indicators => RSI: {rsi_val:.2f}, SMA: {sma_val:.2f}, "
            f"Price: {price:.2f}, ATR: {atr_val:.2f}, Trade Size: {trade_size:.6f}"
        )
        
        # Example signals (adjust thresholds if you want more trades)
        buy_signal = (rsi_val < 30) and (macd_line.iloc[-1] > signal_line.iloc[-1])
        sell_signal = (rsi_val > 70) and (macd_line.iloc[-1] < signal_line.iloc[-1])
        
        # Execute trade if signals
        if trade_size > 0:
            if buy_signal:
                logging.info("Buy signal detected.")
                place_order('buy', trade_size)
            elif sell_signal:
                logging.info("Sell signal detected.")
                place_order('sell', trade_size)
        
        # Check trailing stop logic
        check_profit_loss()
        
        # Sleep ~60s to avoid over-polling. 
        time.sleep(60)

# Run the bot
run_bot()
