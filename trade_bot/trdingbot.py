import ccxt
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = "/Users/will/Desktop/Code/Tradingbot/binanceus_creds.env"
load_dotenv(dotenv_path=dotenv_path)

# Configure Binance API Keys (Use environment variables for security)
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Ensure API keys are loaded correctly
if not API_KEY or not API_SECRET:
    logging.error("API keys not found. Exiting.")
    exit()

# Configure logging
logging.basicConfig(
    filename="trading_bot.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Connect to Binance
exchange = ccxt.binanceus({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'adjustForTimeDifference': True}
})

# Trading parameters
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
RSI_PERIOD = 14
SMA_PERIOD = 200
OVERBOUGHT = 70
OVERSOLD = 30
STOP_LOSS_THRESHOLD = 0.80  # Stop bot if loss > 20%
TAKE_PROFIT_THRESHOLD = 1.20  # Stop bot if profit > 20%
MIN_TRADE_INTERVAL = 300  # 5-minute cooldown between trades

# Track last trade details
last_buy_price = None
cached_balance = None
cached_price = None
initial_balance = None
last_trade_time = 0  # Store last trade timestamp

def fetch_data():
    """Fetch historical market data with retry logic"""
    for _ in range(3):
        try:
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=max(RSI_PERIOD, SMA_PERIOD) + 1)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            logging.warning(f"Error fetching data, retrying: {e}")
            time.sleep(2)
    logging.error("Failed to fetch data after retries")
    return None

def fetch_ticker_price():
    """Fetch the current market price with retries."""
    for _ in range(3):
        try:
            return exchange.fetch_ticker(SYMBOL)['last']
        except Exception as e:
            logging.warning(f"Error fetching ticker price, retrying: {e}")
            time.sleep(2)
    logging.error("Failed to fetch ticker price after retries")
    return None

def fetch_balance():
    """Fetch account balance with retries."""
    for _ in range(3):
        try:
            balance = exchange.fetch_balance()
            if balance:
                return balance
        except Exception as e:
            logging.warning(f"Error fetching balance, retrying: {e}")
            time.sleep(2)
    logging.error("Failed to fetch balance after retries")
    return {"total": {"USDT": 0}, "free": {"ETH": 0, "USDT": 0}}

def calculate_rsi(df):
    """Calculate RSI (Relative Strength Index)"""
    if df is None or df.empty:
        logging.warning("No market data available. Skipping RSI calculation.")
        return None
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_sma(df):
    """Calculate Simple Moving Average for trend confirmation"""
    return df['close'].rolling(window=SMA_PERIOD).mean().iloc[-1]

def check_profit_loss():
    """Check if profit or loss exceeds threshold and stop the bot if needed."""
    global initial_balance
    balance = fetch_balance()
    if not balance:
        return
    
    total_balance = balance['total'].get('USDT', 0)
    balance_change = total_balance / initial_balance if initial_balance else 1
    
    if balance_change <= STOP_LOSS_THRESHOLD:
        logging.info("🚨 Loss threshold exceeded. Stopping bot.")
        exit()
    elif balance_change >= TAKE_PROFIT_THRESHOLD:
        logging.info("🎉 Profit threshold reached. Stopping bot.")
        exit()

def get_trade_size(price, sma):
    """Determine trade size dynamically based on trend strength."""
    trend_strength = (price - sma) / sma
    if trend_strength > 0.1:
        return 0.03  # Strong uptrend
    elif trend_strength > 0.05:
        return 0.02  # Moderate uptrend
    else:
        return 0.01  # Weak uptrend

def place_order(side, trade_size):
    """Place a market order (buy/sell) with dynamic trade size and cooldown."""
    global last_trade_time
    if time.time() - last_trade_time < MIN_TRADE_INTERVAL:
        logging.info("⏳ Skipping trade - Cooldown in effect.")
        return None
    try:
        order = exchange.create_market_order(SYMBOL, side, trade_size)
        logging.info(f"✅ Order placed: {side} {trade_size} ETH")
        last_trade_time = time.time()
        return order
    except Exception as e:
        logging.exception("Order failed")
        return None

def run_bot():
    """Main trading loop"""
    global last_buy_price, cached_balance, cached_price, initial_balance

    balance = fetch_balance()
    if balance:
        initial_balance = balance['total'].get('USDT', 0)
    else:
        logging.error("Failed to fetch initial balance. Exiting.")
        exit()
    
    while True:
        df = fetch_data()
        if df is None:
            time.sleep(60)
            continue
        
        rsi = calculate_rsi(df)
        sma = calculate_sma(df)
        current_price = fetch_ticker_price()
        trade_size = get_trade_size(current_price, sma)
        
        # Log only ETH and USDT balances
        balance = fetch_balance()
        eth_balance = balance['free'].get('ETH', 0) if balance else 0
        usdt_balance = balance['free'].get('USDT', 0) if balance else 0
        logging.info(f"💰 ETH Balance: {eth_balance} | USDT Balance: {usdt_balance}")

        if rsi < OVERSOLD and current_price > sma:
            logging.info("🔵 RSI Low & Above SMA – Buying ETH")
            place_order('buy', trade_size)
        elif rsi > OVERBOUGHT and eth_balance > 0 and current_price < sma:
            logging.info("🔴 RSI High & Below SMA – Selling ETH")
            place_order('sell', trade_size)
        
        check_profit_loss()
        time.sleep(60)

# Run the bot
run_bot()
