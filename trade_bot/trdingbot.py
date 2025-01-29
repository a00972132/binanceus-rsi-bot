import ccxt
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Binance API Keys (Use environment variables for security)
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

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
SMA_PERIOD = 200  # Simple Moving Average for trend confirmation
OVERBOUGHT = 70
OVERSOLD = 30
TRADE_AMOUNT = 0.01  # Reduced trade amount for testing
STOP_LOSS_THRESHOLD = 0.80  # Stop bot if loss > 20%
TAKE_PROFIT_THRESHOLD = 1.20  # Stop bot if profit > 20%

# Track last trade details
last_buy_price = None
cached_balance = None
cached_price = None
initial_balance = None

def fetch_data():
    """Fetch historical market data with retry logic"""
    for _ in range(3):  # Retry up to 3 times
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
            return exchange.fetch_balance()
        except Exception as e:
            logging.warning(f"Error fetching balance, retrying: {e}")
            time.sleep(2)
    logging.error("Failed to fetch balance after retries")
    return None

def calculate_rsi(df):
    """Calculate RSI (Relative Strength Index)"""
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
    current_balance = fetch_balance()
    if current_balance is None:
        return
    
    total_balance = current_balance['total'].get('USDT', 0)
    balance_change = total_balance / initial_balance
    
    if balance_change <= STOP_LOSS_THRESHOLD:
        logging.info("🚨 Loss threshold exceeded. Stopping bot.")
        exit()
    elif balance_change >= TAKE_PROFIT_THRESHOLD:
        logging.info("🎉 Profit threshold reached. Stopping bot.")
        exit()

def place_order(side):
    """Place a market order (buy/sell) with balance check"""
    global last_buy_price, cached_balance, cached_price
    balance = fetch_balance()
    if balance is None:
        logging.error("Cannot fetch balance, skipping trade.")
        return None
    eth_balance = balance['free'].get('ETH', 0)
    usdt_balance = balance['free'].get('USDT', 0)
    current_price = fetch_ticker_price()

    if side == 'buy' and usdt_balance < TRADE_AMOUNT * current_price:
        logging.warning("⚠️ Insufficient USDT balance for buy order.")
        return None
    if side == 'sell' and eth_balance < TRADE_AMOUNT:
        logging.warning("⚠️ Insufficient ETH balance for sell order.")
        return None
    
    try:
        order = exchange.create_market_order(SYMBOL, side, TRADE_AMOUNT)
        logging.info(f"✅ Order placed: {side} {TRADE_AMOUNT} ETH")
        if side == 'buy':
            last_buy_price = fetch_ticker_price()
        cached_balance = None
        cached_price = None
        return order
    except Exception as e:
        logging.exception("Order failed")
        return None

def run_bot():
    """Main trading loop"""
    global last_buy_price, cached_balance, cached_price, initial_balance

    initial_balance = fetch_balance()['total'].get('USDT', 0)
    
    while True:
        df = fetch_data()
        if df is None:
            time.sleep(60)
            continue
        
        rsi = calculate_rsi(df)
        sma = calculate_sma(df)
        current_price = fetch_ticker_price()
        if current_price:
            logging.info(f"📊 RSI: {rsi} | Price: {current_price} | SMA: {sma}")
        
        balance = fetch_balance()
        eth_balance = balance['free'].get('ETH', 0) if balance else 0
        
        if rsi < OVERSOLD and eth_balance == 0 and current_price > sma:
            logging.info("🔵 RSI Low & Above SMA – Buying ETH")
            place_order('buy')
        elif rsi > OVERBOUGHT and eth_balance > 0 and current_price < sma:
            logging.info("🔴 RSI High & Below SMA – Selling ETH")
            place_order('sell')
        
        check_profit_loss()
        time.sleep(60)

# Run the bot
run_bot()
