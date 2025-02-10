import ccxt
import pandas as pd
import time
import logging
import random
from dotenv import load_dotenv
import os

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

# 🔥 Trading Parameters
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
RSI_PERIOD = 14
SMA_PERIOD = 200
OVERBOUGHT = 70
OVERSOLD = 30
STOP_LOSS_THRESHOLD = 0.80  
TAKE_PROFIT_THRESHOLD = 1.20  
TRAILING_STOP_PERCENT = 0.05  # 5% trailing stop-loss
MIN_TRADE_INTERVAL = 300  
MAX_TRADES_PER_HOUR = 3  

# 🛠 Track Trading Data
last_trade_time = 0
trade_count = 0
last_trade_hour = time.strftime('%H')
last_buy_price = None
last_sell_price = None
initial_balance = None
highest_balance = 0  # Fixed issue with uninitialized highest balance

# ✅ Ensure Trading Limits
def can_trade():
    global trade_count, last_trade_hour
    current_hour = time.strftime('%H')
    if current_hour != last_trade_hour:
        trade_count = 0
        last_trade_hour = current_hour
    return trade_count < MAX_TRADES_PER_HOUR

# ✅ Market Data Fetching with Retry Logic
def fetch_data():
    for _ in range(3):
        try:
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=max(RSI_PERIOD, SMA_PERIOD) + 1)
            if bars:
                return pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            logging.warning(f"Error fetching data, retrying: {e}")
            time.sleep(2)
    logging.error("❌ Failed to fetch market data")
    return None

def fetch_ticker_price():
    """Fetch current market price with retries and prevent None errors."""
    for _ in range(3):
        try:
            price_data = exchange.fetch_ticker(SYMBOL)
            price = price_data.get('last')
            if price:
                return price
        except Exception as e:
            logging.warning(f"Error fetching ticker price, retrying: {e}")
            time.sleep(2)
    logging.error("❌ Failed to fetch ticker price")
    return 0  # Return 0 instead of None to prevent calculation errors

def fetch_balance():
    """Fetch account balance with retries."""
    for _ in range(3):
        try:
            balance = exchange.fetch_balance()
            return balance if balance else None
        except Exception as e:
            logging.warning(f"Error fetching balance, retrying: {e}")
            time.sleep(2)
    logging.error("❌ Failed to fetch balance")
    return {"total": {"USDT": 0}, "free": {"ETH": 0, "USDT": 0}}

# 📊 Technical Indicators
def calculate_rsi(df):
    if df is None or df.empty:
        return None
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_sma(df):
    return df['close'].rolling(window=SMA_PERIOD).mean().iloc[-1]

# 🔥 Risk Management
def check_profit_loss():
    """Apply stop-loss and trailing stop-loss."""
    global initial_balance, highest_balance
    balance = fetch_balance()
    if not balance:
        return

    total_balance = balance['total'].get('USDT', 0)
    if total_balance > highest_balance:
        highest_balance = total_balance

    # Apply stop-loss & trailing stop
    if total_balance <= highest_balance * (1 - TRAILING_STOP_PERCENT):
        logging.info("🚨 Trailing Stop-Loss Triggered. Stopping bot.")
        return
    if total_balance <= initial_balance * STOP_LOSS_THRESHOLD:
        logging.info("🚨 Stop-Loss Triggered. Stopping bot.")
        return

# 📈 Trade Size Calculation
def get_trade_size(price, reference_price, rsi):
    price_change = (price - reference_price) / reference_price

    if rsi < 30 and price_change < -0.02:
        return 0.01  
    elif rsi < 25 and price_change < -0.05:
        return 0.02  
    elif rsi < 20 and price_change < -0.10:
        return 0.03  
    elif rsi > 70 and price_change > 0.05:
        return -0.01  
    elif rsi > 75 and price_change > 0.10:
        return -0.02  
    elif rsi > 80 and price_change > 0.15:
        return -0.03  
    return 0.0  

# ✅ Place Orders Using Market Orders (Reverted for Stability)
def place_order(side, trade_size):
    global last_trade_time, trade_count
    if time.time() - last_trade_time < MIN_TRADE_INTERVAL or not can_trade():
        return None

    try:
        order = exchange.create_market_order(SYMBOL, side, abs(trade_size))
        last_trade_time = time.time()
        trade_count += 1
        logging.info(f"✅ Market order placed: {side.upper()} {abs(trade_size)} ETH")
        return order
    except Exception as e:
        logging.exception("❌ Order failed")
        return None

# 🚀 Main Trading Loop
def run_bot():
    global initial_balance, highest_balance, last_buy_price, last_sell_price

    print("🔍 Starting trading bot...")
    balance = fetch_balance()
    if balance:
        initial_balance = balance['total'].get('USDT', 0)
        highest_balance = initial_balance  # Ensure highest balance is set initially
        print(f"✅ Initial balance: {initial_balance}")
    else:
        print("❌ Failed to fetch initial balance. Exiting.")
        exit()

    while True:
        print("🔄 Fetching market data...")
        df = fetch_data()
        if df is None:
            print("⚠ No market data, retrying in 60s...")
            time.sleep(60)
            continue

        print("📊 Calculating indicators...")
        rsi, sma, price = calculate_rsi(df), calculate_sma(df), fetch_ticker_price()
        
        if price == 0:
            print("⚠ No valid price data, retrying in 60s...")
            time.sleep(60)
            continue  

        print(f"📈 RSI: {rsi}, SMA: {sma}, Price: {price}")
        trade_size = get_trade_size(price, last_buy_price or price, rsi)
        print(f"📌 Trade Size: {trade_size}")

        if trade_size != 0:
            print(f"🚀 Placing order: {'BUY' if trade_size > 0 else 'SELL'} {abs(trade_size)} ETH")
            place_order('buy' if trade_size > 0 else 'sell', abs(trade_size))
            last_buy_price = price if trade_size > 0 else last_buy_price
            last_sell_price = price if trade_size < 0 else last_sell_price

        print("🔍 Checking profit/loss...")
        check_profit_loss()
        print("⏳ Sleeping before next trade...")
        time.sleep(random.uniform(1.5, 2.5))


run_bot()
