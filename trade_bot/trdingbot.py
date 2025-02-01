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

# ✅ Fix for Binance Time Sync Issue
exchange = ccxt.binanceus({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'options': {'adjustForTimeDifference': True},
})

# Ensure time sync with Binance
exchange.checkRequiredCredentials()
exchange.load_markets()
exchange.fetch_time()

# Trading parameters
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
RSI_PERIOD = 14
SMA_PERIOD = 200
OVERBOUGHT = 70
OVERSOLD = 30
STOP_LOSS_THRESHOLD = 0.80  # Stop bot if loss > 20%
TAKE_PROFIT_THRESHOLD = 1.20  # Gradual profit-taking at 20%
MIN_TRADE_INTERVAL = 300  # 5-minute cooldown between trades
MAX_TRADES_PER_HOUR = 3  # Limit max trades per hour

# Track last trade details
last_trade_time = 0
trade_count = 0
last_trade_hour = time.strftime('%H')
last_buy_price = None
last_sell_price = None
initial_balance = None

def fetch_data():
    """Fetch historical market data with retry logic"""
    for _ in range(3):
        try:
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=max(RSI_PERIOD, SMA_PERIOD) + 1)
            if not bars:
                logging.warning("⚠ No market data available. Skipping this cycle.")
                return None
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            logging.warning(f"Error fetching data, retrying: {e}")
            time.sleep(2)
    logging.error("❌ Failed to fetch market data after retries")
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
    """Check if profit or loss exceeds threshold and take action accordingly."""
    global initial_balance
    balance = fetch_balance()
    if not balance:
        return
    
    total_balance = balance['total'].get('USDT', 0)
    balance_change = total_balance / initial_balance if initial_balance else 1
    
    if balance_change <= STOP_LOSS_THRESHOLD:
        logging.info("🚨 Stop-Loss Triggered. Stopping bot.")
        exit()
    elif balance_change >= TAKE_PROFIT_THRESHOLD:
        logging.info("🎉 Profit threshold reached. Taking gradual profit.")
        place_order('sell', 0.5)  # Sell 50% instead of stopping bot

def can_trade():
    """Check if the bot can trade based on hourly trade limits."""
    global trade_count, last_trade_hour
    current_hour = time.strftime('%H')

    if current_hour != last_trade_hour:
        trade_count = 0
        last_trade_hour = current_hour

    return trade_count < MAX_TRADES_PER_HOUR

def get_trade_size(price, reference_price, rsi):
    """Dynamically adjust trade size based on price movement & RSI strength."""
    price_change = (price - reference_price) / reference_price

    if rsi < 30 and price_change < -0.02:  # Price dropped 2% & RSI low → Small Buy
        return 0.01
    elif rsi < 25 and price_change < -0.05:  # Price dropped 5% & RSI very low → Moderate Buy
        return 0.02
    elif rsi < 20 and price_change < -0.10:  # Price dropped 10% & RSI extreme → Large Buy
        return 0.03
    elif rsi > 70 and price_change > 0.05:  # Price up 5% & RSI high → Small Sell
        return 0.01
    elif rsi > 75 and price_change > 0.10:  # Price up 10% & RSI very high → Moderate Sell
        return 0.02
    elif rsi > 80 and price_change > 0.15:  # Price up 15% & RSI extreme → Large Sell
        return 0.03
    else:
        return 0.0  # No trade

def place_order(side, trade_size):
    """Place a market order (buy/sell) with cooldown & trade limits."""
    global last_trade_time, trade_count
    if time.time() - last_trade_time < MIN_TRADE_INTERVAL or not can_trade():
        logging.info("⏳ Skipping trade - Cooldown in effect or trade limit reached.")
        return None
    try:
        order = exchange.create_market_order(SYMBOL, side, trade_size)
        last_trade_time = time.time()
        trade_count += 1
        logging.info(f"✅ Order placed: {side} {trade_size} ETH")
        return order
    except Exception as e:
        logging.exception("Order failed")
        return None

def run_bot():
    """Main trading loop"""
    global initial_balance, last_buy_price, last_sell_price

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

        trade_size = get_trade_size(current_price, last_buy_price if last_buy_price else current_price, rsi)

        if rsi < OVERSOLD and current_price > sma:
            place_order('buy', trade_size)
            last_buy_price = current_price
        elif rsi > OVERBOUGHT and current_price < sma:
            place_order('sell', trade_size)
            last_sell_price = current_price

        check_profit_loss()
        time.sleep(random.uniform(1.5, 2.5))  # Randomized delay to avoid API bans

# Run the bot
run_bot()
