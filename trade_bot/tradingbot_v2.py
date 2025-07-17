import ccxt
import pandas as pd
import time
import logging
import os
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from collections import deque

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
logging.getLogger().handlers = []  # Clear any existing handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add file handler with rotation
file_handler = RotatingFileHandler(
    "trading_bot.log",
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

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
TIMEFRAME = '1m'  # Back to 1-minute candles for faster trading
RSI_PERIOD = 14
SMA_PERIOD = 50   # Reduced from 200 to 50 for faster trend signals
TRAILING_STOP_PERCENT = 0.08  # Increased from 0.05 to 0.08 to allow more price movement
MIN_TRADE_INTERVAL = 30  # Reduced to 30 seconds between trades
MAX_TRADES_PER_HOUR = 20  # Increased from 3 to 20
RISK_PER_TRADE = 0.05    # Increased from 0.02 to 0.05 (5% risk per trade)
TAKE_PROFIT_LEVELS = [0.01, 0.02, 0.03]  # Faster take-profits at 1%, 2%, and 3%
POSITION_SCALE_OUT = [0.3, 0.3, 0.4]  # More balanced scaling

# ML Model Parameters
FEATURE_WINDOW = 20
PREDICTION_THRESHOLD = 0.65
MODEL_RETRAIN_INTERVAL = 1000  # retrain every 1000 candles

class MarketRegime:
    TRENDING = 'TRENDING'
    RANGING = 'RANGING'
    VOLATILE = 'VOLATILE'
    UNKNOWN = 'UNKNOWN'

class PredictiveTrader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.historical_data = deque(maxlen=5000)  # Store last 5000 candles
        self.last_train_size = 0
        
    def prepare_features(self, df):
        """Create features for ML model."""
        features = []
        
        # Price-based features
        returns = df['close'].pct_change()
        features.extend([
            returns.rolling(window=5).mean(),
            returns.rolling(window=5).std(),
            (df['high'] - df['low']) / df['low'],  # Volatility
            (df['close'] - df['open']) / df['open'],  # Candle body
        ])
        
        # Technical indicators
        bb = BollingerBands(df['close'])
        features.extend([
            (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()),
            df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean(),
        ])
        
        return np.column_stack(features)
        
    def prepare_labels(self, df):
        """Create binary labels for price direction."""
        future_returns = df['close'].shift(-1).pct_change()
        return (future_returns > 0).astype(int)
        
    def train_model(self, df):
        """Train the ML model on historical data."""
        if len(df) < FEATURE_WINDOW + 10:
            return False
            
        X = self.prepare_features(df)
        y = self.prepare_labels(df)
        
        # Remove NaN values
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:  # Need minimum amount of data
            return False
            
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X[:-1], y[:-1])  # Exclude last point as we don't have its future return
        return True
        
    def predict_direction(self, df):
        """Predict price direction using the trained model."""
        X = self.prepare_features(df.tail(FEATURE_WINDOW))
        X = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X[-1:])
        return probabilities[0][1]  # Probability of price going up

class RiskManager:
    def __init__(self, max_drawdown=0.15, max_position_size=0.3):
        self.max_drawdown = max_drawdown  # Maximum allowed drawdown (15%)
        self.max_position_size = max_position_size  # Maximum position size (30% of portfolio)
        self.peak_balance = 0
        self.current_drawdown = 0
        self.daily_loss = 0
        self.daily_trades = 0
        self.last_trade_day = None
        
    def update_metrics(self, current_balance):
        """Update risk metrics."""
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate current drawdown
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Reset daily metrics if new day
        current_day = time.strftime('%Y-%m-%d')
        if current_day != self.last_trade_day:
            self.daily_loss = 0
            self.daily_trades = 0
            self.last_trade_day = current_day
    
    def can_trade(self, current_balance, proposed_position_size, price):
        """
        Determine if a trade is allowed based on risk parameters.
        """
        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown:
            logging.warning(f"Max drawdown reached: {self.current_drawdown:.2%}")
            return False
        
        # Check position size limit
        position_value = proposed_position_size * price
        if position_value / current_balance > self.max_position_size:
            logging.warning(f"Position size too large: {position_value / current_balance:.2%}")
            return False
        
        # Check daily loss limit (stop trading if lost more than 5% in a day)
        if self.daily_loss < -0.05:
            logging.warning(f"Daily loss limit reached: {self.daily_loss:.2%}")
            return False
        
        # Check daily trade limit
        if self.daily_trades >= MAX_TRADES_PER_HOUR * 24:
            logging.warning("Daily trade limit reached")
            return False
        
        return True
    
    def update_trade_result(self, profit_loss):
        """Update metrics after a trade."""
        self.daily_loss += profit_loss
        self.daily_trades += 1

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
    try:
        data = retry_api_call(lambda: pd.DataFrame(
            exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1000),  # Get 1000 candles for better analysis
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ))
        
        if data is not None and not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return None

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
        return 50  # Return neutral RSI on error

def calculate_sma(df, period=200):
    """Calculate Simple Moving Average."""
    try:
        return df['close'].rolling(window=period).mean().iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating SMA: {str(e)}")
        return df['close'].iloc[-1]  # Return last price on error

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    try:
        df = df.copy()  # Create a copy to avoid modifying original
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['tr'].ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        return atr
    except Exception as e:
        logging.error(f"Error calculating ATR: {str(e)}")
        return (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.1  # Fallback ATR

def calculate_macd(df):
    """Calculate MACD line and Signal line."""
    try:
        macd_line = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line, signal_line
    except Exception as e:
        logging.error(f"Error calculating MACD: {str(e)}")
        # Return neutral MACD values
        return pd.Series([0] * len(df)), pd.Series([0] * len(df))

def calculate_volume_ma(df, period=20):
    """Calculate Volume Moving Average."""
    return df['volume'].rolling(window=period).mean().iloc[-1]

def check_trend(df, sma_val, price):
    """Check if we're in a bullish trend."""
    return (price > sma_val and  # Price above SMA
            df['close'].iloc[-1] > df['close'].iloc[-2])  # Rising price

def check_volume(df):
    """Check if volume is confirming the move."""
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    return current_volume > avg_volume * 1.2  # 20% above average volume

# UPDATED get_trade_size to risk 2% of total account (ETH + USDT)
def get_trade_size(balance, price, atr, rsi_val=50):
    """
    More aggressive position sizing, using higher leverage when conditions align
    """
    if price == 0 or atr == 0:
        return 0
    
    # Get free ETH and USDT
    eth_free = balance['free'].get('ETH', 0)
    usdt_free = balance['free'].get('USDT', 0)

    # Convert ETH to USD + add free USDT => total USD
    eth_value_usd = eth_free * price
    total_account_value_usd = eth_value_usd + usdt_free

    # Risk calculation with momentum multiplier
    risk_amount = total_account_value_usd * RISK_PER_TRADE
    
    # Increase position size if trend is strong
    if abs(rsi_val - 50) > 20:  # Strong trend detected
        risk_amount *= 1.5  # 50% larger position in strong trends
    
    # Stop distance in dollar terms (0.8× ATR for tighter stops)
    stop_distance_usd = atr * price * 0.8
    if stop_distance_usd <= 0:
        return 0

    # Final position size in ETH
    trade_size_eth = risk_amount / stop_distance_usd

    # Enforce minimum trade size
    min_notional = 10
    if trade_size_eth * price < min_notional:
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

def log_trade_metrics(side, trade_size, price, indicators):
    """Log detailed trade metrics."""
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
        
        # Enhanced trade logging
        log_trade_metrics(side, trade_size, price, {
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

def manage_take_profits(current_position, entry_price, current_price):
    """
    Manage take-profit orders for an open position.
    Returns the amount to sell if take-profit is triggered.
    """
    if current_position <= 0 or entry_price <= 0:
        return 0
        
    price_change = (current_price - entry_price) / entry_price
    
    # Check each take-profit level
    for level, scale_out in zip(TAKE_PROFIT_LEVELS, POSITION_SCALE_OUT):
        if price_change >= level:
            amount_to_sell = current_position * scale_out
            logging.info(f"Take profit triggered at {level*100}%, selling {scale_out*100}% of position")
            return amount_to_sell
            
    return 0

def calculate_momentum_score(df):
    """Calculate a momentum score to determine position scaling."""
    # RSI momentum
    rsi = calculate_rsi(df)
    rsi_score = abs(rsi - 50) / 50  # 0 to 1 score based on RSI distance from midpoint
    
    # Price momentum
    price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
    price_score = min(abs(price_change), 1)  # Cap at 1
    
    # Volume momentum
    vol_avg = df['volume'].rolling(10).mean().iloc[-1]
    vol_score = min(df['volume'].iloc[-1] / vol_avg, 2) - 1  # -1 to 1 score
    
    # Combined score (0 to 2)
    return (rsi_score + price_score + max(0, vol_score)) / 1.5

def should_reenter_position(df, last_exit_price, current_price, side='buy'):
    """Determine if we should re-enter a position after exit."""
    if side == 'buy':
        # Re-enter long if price dropped further after our sell
        return (current_price < last_exit_price * 0.995 and  # 0.5% lower
                calculate_momentum_score(df) > 1.2)  # Strong momentum
    else:
        # Re-enter short if price rose further after our buy
        return (current_price > last_exit_price * 1.005 and  # 0.5% higher
                calculate_momentum_score(df) > 1.2)  # Strong momentum

def detect_market_regime(df, window=20):
    """
    Detect the current market regime using volatility and trend metrics.
    """
    # Calculate key metrics
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    atr = calculate_atr(df)
    bb = BollingerBands(df['close'])
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # Get latest values
    current_volatility = volatility.iloc[-1]
    current_bb_width = bb_width.iloc[-1]
    
    # Define thresholds
    high_volatility = current_volatility > volatility.quantile(0.7)
    wide_bb = current_bb_width > bb_width.quantile(0.7)
    
    # Detect trend
    sma_short = df['close'].rolling(window=10).mean()
    sma_long = df['close'].rolling(window=30).mean()
    trend_strength = abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / atr
    
    if high_volatility and wide_bb:
        return MarketRegime.VOLATILE
    elif trend_strength > 1.5:
        return MarketRegime.TRENDING
    elif trend_strength < 0.5:
        return MarketRegime.RANGING
    else:
        return MarketRegime.UNKNOWN

def adjust_parameters_for_regime(regime):
    """
    Adjust trading parameters based on market regime.
    """
    global RISK_PER_TRADE, TRAILING_STOP_PERCENT, TAKE_PROFIT_LEVELS
    
    if regime == MarketRegime.VOLATILE:
        return {
            'risk_per_trade': RISK_PER_TRADE * 0.8,  # Reduce risk in volatile markets
            'trailing_stop': TRAILING_STOP_PERCENT * 1.5,  # Wider stops
            'take_profits': [level * 1.5 for level in TAKE_PROFIT_LEVELS]  # Wider targets
        }
    elif regime == MarketRegime.TRENDING:
        return {
            'risk_per_trade': RISK_PER_TRADE * 1.2,  # Increase risk in trends
            'trailing_stop': TRAILING_STOP_PERCENT * 1.2,  # Wider stops to catch bigger moves
            'take_profits': [level * 1.3 for level in TAKE_PROFIT_LEVELS]  # Wider targets
        }
    elif regime == MarketRegime.RANGING:
        return {
            'risk_per_trade': RISK_PER_TRADE * 0.9,  # Slightly reduced risk
            'trailing_stop': TRAILING_STOP_PERCENT * 0.8,  # Tighter stops
            'take_profits': [level * 0.7 for level in TAKE_PROFIT_LEVELS]  # Closer targets
        }
    else:
        return {
            'risk_per_trade': RISK_PER_TRADE,
            'trailing_stop': TRAILING_STOP_PERCENT,
            'take_profits': TAKE_PROFIT_LEVELS
        }

# Initialize predictive trader
trader = PredictiveTrader()

# Initialize risk manager
risk_manager = RiskManager()

def run_bot():
    global initial_balance, highest_balance
    
    # Initialize variables
    current_position = 0
    entry_price = 0
    last_exit_price = 0
    last_trade_side = None
    candle_count = 0
    
    # Initialize indicators with default values
    rsi_val = 50
    sma_val = 0
    macd_line = pd.Series([0])
    signal_line = pd.Series([0])
    atr_val = 0
    trend_bullish = False
    volume_confirmed = False
    momentum_score = 0
    up_probability = 0.5
    price = 0
    
    # Initial data fetch and model training
    logging.info("Fetching initial data and training ML model...")
    df = fetch_data()
    if df is None or df.empty:
        logging.error("Could not fetch initial data. Exiting.")
        return
        
    # Initial model training
    training_attempts = 0
    max_attempts = 3
    
    while training_attempts < max_attempts:
        if trader.train_model(df):
            logging.info("Initial ML model training successful.")
            break
        training_attempts += 1
        if training_attempts < max_attempts:
            logging.warning(f"Training attempt {training_attempts} failed. Retrying...")
            time.sleep(5)
    else:
        logging.error("Could not train initial ML model after multiple attempts. Exiting.")
        return
    
    logging.info("Initial ML model training complete. Starting trading loop...")
    
    while True:
        try:
            # Reset indicators to default values at start of each loop
            rsi_val = 50
            sma_val = 0
            macd_line = pd.Series([0])
            signal_line = pd.Series([0])
            atr_val = 0
            trend_bullish = False
            volume_confirmed = False
            momentum_score = 0
            up_probability = 0.5
            
            # Fetch and prepare data
            df = fetch_data()
            if df is None or df.empty:
                logging.warning("No market data returned. Sleeping 60s.")
                time.sleep(60)
                continue
            
            # Get current price
            try:
                price = fetch_ticker_price()
                if price == 0:
                    logging.warning("Invalid price received. Sleeping 60s.")
                    time.sleep(60)
                    continue
            except Exception as e:
                logging.error(f"Error fetching price: {str(e)}")
                time.sleep(60)
                continue
            
            # Calculate all technical indicators
            try:
                rsi_val = calculate_rsi(df, RSI_PERIOD)
                sma_val = calculate_sma(df, SMA_PERIOD)
                macd_line, signal_line = calculate_macd(df)
                atr_val = calculate_atr(df, 14)
                
                # Check trend and volume conditions
                trend_bullish = check_trend(df, sma_val, price)
                volume_confirmed = check_volume(df)
                
                # Calculate momentum score
                momentum_score = calculate_momentum_score(df)
                
            except Exception as e:
                logging.error(f"Error calculating indicators: {str(e)}")
                # Continue with default values set at start of loop
            
            # Increment candle count and check if retraining is needed
            candle_count += 1
            if candle_count % MODEL_RETRAIN_INTERVAL == 0:
                logging.info("Retraining ML model...")
                trader.train_model(df)
            
            # Get price prediction
            try:
                up_probability = trader.predict_direction(df)
            except Exception as e:
                logging.error(f"Error in price prediction: {str(e)}")
                up_probability = 0.5  # Neutral prediction on error
            
            # Fetch current balance
            try:
                current_balance = fetch_balance()
                if not current_balance:
                    logging.warning("Could not fetch balance. Sleeping 60s.")
                    time.sleep(60)
                    continue
            except Exception as e:
                logging.error(f"Error fetching balance: {str(e)}")
                time.sleep(60)
                continue
            
            # Log market conditions
            logging.info(f"""
            Market Conditions:
            Price: ${price:.2f}
            RSI: {rsi_val:.2f}
            Trend: {'Bullish' if trend_bullish else 'Bearish'}
            Volume: {'Confirmed' if volume_confirmed else 'Low'}
            Up Probability: {up_probability:.2f}
            Momentum Score: {momentum_score:.2f}
            MACD: {macd_line.iloc[-1]:.6f}
            Signal: {signal_line.iloc[-1]:.6f}
            """)
            
            # Update risk metrics
            risk_manager.update_metrics(current_balance['total']['USDT'])
            
            # Determine trade size with risk checks
            trade_size = get_trade_size(current_balance, price, atr_val, rsi_val)
            
            # Check if trade is allowed by risk manager
            if not risk_manager.can_trade(current_balance['total']['USDT'], trade_size, price):
                logging.info("Trade skipped due to risk management rules")
                trade_size = 0
            
            # Enhanced signals incorporating ML predictions
            buy_signal = (
                up_probability > PREDICTION_THRESHOLD and
                rsi_val < 70 and  # Not overbought
                macd_line.iloc[-1] > signal_line.iloc[-1] and  # MACD crossover
                (trend_bullish or volume_confirmed)  # Either trend or volume confirms
            )
            
            sell_signal = (
                up_probability < (1 - PREDICTION_THRESHOLD) and
                rsi_val > 30 and  # Not oversold
                macd_line.iloc[-1] < signal_line.iloc[-1] and  # MACD crossover
                (not trend_bullish or volume_confirmed)  # Either trend or volume confirms
            )
            
            # Execute trades
            if trade_size > 0:
                if buy_signal:
                    logging.info(f"Buy signal detected (ML Prob: {up_probability:.2f})")
                    if place_order('buy', trade_size):
                        current_position = trade_size
                        entry_price = price
                        last_trade_side = 'buy'
                elif sell_signal:
                    logging.info(f"Sell signal detected (ML Prob: {up_probability:.2f})")
                    if place_order('sell', trade_size):
                        # Calculate and update P&L
                        if entry_price > 0:
                            profit_loss = (price - entry_price) / entry_price
                            risk_manager.update_trade_result(profit_loss)
                        last_exit_price = price
                        last_trade_side = 'sell'
            
            # Manage take-profits
            if current_position > 0:
                tp_amount = manage_take_profits(current_position, entry_price, price)
                if tp_amount > 0:
                    if place_order('sell', tp_amount):
                        current_position -= tp_amount
                        logging.info(f"Take profit executed, remaining position: {current_position:.6f} ETH")
            
            # Check trailing stop
            check_profit_loss()
            
            time.sleep(MIN_TRADE_INTERVAL)
            
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            time.sleep(60)  # Sleep on error

# Run the bot
run_bot()