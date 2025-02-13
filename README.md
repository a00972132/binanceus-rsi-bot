# Trading Bot (Simplified Docs)

Here’s a high-level overview of what the bot does step by step:

Initialization & Setup

Loads your API credentials from an environment file (.env) and configures ccxt to connect to Binance US.
Sets up logging so you can track each loop and each trade.
Defines trading parameters like RSI_PERIOD, SMA_PERIOD, RISK_PER_TRADE, etc.
Infinite Trading Loop
Inside run_bot() the script loops indefinitely:

Fetch Current Data

Calls fetch_data() to get recent OHLCV candles for SYMBOL (ETH/USDT) on a 1-minute timeframe.
Logs a warning and sleeps if data is missing.
Calculate Indicators

RSI (14-period) with Wilder’s smoothing.
SMA (200-period).
MACD (12/26/9).
ATR (14-period) for volatility measurement.
Ticker Price to see the latest ETH/USDT quote.
Position Sizing (2% of Total Account)

Fetches your account balance (both ETH and USDT).
Converts the free ETH into USD value by multiplying by the current ETH price, then adds free USDT for a total USD value.
Takes 2% (RISK_PER_TRADE = 0.02) of that combined total.
Divides by (ATR × price) to get the number of ETH to trade, assuming a 1×ATR “stop distance.”
Enforces a minimum $10 notional to avoid small-trade rejections on BinanceUS.
Check Buy / Sell Signals

Default signals:
Buy if RSI < 30 and MACD > signal_line.
Sell if RSI > 70 and MACD < signal_line.
You can adjust these thresholds if you want more or fewer trades.
Place Orders

If the signal is triggered and the calculated trade size is > 0, it calls place_order().
place_order() checks:
Time constraints (MIN_TRADE_INTERVAL, MAX_TRADES_PER_HOUR).
Bid/ask spread (must be below 0.1%).
Then places a market order (buy or sell) for the computed trade size in ETH.
Trailing Stop Check

If your total USDT balance (including any conversions in Binance’s accounting) falls 5% (TRAILING_STOP_PERCENT = 0.05) below your highest recorded balance, the bot logs a warning and exits (or you could adjust the code to close positions first).
Logging & Sleep

The bot logs each cycle’s indicator values and any trades placed.
Sleeps for 60 seconds (time.sleep(60)) before starting the next loop.
In short, the bot:

Gathers 1-minute candle data for ETH/USDT,
Computes RSI, SMA, MACD, ATR,
Decides if it should buy or sell based on RSI & MACD thresholds,
Sizes each trade to risk 2% of your total ETH + USDT portfolio (via ATR volatility),
Places market orders if the spread and time constraints allow,
Monitors your overall balance for a trailing stop exit,
Repeats every minute.
