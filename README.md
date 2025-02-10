# Trading Bot (Simplified Docs)

## Overview
This bot uses the [ccxt library](https://github.com/ccxt/ccxt) to:
- Connect to **Binance US**  
- Fetch market data (OHLCV)  
- Calculate indicators (RSI, MACD, SMA, ATR)  
- Place **market orders** (buy/sell)  
- Manage basic risk: trailing stop on balance, max trades per hour, etc.

> **Important**: This is for demonstration only. Always test thoroughly with small amounts or a paper account before using real funds.

---
**Key Parameters**
SYMBOL: Pair to trade (e.g., 'ETH/USDT').
TIMEFRAME: Candle interval (e.g., '5m' or '15m').
RISK_PER_TRADE: Fraction of your USDT balance per trade (e.g., 0.02 for 2%).
TRAILING_STOP_PERCENT: If your total balance drops this % from its peak, the bot stops.
MAX_TRADES_PER_HOUR: Limits over-trading; skip new trades if exceeded.

**How It Works**
Fetch Data: Grabs recent OHLCV from Binance US.
Indicators: Calculates RSI, MACD, SMA, and ATR.
Signals:
Buy if RSI < 30 and MACD crosses up.
Sell if RSI > 70 and MACD crosses down.
Order Placement: Places market orders, watching for:
Minimum time since last trade (MIN_TRADE_INTERVAL)
Spread check (skip if bid-ask spread too high)
Trailing Stop: Checks total USDT balance. If it drops below highest_balance * (1 - TRAILING_STOP_PERCENT), the bot exits.
