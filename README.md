# Binance RSI Trading Bot - Wiki

![Binance Trading](https://img.shields.io/badge/Binance-Trading-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

## Overview
The **Binance RSI Trading Bot** is an automated crypto trading system for **ETH/USDT** using **Relative Strength Index (RSI) and Simple Moving Average (SMA)**. It integrates **risk management features** like **stop-loss and take-profit**, ensuring a safer trading experience.

---
## How the Bot works

**RSI Check Frequency**

- The bot checks RSI every minute (time.sleep(60)).

- It uses 5-minute candlesticks (TIMEFRAME = '5m').

**Buy Conditions** (All must be true):

- ✅ RSI < 30 (oversold) + RSI must recover slightly before rebuying (e.g., RSI > 32)
- ✅ Price > 200-SMA (confirming an uptrend)
- ✅ Cooldown period has passed (prevents rapid re-buys)
- ✅ Trade size scales based on trend strength:
  
      🔹 Weak uptrend → Buy 0.01 ETH
      🔹 Moderate uptrend → Buy 0.02 ETH
      🔹 Strong uptrend → Buy 0.03 ETH

 **Sell Conditions** (All must be true):

- 🔴 RSI > 70 (overbought)
- 🔴 Price < 200-SMA (confirming a downtrend)
- 🔴 Holding ETH (eth_balance > 0)
- 🔴 Cooldown period has passed.

**Cooldown & Trade Frequency**

- Default cooldown is 5 minutes (MIN_TRADE_INTERVAL = 300).
- Prevents rapid re-trading even if RSI stays below 30.
- Prevents excessive buys when the market is volatile.
- If RSI stays below 30 and other conditions match, the bot may trade every 5 minutes.
- The 5-minute cooldown prevents excessive buys but doesn't block trading entirely.
- Trade Limit: Maximum 3 trades per hour to prevent overtrading.

**Risk Management**:
   - Stops if portfolio drops 20% from the initial balance.
   - Sells 50% of position at 20% profit.

API Rate-Limit Handling
Added randomized delay (1.5s - 2.5s) to prevent getting flagged by Binance API.

Ensures retries & error handling for API rate limits.
**Logging & Debugging**:
   - **Logs all trade executions, API errors, and price data**.

---
## Setup Instructions
### 1️⃣ Install Dependencies
Ensure Python (3.8 or later) is installed, then run:
```bash
pip install ccxt pandas python-dotenv
```

### 2️⃣ Set Up API Keys
1. Create a **`.env`** file in the bot directory.
2. Add Binance API keys:
```ini
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### 3️⃣ Run the Bot
```bash
python trading_bot.py
```

---
## Customization Guide
### Adjusting Trading Parameters
Modify **RSI and SMA settings**:
```python
RSI_PERIOD = 14
SMA_PERIOD = 200
OVERBOUGHT = 70
OVERSOLD = 30
```

### Changing Risk Management Rules
Modify **stop-loss and take-profit settings**:
```python
STOP_LOSS_THRESHOLD = 0.80  # Stops bot at 20% loss
TAKE_PROFIT_THRESHOLD = 1.20  # Stops bot at 20% profit
```

---
## Troubleshooting & Debugging
- **Check logs**: Review `trading_bot.log` for issues.
- **Verify API keys**: Ensure keys are **correct and activated**.
- **Update dependencies**:
```bash
pip list --outdated
```


