# Binance RSI Trading Bot - Wiki

![Binance Trading](https://img.shields.io/badge/Binance-Trading-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

## Overview
The **Binance RSI Trading Bot** is an automated crypto trading system for **ETH/USDT** using **Relative Strength Index (RSI) and Simple Moving Average (SMA)**. It integrates **risk management features** like **stop-loss and take-profit**, ensuring a safer trading experience.

---
## How the Bot works

🔹 **RSI Check Frequency**

- The bot checks RSI every minute (time.sleep(60)).

- It uses 5-minute candlesticks (TIMEFRAME = '5m').

🔵 **Buy Conditions** (All must be true):

- ✅ RSI < 30 (oversold)
- ✅ Price > 200-SMA (confirming an uptrend)
- ✅ Trade size scales based on trend strength:
      🔹Weak uptrend → Buy 0.01 ETH
      🔹Moderate uptrend → Buy 0.02 ETH
      🔹Strong uptrend → Buy 0.03 ETH

🔴 **Sell Conditions** (All must be true):

- ✅ RSI > 70 (overbought)
- ✅ Price < 200-SMA (confirming a downtrend)
- ✅ Holding ETH (eth_balance > 0)

🔹 **Prevents Over-Trading**

- Won’t buy multiple times in a row.

- Won’t sell unless a buy happened first.

- Logs "No trade executed" if no trade is made.


✅ **Risk Management**:
   - **Stop-Loss**: Stops trading if **balance drops by 20%**.
   - **Take-Profit**: Stops trading if **profit exceeds 20%**.

✅ **Logging & Debugging**:
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


