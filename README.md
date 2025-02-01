# 📌 Binance RSI Trading Bot

A fully automated crypto trading bot that uses **RSI & 200-SMA** to execute trades on **Binance US**.

---

## **💾 Setup & Installation**

### **1️⃣ Install Dependencies**

```bash
pip install ccxt pandas python-dotenv
```

### **2️⃣ Set Up Binance API Keys**

1. Go to **[Binance US API Management](https://www.binance.us/en/usercenter/settings/api-management)**.
2. **Enable trading permissions**.
3. Create a `.env` file and add:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   ```

### **3️⃣ Run the Bot**

```bash
python tradingbot.py
```

---

## **📌 Trading Strategy**

### **🔵 Dynamic Buy Strategy**:

✅ **Buy more as price drops, but only if RSI confirms oversold conditions.**

- **If RSI < 30 and price drops 2% → Buy 0.01 ETH**
- **If RSI < 25 and price drops 5% → Buy 0.02 ETH**
- **If RSI < 20 and price drops 10% → Buy 0.03 ETH**

✅ **Uses SMA as confirmation:** Only buy if **price > 200-SMA** to avoid catching a falling knife.\
✅ **Cooldown period prevents excessive buys.**\
✅ **Scaling stops after 3-4 buy trades to prevent overexposure.**

### **🔴 Dynamic Sell Strategy**:

✅ **Sell more as price rises, but only if RSI confirms overbought conditions.**

- **If RSI > 70 and price rises 5% → Sell 0.01 ETH**
- **If RSI > 75 and price rises 10% → Sell 0.02 ETH**
- **If RSI > 80 and price rises 15% → Sell 0.03 ETH**

✅ **Uses SMA as confirmation:** Only sell if **price < 200-SMA** to confirm a downtrend.\
✅ **Gradually exits positions instead of dumping all at once.**

### **💰 Stop-Loss & Take-Profit**

🚨 **Stops if portfolio drops 20% from the initial balance.**\
🎉 **Instead of stopping at 20% profit, the bot gradually scales out of trades.**\
✅ **Ensures the bot does not fully stop on the second sell but continues trading dynamically.**

### **📌 How RSI is Calculated**

RSI (**Relative Strength Index**) measures the strength of price movements.

**Formula:**
\[
RSI = 100 - \left( \frac{100}{1 + RS} \right)
\]
Where **RS (Relative Strength)** = **Average Gain / Average Loss**

**Example RSI Calculation (14 periods):**

| **Day** | **Closing Price** | **Change** | **Gain** | **Loss** |
|---------|-----------------|------------|----------|----------|
| 1       | 100             | -          | -        | -        |
| 2       | 102             | **+2**     | 2        | 0        |
| 3       | 101             | **-1**     | 0        | 1        |
| 4       | 105             | **+4**     | 4        | 0        |
| 5       | 103             | **-2**     | 0        | 2        |
| 6       | 106             | **+3**     | 3        | 0        |
| 7       | 108             | **+2**     | 2        | 0        |
| 8       | 107             | **-1**     | 0        | 1        |
| 9       | 109             | **+2**     | 2        | 0        |
| 10      | 111             | **+2**     | 2        | 0        |
| 11      | 112             | **+1**     | 1        | 0        |
| 12      | 110             | **-2**     | 0        | 2        |
| 13      | 109             | **-1**     | 0        | 1        |
| 14      | 110             | **+1**     | 1        | 0        |

**Final RSI Value:** RSI = **69.51**

✅ **RSI < 30 → Buy signal** (oversold)
✅ **RSI > 70 → Sell signal** (overbought)

### **🛠️ Cooldown & Trade Frequency**

- **Default cooldown is 5 minutes** (`MIN_TRADE_INTERVAL = 300`).
- Prevents rapid re-trading even if RSI stays below 30.
- Prevents excessive buys when the market is volatile.
- **If RSI stays below 30 and other conditions match, the bot may trade every 5 minutes.**
- **The 5-minute cooldown prevents excessive buys but doesn't block trading entirely.**
- **Trade Limit: Maximum 3 trades per hour to prevent overtrading.**

### **🛠️ API Rate-Limit Handling**

- **Added randomized delay (1.5s - 2.5s)** to prevent getting flagged by Binance API.
- **Ensures retries & error handling** for API rate limits.

---

## **📌 Logging & Monitoring**

Every cycle, the bot logs the following details:

```plaintext
📊 Market Data:
- RSI: 45.2
- Price: 3121.23
- SMA: 3109.84
- Trade Size: 0.01

💰 Account Balance:
- ETH: 0.30
- USDT: 1980.00

⏳ No trade executed this cycle.
```

### **💡 How to View Logs**

To check recent trades and bot activity:

```bash
cat trading_bot.log
```

To filter for only executed trades:

```bash
grep 'Order placed' trading_bot.log
```

---

