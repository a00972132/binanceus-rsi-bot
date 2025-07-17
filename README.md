ETH_Bot.py Wiki

Overview
This bot is an automated trading system for ETH/USDT on Binance US. It combines technical analysis, machine learning, and robust risk management to make buy/sell decisions and manage open positions.


Key Features
Exchange: Uses ccxt to interact with Binance US.
Technical Indicators: RSI, SMA, MACD, ATR, Bollinger Bands, Volume MA.
Machine Learning: Random Forest classifier predicts next-candle direction using engineered features.
Risk Management: Enforces max drawdown, position size, daily loss, and trade frequency limits.
Market Regime Detection: Adjusts risk and take-profit parameters based on volatility and trend.
Logging: Rotates log files and outputs to both file and console.
Order Execution: Places market orders with spread checks and interval controls.
Take-Profit & Trailing Stop: Scales out of positions at multiple profit levels and exits on portfolio drawdown.


Main Components
1. PredictiveTrader
Prepares features and labels for ML.
Trains and predicts price direction.
2. RiskManager
Tracks drawdown, daily loss, and trade count.
Decides if a trade is allowed.
3. Technical Analysis Functions
Calculate RSI, SMA, MACD, ATR, Bollinger Bands, and volume ratios.
Detects trend and volume confirmation.
4. Trading Logic
Fetches market data and account balance.
Generates buy/sell signals using ML and TA.
Manages open positions, take-profits, and trailing stops.