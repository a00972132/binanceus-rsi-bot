# BinanceUS Trend Pullback Bot

This repo now contains a smaller long-only trading bot and a matching backtest.

The strategy is intentionally simple:
- Only trade in an uptrend: price > fast SMA > slow SMA
- Enter on a pullback: RSI returns to a defined zone near the fast SMA
- Require momentum to turn back up: MACD histogram positive and improving
- Risk a fixed fraction of equity per trade
- Hold one position at a time
- Exit on stop loss, target, profit protection, or trend failure

## Quick Start

- Create the virtualenv and install dependencies:
  - `make install`
- Start the Streamlit dashboard:
  - `make run`
- Start just the bot in the background:
  - `make bot-start`
- Stop the bot:
  - `make bot-stop`

## Configure Credentials

- Copy `config/.env.example` to `.env`
- Set:
  - `BINANCE_API_KEY`
  - `BINANCE_API_SECRET`
- Optional:
  - `BOT_SYMBOL`
  - `BOT_TIMEFRAME`
  - `BOT_ENV_FILE`

## Core Environment Variables

- `BOT_SYMBOL`: pair like `ETH/USDT`
- `BOT_TIMEFRAME`: candle timeframe like `15m` or `1h`
- `BOT_PAPER_TRADING`: `true` to avoid live orders
- `BOT_FAST_SMA_PERIOD`
- `BOT_SLOW_SMA_PERIOD`
- `BOT_RSI_ENTRY_MIN`
- `BOT_RSI_ENTRY_MAX`
- `BOT_RSI_EXIT_MIN`
- `BOT_RISK_PER_TRADE`
- `BOT_MAX_POSITION_FRACTION`
- `BOT_STOP_ATR_MULT`
- `BOT_TARGET_R_MULTIPLE`
- `BOT_MAX_SPREAD_PERCENT`
- `BOT_MIN_TRADE_INTERVAL`
- `BOT_MAX_TRADES_PER_DAY`

## Backtest

- Run:
  - `make backtest`
- Or customize:
  - `.venv/bin/python utils/backtest_strategy.py --symbol ETH/USDT --timeframe 15m --days 120`

The backtest uses public Binance US candles and the same simplified strategy logic as the live bot.

## Project Structure

- `trade_bot/trading_bot.py`: live bot logic
- `app.py`: small Streamlit dashboard
- `utils/backtest_strategy.py`: offline strategy test
- `utils/test_connection.py`: API connectivity check

## Important

No bot can guarantee profit. Use paper trading and backtests first, then compare strategy return against buy-and-hold before risking live capital.
