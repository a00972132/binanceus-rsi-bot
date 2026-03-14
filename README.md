# BinanceUS Breakout Bot

This repo now contains a smaller long-only breakout bot and a matching backtest.

The strategy is intentionally simple:
- Only trade in an uptrend: price > fast SMA > slow SMA
- Enter on strength: RSI stays in a momentum zone and price breaks above the recent high
- Require momentum confirmation: MACD histogram positive and improving
- Risk a larger fixed fraction of equity per trade
- Allow a limited add-on into winning trades
- Exit on stop loss, ATR trailing stop, or trend failure

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
- `BOT_BREAKOUT_LOOKBACK`
- `BOT_ADD_ON_ENABLED`
- `BOT_MAX_ADD_ONS`
- `BOT_ADD_ON_TRIGGER_R`
- `BOT_ADD_ON_RISK_FRACTION`
- `BOT_MAX_SPREAD_PERCENT`
- `BOT_MIN_TRADE_INTERVAL`
- `BOT_MAX_TRADES_PER_DAY`

## Backtest

- Run:
  - `make backtest`
- Or customize:
  - `.venv/bin/python utils/backtest_strategy.py --symbol ETH/USDT --timeframe 1h --days 120`

The backtest uses public Binance US candles and the same breakout/trailing logic as the live bot.

## Research Loop

- Offline strategy research lives under `research/`
- Mutable candidate:
  - `research/candidate_strategy.py`
- Fixed evaluator:
  - `research/evaluator.py`
- Run one research evaluation:
  - `make research-eval`
  - or `.venv/bin/python -m research.run_experiment --tag test`
- Results are written to `research/results/latest.json` and `research/results/results.tsv`

This borrows the core autoresearch idea safely: the agent can iterate on a small strategy surface while evaluation stays fixed and the live bot stays isolated.

## Project Structure

- `trade_bot/trading_bot.py`: live bot logic
- `app.py`: small Streamlit dashboard
- `utils/backtest_strategy.py`: offline strategy test
- `utils/test_connection.py`: API connectivity check

## Important

No bot can guarantee profit. Use paper trading and backtests first, then compare strategy return against buy-and-hold before risking live capital.
