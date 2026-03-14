# BinanceUS ETH Context Bot

This repo now contains a smaller long-only ETH bot, a matching backtest, and an offline research loop.

The current paper-trading default strategy is:
- Only trade ETH when BTC is in a `4h` uptrend
- Only trade ETH when the `ETH/BTC` relative-strength ratio is also in a `4h` uptrend
- Enter ETH on `1h` volatility-expansion breakout conditions
- Risk a fixed fraction of equity per trade
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
- `BOT_MIN_VOLUME_RATIO`
- `BOT_MIN_ATR_RATIO`
- `BOT_REGIME_SYMBOL`
- `BOT_REGIME_TIMEFRAME`
- `BOT_REGIME_FAST_SMA_PERIOD`
- `BOT_REGIME_SLOW_SMA_PERIOD`
- `BOT_RELATIVE_STRENGTH_SYMBOL`
- `BOT_RELATIVE_STRENGTH_TIMEFRAME`
- `BOT_RELATIVE_STRENGTH_FAST_SMA_PERIOD`
- `BOT_RELATIVE_STRENGTH_SLOW_SMA_PERIOD`
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

The backtest uses public Binance US candles and the same strategy family as the live bot, including cross-asset BTC context filters.

## Research Loop

- Offline strategy research lives under `research/`
- Mutable candidate:
  - `research/candidate_strategy.py`
- Fixed evaluator:
  - `research/evaluator.py`
- Run one research evaluation:
  - `make research-eval`
  - or `.venv/bin/python -m research.run_experiment --tag test`
- Results are written to `research/results/latest.json`, `research/results/latest.md`, and `research/results/results.tsv`

This borrows the core autoresearch idea safely: the agent can iterate on a small strategy surface while evaluation stays fixed and the live bot stays isolated.

## Project Structure

- `trade_bot/trading_bot.py`: live bot logic
- `app.py`: small Streamlit dashboard
- `utils/backtest_strategy.py`: offline strategy test
- `utils/test_connection.py`: API connectivity check

## Important

No bot can guarantee profit. The current candidate is promoted for paper trading by the repo's evaluator, not for live capital. Keep it in paper mode until live paper results match the research character.
