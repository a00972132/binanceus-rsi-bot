# BinanceUS RSI Bot — Streamlit Dashboard

An organized, easy-to-run trading bot project with a Streamlit dashboard, safe credential handling, and simple tooling.

## Quick Start

- Create a virtual environment and install:
  - `make install`
- Start the dashboard (foreground):
  - `make run`
- Or start in background:
  - `make start` (URL: http://localhost:8501)
- Stop background dashboard:
  - `make stop`

## Configure Credentials

- Copy `config/.env.example` to project root as `.env` and fill in values:
  - `BINANCE_API_KEY`, `BINANCE_API_SECRET`
  - Optional: `BOT_SYMBOL`, `BOT_TIMEFRAME`
- Alternatively, set `BOT_ENV_FILE` to point at an absolute path of your env file.
- The repo `.gitignore` excludes `.env` and keeps secrets out of Git.

## Project Structure

- `app.py`: Streamlit dashboard to monitor and control the bot
- `trade_bot/tradingbot_v2.py`: Main trading bot logic (used by the app)
- `trade_bot/tradingbot_v2.py`: Primary/only bot used by the app
- `utils/test_connection.py`: Quick connection test to BinanceUS
- `config/.env.example`: Example environment file
- `logs/`: Runtime logs (ignored by Git)
- `run/`: PID files for background processes (ignored by Git)
- `Makefile`: Common tasks (install, run, start/stop, logs)

## Streamlit Dashboard

- Start/stop the bot from the sidebar
- Set Symbol and Timeframe before starting the bot
- Live metrics: price, balances, indicators (RSI, MACD), candlesticks
- Log tail for quick debugging

## Environment Variables

- `BINANCE_API_KEY` / `BINANCE_API_SECRET`: Required
- `BOT_SYMBOL`: Trading pair (e.g., `ETH/USDT`)
- `BOT_TIMEFRAME`: Candle timeframe (e.g., `1m`, `5m`, `1h`)
- `BOT_ENV_FILE`: Absolute path to a `.env` file to load

## Logs and PIDs

- Bot logs: `logs/trading_bot.log`
- Streamlit logs: `logs/streamlit-app.log`
- PIDs stored under `run/`

## Git Hygiene

- `.gitignore` excludes `.env`, `logs/`, `run/`, `.venv/`, and OS/editor files
- Safe to commit and push without exposing secrets

## Notes

- The dashboard uses `trade_bot/tradingbot_v2.py`.
The legacy `trdingbot.py` has been removed to avoid confusion. All code paths use `tradingbot_v2.py`.
- If you want paper-trading mode or additional config, open an issue or ask to extend the UI.
