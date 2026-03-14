# Trading Autoresearch

This folder adapts the `autoresearch` methodology to the ETH bot.

## Goal

Improve the offline strategy score without touching the live trading bot once the research harness is set.

## In Scope

- `research/candidate_strategy.py`

## Fixed Files

Do not modify these during normal experiment loops:

- `research/evaluator.py`
- `research/run_experiment.py`
- `utils/backtest_strategy.py`
- `trade_bot/trading_bot.py`

## Method

1. Edit only `research/candidate_strategy.py`.
2. Run one experiment:
   - `.venv/bin/python research/run_experiment.py --tag test`
3. Check `research/results/latest.json` and `research/results/latest.md`.
4. Keep a candidate only if:
   - `promote_to_paper` is `true`, or
   - the total `score` improves over the current baseline.
5. Discard candidates that worsen the score or materially increase drawdown.

## Guardrails

- This is offline research only.
- No live trading changes are allowed in the experiment loop.
- Do not change the evaluator and candidate at the same time.
- Prefer simpler changes over fragile ones.
- Treat tuning windows as idea-shaping only. Holdout windows are the promotion gate.

## What To Search

- SMA window combinations
- RSI entry and exit zones
- breakout lookback and entry buffer
- volume confirmation window and threshold
- higher-timeframe regime filters
- stop distance, risk, and max position fraction

## What Not To Do

- Do not optimize on one short window only.
- Do not optimize directly against holdout windows.
- Do not use raw PnL alone.
- Do not promote directly from offline to live.
- Do not rewrite execution logic as part of strategy search.
