# Trading Autoresearch

This folder adapts the `autoresearch` methodology to the ETH bot.

## Goal

Improve the offline strategy score without touching the live trading bot or the fixed evaluator.

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
3. Check `research/results/latest.json`.
4. Keep a candidate only if:
   - `promote_to_paper` is `true`, or
   - the total `score` improves over the current baseline.
5. Discard candidates that worsen the score or materially increase drawdown.

## Guardrails

- This is offline research only.
- No live trading changes are allowed in the experiment loop.
- Do not change the evaluator and candidate at the same time.
- Prefer simpler changes over fragile ones.

## What To Search

- SMA window combinations
- RSI entry and exit zones
- stop distance and target multiple
- entry buffer and min stop percentage
- risk and max position fraction

## What Not To Do

- Do not optimize on one short window only.
- Do not use raw PnL alone.
- Do not promote directly from offline to live.
- Do not rewrite execution logic as part of strategy search.
