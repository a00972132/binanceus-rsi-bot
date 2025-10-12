PY=.venv/bin/python
PIP=.venv/bin/pip
STREAMLIT=.venv/bin/streamlit

.PHONY: venv install run start stop logs bot-start bot-stop bot-logs

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt

# Run Streamlit in the foreground (opens browser)
run:
	. .venv/bin/activate && $(STREAMLIT) run app.py

# Start Streamlit in background (headless)
start:
	. .venv/bin/activate && nohup $(STREAMLIT) run app.py --server.headless true --server.port 8501 > .streamlit-app.log 2>&1 & echo $$! > .streamlit-app.pid && sleep 1 && echo "Started Streamlit (pid $$(cat .streamlit-app.pid)) on http://localhost:8501"

stop:
	@if [ -f .streamlit-app.pid ]; then \
		kill $$(cat .streamlit-app.pid) 2>/dev/null || true; \
		rm -f .streamlit-app.pid; \
		echo "Stopped Streamlit"; \
	else \
		echo "No Streamlit PID file found."; \
	fi

logs:
	@tail -n 200 -f .streamlit-app.log

# Start/stop the trading bot as a background process
bot-start:
	. .venv/bin/activate && nohup $(PY) -u trade_bot/tradingbot_v2.py > trading_bot.out 2>&1 & echo $$! > bot.pid && sleep 1 && echo "Started bot (pid $$(cat bot.pid))"

bot-stop:
	@if [ -f bot.pid ]; then \
		kill $$(cat bot.pid) 2>/dev/null || true; \
		rm -f bot.pid; \
		echo "Stopped bot"; \
	else \
		echo "No bot PID file found."; \
	fi

bot-logs:
	@tail -n 200 -f trading_bot.out

