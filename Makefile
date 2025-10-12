PY=.venv/bin/python
PIP=.venv/bin/pip
STREAMLIT=.venv/bin/streamlit

.PHONY: venv install run start stop logs bot-start bot-stop bot-logs

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt

# Ensure directories
prepare:
	mkdir -p logs run

# Run Streamlit in the foreground (opens browser)
run: prepare
	. .venv/bin/activate && $(STREAMLIT) run app.py

# Start Streamlit in background (headless)
start: prepare
	. .venv/bin/activate && nohup $(STREAMLIT) run app.py --server.headless true --server.port 8501 > logs/streamlit-app.log 2>&1 & echo $$! > run/streamlit-app.pid && sleep 1 && echo "Started Streamlit (pid $$(cat run/streamlit-app.pid)) on http://localhost:8501"

stop:
	@if [ -f run/streamlit-app.pid ]; then \
		kill $$(cat run/streamlit-app.pid) 2>/dev/null || true; \
		rm -f run/streamlit-app.pid; \
		echo "Stopped Streamlit"; \
	else \
		echo "No Streamlit PID file found."; \
	fi

logs:
	@tail -n 200 -f logs/streamlit-app.log

# Start/stop the trading bot as a background process
bot-start: prepare
	. .venv/bin/activate && nohup $(PY) -u trade_bot/trading_bot.py > logs/trading_bot.out 2>&1 & echo $$! > run/bot.pid && sleep 1 && echo "Started bot (pid $$(cat run/bot.pid))"

bot-stop:
	@if [ -f run/bot.pid ]; then \
		kill $$(cat run/bot.pid) 2>/dev/null || true; \
		rm -f run/bot.pid; \
		echo "Stopped bot"; \
	else \
		echo "No bot PID file found."; \
	fi

bot-logs:
	@tail -n 200 -f logs/trading_bot.out
