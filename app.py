import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    st_autorefresh = None


REPO_DIR = Path(__file__).parent
BOT_MODULE_PATH = REPO_DIR / "trade_bot" / "trading_bot.py"
LOG_PATH = REPO_DIR / "logs" / "trading_bot.log"
PID_PATH = REPO_DIR / "run" / "bot.pid"
STATE_PATH = REPO_DIR / "run" / "position_state.json"
TIMEFRAME_OPTIONS = ["15m", "30m", "1h", "4h", "1d"]


@st.cache_data(show_spinner=False, ttl=2)
def read_log_tail(lines: int = 200) -> str:
    try:
        if not LOG_PATH.exists():
            return "No log file yet."
        with LOG_PATH.open("r", errors="ignore") as file:
            return "".join(file.readlines()[-lines:])
    except Exception as exc:
        return f"Could not read logs: {exc}"


def read_state() -> Dict[str, float]:
    try:
        if not STATE_PATH.exists():
            return {}
        with STATE_PATH.open("r") as file:
            return json.load(file)
    except Exception:
        return {}


def read_pid() -> Optional[int]:
    try:
        if PID_PATH.exists():
            return int(PID_PATH.read_text().strip())
    except Exception:
        return None
    return None


def write_pid(pid: int) -> None:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(pid))


def is_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def start_bot(extra_env: Dict[str, str]) -> Optional[int]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in extra_env.items()})
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(BOT_MODULE_PATH)],
            cwd=str(REPO_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        write_pid(proc.pid)
        return proc.pid
    except Exception as exc:
        st.error(f"Could not start bot: {exc}")
        return None


def stop_bot(pid: Optional[int]) -> None:
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        pass
    PID_PATH.unlink(missing_ok=True)


def load_bot_module():
    from trade_bot import trading_bot as bot

    return bot


def render_sidebar(pid_running: bool, pid: Optional[int], symbol: str, timeframe: str) -> Dict[str, str]:
    with st.sidebar:
        st.header("Bot")
        symbol = st.text_input("Symbol", value=symbol)
        timeframe = st.selectbox(
            "Timeframe",
            TIMEFRAME_OPTIONS,
            index=TIMEFRAME_OPTIONS.index(timeframe) if timeframe in TIMEFRAME_OPTIONS else 2,
        )
        paper = st.checkbox("Paper trading", value=bool(st.session_state.get("paper_trading", True)))
        diagnostics = st.checkbox("Verbose logs", value=bool(st.session_state.get("diagnostics", False)))

        st.subheader("Strategy")
        fast_sma = st.number_input("Fast SMA", min_value=10, max_value=200, value=int(st.session_state.get("fast_sma", 20)))
        slow_sma = st.number_input("Slow SMA", min_value=50, max_value=400, value=int(st.session_state.get("slow_sma", 100)))
        rsi_entry_min = st.slider("RSI entry min", 10, 70, int(st.session_state.get("rsi_entry_min", 35)))
        rsi_entry_max = st.slider("RSI entry max", 20, 80, int(st.session_state.get("rsi_entry_max", 55)))
        rsi_exit_min = st.slider("RSI exit min", 40, 90, int(st.session_state.get("rsi_exit_min", 68)))
        risk_per_trade = st.slider("Risk per trade (%)", 0.25, 3.0, float(st.session_state.get("risk_per_trade", 1.0)), 0.25)
        max_position_fraction = st.slider("Max position size (%)", 5.0, 50.0, float(st.session_state.get("max_position_fraction", 20.0)), 1.0)
        stop_atr_mult = st.slider("ATR stop multiple", 1.0, 4.0, float(st.session_state.get("stop_atr_mult", 1.5)), 0.1)
        target_r_multiple = st.slider("Target R multiple", 1.0, 4.0, float(st.session_state.get("target_r_multiple", 2.0)), 0.1)
        max_spread = st.slider("Max spread (%)", 0.05, 0.50, float(st.session_state.get("max_spread", 0.20)), 0.01)

        st.subheader("Execution")
        min_trade_interval = st.slider("Loop interval (sec)", 300, 7200, int(st.session_state.get("min_trade_interval", 3600)), 300)
        max_trades_per_day = st.slider("Max trades per day", 1, 10, int(st.session_state.get("max_trades_per_day", 3)))

        st.session_state.update(
            {
                "paper_trading": paper,
                "diagnostics": diagnostics,
                "fast_sma": fast_sma,
                "slow_sma": slow_sma,
                "rsi_entry_min": rsi_entry_min,
                "rsi_entry_max": rsi_entry_max,
                "rsi_exit_min": rsi_exit_min,
                "risk_per_trade": risk_per_trade,
                "max_position_fraction": max_position_fraction,
                "stop_atr_mult": stop_atr_mult,
                "target_r_multiple": target_r_multiple,
                "max_spread": max_spread,
                "min_trade_interval": min_trade_interval,
                "max_trades_per_day": max_trades_per_day,
            }
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start", type="primary", disabled=pid_running, use_container_width=True):
                new_pid = start_bot(
                    {
                        "BOT_SYMBOL": symbol,
                        "BOT_TIMEFRAME": timeframe,
                        "BOT_PAPER_TRADING": "true" if paper else "false",
                        "BOT_DIAGNOSTICS": "true" if diagnostics else "false",
                        "BOT_FAST_SMA_PERIOD": fast_sma,
                        "BOT_SLOW_SMA_PERIOD": slow_sma,
                        "BOT_RSI_ENTRY_MIN": rsi_entry_min,
                        "BOT_RSI_ENTRY_MAX": rsi_entry_max,
                        "BOT_RSI_EXIT_MIN": rsi_exit_min,
                        "BOT_RISK_PER_TRADE": risk_per_trade / 100.0,
                        "BOT_MAX_POSITION_FRACTION": max_position_fraction / 100.0,
                        "BOT_STOP_ATR_MULT": stop_atr_mult,
                        "BOT_TARGET_R_MULTIPLE": target_r_multiple,
                        "BOT_MAX_SPREAD_PERCENT": max_spread,
                        "BOT_MIN_TRADE_INTERVAL": min_trade_interval,
                        "BOT_MAX_TRADES_PER_DAY": max_trades_per_day,
                    }
                )
                if new_pid:
                    st.session_state["bot_pid"] = new_pid
                    st.rerun()
        with col2:
            if st.button("Stop", disabled=not pid_running, use_container_width=True):
                stop_bot(pid)
                st.session_state["bot_pid"] = None
                st.rerun()

        refresh_enabled = st.checkbox("Auto-refresh", value=bool(st.session_state.get("auto_refresh", True)))
        refresh_sec = st.slider("Refresh seconds", 2, 30, int(st.session_state.get("refresh_sec", 5)))
        st.session_state["auto_refresh"] = refresh_enabled
        st.session_state["refresh_sec"] = refresh_sec

    return {"symbol": symbol, "timeframe": timeframe}


def render_chart(df: pd.DataFrame, bot) -> None:
    if df.empty:
        st.info("No market data available.")
        return
    plot_df = df.tail(250).copy()
    close = plot_df["close"]
    plot_df["fast_sma"] = close.rolling(bot.FAST_SMA_PERIOD).mean()
    plot_df["slow_sma"] = close.rolling(bot.SLOW_SMA_PERIOD).mean()
    plot_df["rsi"] = close.copy()
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_up = up.ewm(com=bot.RSI_PERIOD - 1, adjust=False).mean()
    avg_down = down.ewm(com=bot.RSI_PERIOD - 1, adjust=False).mean()
    rs = avg_up / avg_down
    plot_df["rsi"] = 100 - (100 / (1 + rs))

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["close"], name="Close"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["fast_sma"], name=f"SMA {bot.FAST_SMA_PERIOD}"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["slow_sma"], name=f"SMA {bot.SLOW_SMA_PERIOD}"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"], name="RSI"))
        rsi_fig.add_hline(y=bot.RSI_ENTRY_MIN, line_dash="dot", line_color="green")
        rsi_fig.add_hline(y=bot.RSI_ENTRY_MAX, line_dash="dot", line_color="green")
        rsi_fig.add_hline(y=bot.RSI_EXIT_MIN, line_dash="dot", line_color="red")
        rsi_fig.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 100]), template="plotly_white")
        st.plotly_chart(rsi_fig, use_container_width=True)
    except Exception:
        st.line_chart(plot_df[["close", "fast_sma", "slow_sma"]])
        st.line_chart(plot_df["rsi"])


def main() -> None:
    st.set_page_config(page_title="BinanceUS RSI Bot", layout="wide")
    pid = st.session_state.get("bot_pid") or read_pid()
    pid_running = is_running(pid)
    if pid_running and pid:
        st.session_state["bot_pid"] = pid

    if st.session_state.get("auto_refresh", True):
        if st_autorefresh:
            st_autorefresh(interval=int(st.session_state.get("refresh_sec", 5) * 1000), key="refresh")

    bot = load_bot_module()
    symbol = st.session_state.get("symbol", getattr(bot, "SYMBOL", "ETH/USDT"))
    timeframe = st.session_state.get("timeframe", getattr(bot, "TIMEFRAME", "1h"))
    selected = render_sidebar(pid_running, pid, symbol, timeframe)

    try:
        bot.SYMBOL = selected["symbol"]
        bot.TIMEFRAME = selected["timeframe"]
    except Exception:
        pass

    st.title("BinanceUS RSI Bot")
    st.caption(f"Status: {'Running' if pid_running else 'Stopped'}{f' (PID {pid})' if pid_running and pid else ''}")

    price = bot.fetch_ticker_price()
    balance = bot.fetch_balance() or {}
    quote_total = float((balance.get("total") or {}).get(bot.QUOTE_ASSET, 0.0))
    base_total = float((balance.get("total") or {}).get(bot.BASE_ASSET, 0.0))
    equity = quote_total + (base_total * price)
    state = read_state()

    metrics = st.columns(4)
    metrics[0].metric(selected["symbol"], f"${price:,.2f}" if price else "-")
    metrics[1].metric(f"{bot.QUOTE_ASSET} Total", f"${quote_total:,.2f}")
    metrics[2].metric(f"{bot.BASE_ASSET} Total", f"{base_total:,.6f}")
    metrics[3].metric("Equity", f"${equity:,.2f}" if equity else "-")

    df = bot.fetch_data()
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        snapshot = bot.build_snapshot(df.iloc[:-1] if len(df) > 1 else df)
        info_cols = st.columns(4)
        info_cols[0].metric("Fast / Slow SMA", f"{snapshot['fast_sma']:.2f} / {snapshot['slow_sma']:.2f}")
        info_cols[1].metric("RSI", f"{snapshot['rsi']:.2f}")
        info_cols[2].metric("ATR", f"{snapshot['atr']:.2f}")
        info_cols[3].metric("MACD Hist", f"{snapshot['macd_hist']:.4f}")

        state_open = bool(state.get("quantity", 0) > 0)
        entry_ok, reason = bot.should_enter_long(snapshot)
        st.subheader("Strategy State")
        st.write(
            {
                "trend_up": snapshot["price"] > snapshot["fast_sma"] > snapshot["slow_sma"],
                "entry_ready": entry_ok and not state_open,
                "entry_reason": "position already open" if state_open else reason,
                "position_open": state_open,
                "saved_position": state,
            }
        )
        render_chart(df, bot)
    else:
        st.info("Market data unavailable.")

    st.subheader("Bot Logs")
    st.code(read_log_tail(250), language="log")


if __name__ == "__main__":
    main()
