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


REPO_DIR = Path(__file__).parent
BOT_MODULE_PATH = REPO_DIR / "trade_bot" / "trading_bot.py"
LOG_PATH = REPO_DIR / "logs" / "trading_bot.log"
PID_PATH = REPO_DIR / "run" / "bot.pid"
STATE_PATH = REPO_DIR / "run" / "position_state.json"
TIMEFRAME_OPTIONS = ["15m", "30m", "1h", "4h", "1d"]
DEFAULTS = {
    "paper_trading": True,
    "diagnostics": True,
    "fast_sma": 20,
    "slow_sma": 100,
    "rsi_entry_min": 52,
    "rsi_entry_max": 75,
    "rsi_exit_min": 45,
    "risk_per_trade": 2.0,
    "max_position_fraction": 40.0,
    "stop_atr_mult": 2.5,
    "breakout_lookback": 15,
    "add_on_enabled": False,
    "max_add_ons": 0,
    "add_on_trigger_r": 1.0,
    "add_on_risk_fraction": 0.0,
    "max_spread": 0.20,
    "min_trade_interval": 3600,
    "max_trades_per_day": 3,
}


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


def get_setting(name: str):
    return st.session_state.get(name, DEFAULTS[name])


def render_sidebar(pid_running: bool, pid: Optional[int], symbol: str, timeframe: str) -> Dict[str, str]:
    with st.sidebar:
        st.header("Controls")
        symbol = st.text_input("Symbol", value=symbol)
        timeframe = st.selectbox(
            "Timeframe",
            TIMEFRAME_OPTIONS,
            index=TIMEFRAME_OPTIONS.index(timeframe) if timeframe in TIMEFRAME_OPTIONS else 2,
        )
        paper = st.checkbox("Paper trading", value=bool(get_setting("paper_trading")))
        diagnostics = st.checkbox("Show setup checks", value=bool(get_setting("diagnostics")))
        st.caption("Dashboard refresh is manual to avoid Streamlit component issues.")

        st.caption("Basic strategy")
        risk_per_trade = st.slider("Risk per trade (%)", 0.5, 5.0, float(get_setting("risk_per_trade")), 0.25)
        max_position_fraction = st.slider("Max position size (%)", 10.0, 80.0, float(get_setting("max_position_fraction")), 1.0)
        breakout_lookback = st.slider("Breakout lookback", 5, 60, int(get_setting("breakout_lookback")))
        add_on_enabled = st.checkbox("Allow winner add-ons", value=bool(get_setting("add_on_enabled")))

        with st.expander("Advanced settings"):
            fast_sma = st.number_input("Fast SMA", min_value=10, max_value=200, value=int(get_setting("fast_sma")))
            slow_sma = st.number_input("Slow SMA", min_value=50, max_value=400, value=int(get_setting("slow_sma")))
            rsi_entry_min = st.slider("RSI strength min", 30, 80, int(get_setting("rsi_entry_min")))
            rsi_entry_max = st.slider("RSI strength max", 40, 90, int(get_setting("rsi_entry_max")))
            rsi_exit_min = st.slider("Trend-failure RSI", 20, 70, int(get_setting("rsi_exit_min")))
            stop_atr_mult = st.slider("ATR trail multiple", 1.0, 5.0, float(get_setting("stop_atr_mult")), 0.1)
            max_add_ons = st.slider("Max add-ons", 0, 3, int(get_setting("max_add_ons")))
            add_on_trigger_r = st.slider("Add-on trigger (R)", 0.5, 3.0, float(get_setting("add_on_trigger_r")), 0.25)
            add_on_risk_fraction = st.slider(
                "Add-on size (% of base risk)", 10.0, 100.0, float(get_setting("add_on_risk_fraction")), 5.0
            )
            max_spread = st.slider("Max spread (%)", 0.05, 0.50, float(get_setting("max_spread")), 0.01)
            min_trade_interval = st.slider("Loop interval (sec)", 300, 7200, int(get_setting("min_trade_interval")), 300)
            max_trades_per_day = st.slider("Max trades per day", 1, 10, int(get_setting("max_trades_per_day")))

        if "fast_sma" not in st.session_state:
            fast_sma = DEFAULTS["fast_sma"]
            slow_sma = DEFAULTS["slow_sma"]
            rsi_entry_min = DEFAULTS["rsi_entry_min"]
            rsi_entry_max = DEFAULTS["rsi_entry_max"]
            rsi_exit_min = DEFAULTS["rsi_exit_min"]
            stop_atr_mult = DEFAULTS["stop_atr_mult"]
            max_add_ons = DEFAULTS["max_add_ons"]
            add_on_trigger_r = DEFAULTS["add_on_trigger_r"]
            add_on_risk_fraction = DEFAULTS["add_on_risk_fraction"]
            max_spread = DEFAULTS["max_spread"]
            min_trade_interval = DEFAULTS["min_trade_interval"]
            max_trades_per_day = DEFAULTS["max_trades_per_day"]

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
                "breakout_lookback": breakout_lookback,
                "add_on_enabled": add_on_enabled,
                "max_add_ons": max_add_ons,
                "add_on_trigger_r": add_on_trigger_r,
                "add_on_risk_fraction": add_on_risk_fraction,
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
                        "BOT_BREAKOUT_LOOKBACK": breakout_lookback,
                        "BOT_ADD_ON_ENABLED": "true" if add_on_enabled else "false",
                        "BOT_MAX_ADD_ONS": max_add_ons,
                        "BOT_ADD_ON_TRIGGER_R": add_on_trigger_r,
                        "BOT_ADD_ON_RISK_FRACTION": add_on_risk_fraction / 100.0,
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
    st.set_page_config(page_title="BinanceUS Breakout Bot", layout="wide")
    pid = st.session_state.get("bot_pid") or read_pid()
    pid_running = is_running(pid)
    if pid_running and pid:
        st.session_state["bot_pid"] = pid

    bot = load_bot_module()
    symbol = st.session_state.get("symbol", getattr(bot, "SYMBOL", "ETH/USDT"))
    timeframe = st.session_state.get("timeframe", getattr(bot, "TIMEFRAME", "1h"))
    selected = render_sidebar(pid_running, pid, symbol, timeframe)

    try:
        bot.SYMBOL = selected["symbol"]
        bot.TIMEFRAME = selected["timeframe"]
    except Exception:
        pass

    st.title("BinanceUS ETH Context Bot")
    st.caption("Paper-trading dashboard for the BTC-regime and ETH relative-strength strategy.")
    if st.button("Refresh", use_container_width=False):
        st.rerun()

    summary_cols = st.columns([1.2, 1, 1, 1])
    summary_cols[0].metric("Bot status", "Running" if pid_running else "Stopped", f"PID {pid}" if pid_running and pid else None)

    price = bot.fetch_ticker_price()
    balance = bot.fetch_balance() or {}
    quote_total = float((balance.get("total") or {}).get(bot.QUOTE_ASSET, 0.0))
    base_total = float((balance.get("total") or {}).get(bot.BASE_ASSET, 0.0))
    equity = quote_total + (base_total * price)
    state = read_state()

    summary_cols[1].metric(selected["symbol"], f"${price:,.2f}" if price else "-")
    summary_cols[2].metric("Account equity", f"${equity:,.2f}" if equity else "-")
    state = read_state()
    position_qty = float(state.get("quantity", 0.0))
    summary_cols[3].metric("Position", f"{position_qty:.6f} {bot.BASE_ASSET}" if position_qty else "Flat")

    df = bot.fetch_data()
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        snapshot = bot.build_snapshot(df.iloc[:-1] if len(df) > 1 else df)
        snapshot.update(bot.build_entry_context())
        state_open = bool(state.get("quantity", 0) > 0)
        entry_ok, reason = bot.should_enter_long(snapshot)
        btc_regime = snapshot.get("regime_trend_up", True)
        eth_vs_btc = snapshot.get("relative_strength_up", True)
        trend_up = snapshot["price"] > snapshot["fast_sma"] > snapshot["slow_sma"]
        rsi_strength = bot.RSI_ENTRY_MIN <= snapshot["rsi"] <= bot.RSI_ENTRY_MAX
        breakout = snapshot["price"] >= snapshot["breakout_level"] * (1 + bot.ENTRY_BUFFER_PCT)
        volume_ready = snapshot["volume"] >= snapshot["volume_avg"] * bot.MIN_VOLUME_RATIO
        atr_expand = (snapshot["atr"] / snapshot["atr_avg_range"]) >= bot.MIN_ATR_RATIO if snapshot["atr_avg_range"] > 0 else False
        add_on_ready = state_open and bool(state) and bot.should_add_on(snapshot, bot.PositionState(**state))[0]

        st.subheader("What The Bot Sees")
        setup_cols = st.columns(6)
        setup_cols[0].metric("BTC regime", "Ready" if btc_regime else "Blocked")
        setup_cols[1].metric("ETH/BTC strength", "Ready" if eth_vs_btc else "Blocked")
        setup_cols[2].metric("ETH trend", "Ready" if trend_up else "Blocked")
        setup_cols[3].metric("RSI", "Ready" if rsi_strength else "Blocked", f"{snapshot['rsi']:.1f}")
        setup_cols[4].metric("Breakout", "Ready" if breakout else "Waiting", f"{snapshot['breakout_level']:.2f}")
        setup_cols[5].metric("Expansion", "Ready" if volume_ready and atr_expand else "Blocked")

        status_cols = st.columns(3)
        status_cols[0].metric("Entry setup", "Ready" if entry_ok and not state_open else "No")
        status_cols[1].metric("Add-on setup", "Ready" if add_on_ready else "No")
        status_cols[2].metric("Position state", "Open" if state_open else "Flat")

        with st.expander("Current settings"):
            st.json(
                {
                    "symbol": selected["symbol"],
                    "timeframe": selected["timeframe"],
                    "paper_trading": get_setting("paper_trading"),
                    "risk_per_trade_pct": get_setting("risk_per_trade"),
                    "max_position_fraction_pct": get_setting("max_position_fraction"),
                    "breakout_lookback": get_setting("breakout_lookback"),
                    "btc_regime_symbol": getattr(bot, "REGIME_SYMBOL", "BTC/USDT"),
                    "relative_strength_symbol": getattr(bot, "RELATIVE_STRENGTH_SYMBOL", "BTC/USDT"),
                    "winner_add_ons": get_setting("add_on_enabled"),
                    "saved_position": state,
                    "entry_reason": "position already open" if state_open else reason,
                }
            )
        render_chart(df, bot)
    else:
        st.info("Market data unavailable.")

    st.subheader("Bot Logs")
    st.code(read_log_tail(250), language="log")


if __name__ == "__main__":
    main()
