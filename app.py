import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    # Provided by streamlit-autorefresh package
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    st_autorefresh = None  # fallback


REPO_DIR = Path(__file__).parent
# Paths and constants
BOT_MODULE_PATH = REPO_DIR / "trade_bot" / "trading_bot.py"
LOG_PATH = REPO_DIR / "logs" / "trading_bot.log"
PID_PATH = REPO_DIR / "run" / "bot.pid"

# Common timeframe options for Binance US (via CCXT)
TIMEFRAME_OPTIONS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]


@st.cache_data(show_spinner=False, ttl=1)
def _read_log_tail(max_lines: int = 200) -> str:
    try:
        if not LOG_PATH.exists():
            return "No log file yet. Start the bot to generate logs."
        with LOG_PATH.open("r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception as e:
        return f"Error reading log: {e}"


def _write_pid(pid: int) -> None:
    try:
        PID_PATH.parent.mkdir(parents=True, exist_ok=True)
        PID_PATH.write_text(str(pid))
    except Exception:
        pass


def _safe_rerun() -> None:
    try:
        # Streamlit 1.20+
        st.rerun()
    except Exception:
        try:
            # Older Streamlit
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass


def _read_pid() -> Optional[int]:
    try:
        if PID_PATH.exists():
            return int(PID_PATH.read_text().strip())
    except Exception:
        return None
    return None


def _is_process_running(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        # Signal 0 checks for existence without killing
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but not permitted; treat as running
        return True


def _start_bot_process(extra_env: Optional[Dict[str, str]] = None) -> Optional[int]:
    if not BOT_MODULE_PATH.exists():
        st.error(f"Trading bot not found at {BOT_MODULE_PATH}")
        return None
    try:
        # Use unbuffered output and inherit environment
        env = os.environ.copy()
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})
        proc = subprocess.Popen(
            [sys.executable, "-u", str(BOT_MODULE_PATH)],
            cwd=str(REPO_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # separate process group
        )
        _write_pid(proc.pid)
        return proc.pid
    except Exception as e:
        st.error(f"Failed to start bot: {e}")
        return None


def _stop_bot_process(pid: Optional[int]) -> bool:
    if not pid:
        return True
    try:
        # Terminate process group for clean stop
        os.killpg(pid, signal.SIGTERM)
        # Wait a moment then force kill if needed
        time.sleep(0.8)
        if _is_process_running(pid):
            os.killpg(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except Exception as e:
        st.warning(f"Could not stop process {pid}: {e}")
        return False
    finally:
        try:
            if PID_PATH.exists():
                PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass


def _import_bot_module():
    try:
        # Local import to avoid heavy module at global scope
        from trade_bot import trading_bot as bot
        return bot
    except Exception as e:
        st.error(
            "Failed to import trading bot module. Ensure dependencies are installed.\n"
            f"Error: {e}"
        )
        return None


def _compute_series_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if df is None or df.empty:
        return out
    close = df["close"].astype(float)
    # SMA
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    # RSI (Wilder)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(com=14 - 1, adjust=False).mean()
    roll_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = roll_up / roll_down
    out["rsi14"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"] = macd_line
    out["signal"] = signal
    return out


def _render_header(pid_running: bool, pid: Optional[int], symbol: str, timeframe: str) -> None:
    st.title("BinanceUS RSI Bot Dashboard")
    status_badge = "🟢 Running" if pid_running else "🔴 Stopped"
    st.caption(f"Status: {status_badge}{f' (PID {pid})' if pid_running and pid else ''}")
    # Try to read paper-trading flag from bot module
    paper = False
    try:
        from trade_bot import trading_bot as bot_mod
        paper = bool(getattr(bot_mod, 'PAPER_TRADING', False))
    except Exception:
        pass
    st.caption(f"Symbol: {symbol} • Timeframe: {timeframe} • Paper: {'On' if paper else 'Off'}")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Refresh now", use_container_width=True):
            _safe_rerun()


def _render_sidebar(pid_running: bool, pid: Optional[int], symbol: str, timeframe: str) -> Tuple[str, str]:
    with st.sidebar:
        st.header("Controls")
        st.subheader("Bot Settings")
        symbol = st.text_input("Symbol", value=symbol, help="e.g., ETH/USDT, BTC/USDT")
        timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, index=(TIMEFRAME_OPTIONS.index(timeframe) if timeframe in TIMEFRAME_OPTIONS else 0))
        st.session_state['symbol'] = symbol
        st.session_state['timeframe'] = timeframe
        st.caption("Settings apply when you start the bot. Restart to take effect.")

        st.subheader("Strategy Tuning")
        aggr = st.selectbox("Aggressiveness", ["conservative", "balanced", "aggressive"], index=1,
                             help="Drives thresholds and trade frequency")
        threshold = st.slider("Model threshold", 0.50, 0.80, 0.65, 0.01,
                              help="Minimum ML up-probability to consider a buy (sell uses 1-threshold)")
        confirms = st.slider("Confirmations required", 1, 3, 2,
                             help="Count among [MACD, trend, volume]")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Bot", type="primary", disabled=pid_running):
                new_pid = _start_bot_process({
                    'BOT_SYMBOL': symbol,
                    'BOT_TIMEFRAME': timeframe,
                    'BOT_AGGRESSIVENESS': aggr,
                    'BOT_PREDICTION_THRESHOLD': threshold,
                    'BOT_CONFIRMATIONS_REQUIRED_BUY': confirms,
                    'BOT_CONFIRMATIONS_REQUIRED_SELL': confirms,
                })
                if new_pid:
                    st.session_state["bot_pid"] = new_pid
                    _safe_rerun()
        with col2:
            if st.button("Stop Bot", disabled=not pid_running):
                _stop_bot_process(pid)
                st.session_state["bot_pid"] = None
                _safe_rerun()

        st.divider()
        st.subheader("Refresh")
        auto_enabled = st.checkbox("Auto-refresh", value=True)
        refresh_sec = st.slider("Interval (sec)", 2, 30, 5, 1)
        st.session_state["auto_refresh_enabled"] = auto_enabled
        st.session_state["refresh_sec"] = refresh_sec
        st.caption("Auto-refresh updates price, balances, and logs.")

        st.divider()
        st.subheader("Paths")
        st.code(str(BOT_MODULE_PATH), language="bash")
        st.code(str(LOG_PATH), language="bash")

        st.divider()
        st.subheader("Logs")
        log_lines = st.slider("Show last lines", 20, 1000, 200, 10)
        newest_first = st.checkbox("Newest first", value=True)
        level_choice = st.selectbox("Level filter", ["All", "ERROR", "WARNING", "INFO", "DEBUG"], index=0)
        st.session_state["log_lines"] = log_lines
        st.session_state["log_newest_first"] = newest_first
        st.session_state["log_level_filter"] = level_choice

        return symbol, timeframe

def main():
    # Keep a lightweight periodic refresh (if available)
    refresh_sec = st.session_state.get("refresh_sec", 5)
    auto_enabled = st.session_state.get("auto_refresh_enabled", True)
    try:
        if auto_enabled and st_autorefresh:
            st_autorefresh(interval=int(refresh_sec * 1000), limit=None, key="auto_refresh")
        elif auto_enabled:
            # HTML meta refresh fallback if plugin unavailable
            st.markdown(
                f"<meta http-equiv='refresh' content='{int(refresh_sec)}'>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # Track PID in session + file for resilience
    pid = st.session_state.get("bot_pid") or _read_pid()
    pid_running = _is_process_running(pid)
    if pid_running and pid:
        st.session_state["bot_pid"] = pid
    elif pid and not pid_running:
        # Clear stale PID
        st.session_state["bot_pid"] = None
        try:
            PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    bot = _import_bot_module()
    if not bot:
        st.stop()

    # Selected settings
    selected_symbol = st.session_state.get('symbol', getattr(bot, 'SYMBOL', 'ETH/USDT'))
    selected_timeframe = st.session_state.get('timeframe', getattr(bot, 'TIMEFRAME', '1m'))
    _render_header(pid_running, pid, selected_symbol, selected_timeframe)
    st.caption(f"Last refresh: {time.strftime('%H:%M:%S')}")
    selected_symbol, selected_timeframe = _render_sidebar(pid_running, pid, selected_symbol, selected_timeframe)

    # Apply selection to imported bot module for UI data
    try:
        bot.SYMBOL = selected_symbol
        bot.TIMEFRAME = selected_timeframe
    except Exception:
        pass

    # Top metrics
    # Fetch shared data once
    try:
        price = float(bot.fetch_ticker_price())
    except Exception:
        price = 0.0
    try:
        bal = bot.fetch_balance() or {}
        usdt_total = float(bal.get("total", {}).get("USDT", 0.0))
        eth_total = float(bal.get("total", {}).get("ETH", 0.0))
    except Exception:
        usdt_total, eth_total = 0.0, 0.0

    cols = st.columns(4)
    cols[0].metric(selected_symbol, f"${price:,.2f}")
    cols[1].metric("USDT Total", f"${usdt_total:,.2f}" if usdt_total else "-")
    cols[2].metric("ETH Total", f"{eth_total:,.6f}" if eth_total else "-")
    cols[3].metric("Bot PID", f"{pid if pid_running else '—'}")

    # Market snapshot and charts
    st.subheader("Market Snapshot")
    df = None
    try:
        df = bot.fetch_data()
    except Exception as e:
        st.warning(f"Could not fetch OHLCV: {e}")

    if df is not None and not df.empty:
        # Keep last 400 rows for speed
        dfv = df.tail(400).copy()
        inds = _compute_series_indicators(dfv)

        # Price + SMA chart
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=dfv.index,
                open=dfv['open'], high=dfv['high'], low=dfv['low'], close=dfv['close'],
                name='Price'))
            if "sma50" in inds:
                fig.add_trace(go.Scatter(x=dfv.index, y=inds["sma50"], name="SMA 50", line=dict(color="#1f77b4")))
            if "sma200" in inds:
                fig.add_trace(go.Scatter(x=dfv.index, y=inds["sma200"], name="SMA 200", line=dict(color="#ff7f0e")))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly not available: {e}")
            st.line_chart(dfv["close"])

        # RSI and MACD panels
        c1, c2 = st.columns(2)
        with c1:
            st.caption("RSI(14)")
            if "rsi14" in inds:
                st.line_chart(inds["rsi14"])
        with c2:
            st.caption("MACD")
            if "macd" in inds and "signal" in inds:
                macd_df = pd.DataFrame({"macd": inds["macd"], "signal": inds["signal"]}, index=dfv.index)
                st.line_chart(macd_df)

        # Latest indicators summary
        try:
            rsi_val = float(bot.calculate_rsi(df))
            sma_val = float(bot.calculate_sma(df, 200))
            macd_line, signal_line = bot.calculate_macd(df)
            latest_macd = float(macd_line.iloc[-1])
            latest_signal = float(signal_line.iloc[-1])
            cols2 = st.columns(4)
            cols2[0].metric("RSI(14)", f"{rsi_val:.2f}")
            cols2[1].metric("SMA(200)", f"{sma_val:,.2f}")
            cols2[2].metric("MACD", f"{latest_macd:.4f}")
            cols2[3].metric("Signal", f"{latest_signal:.4f}")
        except Exception:
            pass
    else:
        st.info("No market data available right now.")

    # Logs viewer
    st.subheader("Bot Logs")
    want_lines = int(st.session_state.get("log_lines", 200))
    newest_first = bool(st.session_state.get("log_newest_first", True))
    level_choice = str(st.session_state.get("log_level_filter", "All"))
    raw_tail = _read_log_tail(want_lines)
    lines = raw_tail.splitlines()
    # Optional filter by level
    if level_choice != "All":
        tag = f" - {level_choice} - "
        lines = [ln for ln in lines if tag in ln]
    # Newest first ordering
    if newest_first:
        lines = list(reversed(lines))
    view_text = "\n".join(lines)
    st.code(view_text, language="log")
    # Download button for the shown snippet
    st.download_button(
        label="Download shown logs",
        data=view_text,
        file_name="log_tail.txt",
        mime="text/plain",
    )

    # Footer
    st.caption("Tip: Use the sidebar to start/stop the bot and adjust refresh.")


if __name__ == "__main__":
    main()
