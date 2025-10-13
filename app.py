import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import re
from datetime import datetime
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


def _parse_trade_markers(df_index: pd.Index, max_lines: int = 3000) -> Tuple[List[pd.Timestamp], List[float], List[pd.Timestamp], List[float]]:
    """Parse recent executed trades from the log and align them to chart index.
    Returns (buy_x, buy_y, sell_x, sell_y).
    """
    buy_x: List[pd.Timestamp] = []
    buy_y: List[float] = []
    sell_x: List[pd.Timestamp] = []
    sell_y: List[float] = []
    if not LOG_PATH.exists():
        return buy_x, buy_y, sell_x, sell_y
    try:
        with LOG_PATH.open('r', errors='ignore') as f:
            lines = f.readlines()[-max_lines:]
    except Exception:
        return buy_x, buy_y, sell_x, sell_y
    ts_re = re.compile(r'^(\d{4}-\d{2}-\d{2} [0-9:]{8})\s*-')
    side_re = re.compile(r'^\s*Side:\s*(BUY|SELL)', re.IGNORECASE)
    price_re = re.compile(r'^\s*Price:\s*\$?([0-9,.]+)')
    current_ts: Optional[pd.Timestamp] = None
    cur_side: Optional[str] = None
    cur_price: Optional[float] = None
    def flush():
        nonlocal cur_side, cur_price, current_ts
        if cur_side and cur_price and current_ts is not None and len(df_index) > 0:
            # align to nearest chart index
            try:
                ts = pd.to_datetime(current_ts)
                pos = df_index.get_indexer([ts], method='nearest')
                if pos is not None and len(pos) and pos[0] != -1:
                    aligned_ts = df_index[pos[0]]
                    if abs((aligned_ts - ts).total_seconds()) <= 600:  # within 10 minutes
                        if cur_side.upper().startswith('B'):
                            buy_x.append(aligned_ts)
                            buy_y.append(float(cur_price))
                        else:
                            sell_x.append(aligned_ts)
                            sell_y.append(float(cur_price))
            except Exception:
                pass
        cur_side = None
        cur_price = None
    for line in lines:
        m = ts_re.match(line)
        if m:
            # starting a new record
            # flush any pending block
            flush()
            try:
                current_ts = pd.to_datetime(m.group(1))
            except Exception:
                current_ts = None
            continue
        sm = side_re.search(line)
        if sm:
            cur_side = sm.group(1).upper()
            continue
        pm = price_re.search(line)
        if pm:
            try:
                cur_price = float(pm.group(1).replace(',', ''))
            except Exception:
                cur_price = None
            continue
        if '====' in line:
            # end of block
            flush()
    # final flush
    flush()
    return buy_x, buy_y, sell_x, sell_y


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
        # Presets by aggressiveness
        presets = {
            'conservative': {
                'threshold': 0.70, 'confirms': 3,
                'rsi_buy_max': 35.0, 'rsi_sell_min': 65.0,
                'spread_normal': 0.10, 'spread_volatile': 0.18,
                'min_interval': 45, 'max_trades_per_hour': 12,
            },
            'balanced': {
                'threshold': 0.65, 'confirms': 2,
                'rsi_buy_max': 40.0, 'rsi_sell_min': 60.0,
                'spread_normal': 0.12, 'spread_volatile': 0.22,
                'min_interval': 30, 'max_trades_per_hour': 20,
            },
            'aggressive': {
                'threshold': 0.58, 'confirms': 1,
                'rsi_buy_max': 45.0, 'rsi_sell_min': 55.0,
                'spread_normal': 0.15, 'spread_volatile': 0.25,
                'min_interval': 20, 'max_trades_per_hour': 30,
            },
        }
        # Initialize session tuning state once
        if '_tuning_init' not in st.session_state:
            st.session_state['aggr'] = 'balanced'
            for k, v in presets['balanced'].items():
                st.session_state[k] = v
            st.session_state['_prev_aggr'] = 'balanced'
            st.session_state['_tuning_init'] = True
        # Aggressiveness selector with auto-apply presets
        aggr = st.selectbox(
            "Aggressiveness", ["conservative", "balanced", "aggressive"],
            index=["conservative","balanced","aggressive"].index(st.session_state.get('aggr','balanced')),
            help="Presets thresholds and trade cadence"
        )
        if st.session_state.get('_prev_aggr') != aggr:
            for k, v in presets[aggr].items():
                st.session_state[k] = v
            st.session_state['aggr'] = aggr
            st.session_state['_prev_aggr'] = aggr
        # Threshold and confirmations (bound to session state; safe defaults)
        st.slider("Model threshold", 0.50, 0.80,
                  value=float(st.session_state.get('threshold', presets[aggr]['threshold'])), step=0.01,
                  help="Minimum ML up-probability to consider a buy (sell uses 1-threshold)", key='threshold')
        st.slider("Confirmations required", 1, 3,
                  value=int(st.session_state.get('confirms', presets[aggr]['confirms'])),
                  help="Count among [MACD, trend, volume]", key='confirms')
        colr1, colr2 = st.columns(2)
        with colr1:
            st.slider("RSI buy max", 20, 80,
                      value=int(st.session_state.get('rsi_buy_max', presets[aggr]['rsi_buy_max'])), step=1,
                      help="Consider buys when RSI is below this", key='rsi_buy_max')
        with colr2:
            st.slider("RSI sell min", 20, 80,
                      value=int(st.session_state.get('rsi_sell_min', presets[aggr]['rsi_sell_min'])), step=1,
                      help="Consider sells when RSI is above this", key='rsi_sell_min')

        # Spread + cadence tuning
        st.caption("Execution guards")
        spc1, spc2 = st.columns(2)
        with spc1:
            spread_normal = st.slider(
                "Max spread (normal) %", 0.05, 0.50,
                value=float(st.session_state.get('spread_normal', presets[aggr]['spread_normal'])), step=0.01,
                help="Skip trades if order book spread is above this in normal regime"
            )
            min_interval = st.slider(
                "Min trade interval (sec)", 5, 120,
                value=int(st.session_state.get('min_interval', presets[aggr]['min_interval'])), step=1,
                help="Minimum seconds between trades"
            )
        with spc2:
            spread_volatile = st.slider(
                "Max spread (volatile) %", 0.10, 0.80,
                value=float(st.session_state.get('spread_volatile', presets[aggr]['spread_volatile'])), step=0.01,
                help="Skip trades if spread above this in volatile regime"
            )
            max_trades_per_hour = st.slider(
                "Max trades per hour", 1, 60,
                value=int(st.session_state.get('max_trades_per_hour', presets[aggr]['max_trades_per_hour'])), step=1,
                help="Hard cap on trading frequency"
            )
        # Persist for UI reference
        st.session_state['spread_normal'] = float(spread_normal)
        st.session_state['spread_volatile'] = float(spread_volatile)
        st.session_state['min_interval'] = int(min_interval)
        st.session_state['max_trades_per_hour'] = int(max_trades_per_hour)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Bot", type="primary", disabled=pid_running):
                new_pid = _start_bot_process({
                    'BOT_SYMBOL': symbol,
                    'BOT_TIMEFRAME': timeframe,
                    'BOT_AGGRESSIVENESS': st.session_state['aggr'],
                    'BOT_PREDICTION_THRESHOLD': st.session_state['threshold'],
                    'BOT_CONFIRMATIONS_REQUIRED_BUY': st.session_state['confirms'],
                    'BOT_CONFIRMATIONS_REQUIRED_SELL': st.session_state['confirms'],
                    'BOT_RSI_BUY_MAX': st.session_state['rsi_buy_max'],
                    'BOT_RSI_SELL_MIN': st.session_state['rsi_sell_min'],
                    'BOT_MAX_SPREAD_PERCENT_NORMAL': st.session_state['spread_normal'],
                    'BOT_MAX_SPREAD_PERCENT_VOLATILE': st.session_state['spread_volatile'],
                    'BOT_MIN_TRADE_INTERVAL': st.session_state['min_interval'],
                    'BOT_MAX_TRADES_PER_HOUR': st.session_state['max_trades_per_hour'],
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

        # Indicator toggles
        st.caption("Chart display options")
        opt_col1, opt_col2, opt_col3, opt_col4, opt_col5 = st.columns(5)
        with opt_col1:
            show_sma = st.checkbox("Show SMAs", value=True)
        with opt_col2:
            show_vol = st.checkbox("Show Volume", value=True)
        with opt_col3:
            show_rsi = st.checkbox("Show RSI", value=True)
        with opt_col4:
            show_macd = st.checkbox("Show MACD", value=True)
        with opt_col5:
            show_trades = st.checkbox("Show Trades", value=True)

        # Price + SMA chart
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=dfv.index,
                open=dfv['open'], high=dfv['high'], low=dfv['low'], close=dfv['close'],
                name='Price'))
            if show_sma and "sma50" in inds:
                fig.add_trace(go.Scatter(x=dfv.index, y=inds["sma50"], name="SMA 50", line=dict(color="#1f77b4")))
            if show_sma and "sma200" in inds:
                fig.add_trace(go.Scatter(x=dfv.index, y=inds["sma200"], name="SMA 200", line=dict(color="#ff7f0e")))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            # Overlay executed trade markers from logs
            if show_trades:
                bx, by, sx, sy = _parse_trade_markers(dfv.index)
                if bx:
                    fig.add_trace(go.Scatter(
                        x=bx, y=by, name='Buy', mode='markers',
                        marker=dict(color='#16a34a', symbol='triangle-up', size=10),
                        hovertemplate='Buy @ %{y:.2f}<extra></extra>'
                    ))
                if sx:
                    fig.add_trace(go.Scatter(
                        x=sx, y=sy, name='Sell', mode='markers',
                        marker=dict(color='#dc2626', symbol='triangle-down', size=10),
                        hovertemplate='Sell @ %{y:.2f}<extra></extra>'
                    ))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly not available: {e}")
            st.line_chart(dfv["close"])

        # Volume bars
        if show_vol and 'volume' in dfv:
            try:
                import plotly.graph_objects as go
                vfig = go.Figure()
                vfig.add_trace(go.Bar(x=dfv.index, y=dfv['volume'], name='Volume', marker_color='#9ca3af'))
                vfig.update_layout(height=120, margin=dict(l=10, r=10, t=0, b=10))
                st.plotly_chart(vfig, use_container_width=True)
            except Exception:
                st.bar_chart(dfv['volume'])

        # RSI and MACD panels with thresholds and histogram
        try:
            if show_rsi and "rsi14" in inds:
                import plotly.graph_objects as go
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=dfv.index, y=inds['rsi14'], name='RSI(14)', line=dict(color="#374151")))
                # Thresholds
                rsi_buy_max = float(st.session_state.get('rsi_buy_max', getattr(bot, 'RSI_BUY_MAX', 70.0)))
                rsi_sell_min = float(st.session_state.get('rsi_sell_min', getattr(bot, 'RSI_SELL_MIN', 30.0)))
                rsi_fig.add_hline(y=rsi_buy_max, line=dict(color="#16a34a", width=1, dash='dot'))
                rsi_fig.add_hline(y=rsi_sell_min, line=dict(color="#dc2626", width=1, dash='dot'))
                # Shaded zones
                rsi_fig.add_shape(type="rect", xref="x", yref="y",
                                  x0=dfv.index.min(), x1=dfv.index.max(), y0=0, y1=rsi_buy_max,
                                  fillcolor="#16a34a", opacity=0.06, line_width=0)
                rsi_fig.add_shape(type="rect", xref="x", yref="y",
                                  x0=dfv.index.min(), x1=dfv.index.max(), y0=rsi_sell_min, y1=100,
                                  fillcolor="#dc2626", opacity=0.06, line_width=0)
                rsi_fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0,100]))
                st.plotly_chart(rsi_fig, use_container_width=True)

            if show_macd and "macd" in inds and "signal" in inds:
                import plotly.graph_objects as go
                macd_fig = go.Figure()
                macd_line = inds['macd']
                signal_line = inds['signal']
                hist = macd_line - signal_line
                colors = ['#16a34a' if v >= 0 else '#dc2626' for v in hist]
                macd_fig.add_trace(go.Bar(x=dfv.index, y=hist, name='Histogram', marker_color=colors))
                macd_fig.add_trace(go.Scatter(x=dfv.index, y=macd_line, name='MACD', line=dict(color="#2563eb")))
                macd_fig.add_trace(go.Scatter(x=dfv.index, y=signal_line, name='Signal', line=dict(color="#f59e0b")))
                macd_fig.add_hline(y=0, line=dict(color="#9ca3af", width=1))
                macd_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(macd_fig, use_container_width=True)
        except Exception:
            pass

        # Latest indicators summary with criteria coloring
        try:
            rsi_val = float(bot.calculate_rsi(df))
            sma_period = int(getattr(bot, 'SMA_PERIOD', 50))
            sma_val = float(bot.calculate_sma(df, sma_period))
            macd_line, signal_line = bot.calculate_macd(df)
            latest_macd = float(macd_line.iloc[-1])
            latest_signal = float(signal_line.iloc[-1])

            # Try to compute ML probability (train quickly in UI context)
            up_prob = None
            try:
                if hasattr(bot, 'trader'):
                    # safe train; ignore failures
                    bot.trader.train_model(df)
                    up_prob = float(bot.trader.predict_direction(df))
            except Exception:
                up_prob = None

            # Criteria thresholds (prefer UI session, fallback to bot)
            rsi_buy_max = float(st.session_state.get('rsi_buy_max', getattr(bot, 'RSI_BUY_MAX', 70.0)))
            rsi_sell_min = float(st.session_state.get('rsi_sell_min', getattr(bot, 'RSI_SELL_MIN', 30.0)))
            pred_thresh = float(getattr(bot, 'PREDICTION_THRESHOLD', 0.65))
            confirms_required_buy = int(getattr(bot, 'CONFIRMATIONS_REQUIRED_BUY', 2))
            confirms_required_sell = int(getattr(bot, 'CONFIRMATIONS_REQUIRED_SELL', 2))

            # Signals/confirmations
            macd_up = latest_macd > latest_signal
            macd_down = latest_macd < latest_signal
            try:
                trend_bullish = bool(bot.check_trend(df, sma_val, price))
            except Exception:
                trend_bullish = bool(price > sma_val)
            trend_bearish = not trend_bullish
            try:
                volume_confirmed = bool(bot.check_volume(df))
            except Exception:
                volume_confirmed = False

            # Confirmations count
            confs_buy_met = int(sum(1 for c in [macd_up, trend_bullish, volume_confirmed] if c))
            confs_sell_met = int(sum(1 for c in [macd_down, trend_bearish, volume_confirmed] if c))

            # Helpers for color
            def colorize(text: str, ok: bool) -> str:
                color = "#16a34a" if ok else "#dc2626"  # green/red
                return f"<span style='color:{color};font-weight:600'>{text}</span>"
            def color3(text: str, state: str) -> str:
                # state: 'good'|'neutral'|'bad'
                c = {'good': '#16a34a', 'neutral': '#f59e0b', 'bad': '#dc2626'}.get(state, '#dc2626')
                return f"<span style='color:{c};font-weight:600'>{text}</span>"

            # ML probability badge
            ml_ok_buy = (up_prob is not None) and (up_prob > pred_thresh)
            ml_ok_sell = (up_prob is not None) and (up_prob < (1 - pred_thresh))
            ml_text = "N/A" if up_prob is None else f"{up_prob:.2f}"
            st.markdown(
                f"Model up-prob: {colorize(ml_text + ' vs ' + str(round(pred_thresh,2)), ml_ok_buy)}",
                unsafe_allow_html=True,
            )

            # Compact decision summary badge (Buy-leaning / Sell-leaning / Neutral)
            buy_votes = 0
            sell_votes = 0
            # Model vote
            if up_prob is not None:
                if up_prob > pred_thresh:
                    buy_votes += 1
                elif up_prob < (1 - pred_thresh):
                    sell_votes += 1
            # RSI vote
            if rsi_val < rsi_buy_max:
                buy_votes += 1
            if rsi_val > rsi_sell_min:
                sell_votes += 1
            # Trend vote
            if price > sma_val:
                buy_votes += 1
            elif price < sma_val:
                sell_votes += 1
            # MACD vote
            if macd_up:
                buy_votes += 1
            elif macd_down:
                sell_votes += 1

            score = buy_votes - sell_votes
            # Also consider confirmations directly
            buy_ok = (confs_buy_met >= confirms_required_buy) and (up_prob is None or up_prob > pred_thresh)
            sell_ok = (confs_sell_met >= confirms_required_sell) and (up_prob is None or up_prob < (1 - pred_thresh))

            def pill(text: str, state: str) -> str:
                if state == 'buy':
                    bg, fg = '#16a34a', '#ffffff'  # green
                elif state == 'sell':
                    bg, fg = '#dc2626', '#ffffff'  # red
                else:
                    bg, fg = '#f59e0b', '#111111'  # yellow
                return f"<span style='background-color:{bg};color:{fg};padding:4px 10px;border-radius:999px;font-weight:700'>{text}</span>"

            if score >= 2 or buy_ok:
                summary = pill('Buy leaning', 'buy')
            elif score <= -2 or sell_ok:
                summary = pill('Sell leaning', 'sell')
            else:
                summary = pill('Neutral', 'neutral')

            st.markdown(f"Decision Summary: {summary}", unsafe_allow_html=True)

            # Buy and Sell panels side-by-side
            bc, sc = st.columns(2)
            with bc:
                st.caption("Buy Criteria")
                # Three-state color for RSI: green (< sell_min), yellow (between), red (>= buy_max)
                if rsi_val <= rsi_sell_min:
                    rsi_state_buy = 'good'
                elif rsi_val < rsi_buy_max:
                    rsi_state_buy = 'neutral'
                else:
                    rsi_state_buy = 'bad'
                st.markdown(
                    f"RSI(14): {color3(f'{rsi_val:.2f} < {rsi_buy_max:.0f}', rsi_state_buy)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Trend (Price>SMA{str(sma_period)}): {colorize(f'{price:,.2f} > {sma_val:,.2f}', price > sma_val)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"MACD>Signal: {colorize(f'{latest_macd:.4f} > {latest_signal:.4f}', macd_up)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Volume boost: {colorize('True' if volume_confirmed else 'False', volume_confirmed)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Confirmations: {colorize(f'{confs_buy_met}/{confirms_required_buy}', confs_buy_met >= confirms_required_buy)}",
                    unsafe_allow_html=True,
                )

            with sc:
                st.caption("Sell Criteria")
                # Three-state color for RSI: green (> buy_max), yellow (between), red (<= sell_min)
                if rsi_val >= rsi_buy_max:
                    rsi_state_sell = 'good'
                elif rsi_val > rsi_sell_min:
                    rsi_state_sell = 'neutral'
                else:
                    rsi_state_sell = 'bad'
                st.markdown(
                    f"RSI(14): {color3(f'{rsi_val:.2f} > {rsi_sell_min:.0f}', rsi_state_sell)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Trend (Price<SMA{str(sma_period)}): {colorize(f'{price:,.2f} < {sma_val:,.2f}', price < sma_val)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"MACD<Signal: {colorize(f'{latest_macd:.4f} < {latest_signal:.4f}', macd_down)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Volume boost: {colorize('True' if volume_confirmed else 'False', volume_confirmed)}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"Confirmations: {colorize(f'{confs_sell_met}/{confirms_required_sell}', confs_sell_met >= confirms_required_sell)}",
                    unsafe_allow_html=True,
                )

            # Legend / tips
            with st.expander("Legend: How to read colors", expanded=False):
                st.markdown(
                    "- Green = condition strongly supports action (buy/sell)\n"
                    "- Yellow = neutral zone (neither strong buy nor sell)\n"
                    "- Red = condition does not support action\n"
                    "- Model up-prob must exceed threshold for buys (or be below 1-threshold for sells)\n"
                    "- Confirmations count checks MACD direction, Trend alignment (Price vs SMA), and Volume boost"
                )
        except Exception:
            pass
    else:
        st.info("No market data available right now.")

    # Logs viewer
    st.subheader("Performance")
    try:
        eq_price = price
        eq_bal = bal if 'bal' in locals() else (bot.fetch_balance() or {})
        usdt_total = float(eq_bal.get("total", {}).get("USDT", 0.0))
        eth_total = float(eq_bal.get("total", {}).get("ETH", 0.0))
        equity = usdt_total + eth_total * eq_price
        base = st.session_state.get('perf_base_equity')
        if base is None:
            st.session_state['perf_base_equity'] = equity
            base = equity
        peak = st.session_state.get('perf_peak_equity', equity)
        if equity > peak:
            peak = equity
            st.session_state['perf_peak_equity'] = peak
        dd = 0.0 if peak <= 0 else (peak - equity) / peak
        cols_perf = st.columns(4)
        cols_perf[0].metric("Equity", f"${equity:,.2f}")
        cols_perf[1].metric("PnL (since baseline)", f"${(equity - base):,.2f}")
        cols_perf[2].metric("Drawdown", f"{dd*100:.2f}%")
        cols_perf[3].button("Reset baseline", on_click=lambda: st.session_state.update({'perf_base_equity': equity, 'perf_peak_equity': equity}))
    except Exception:
        st.caption("Performance metrics unavailable")

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
