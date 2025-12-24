import os
import json
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
PERF_STATE_PATH = REPO_DIR / "run" / "perf_state.json"

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


def _load_perf_state() -> Dict[str, float]:
    try:
        if PERF_STATE_PATH.exists():
            with PERF_STATE_PATH.open("r") as f:
                data = json.load(f)
            # ensure numeric types
            out: Dict[str, float] = {}
            for k in ("base", "peak"):
                v = data.get(k)
                try:
                    out[k] = float(v)
                except Exception:
                    pass
            return out
    except Exception:
        pass
    return {}


def _save_perf_state(base: Optional[float], peak: Optional[float]) -> None:
    try:
        PERF_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, float] = {}
        if base is not None:
            data["base"] = float(base)
        if peak is not None:
            data["peak"] = float(peak)
        with PERF_STATE_PATH.open("w") as f:
            json.dump(data, f)
    except Exception:
        pass


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
    paper_cfg = st.session_state.get("paper_trading")
    paper_show = paper if paper_cfg is None else bool(paper_cfg)
    st.caption(f"Symbol: {symbol} • Timeframe: {timeframe} • Paper: {'On' if paper_show else 'Off'}")
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

        # Volume and soft entries/exits
        st.caption("Volume filter and soft entries/exits")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            vol_boost = st.slider(
                "Volume boost multiplier", 1.00, 1.50,
                value=float(st.session_state.get('volume_boost', 1.10)), step=0.01,
                help="Require current volume > (multiplier × 20-period avg)"
            )
            soft_buy_delta = st.slider(
                "Soft buy delta", 0.00, 0.15,
                value=float(st.session_state.get('soft_buy_delta', 0.07)), step=0.01,
                help="Allows buy if ML prob ≥ (threshold − delta) with reduced size"
            )
            soft_buy_size = st.slider(
                "Soft buy size factor", 0.10, 1.00,
                value=float(st.session_state.get('soft_buy_size', 0.50)), step=0.05,
                help="Fraction of normal size for soft buys"
            )
        with vcol2:
            soft_sell_delta = st.slider(
                "Soft sell delta", 0.00, 0.15,
                value=float(st.session_state.get('soft_sell_delta', 0.07)), step=0.01,
                help="Allows sell if ML prob ≤ (1 − threshold + delta) with reduced size"
            )
            soft_sell_size = st.slider(
                "Soft sell size factor", 0.10, 1.00,
                value=float(st.session_state.get('soft_sell_size', 0.60)), step=0.05,
                help="Fraction of normal size for soft sells"
            )
            sell_full_bear = st.checkbox(
                "Sell full on strong bearish", value=bool(st.session_state.get('sell_full_bear', True)),
                help="If model probability is decisively low and momentum is down, close entire position"
            )

        # Exits and protection
        st.caption("Exits and protection")
        xcol1, xcol2 = st.columns(2)
        with xcol1:
            pos_trail_pct = st.slider(
                "Per-position trailing %", 0.03, 0.20,
                value=float(st.session_state.get('pos_trail_pct', 0.08)), step=0.01,
                help="Trail peak price since entry; widened in trends/volatility"
            )
            min_hold_sec = st.slider(
                "Min hold after buy (sec)", 0, 600,
                value=int(st.session_state.get('min_hold_sec', 120)), step=10,
                help="Block normal sells for at least this long after a buy (strong/ATR/trailing exits still allowed)"
            )
            no_churn_guard = st.checkbox(
                "No-churn until small profit", value=bool(st.session_state.get('no_churn_guard', True)),
                help="Prevents selling before a small profit unless strong-bearish or stop triggers"
            )
            min_profit_pct = st.slider(
                "Min profit to allow sell (%)", 0.0, 1.0,
                value=float(st.session_state.get('min_profit_pct', 0.30)), step=0.05,
                help="Sells allowed after price has risen by at least this percent from entry"
            )
        with xcol2:
            atr_stop_mult = st.slider(
                "ATR stop multiplier", 1.0, 4.0,
                value=float(st.session_state.get('atr_stop_mult', 2.0)), step=0.1,
                help="Hard stop at entry − ATR × multiplier"
            )
            sell_hyst = st.slider(
                "Sell probability hysteresis", 0.00, 0.15,
                value=float(st.session_state.get('sell_hyst', 0.05)), step=0.01,
                help="Extra buffer below 1−threshold to trigger sells; reduces flip-flops"
            )
        atr_within_hold = st.slider(
            "ATR trigger within hold", 0.00, 1.00,
            value=float(st.session_state.get('atr_within_hold', 0.25)), step=0.05,
            help="Within the hold window, allow sells if price ≤ entry − ATR × this value"
        )
        no_churn_atr_mult = st.slider(
            "ATR override for no‑churn", 0.00, 1.00,
            value=float(st.session_state.get('no_churn_atr_mult', 0.50)), step=0.05,
            help="Outside the hold window, allow a sell before profit if price ≤ entry − ATR × this value"
        )
        strong_sell_delta = st.slider(
            "Strong sell delta", 0.00, 0.20,
            value=float(st.session_state.get('strong_sell_delta', 0.10)), step=0.01,
            help="Full exit if prob ≤ (1 − threshold − delta) with bearish momentum"
        )
        diagnostics = st.checkbox("Verbose diagnostics in logs", value=bool(st.session_state.get('diagnostics', True)))

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
        st.session_state['volume_boost'] = float(vol_boost)
        st.session_state['soft_buy_delta'] = float(soft_buy_delta)
        st.session_state['soft_buy_size'] = float(soft_buy_size)
        st.session_state['soft_sell_delta'] = float(soft_sell_delta)
        st.session_state['soft_sell_size'] = float(soft_sell_size)
        st.session_state['sell_full_bear'] = bool(sell_full_bear)
        st.session_state['pos_trail_pct'] = float(pos_trail_pct)
        st.session_state['atr_stop_mult'] = float(atr_stop_mult)
        st.session_state['strong_sell_delta'] = float(strong_sell_delta)
        st.session_state['diagnostics'] = bool(diagnostics)
        st.session_state['min_hold_sec'] = int(min_hold_sec)
        st.session_state['sell_hyst'] = float(sell_hyst)
        st.session_state['atr_within_hold'] = float(atr_within_hold)
        st.session_state['no_churn_guard'] = bool(no_churn_guard)
        st.session_state['min_profit_pct'] = float(min_profit_pct)
        st.session_state['no_churn_atr_mult'] = float(no_churn_atr_mult)

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
                    'BOT_VOLUME_BOOST': st.session_state['volume_boost'],
                    'BOT_SOFT_BUY_DELTA': st.session_state['soft_buy_delta'],
                    'BOT_SOFT_BUY_SIZE_FACTOR': st.session_state['soft_buy_size'],
                    'BOT_SOFT_SELL_DELTA': st.session_state['soft_sell_delta'],
                    'BOT_SOFT_SELL_SIZE_FACTOR': st.session_state['soft_sell_size'],
                    'BOT_SELL_FULL_ON_STRONG_BEAR': 'true' if st.session_state['sell_full_bear'] else 'false',
                    'BOT_POS_TRAIL_PCT': st.session_state['pos_trail_pct'],
                    'BOT_STOP_ATR_MULT': st.session_state['atr_stop_mult'],
                    'BOT_STRONG_SELL_DELTA': st.session_state['strong_sell_delta'],
                    'BOT_DIAGNOSTICS': 'true' if st.session_state['diagnostics'] else 'false',
                    'BOT_MIN_HOLD_SEC': st.session_state['min_hold_sec'],
                    'BOT_SELL_HYSTERESIS': st.session_state['sell_hyst'],
                    'BOT_SELL_ATR_WITHIN_HOLD': st.session_state['atr_within_hold'],
                    'BOT_NO_CHURN_BEFORE_PROFIT': 'true' if st.session_state['no_churn_guard'] else 'false',
                    'BOT_MIN_PROFIT_PCT': st.session_state['min_profit_pct'] / 100.0,
                    'BOT_NO_CHURN_ATR_MULT': st.session_state['no_churn_atr_mult'],
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


def _render_sidebar_simple(pid_running: bool, pid: Optional[int], symbol: str, timeframe: str) -> Tuple[str, str]:
    with st.sidebar:
        st.header("Bot")

        symbol = st.text_input("Symbol", value=symbol, help="e.g., ETH/USDT")
        timeframe = st.selectbox(
            "Timeframe",
            TIMEFRAME_OPTIONS,
            index=(TIMEFRAME_OPTIONS.index(timeframe) if timeframe in TIMEFRAME_OPTIONS else 0),
        )
        st.session_state["symbol"] = symbol
        st.session_state["timeframe"] = timeframe

        st.session_state.setdefault("aggr", "balanced")
        mode = st.selectbox(
            "Mode",
            ["conservative", "balanced", "aggressive"],
            index=["conservative", "balanced", "aggressive"].index(st.session_state.get("aggr", "balanced")),
        )
        st.session_state["aggr"] = mode

        paper = st.checkbox(
            "Paper trading (no real orders)",
            value=bool(st.session_state.get("paper_trading", True)),
        )
        st.session_state["paper_trading"] = paper
        if not paper:
            st.warning("Live trading is ON. Double-check your API keys and risk settings.")

        diagnostics = st.checkbox(
            "Verbose diagnostics",
            value=bool(st.session_state.get("diagnostics", False)),
            help="Adds extra reasoning to the logs; can be noisy.",
        )
        st.session_state["diagnostics"] = diagnostics

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start", type="primary", disabled=pid_running, use_container_width=True):
                new_pid = _start_bot_process(
                    {
                        "BOT_SYMBOL": symbol,
                        "BOT_TIMEFRAME": timeframe,
                        "BOT_AGGRESSIVENESS": st.session_state["aggr"],
                        "BOT_PAPER_TRADING": "true" if st.session_state["paper_trading"] else "false",
                        "BOT_DIAGNOSTICS": "true" if st.session_state["diagnostics"] else "false",
                    }
                )
                if new_pid:
                    st.session_state["bot_pid"] = new_pid
                    _safe_rerun()
        with col2:
            if st.button("Stop", disabled=not pid_running, use_container_width=True):
                _stop_bot_process(pid)
                st.session_state["bot_pid"] = None
                _safe_rerun()

        with st.expander("Display", expanded=False):
            auto_enabled = st.checkbox("Auto-refresh", value=bool(st.session_state.get("auto_refresh_enabled", True)))
            refresh_sec = st.slider(
                "Refresh interval (sec)",
                2,
                30,
                int(st.session_state.get("refresh_sec", 5)),
                1,
            )
            st.session_state["auto_refresh_enabled"] = auto_enabled
            st.session_state["refresh_sec"] = refresh_sec

            st.divider()
            log_lines = st.slider("Log lines", 20, 1000, int(st.session_state.get("log_lines", 200)), 10)
            newest_first = st.checkbox("Newest first", value=bool(st.session_state.get("log_newest_first", True)))
            level_choice = st.selectbox(
                "Level filter",
                ["All", "ERROR", "WARNING", "INFO", "DEBUG"],
                index=["All", "ERROR", "WARNING", "INFO", "DEBUG"].index(st.session_state.get("log_level_filter", "All")),
            )
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
    selected_symbol, selected_timeframe = _render_sidebar_simple(pid_running, pid, selected_symbol, selected_timeframe)

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

        # Organize charts into tabs for clarity
        t_price, t_volume, t_rsi, t_macd = st.tabs(["Price", "Volume", "RSI", "MACD"])

        # Price + SMAs + Trades
        with t_price:
            st.subheader("Price (" + selected_symbol + ") + SMAs")
            with st.expander("What is this?", expanded=False):
                st.markdown(
                    "- Shows the coin's price over time as candlesticks (each bar is one time block).\n"
                    "- Colored SMA lines (50/200) show average price and trend direction.\n"
                    "- Price above SMA lines suggests an uptrend; below suggests a downtrend.\n"
                    "- Optional markers show where the bot executed buys (green) and sells (red)."
                )
            show_sma = st.checkbox("Show SMAs", value=True, key="opt_show_sma")
            show_trades = st.checkbox("Show Trades", value=True, key="opt_show_trades")
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
                if show_trades:
                    bx, by, sx, sy = _parse_trade_markers(dfv.index)
                    if bx:
                        fig.add_trace(go.Scatter(x=bx, y=by, name='Buy', mode='markers',
                                                 marker=dict(color='#16a34a', symbol='triangle-up', size=10),
                                                 hovertemplate='Buy @ %{y:.2f}<extra></extra>'))
                    if sx:
                        fig.add_trace(go.Scatter(x=sx, y=sy, name='Sell', mode='markers',
                                                 marker=dict(color='#dc2626', symbol='triangle-down', size=10),
                                                 hovertemplate='Sell @ %{y:.2f}<extra></extra>'))
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                                  hovermode='x unified', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Plotly not available: {e}")
                st.line_chart(dfv["close"])

        # Volume
        with t_volume:
            st.subheader("Volume")
            with st.expander("What is this?", expanded=False):
                st.markdown(
                    "- Bars show how much was traded in each time block.\n"
                    "- Taller bars = more participation; small bars = quiet trading.\n"
                    "- Moves on higher volume are generally more reliable than thin moves."
                )
            try:
                import plotly.graph_objects as go
                vfig = go.Figure()
                vfig.add_trace(go.Bar(x=dfv.index, y=dfv['volume'], name='Volume', marker_color='#9ca3af'))
                vfig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10), template='plotly_white')
                st.plotly_chart(vfig, use_container_width=True)
            except Exception:
                st.bar_chart(dfv['volume'])

        # RSI
        with t_rsi:
            st.subheader("RSI(14) with thresholds")
            with st.expander("What is this?", expanded=False):
                st.markdown(
                    "- RSI is a 0–100 gauge of recent price speed. Lower = cooler, higher = hotter.\n"
                    "- Green zone (below Buy max): friendlier for buys. Red zone (above Sell min): friendlier for sells.\n"
                    "- The middle zone is neutral — neither strong buy nor sell by itself."
                )
            try:
                import plotly.graph_objects as go
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=dfv.index, y=inds['rsi14'], name='RSI(14)', line=dict(color="#374151")))
                rsi_buy_max = float(st.session_state.get('rsi_buy_max', getattr(bot, 'RSI_BUY_MAX', 70.0)))
                rsi_sell_min = float(st.session_state.get('rsi_sell_min', getattr(bot, 'RSI_SELL_MIN', 30.0)))
                rsi_fig.add_hline(y=rsi_buy_max, line=dict(color="#16a34a", width=1, dash='dot'))
                rsi_fig.add_hline(y=rsi_sell_min, line=dict(color="#dc2626", width=1, dash='dot'))
                rsi_fig.add_shape(type="rect", xref="x", yref="y",
                                  x0=dfv.index.min(), x1=dfv.index.max(), y0=0, y1=rsi_buy_max,
                                  fillcolor="#16a34a", opacity=0.06, line_width=0)
                rsi_fig.add_shape(type="rect", xref="x", yref="y",
                                  x0=dfv.index.min(), x1=dfv.index.max(), y0=rsi_sell_min, y1=100,
                                  fillcolor="#dc2626", opacity=0.06, line_width=0)
                rsi_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0,100]),
                                      template='plotly_white')
                st.plotly_chart(rsi_fig, use_container_width=True)
            except Exception:
                if 'rsi14' in inds:
                    st.line_chart(inds['rsi14'])

        # MACD
        with t_macd:
            st.subheader("MACD (line, signal, histogram)")
            with st.expander("What is this?", expanded=False):
                st.markdown(
                    "- MACD combines direction and strength of the move (momentum).\n"
                    "- Green histogram bars (above zero) = upward momentum; red bars (below zero) = downward momentum.\n"
                    "- MACD crossing above Signal often marks momentum turning up (and vice versa)."
                )
            try:
                import plotly.graph_objects as go
                macd_line = inds['macd']
                signal_line = inds['signal']
                hist = macd_line - signal_line
                colors = ['#16a34a' if v >= 0 else '#dc2626' for v in hist]
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Bar(x=dfv.index, y=hist, name='Histogram', marker_color=colors))
                macd_fig.add_trace(go.Scatter(x=dfv.index, y=macd_line, name='MACD', line=dict(color="#2563eb")))
                macd_fig.add_trace(go.Scatter(x=dfv.index, y=signal_line, name='Signal', line=dict(color="#f59e0b")))
                macd_fig.add_hline(y=0, line=dict(color="#9ca3af", width=1))
                macd_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), template='plotly_white')
                st.plotly_chart(macd_fig, use_container_width=True)
            except Exception:
                if 'macd' in inds and 'signal' in inds:
                    st.line_chart(pd.DataFrame({"macd": inds["macd"], "signal": inds["signal"]}, index=dfv.index))

        # Latest indicators summary + trade checklist (simplified)
        try:
            rsi_val = float(bot.calculate_rsi(df))
            sma_period = int(getattr(bot, 'SMA_PERIOD', 50))
            sma_val = float(bot.calculate_sma(df, sma_period))
            macd_line, signal_line = bot.calculate_macd(df)
            latest_macd = float(macd_line.iloc[-1])
            latest_signal = float(signal_line.iloc[-1])

            # Try to compute ML probability (train quickly in UI context)
            try:
                if hasattr(bot, 'trader'):
                    # safe train; ignore failures
                    bot.trader.train_model(df)
                    up_prob = float(bot.trader.predict_direction(df))
                else:
                    up_prob = 0.5
            except Exception:
                up_prob = 0.5

            # Criteria thresholds (prefer UI session, fallback to bot)
            rsi_buy_max = float(st.session_state.get('rsi_buy_max', getattr(bot, 'RSI_BUY_MAX', 70.0)))
            rsi_sell_min = float(st.session_state.get('rsi_sell_min', getattr(bot, 'RSI_SELL_MIN', 30.0)))
            pred_thresh = float(st.session_state.get('threshold', getattr(bot, 'PREDICTION_THRESHOLD', 0.65)))
            confirms_required_buy = int(st.session_state.get('confirms', getattr(bot, 'CONFIRMATIONS_REQUIRED_BUY', 2)))
            confirms_required_sell = int(st.session_state.get('confirms', getattr(bot, 'CONFIRMATIONS_REQUIRED_SELL', 2)))

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
            vol_mult = float(st.session_state.get('volume_boost', 1.10))

            # Confirmations count
            confs_buy_met = int(sum(1 for c in [macd_up, trend_bullish, volume_confirmed] if c))
            confs_sell_met = int(sum(1 for c in [macd_down, trend_bearish, volume_confirmed] if c))
            st.subheader("Trade Checklist")
            st.caption("This matches the bot's gating: signal checks + execution guards.")

            # Trade checklist: end-to-end checklist for Buy and Sell
            try:
                # Balance and portfolio
                try:
                    balance = bal if 'bal' in locals() else (bot.fetch_balance() or {})
                except Exception:
                    balance = {}
                usdt_total_g = float((balance.get('total') or {}).get('USDT', 0.0))
                eth_total_g = float((balance.get('total') or {}).get('ETH', 0.0))
                free_eth = float((balance.get('free') or {}).get('ETH', 0.0))
                portfolio_value = usdt_total_g + eth_total_g * price if 'price' in locals() else 0.0

                # ATR and momentum for sizing
                atr_val = float(bot.calculate_atr(df))
                momentum_score = float(bot.calculate_momentum_score(df))

                # Estimate size with current bot logic
                try:
                    est_size = float(bot.get_trade_size(
                        balance, float(price), atr_val,
                        rsi_val=float(rsi_val), momentum_score_val=momentum_score
                    )) if price else 0.0
                except Exception:
                    est_size = 0.0
                est_value = est_size * float(price) if price else 0.0

                # Spread gate
                try:
                    bid, ask, spread = bot.get_order_book_spread()
                    mid = (bid + ask) / 2 if bid and ask else float(price)
                    base_p = mid if mid and mid > 0 else ask
                    spread_pct = (spread / base_p) * 100 if base_p else 100.0
                except Exception:
                    spread_pct = 999.0
                try:
                    regime = bot.detect_market_regime(df)
                except Exception:
                    regime = getattr(bot, 'MarketRegime', type('X', (), {'VOLATILE':'VOLATILE'})).VOLATILE
                spread_cap = float(st.session_state.get('spread_volatile' if regime == bot.MarketRegime.VOLATILE else 'spread_normal', 0.22))
                spread_ok = spread_pct <= spread_cap

                # Risk cap (position size vs equity)
                try:
                    max_pos_frac = float(getattr(bot.risk_manager, 'max_position_size', 0.3))
                except Exception:
                    max_pos_frac = 0.3
                pos_cap_value = portfolio_value * max_pos_frac
                # Compute clamped size/value same as bot
                cap_size_eth = (pos_cap_value / float(price)) if price else 0.0
                clamped_size = min(est_size, cap_size_eth) if cap_size_eth > 0 else est_size
                clamped_value = clamped_size * float(price) if price else 0.0
                pos_cap_ok = clamped_value <= pos_cap_value if pos_cap_value > 0 else False

                # Gates per side
                ml_buy_ok = up_prob > pred_thresh
                ml_sell_ok = up_prob < (1 - pred_thresh)
                rsi_buy_ok = rsi_val < rsi_buy_max
                rsi_sell_ok = rsi_val > rsi_sell_min
                conf_buy_ok = (confs_buy_met >= confirms_required_buy)
                conf_sell_ok = (confs_sell_met >= confirms_required_sell)
                size_ok = est_size > 0
                have_eth = free_eth > 0.0

                def _pf(ok: bool) -> str:
                    return "PASS" if ok else "FAIL"

                checklist_rows = [
                    {
                        "Check": "Model up-prob",
                        "Value": f"{up_prob:.2f} (buy>{pred_thresh:.2f}, sell<{(1 - pred_thresh):.2f})",
                        "Buy": _pf(ml_buy_ok),
                        "Sell": _pf(ml_sell_ok),
                    },
                    {
                        "Check": "RSI(14)",
                        "Value": f"{rsi_val:.1f} (buy<{rsi_buy_max:.0f}, sell>{rsi_sell_min:.0f})",
                        "Buy": _pf(rsi_buy_ok),
                        "Sell": _pf(rsi_sell_ok),
                    },
                    {
                        "Check": f"Trend vs SMA{sma_period}",
                        "Value": f"price={price:,.2f}, sma={sma_val:,.2f}",
                        "Buy": _pf(price > sma_val),
                        "Sell": _pf(price < sma_val),
                    },
                    {
                        "Check": "MACD",
                        "Value": f"macd={latest_macd:.4f}, signal={latest_signal:.4f}",
                        "Buy": _pf(macd_up),
                        "Sell": _pf(macd_down),
                    },
                    {
                        "Check": f"Volume confirm (x{vol_mult:.2f})",
                        "Value": "confirmed" if volume_confirmed else "not confirmed",
                        "Buy": _pf(volume_confirmed),
                        "Sell": _pf(volume_confirmed),
                    },
                    {
                        "Check": "Confirmations",
                        "Value": f"buy={confs_buy_met}/{confirms_required_buy}, sell={confs_sell_met}/{confirms_required_sell}",
                        "Buy": _pf(conf_buy_ok),
                        "Sell": _pf(conf_sell_ok),
                    },
                    {
                        "Check": "Spread",
                        "Value": f"{spread_pct:.2f}% (cap {spread_cap:.2f}%, {regime})",
                        "Buy": _pf(spread_ok),
                        "Sell": _pf(spread_ok),
                    },
                    {
                        "Check": "Size / Holdings",
                        "Value": f"est_size={est_size:.6f} ETH, free_eth={free_eth:.6f}",
                        "Buy": _pf(size_ok),
                        "Sell": _pf(have_eth),
                    },
                    {
                        "Check": "Position cap",
                        "Value": f"cap=${pos_cap_value:,.2f}, will_use=${clamped_value:,.2f}",
                        "Buy": _pf(pos_cap_ok),
                        "Sell": _pf(pos_cap_ok),
                    },
                ]
                checklist_df = pd.DataFrame(checklist_rows)
                try:
                    def _pass_fail_style(v: object) -> str:
                        if v == "PASS":
                            return "color: #16a34a; font-weight: 700;"
                        if v == "FAIL":
                            return "color: #dc2626; font-weight: 700;"
                        return ""

                    styled = checklist_df.style.applymap(_pass_fail_style, subset=["Buy", "Sell"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                except Exception:
                    st.table(checklist_df)

                buy_allowed = all([ml_buy_ok, rsi_buy_ok, conf_buy_ok, spread_ok, size_ok, pos_cap_ok])
                sell_allowed = all([ml_sell_ok, rsi_sell_ok, conf_sell_ok, spread_ok, have_eth, pos_cap_ok])
                c1, c2 = st.columns(2)
                c1.metric("Buy allowed now", "YES" if buy_allowed else "NO")
                c2.metric("Sell allowed now", "YES" if sell_allowed else "NO")

                # Blocking reasons
                buy_reasons = []
                if not ml_buy_ok: buy_reasons.append("ML below threshold")
                if not rsi_buy_ok: buy_reasons.append("RSI gate")
                if not conf_buy_ok: buy_reasons.append("confirmations")
                if not spread_ok: buy_reasons.append("spread")
                if not size_ok: buy_reasons.append("size")
                # If clamped_value fits the cap, position cap will not block
                # (the order will use the clamped size). Only add if even clamped cannot fit.
                if clamped_value > pos_cap_value: buy_reasons.append("position cap")
                sell_reasons = []
                if not ml_sell_ok: sell_reasons.append("ML not < 1-threshold")
                if not rsi_sell_ok: sell_reasons.append("RSI gate")
                if not conf_sell_ok: sell_reasons.append("confirmations")
                if not spread_ok: sell_reasons.append("spread")
                if not have_eth: sell_reasons.append("no free ETH")
                if clamped_value > pos_cap_value: sell_reasons.append("position cap")
                if buy_reasons or sell_reasons:
                    st.caption(
                        "Blocking reasons — "
                        + (f"Buy: {', '.join(buy_reasons)}; " if buy_reasons else "")
                        + (f"Sell: {', '.join(sell_reasons)}" if sell_reasons else "")
                    )
            except Exception:
                pass
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

        # Initialize baseline/peak from persisted state if not present in session
        if 'perf_base_equity' not in st.session_state or 'perf_peak_equity' not in st.session_state:
            persisted = _load_perf_state()
            if 'base' in persisted:
                st.session_state['perf_base_equity'] = float(persisted['base'])
            if 'peak' in persisted:
                st.session_state['perf_peak_equity'] = float(persisted['peak'])

        base = st.session_state.get('perf_base_equity')
        if base is None:
            st.session_state['perf_base_equity'] = equity
            base = equity
            _save_perf_state(base, st.session_state.get('perf_peak_equity'))

        peak = st.session_state.get('perf_peak_equity', equity)
        if equity > peak:
            peak = equity
            st.session_state['perf_peak_equity'] = peak
            _save_perf_state(st.session_state.get('perf_base_equity'), peak)

        dd = 0.0 if peak <= 0 else (peak - equity) / peak
        cols_perf = st.columns(4)
        cols_perf[0].metric("Equity", f"${equity:,.2f}")
        cols_perf[1].metric("PnL (since baseline)", f"${(equity - base):,.2f}")
        cols_perf[2].metric("Drawdown", f"{dd*100:.2f}%")
        def _reset_perf():
            st.session_state.update({'perf_base_equity': equity, 'perf_peak_equity': equity})
            _save_perf_state(equity, equity)
        cols_perf[3].button("Reset baseline", on_click=_reset_perf)
        # Small caption for transparency
        st.caption(f"Baseline: ${base:,.2f} • Peak: ${peak:,.2f}")
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
