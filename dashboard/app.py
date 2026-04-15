"""
AI Trading Assistant — Terminal UI
Run with: streamlit run dashboard/app.py
"""
import sys
sys.path.insert(0, "..")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import duckdb
import time

from config import settings
from data.fetcher import fetch_historical, get_live_price
from strategies.aggregator import SignalAggregator
from risk.manager import RiskManager
from execution.paper import PaperBroker
from ai.agent import TradingAgent
from data.store import init_db, DB_PATH, get_open_positions
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.engine_5m import BacktestEngine5m, run_multi_market, FUTURES_UNIVERSE
from backtesting.engine_aplus import BacktestEngineAPlus, APLUS_UNIVERSE
from backtesting.models import BacktestTrade as BtTrade
from utils.indicators import add_all, get_key_levels

init_db()

st.set_page_config(
    page_title="AI Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────── #
# Design system — CSS only (no HTML layout blocks)                    #
# ─────────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080b12 !important;
}
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 1rem 1.5rem 1rem 1.5rem !important; max-width: 100% !important; }
section.main > div { padding-top: 0.5rem !important; }

/* Typography */
h1, h2, h3, h4 { color: #e2e8f0 !important; font-weight: 700 !important; }

/* Tabs */
div[data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #1e2a3a !important;
    border-radius: 0 !important;
    gap: 0 !important;
}
div[data-baseweb="tab"] {
    background: transparent !important;
    color: #5a6a7a !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 10px 22px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[aria-selected="true"][data-baseweb="tab"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}
div[data-baseweb="tab-panel"] { background: #080b12 !important; padding-top: 16px !important; }

/* Input fields */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.12) !important;
}
div[data-baseweb="select"] > div {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}
label[data-testid="stWidgetLabel"],
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    color: #5a6a7a !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(56,139,253,0.25) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(56,139,253,0.4) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0d1117 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] > div {
    color: #5a6a7a !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="stMetricValue"] > div {
    color: #e2e8f0 !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* Expander */
details {
    background: #0d1117 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 8px !important;
    margin: 6px 0 !important;
}
details summary {
    color: #768390 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    padding: 10px 14px !important;
}
details > div { padding: 0 14px 12px 14px !important; }

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a3a !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table { background: #0d1117 !important; }
[data-testid="stDataFrame"] th {
    background: #161b22 !important;
    color: #5a6a7a !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    border-bottom: 1px solid #1e2a3a !important;
}
[data-testid="stDataFrame"] td {
    color: #c9d1d9 !important;
    font-size: 0.78rem !important;
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    border-bottom: 1px solid #161b22 !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
    border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
    background: #161b22 !important;
    border-radius: 4px !important;
}

/* Toggle */
[data-testid="stToggle"] span {
    font-size: 0.75rem !important;
    color: #768390 !important;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: #0d1117 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    margin: 4px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* Dividers */
hr { border-color: #1e2a3a !important; margin: 12px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080b12; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #30363d; }

/* Success/Warning/Error custom */
div[data-testid="stNotification"] { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────── #
# Sound alert                                                          #
# ─────────────────────────────────────────────────────────────────── #
ALERT_JS = """<script>
(function(){try{
    var c=new(window.AudioContext||window.webkitAudioContext)();
    [0,0.15,0.3].forEach(function(t){
        var o=c.createOscillator(),g=c.createGain();
        o.connect(g);g.connect(c.destination);
        o.frequency.value=880;
        g.gain.setValueAtTime(0.25,c.currentTime+t);
        g.gain.exponentialRampToValueAtTime(0.001,c.currentTime+t+0.1);
        o.start(c.currentTime+t);o.stop(c.currentTime+t+0.1);
    });
}catch(e){}}());
</script>"""

# ─────────────────────────────────────────────────────────────────── #
# Session state                                                        #
# ─────────────────────────────────────────────────────────────────── #
_defaults = {
    "broker":             PaperBroker(risk_manager=RiskManager()),
    "agent":              TradingAgent(),
    "last_decision":      None,
    "chat_history":       [],
    "df":                 None,
    "agg_signal":         None,
    "scanner_results":    [],
    "alerted_symbols":    set(),
    "current_symbol":     "ES",
    "current_timeframe":  "5m",
    "current_market":     "futures",
    "last_analysis_time": 0.0,   # epoch float — tracks when last full analysis ran
    "live_price":         None,  # latest fast_info snapshot
    "bt_result":          None,  # last BacktestResult
    # Autopilot
    "ap_armed":           False,
    "ap_auto_trade":      True,
    "ap_last_signal":     None,   # last live_signal() result
    "ap_last_check":      0.0,    # epoch of last check
    "ap_activity":        [],     # list of {time, type, msg} dicts
    "ap_executed":        set(),  # set of signal hashes already traded
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

broker    = st.session_state.broker
portfolio = broker.portfolio_summary()
is_paper  = settings.is_paper

# ─────────────────────────────────────────────────────────────────── #
# Top header — native Streamlit layout                                 #
# ─────────────────────────────────────────────────────────────────── #
hdr_left, hdr_right = st.columns([1, 3])

with hdr_left:
    mode_color = "#2ea043" if is_paper else "#f85149"
    mode_label = "● PAPER MODE" if is_paper else "● LIVE MODE"
    st.markdown(
        f"## 📈 AI Trading Terminal\n"
        f'<span style="color:{mode_color};font-size:0.72rem;font-weight:700;'
        f'letter-spacing:1px;">{mode_label}</span>',
        unsafe_allow_html=True,
    )

with hdr_right:
    pnl      = portfolio["daily_pnl"]
    pnl_sign = "+" if pnl >= 0 else ""
    halt_txt = "  🛑 **HALTED**" if broker.risk_manager.is_halted else ""
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Balance",    f"${portfolio['account_balance']:,.0f}")
    a2.metric("Day P&L",   f"{pnl_sign}${pnl:,.2f}")
    a3.metric("Positions",  str(portfolio["open_positions"]))
    a4.metric("Mode",       "Paper" if is_paper else "Live")
    if broker.risk_manager.is_halted:
        st.error("🛑 Trading halted — daily loss limit reached")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────── #
# Tabs                                                                 #
# ─────────────────────────────────────────────────────────────────── #
tab_autopilot, tab_analysis, tab_scanner, tab_backtest, tab_journal = st.tabs(
    ["🤖  Autopilot", "📊  Analysis", "🔍  Scanner", "⚡  Backtest", "📓  Journal"]
)


# ─────────────────────────────────────────────────────────────────── #
# HELPERS                                                              #
# ─────────────────────────────────────────────────────────────────── #
def _rec_html(agg) -> None:
    """Render the big coloured recommendation card."""
    rec = agg.recommendation
    styles = {
        "STRONG BUY":         ("0d2818", "3fb950", "3fb950"),
        "BUY":                ("0d1f14", "238636", "2ea043"),
        "STRONG SELL":        ("2d0f0f", "f85149", "f85149"),
        "SELL":               ("1f0d0d", "da3633", "da3633"),
        "WAIT":               ("0d1117", "30363d", "484f58"),
        "WEAK SIGNAL — WAIT": ("0d1117", "30363d", "484f58"),
    }
    bg, border, text = styles.get(rec, ("0d1117","30363d","484f58"))

    # Sub-line: richer message when nothing is firing
    if agg.signal_count == 0:
        sub = (f"0/{agg.total_strategies} strategies see a setup right now &nbsp;·&nbsp; "
               f"score {agg.composite_score:+.2f} &nbsp;·&nbsp; "
               f"see breakdown below for each reason")
    else:
        sub = (f"{agg.signal_count}/{agg.total_strategies} strategies agree &nbsp;·&nbsp; "
               f"score {agg.composite_score:+.2f} &nbsp;·&nbsp; conf {agg.confidence:.0%}")

    st.markdown(f"""
<div style="
    background:#{bg};border:1px solid #{border};border-radius:8px;
    padding:20px 24px 16px 24px;text-align:center;margin-bottom:14px;">
  <div style="font-size:2rem;font-weight:900;letter-spacing:3px;
              color:#{text};line-height:1;margin-bottom:6px;">{rec}</div>
  <div style="font-family:'SF Mono','Fira Code',monospace;font-size:0.7rem;
              color:#768390;">{sub}</div>
</div>""", unsafe_allow_html=True)


def _levels_html(entry, stop, target, rr_str) -> None:
    """Render the 4-box trade levels grid."""
    def box(label, value, color):
        return (f'<div style="background:#0d1117;border:1px solid #1e2a3a;'
                f'border-left:3px solid #{color};border-radius:6px;padding:10px 14px;">'
                f'<div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.8px;color:#5a6a7a;margin-bottom:3px;">{label}</div>'
                f'<div style="font-family:\'SF Mono\',\'Fira Code\',monospace;'
                f'font-size:1rem;font-weight:600;color:#{color};">{value}</div>'
                f'</div>')

    st.markdown(
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px;">'
        f'{box("Entry",  f"{entry:.2f}" if entry else "—",  "388bfd")}'
        f'{box("Stop",   f"{stop:.2f}"  if stop  else "—",  "f85149")}'
        f'{box("Target", f"{target:.2f}" if target else "—", "3fb950")}'
        f'{box("R:R",    rr_str,                             "d29922")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _strategy_pills(agg) -> None:
    """Render strategy pill badges."""
    html = '<div style="margin-bottom:10px;line-height:2.2;">'
    for s in agg.agreeing_strategies:
        html += (f'<span style="display:inline-block;font-size:0.62rem;font-weight:700;'
                 f'padding:2px 8px;border-radius:3px;margin:2px;text-transform:uppercase;'
                 f'letter-spacing:0.3px;background:#0d2818;border:1px solid #238636;'
                 f'color:#3fb950;">{s}</span>')
    for s in agg.disagreeing_strategies:
        html += (f'<span style="display:inline-block;font-size:0.62rem;font-weight:700;'
                 f'padding:2px 8px;border-radius:3px;margin:2px;text-transform:uppercase;'
                 f'letter-spacing:0.3px;background:#2d0f0f;border:1px solid #da3633;'
                 f'color:#f85149;">{s}</span>')
    for s in agg.neutral_strategies:
        html += (f'<span style="display:inline-block;font-size:0.62rem;font-weight:600;'
                 f'padding:2px 8px;border-radius:3px;margin:2px;text-transform:uppercase;'
                 f'letter-spacing:0.3px;background:#161b22;border:1px solid #21262d;'
                 f'color:#484f58;">{s}</span>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _metric_box(label: str, value: str, color: str) -> str:
    """HTML metric card for use inside a CSS grid (backtest summary row)."""
    return (
        f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-top:2px solid #{color};'
        f'border-radius:6px;padding:12px 14px;text-align:center;">'
        f'<div style="font-size:0.58rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#5a6a7a;margin-bottom:4px;">{label}</div>'
        f'<div style="font-family:\'SF Mono\',monospace;font-size:1.1rem;'
        f'font-weight:700;color:#{color};">{value}</div>'
        f'</div>'
    )


def _section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;color:#5a6a7a;border-bottom:1px solid #1e2a3a;'
        f'padding-bottom:5px;margin:12px 0 8px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


def render_signal_panel(agg, decision):
    """Full right-hand signal panel. Returns (entry, stop, target)."""
    _rec_html(agg)

    entry_price  = decision.entry     if decision and decision.entry     else agg.suggested_entry
    stop_price   = decision.stop_loss if decision and decision.stop_loss else agg.suggested_stop
    target_price = decision.target    if decision and decision.target    else agg.suggested_target

    if entry_price and stop_price and target_price:
        risk   = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        rr_str = f"{reward/risk:.1f}:1" if risk > 0 else "—"
    elif decision and decision.r_ratio:
        rr_str = f"{decision.r_ratio}:1"
    else:
        rr_str = "—"

    _levels_html(entry_price, stop_price, target_price, rr_str)

    _section_label("Strategies")
    _strategy_pills(agg)

    if decision and decision.raw_response:
        _section_label("Claude's Analysis")
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-radius:6px;'
            f'padding:12px 14px;font-size:0.78rem;line-height:1.65;color:#c9d1d9;'
            f'max-height:280px;overflow-y:auto;">'
            f'{decision.raw_response}'
            f'</div>',
            unsafe_allow_html=True,
        )

    can_trade = (agg.direction in ("BUY","SELL")
                 and entry_price and stop_price and target_price
                 and agg.recommendation not in ("WAIT", "WEAK SIGNAL — WAIT"))

    if can_trade:
        trade_dir = (decision.action if decision and decision.action in ("BUY","SELL")
                     else agg.direction)
        strat_lbl = (decision.lead_strategy if decision and decision.lead_strategy
                     else ", ".join(agg.agreeing_strategies[:3]))
        ai_rsn    = (decision.reasoning if decision and decision.reasoning
                     else f"Aggregated: {agg.recommendation} | score {agg.composite_score:+.2f}")
        btn_icon  = "🟢" if trade_dir == "BUY" else "🔴"
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"{btn_icon} Execute {trade_dir} — Paper Trade",
                     type="primary", use_container_width=True, key="exec_trade"):
            ok, msg, _ = broker.open_position(
                symbol=st.session_state.current_symbol,
                direction=trade_dir,
                entry_price=entry_price,
                stop_loss=stop_price,
                take_profit=target_price,
                strategy_used=strat_lbl,
                ai_reasoning=ai_rsn,
            )
            st.success(msg) if ok else st.error(msg)

    return entry_price, stop_price, target_price


def build_chart(df, entry=None, stop=None, target=None, open_positions=None):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.19, 0.19],
        vertical_spacing=0.018,
        subplot_titles=("", "", ""),
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"],
        low=df["low"],  close=df["close"],
        name="", showlegend=False,
        increasing=dict(line=dict(color="#3fb950", width=1), fillcolor="#163a22"),
        decreasing=dict(line=dict(color="#f85149", width=1), fillcolor="#3a1318"),
    ), row=1, col=1)

    # EMAs
    for p, col, w in [(8,"#d29922",1.0),(21,"#388bfd",1.2),(55,"#8957e5",1.4)]:
        if f"ema_{p}" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df[f"ema_{p}"],
                name=f"EMA {p}", line=dict(color=col, width=w), opacity=0.85,
            ), row=1, col=1)

    # VWAP
    if "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vwap"],
            name="VWAP", line=dict(color="#cdd9e5", width=1.1, dash="dot"), opacity=0.55,
        ), row=1, col=1)

    # S/R levels
    try:
        lvls = get_key_levels(df)
        for l in lvls["resistance"][:3]:
            fig.add_hline(y=l, line_dash="dot", line_color="#f85149", line_width=0.8,
                          opacity=0.45, row=1, col=1,
                          annotation_text=f"R {l:.1f}",
                          annotation_font=dict(size=9, color="#f85149"),
                          annotation_position="right")
        for l in lvls["support"][:3]:
            fig.add_hline(y=l, line_dash="dot", line_color="#3fb950", line_width=0.8,
                          opacity=0.45, row=1, col=1,
                          annotation_text=f"S {l:.1f}",
                          annotation_font=dict(size=9, color="#3fb950"),
                          annotation_position="right")
    except Exception:
        pass

    # Trade lines
    if entry:
        fig.add_hline(y=entry, line_color="#388bfd", line_width=1.5,
                      annotation_text=f" ENTRY {entry:.2f}",
                      annotation_font=dict(size=9, color="#388bfd"),
                      annotation_position="right", row=1, col=1)
    if stop:
        fig.add_hline(y=stop, line_color="#f85149", line_width=1.5,
                      annotation_text=f" STOP {stop:.2f}",
                      annotation_font=dict(size=9, color="#f85149"),
                      annotation_position="right", row=1, col=1)
    if target:
        fig.add_hline(y=target, line_color="#3fb950", line_width=1.5,
                      annotation_text=f" TARGET {target:.2f}",
                      annotation_font=dict(size=9, color="#3fb950"),
                      annotation_position="right", row=1, col=1)
    if entry and stop:
        fig.add_hrect(y0=min(entry,stop), y1=max(entry,stop),
                      fillcolor="#f85149", opacity=0.05, line_width=0, row=1, col=1)
    if entry and target:
        fig.add_hrect(y0=min(entry,target), y1=max(entry,target),
                      fillcolor="#3fb950", opacity=0.04, line_width=0, row=1, col=1)

    # Open position lines (one per live trade, dashed)
    if open_positions:
        for pos in open_positions:
            pos_color = "#3fb950" if pos["direction"] == "LONG" else "#f85149"
            fig.add_hline(
                y=pos["entry_price"], line_color=pos_color,
                line_width=1.2, line_dash="dash", opacity=0.6, row=1, col=1,
                annotation_text=f" {pos['direction']} {pos['symbol']} x{pos['qty']}",
                annotation_font=dict(size=9, color=pos_color),
                annotation_position="right",
            )
            if pos.get("stop_loss"):
                fig.add_hline(
                    y=pos["stop_loss"], line_color="#f85149",
                    line_width=0.8, line_dash="dot", opacity=0.4, row=1, col=1,
                )
            if pos.get("take_profit"):
                fig.add_hline(
                    y=pos["take_profit"], line_color="#3fb950",
                    line_width=0.8, line_dash="dot", opacity=0.4, row=1, col=1,
                )

    # Volume
    vc = ["#163a22" if c>=o else "#3a1318" for c,o in zip(df["close"],df["open"])]
    vl = ["#3fb950" if c>=o else "#f85149" for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"],
                         marker_color=vc, marker_line_color=vl,
                         marker_line_width=0.4, showlegend=False), row=2, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["rsi"],
                                 line=dict(color="#d29922", width=1.2),
                                 showlegend=False), row=3, col=1)
        for lvl, col in [(70,"#f85149"),(30,"#3fb950")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=col,
                          line_width=0.7, opacity=0.5, row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="#f85149", opacity=0.04, line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="#3fb950", opacity=0.04, line_width=0, row=3, col=1)

    _grid = dict(gridcolor="#161b22", zerolinecolor="#1e2a3a", showgrid=True)
    fig.update_layout(
        height=590,
        template="plotly_dark",
        paper_bgcolor="#080b12",
        plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.008,
                    font=dict(size=10, color="#768390"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=95, t=6, b=8),
        font=dict(size=9, color="#5a6a7a"),
        xaxis=_grid, yaxis=_grid,
        xaxis2=_grid, yaxis2=_grid,
        xaxis3=_grid, yaxis3=_grid,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 9
    return fig


# ================================================================== #
# TAB: AUTOPILOT                                                       #
# ================================================================== #
with tab_autopilot:

    # ── Header ───────────────────────────────────────────────────── #
    st.markdown(
        '<h2 style="margin-bottom:4px;">🤖 Autopilot</h2>'
        '<div style="font-size:0.82rem;color:#768390;margin-bottom:12px;">'
        'Watches ES=F (S&amp;P 500 E-mini futures) and alerts you — or trades '
        'automatically — when a validated strategy fires a setup.</div>',
        unsafe_allow_html=True,
    )

    # ── Mode selector ─────────────────────────────────────────────── #
    ap_mode = st.radio(
        "Trading mode",
        ["1h Mode  (EMA55/EMA21 — 2-4 trades/month, 4-8h holds)",
         "5m Mode  (ORB + VWAP + EMA Stack — 2-3 trades/day, same-session holds)",
         "A+ Mode  (IB Breakout + VWAP-85 Retest — 1-2 precision setups/day)"],
        key="ap_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    _ap_is_5m    = ap_mode.startswith("5m")
    _ap_is_aplus = ap_mode.startswith("A+")

    # ── Controls ─────────────────────────────────────────────────── #
    ap_col1, ap_col2, ap_col3 = st.columns([1.8, 2.2, 3])
    with ap_col1:
        ap_armed = st.toggle(
            "⚡ Autopilot Armed",
            value=st.session_state["ap_armed"],
            key="ap_armed_toggle",
            help="When armed, the system auto-checks for signals every 5 minutes",
        )
        st.session_state["ap_armed"] = ap_armed
    with ap_col2:
        ap_auto_trade = st.toggle(
            "🤖 Auto-Execute Paper Trades",
            value=st.session_state["ap_auto_trade"],
            key="ap_auto_trade_toggle",
            help="Automatically places paper trades when a signal fires — no manual click needed",
        )
        st.session_state["ap_auto_trade"] = ap_auto_trade
    with ap_col3:
        if ap_armed:
            st.markdown(
                '<div style="padding:8px 0;">'
                '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                'background:#3fb950;margin-right:6px;animation:pulse 1.5s infinite;"></span>'
                '<style>@keyframes pulse{0%{opacity:1}50%{opacity:0.35}100%{opacity:1}}</style>'
                '<span style="font-size:0.75rem;color:#3fb950;font-weight:700;">'
                'ARMED — checks every 5 min during market hours</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding:8px 0;font-size:0.75rem;color:#484f58;">'
                '⬤ &nbsp;Disarmed — toggle to activate auto-monitoring</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Auto-refresh when armed ───────────────────────────────────── #
    if ap_armed:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=5 * 60 * 1000, key="ap_refresh")  # every 5 min
        except ImportError:
            pass

    # ── Check Now button ─────────────────────────────────────────── #
    ap_btn_c1, ap_btn_c2, ap_btn_c3 = st.columns([1.5, 1.5, 4])
    with ap_btn_c1:
        ap_manual_check = st.button("🔍 Check Now", type="primary",
                                    use_container_width=True, key="ap_check_btn")
    with ap_btn_c2:
        ap_clear_btn = st.button("🗑 Clear Log", use_container_width=True, key="ap_clear_btn")
        if ap_clear_btn:
            st.session_state["ap_activity"]  = []
            st.session_state["ap_executed"]  = set()
            st.session_state["ap_last_signal"] = None
            st.rerun()

    # ── Signal check logic ────────────────────────────────────────── #
    _ap_now = time.time()
    _ap_should_check = (
        ap_armed and (
            st.session_state["ap_last_check"] == 0.0 or
            (_ap_now - st.session_state["ap_last_check"]) >= 300
        )
    )

    if ap_manual_check or _ap_should_check:
        _ap_tf_label = "5m" if (_ap_is_5m or _ap_is_aplus) else "1h"
        with st.spinner(f"Fetching ES=F {_ap_tf_label} data and checking for setups…"):
            try:
                _df_ap   = fetch_historical("ES=F", _ap_tf_label, days_back=58 if (_ap_is_5m or _ap_is_aplus) else 365)
                _snap_ap = get_live_price("ES")
                _cur_px  = _snap_ap.get("last_price") if _snap_ap else None

                if _ap_is_aplus:
                    _engine_ap = BacktestEngineAPlus(
                        min_adx=18.0, min_score=0.65,
                        allow_short=False, require_macro_confirm=False,
                    )
                elif _ap_is_5m:
                    _engine_ap = BacktestEngine5m(
                        warmup=50, max_bars=24, rr_ratio=2.0, atr_stop=1.5,
                        min_score=0.50, allow_short=True, min_adx=18.0, rth_only=True,
                    )
                else:
                    _engine_ap = BacktestEngine(
                        warmup=200, max_bars=24, rr_ratio=2.0, atr_stop=1.0,
                        min_score=0.55, allow_short=True, min_adx=25.0, rth_only=True,
                    )
                _sig = _engine_ap.live_signal(_df_ap, current_price=_cur_px)

                st.session_state["ap_last_signal"] = _sig
                st.session_state["ap_last_check"]  = _ap_now

                _chk_time = datetime.now().strftime("%I:%M %p")

                if _sig:
                    _sig_hash = f"{_sig['direction']}_{_sig['strategy']}_{_sig['bar_time']}"
                    # Log the signal
                    st.session_state["ap_activity"].insert(0, {
                        "time": _chk_time, "type": "signal",
                        "msg":  f"{'🟢' if _sig['direction']=='BUY' else '🔴'} "
                                f"{_sig['direction']} signal — {_sig['strategy']} "
                                f"(score {_sig['score']:.2f})",
                    })
                    # Auto-execute if enabled and not already traded
                    if ap_auto_trade and _sig_hash not in st.session_state["ap_executed"]:
                        _ok, _msg, _pos = broker.open_position(
                            symbol        = "ES",
                            direction     = _sig["direction"],
                            entry_price   = _sig["entry"],
                            stop_loss     = _sig["stop"],
                            take_profit   = _sig["target"],
                            strategy_used = _sig["strategy"],
                            ai_reasoning  = (
                                f"Autopilot: {_sig['strategy_label']} | "
                                f"regime={_sig['regime']} | score={_sig['score']} | "
                                f"adx={_sig['adx']}"
                            ),
                        )
                        st.session_state["ap_executed"].add(_sig_hash)
                        st.session_state["ap_activity"].insert(0, {
                            "time": _chk_time,
                            "type": "trade" if _ok else "error",
                            "msg":  f"{'✅ Auto-traded' if _ok else '❌ Trade failed'}: {_msg[:100]}",
                        })
                        if _ok:
                            st.toast(
                                f"{'🟢' if _sig['direction']=='BUY' else '🔴'} "
                                f"Paper trade executed — {_sig['direction']} ES @ {_sig['entry']:.2f}",
                                icon="🤖",
                            )
                            st.components.v1.html(ALERT_JS, height=0)
                else:
                    _strat_names = ("IB Breakout + VWAP-85" if _ap_is_aplus
                                    else "ORB / VWAP / EMA Stack" if _ap_is_5m
                                    else "EMA55 / EMA21 / VWAP")
                    st.session_state["ap_activity"].insert(0, {
                        "time": _chk_time, "type": "check",
                        "msg":  f"⚪ No setup — all strategies NEUTRAL ({_strat_names})",
                    })

            except Exception as _ap_err:
                st.warning(f"Signal check failed: {_ap_err}")

    # ── Display current signal ─────────────────────────────────────── #
    _sig     = st.session_state["ap_last_signal"]
    _last_chk = st.session_state["ap_last_check"]
    _chk_str = datetime.fromtimestamp(_last_chk).strftime("%I:%M %p") if _last_chk > 0 else "—"

    if _sig:
        _dc   = "3fb950" if _sig["direction"] == "BUY" else "f85149"
        _icon = "🟢" if _sig["direction"] == "BUY" else "🔴"
        _dw   = _sig["direction"]
        _act  = "BUY at market" if _dw == "BUY" else "SELL SHORT at market"
        _rpx  = _sig["risk_pts"]  * 50   # ES = $50/point
        _tpx  = _rpx * _sig["rr"]

        st.markdown(f"""
<div style="background:{'#0a1f0f' if _dw=='BUY' else '#1f0a0a'};
            border:2px solid #{_dc};border-radius:12px;
            padding:24px 28px;margin-bottom:20px;">
  <div style="font-size:2.4rem;font-weight:900;letter-spacing:3px;
              color:#{_dc};line-height:1.1;margin-bottom:6px;">
    {_icon} {_dw} SIGNAL
  </div>
  <div style="font-size:1.05rem;font-weight:600;color:#e2e8f0;margin-bottom:4px;">
    ES=F &nbsp;·&nbsp; S&P 500 E-mini Futures
  </div>
  <div style="font-size:0.78rem;color:#768390;line-height:1.6;">
    Strategy: <strong style="color:#c9d1d9;">{_sig['strategy_label']}</strong><br>
    Market regime: <strong style="color:#c9d1d9;">{_sig['regime'].upper()}</strong>
    &nbsp;·&nbsp; ADX: {_sig['adx']:.1f} &nbsp;·&nbsp; Score: {_sig['score']:.2f}<br>
    Signal bar closed at: {_sig['bar_time']} &nbsp;·&nbsp; Checked: {_chk_str}
  </div>
</div>
""", unsafe_allow_html=True)

        _section_label("📋 Exact Steps — What To Do Right Now")
        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;
            padding:22px 26px;margin-bottom:16px;">

  <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:14px;">
    <div style="background:{'#0d2230' if _dw=='BUY' else '#2d0f0f'};border:1px solid #{_dc};
                border-radius:50%;width:32px;height:32px;min-width:32px;
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:1rem;color:#{_dc};">1</div>
    <div>
      <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:2px;">
        {_act} when the next 1h bar opens
      </div>
      <div style="font-family:'SF Mono',monospace;font-size:1.2rem;font-weight:800;color:#{_dc};">
        ~{_sig['entry']:,.2f}
      </div>
      <div style="font-size:0.72rem;color:#5a6a7a;margin-top:2px;">
        Approximate — use the current market price; you have ~30 min from bar open
      </div>
    </div>
  </div>

  <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:14px;">
    <div style="background:#2d0f0f;border:1px solid #f85149;
                border-radius:50%;width:32px;height:32px;min-width:32px;
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:1rem;color:#f85149;">2</div>
    <div>
      <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:2px;">
        Set your Stop Loss at
      </div>
      <div style="font-family:'SF Mono',monospace;font-size:1.2rem;font-weight:800;color:#f85149;">
        {_sig['stop']:,.2f}
      </div>
      <div style="font-size:0.72rem;color:#5a6a7a;margin-top:2px;">
        Max loss if trade goes wrong: {_sig['risk_pts']:.1f} pts ≈ <strong style="color:#f85149;">${_rpx:,.0f}</strong> per contract
      </div>
    </div>
  </div>

  <div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:14px;">
    <div style="background:#0a1f0f;border:1px solid #3fb950;
                border-radius:50%;width:32px;height:32px;min-width:32px;
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:1rem;color:#3fb950;">3</div>
    <div>
      <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:2px;">
        Set your Profit Target at
      </div>
      <div style="font-family:'SF Mono',monospace;font-size:1.2rem;font-weight:800;color:#3fb950;">
        {_sig['target']:,.2f}
      </div>
      <div style="font-size:0.72rem;color:#5a6a7a;margin-top:2px;">
        Expected gain: {_sig['risk_pts']*_sig['rr']:.1f} pts ≈ <strong style="color:#3fb950;">${_tpx:,.0f}</strong> per contract
      </div>
    </div>
  </div>

  <div style="display:flex;align-items:flex-start;gap:14px;">
    <div style="background:#1a1600;border:1px solid #d29922;
                border-radius:50%;width:32px;height:32px;min-width:32px;
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:1rem;color:#d29922;">4</div>
    <div>
      <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:2px;">
        Walk away — let the trade work
      </div>
      <div style="font-size:0.72rem;color:#5a6a7a;margin-top:2px;">
        Risk/Reward: <strong style="color:#d29922;">{_sig['rr']:.1f}:1</strong> &nbsp;·&nbsp;
        Stop and target are already set — no need to watch the screen
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

        # Manual execute button (shown when auto-trade is OFF)
        if not ap_auto_trade:
            _sig_hash = f"{_sig['direction']}_{_sig['strategy']}_{_sig['bar_time']}"
            _already  = _sig_hash in st.session_state["ap_executed"]
            if _already:
                st.info("✅ This signal has already been paper traded.")
            else:
                if st.button(
                    f"{_icon} Execute Paper Trade — {_dw} ES=F @ {_sig['entry']:,.2f}",
                    type="primary", use_container_width=True, key="ap_exec_btn",
                ):
                    _ok, _msg, _ = broker.open_position(
                        symbol="ES", direction=_sig["direction"],
                        entry_price=_sig["entry"], stop_loss=_sig["stop"],
                        take_profit=_sig["target"], strategy_used=_sig["strategy"],
                        ai_reasoning=f"Autopilot manual: {_sig['strategy_label']}",
                    )
                    if _ok:
                        st.session_state["ap_executed"].add(_sig_hash)
                        st.session_state["ap_activity"].insert(0, {
                            "time": datetime.now().strftime("%I:%M %p"),
                            "type": "trade",
                            "msg":  f"✅ Manual trade: {_msg[:100]}",
                        })
                        st.success(f"✅ Paper trade executed: {_msg}")
                        st.components.v1.html(ALERT_JS, height=0)
                    else:
                        st.error(f"❌ {_msg}")

    else:
        # NO SIGNAL state
        _regime_hint = ""
        if _last_chk > 0:
            try:
                _df_hint = fetch_historical("ES=F", "1h", days_back=30)
                _df_hint = add_all(_df_hint.copy())
                if "adx" in _df_hint.columns:
                    _adx_now = _df_hint["adx"].iloc[-1]
                    _ema55   = _df_hint["ema_55"].iloc[-1] if "ema_55" in _df_hint.columns else 0
                    _ema200  = _df_hint["ema_200"].iloc[-1] if "ema_200" in _df_hint.columns else 0
                    _trend_ok = _ema55 > _ema200 * 1.001 if _ema55 and _ema200 else False
                    _regime_hint = (
                        f"ADX: {_adx_now:.1f} &nbsp;·&nbsp; "
                        f"Trend: {'✅ Intact (EMA55 > EMA200)' if _trend_ok else '⚠ Not confirmed'}"
                    )
            except Exception:
                pass

        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #1e2a3a;border-radius:12px;
            padding:48px 28px;text-align:center;margin-bottom:20px;">
  <div style="font-size:3.5rem;margin-bottom:12px;opacity:0.35;">⚪</div>
  <div style="font-size:1.5rem;font-weight:800;color:#484f58;letter-spacing:2px;
              margin-bottom:10px;">NO SETUP RIGHT NOW</div>
  <div style="font-size:0.8rem;color:#5a6a7a;margin-bottom:8px;">
    {"Last checked: " + _chk_str if _last_chk > 0 else "Click <strong>Check Now</strong> or enable Autopilot"}
    {"&nbsp;·&nbsp;" + _regime_hint if _regime_hint else ""}
  </div>
  <div style="font-size:0.75rem;color:#30363d;max-width:440px;
              margin:12px auto 0 auto;line-height:1.7;">
    {"In A+ mode, expect 1–2 precision setups per day. Entry window: 10:15–11:30 AM and 2:00–3:55 PM ET. Requires IB breakout + VWAP-85 retest + candle pattern — all three must align." if _ap_is_aplus else "In 5m mode, setups appear 2–3× per day during active market hours. If nothing is showing, the market may be ranging or you may be outside the 9:30am–noon window for ORB." if _ap_is_5m else "Completely normal. The system only fires when a high-quality setup appears — roughly 2× per month. Quality over quantity."}</div>
</div>
""", unsafe_allow_html=True)

        if _last_chk == 0.0:
            st.markdown("""
<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
            padding:16px 20px;text-align:center;font-size:0.78rem;color:#768390;">
  👆 Click <strong style="color:#388bfd;">Check Now</strong> above to run the first signal check.
  Or enable <strong style="color:#3fb950;">Autopilot Armed</strong> to auto-check every 5 minutes.
</div>
""", unsafe_allow_html=True)

    # ── Activity log ─────────────────────────────────────────────── #
    if st.session_state["ap_activity"]:
        st.markdown("---")
        _section_label("Activity Log")
        for _entry in st.session_state["ap_activity"][:15]:
            _lc = {"signal": "388bfd", "trade": "3fb950", "error": "f85149", "check": "484f58"}.get(
                _entry["type"], "484f58"
            )
            st.markdown(
                f'<div style="font-size:0.78rem;padding:5px 10px;'
                f'border-left:2px solid #{_lc};margin:3px 0;color:#c9d1d9;">'
                f'<span style="color:#5a6a7a;font-family:monospace;">{_entry["time"]}</span>'
                f'&nbsp;&nbsp;{_entry["msg"]}</div>',
                unsafe_allow_html=True,
            )

    # ── How it works (collapsible) ────────────────────────────────── #
    st.markdown("---")
    with st.expander("📖 How it works — plain English guide", expanded=False):
        st.markdown("""
**What is this watching?**

ES=F — the S&P 500 E-mini futures contract. One of the most liquid markets in the world,
trading weekdays 9am–4pm ET.

---

**1h Mode — swing intraday (EMA55/EMA21 Bounce)**

| Strategy | What it means | Win Rate | Profit Factor |
|---|---|---|---|
| EMA55 Bounce | S&P pulls back to 55-period MA in uptrend, then bounces | ~47% | 1.74 |
| EMA21 Bounce | Shallower pullback to the faster 21-period MA | ~48% | 1.61 |
| PDHL Breakout | Price breaks above/below prior day's high/low after compression | ~50% | 1.58 |

✅ **~2 trades/month** · Holds 4-8 hours · 2:1 R:R · Stop = 1 ATR (~15-25 pts ≈ $750-$1,250)

---

**5m Mode — true day trading (ORB + VWAP + EMA Stack)**

| Strategy | What it means | Expected WR |
|---|---|---|
| ORB Breakout | Price breaks out of the tight opening range (9:30-10am) between 10am-noon | ~52-58% |
| VWAP Pullback | Price trending above VWAP, dips back to it, bounces back up | ~55-62% |
| EMA21 Stack | EMA8 > EMA21 > EMA55 stacked; price dips to EMA21 and bounces | ~50-56% |

✅ **2-3 trades/day** · Same-session holds (max 2h, closes by 4pm) · 1.5-2:1 R:R

---

**The ES contract:** $50 per point. A 15-point stop = $750 risk. A 30-point target = $1,500 gain.

**Market hours:** 9:00 AM – 4:00 PM ET, Monday–Friday only.

**No overnight holds** — all trades close same day by 4pm ET.

---

**Paper trading first:** All trades here are PAPER (simulated). No real money until
you've seen consistent results over several months. When ready to go live, connect
your IBKR account in the Settings (coming soon).
        """)


# ================================================================== #
# TAB: ANALYSIS                                                        #
# ================================================================== #
with tab_analysis:

    # ── Controls ─────────────────────────────────────────────────── #
    c1, c2, c3, c4, c5, c6 = st.columns([2.2, 1.1, 1.1, 0.8, 1.6, 1.4])
    with c1: symbol    = st.text_input("Symbol",    value=st.session_state.current_symbol)
    with c2: timeframe = st.selectbox("Timeframe",  ["1m","5m","15m","30m","1h","1d"],
                                      index=["1m","5m","15m","30m","1h","1d"].index(
                                          st.session_state.current_timeframe))
    with c3: market    = st.selectbox("Market",     ["futures","equities"],
                                      index=["futures","equities"].index(
                                          st.session_state.current_market))
    with c4: days_back = st.number_input("Days",    min_value=5, max_value=90, value=20)
    with c5: run_btn   = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    with c6: live_mode = st.toggle("⚡ Live (60s)", value=False)

    # ── Live mode: dual-tier refresh ─────────────────────────────── #
    # Tier 1 — price ticker: every 10s, lightweight fast_info call
    # Tier 2 — strategy refresh: every 60s, re-fetches bars + strategies (no Claude)
    # Claude only runs on manual "Run Analysis" click
    if live_mode:
        try:
            from streamlit_autorefresh import st_autorefresh

            price_col, strat_col = st.columns([1, 2])
            with price_col:
                price_secs = st.select_slider(
                    "Price refresh",
                    options=[5, 10, 15, 30],
                    value=10,
                    format_func=lambda x: f"{x}s",
                    key="price_rf",
                )
            with strat_col:
                strat_secs = st.select_slider(
                    "Strategy refresh",
                    options=[30, 60, 120, 300],
                    value=60,
                    format_func=lambda x: f"{x}s",
                    key="strat_rf",
                )
            # Fast price refresh (drives lightweight price banner)
            st_autorefresh(interval=price_secs * 1000, key="price_refresh")

        except ImportError:
            st.caption("Install streamlit-autorefresh for live mode")
            strat_secs = 60

    # ── Run — full analysis (manual button or first-load in live mode) ── #
    now = time.time()
    auto_strat_refresh = (
        live_mode
        and st.session_state.df is not None
        and (now - st.session_state.last_analysis_time) >= (st.session_state.get("strat_rf", 60))
    )

    if run_btn or (live_mode and st.session_state.df is None) or auto_strat_refresh:
        sym = symbol.strip().upper()
        st.session_state.current_symbol    = sym
        st.session_state.current_timeframe = timeframe
        st.session_state.current_market    = market
        is_manual = bool(run_btn)
        if is_manual:
            st.session_state.last_decision = None  # clear Claude on manual re-run only

        prog = st.progress(0, text="Fetching latest bars…")
        try:
            df_raw = fetch_historical(sym, timeframe, days_back)
            df     = add_all(df_raw.copy())
            st.session_state.df = df
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            prog.empty()
            st.stop()

        prog.progress(40, text="Running 10 strategies…")
        agg = SignalAggregator(market=market).run(df_raw, symbol=sym, timeframe=timeframe)
        st.session_state.agg_signal        = agg
        st.session_state.last_analysis_time = time.time()

        akey = f"{sym}_{timeframe}"
        if abs(agg.composite_score) >= 0.5 and akey not in st.session_state.alerted_symbols:
            st.session_state.alerted_symbols.add(akey)
            st.toast(f"{'🟢' if agg.direction=='BUY' else '🔴'} {agg.recommendation} on {sym}!", icon="🔔")
            st.components.v1.html(ALERT_JS, height=0)

        # Claude only on manual press — skip on autorefresh to keep it fast
        if is_manual:
            prog.progress(66, text="Claude is analyzing the setup…")
            try:
                agent    = st.session_state.agent
                decision = agent.analyze(
                    symbol=sym, timeframe=timeframe,
                    aggregated_signal={
                        "direction":              agg.direction,
                        "composite_score":        agg.composite_score,
                        "confidence":             agg.confidence,
                        "recommendation":         agg.recommendation,
                        "agreeing_strategies":    agg.agreeing_strategies,
                        "disagreeing_strategies": agg.disagreeing_strategies,
                        "neutral_strategies":     agg.neutral_strategies,
                        "individual_signals":     [
                            {"strategy": s.strategy, "direction": s.direction,
                             "confidence": s.confidence, "reasoning": s.reasoning}
                            for s in agg.individual_signals],
                        "suggested_entry":  agg.suggested_entry,
                        "suggested_stop":   agg.suggested_stop,
                        "suggested_target": agg.suggested_target,
                    },
                    portfolio_summary=portfolio,
                )
                st.session_state.last_decision = decision
                st.session_state.agent         = agent
            except Exception as e:
                st.warning(f"Claude unavailable ({e}) — strategy signals still shown.")

        prog.progress(100, text="Done")
        time.sleep(0.2)
        prog.empty()

    # ── Fast price poll (runs on every page refresh, very cheap) ── #
    if live_mode and st.session_state.current_symbol:
        try:
            snap = get_live_price(st.session_state.current_symbol)
            st.session_state.live_price = snap

            # Auto-check stops and targets for open positions
            if snap and snap.get("last_price"):
                close_msgs = broker.check_stops_and_targets(
                    st.session_state.current_symbol, snap["last_price"]
                )
                for msg in close_msgs:
                    icon = "🎯" if "TARGET" in msg else "🛑"
                    st.toast(msg, icon=icon)
                if close_msgs:
                    st.rerun()
        except Exception:
            pass

    # ── Output ───────────────────────────────────────────────────── #
    if st.session_state.agg_signal is not None:
        agg      = st.session_state.agg_signal
        decision = st.session_state.last_decision
        df       = st.session_state.df

        chart_col, sig_col = st.columns([3, 1.2], gap="small")

        with sig_col:
            entry_p, stop_p, target_p = render_signal_panel(agg, decision)

            # Strategy breakdown — always expanded when nothing is firing
            all_neutral = all(s.direction == "NEUTRAL" for s in agg.individual_signals)
            with st.expander("All Strategy Signals", expanded=all_neutral):
                rows = []
                for s in sorted(agg.individual_signals,
                                key=lambda x: (x.direction=="NEUTRAL", x.direction)):
                    em = "🟢" if s.direction=="BUY" else "🔴" if s.direction=="SELL" else "⚪"
                    rows.append({
                        "Strategy":   s.strategy,
                        "Signal":     f"{em} {s.direction}",
                        "Conf":       f"{s.confidence:.0%}",
                        "Reason":     s.reasoning[:80]+"…" if len(s.reasoning)>80 else s.reasoning,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if all_neutral:
                    st.markdown(
                        '<div style="font-size:0.72rem;color:#5a6a7a;margin-top:8px;'
                        'padding:8px 10px;background:#161b22;border-radius:4px;">'
                        '💡 <strong style="color:#768390;">No setup right now</strong> — '
                        'each strategy listed its exact reason above. '
                        'Try a different timeframe (1h often shows cleaner structure) '
                        'or wait for price to reach a key level.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

        with chart_col:
            if df is not None:
                sym_d  = st.session_state.current_symbol
                tf_d   = st.session_state.current_timeframe
                snap   = st.session_state.live_price  # fast_info snapshot (live mode)
                bar_last = df["close"].iloc[-1]
                bar_prev = df["close"].iloc[-2]
                bar_chg  = bar_last - bar_prev
                bar_chgp = bar_chg / bar_prev * 100

                # Live price takes priority over bar close when available
                display_price = snap["last_price"] if snap and snap.get("last_price") else bar_last
                disp_chg  = display_price - bar_prev
                disp_chgp = disp_chg / bar_prev * 100
                cc = "#3fb950" if disp_chg >= 0 else "#f85149"
                cs = "+" if disp_chg >= 0 else ""

                # ── Price ticker bar ─────────────────────────────── #
                bid_ask = ""
                if snap and snap.get("bid") and snap.get("ask"):
                    bid_ask = (f'<span style="font-size:0.72rem;color:#5a6a7a;">'
                               f'B {snap["bid"]:.2f} &nbsp;/&nbsp; A {snap["ask"]:.2f}</span>')

                day_range = ""
                if snap and snap.get("day_low") and snap.get("day_high"):
                    day_range = (f'<span style="font-size:0.7rem;color:#5a6a7a;">'
                                 f'Range &nbsp;{snap["day_low"]:.2f} – {snap["day_high"]:.2f}</span>')

                live_dot = ""
                if live_mode:
                    live_dot = ('<span style="display:inline-block;width:7px;height:7px;'
                                'border-radius:50%;background:#3fb950;margin-right:5px;'
                                'animation:pulse 1.5s infinite;"></span>'
                                '<style>@keyframes pulse{'
                                '0%{opacity:1}50%{opacity:0.3}100%{opacity:1}}</style>'
                                '<span style="font-size:0.65rem;color:#3fb950;'
                                'font-weight:700;letter-spacing:0.5px;">LIVE</span>')
                else:
                    live_dot = (f'<span style="font-size:0.65rem;color:#484f58;">'
                                f'chart as of {time.strftime("%H:%M:%S")}</span>')

                data_note = ('<span style="font-size:0.6rem;color:#30363d;margin-left:auto;" '
                             'title="yfinance intraday data is ~1-2 min delayed. '
                             'Connect IBKR for real-time.">'
                             '⚠ ~1-2 min delayed · IBKR = real-time</span>')

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;'
                    f'margin-bottom:8px;padding:8px 10px;background:#0d1117;'
                    f'border:1px solid #1e2a3a;border-radius:6px;">'
                    f'<span style="font-size:1.25rem;font-weight:800;color:#e2e8f0;">{sym_d}</span>'
                    f'<span style="font-size:0.68rem;color:#5a6a7a;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:1px;padding:2px 6px;'
                    f'background:#161b22;border-radius:3px;">{tf_d}</span>'
                    f'<span style="font-family:\'SF Mono\',monospace;font-size:1.3rem;'
                    f'font-weight:700;color:#e2e8f0;">{display_price:,.2f}</span>'
                    f'<span style="font-family:\'SF Mono\',monospace;font-size:0.88rem;'
                    f'color:{cc};font-weight:600;">{cs}{disp_chg:.2f} ({cs}{disp_chgp:.2f}%)</span>'
                    f'{bid_ask}'
                    f'{day_range}'
                    f'<span style="margin-left:auto;display:flex;align-items:center;gap:10px;">'
                    f'{live_dot}&nbsp;{data_note}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Fetch open positions to show on chart + in panel
                try:
                    open_pos_df = get_open_positions()
                    open_pos_list = open_pos_df[
                        open_pos_df["symbol"] == st.session_state.current_symbol
                    ].to_dict("records") if not open_pos_df.empty else []
                except Exception:
                    open_pos_list = []

                fig = build_chart(df, entry_p, stop_p, target_p,
                                  open_positions=open_pos_list)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

                # ── Open Positions Panel ──────────────────────────── #
                all_open = broker.open_positions
                if all_open:
                    _section_label(f"Open Positions  ({len(all_open)})")
                    for pos in all_open:
                        upnl       = pos.pnl_at_price(display_price)
                        upnl_sign  = "+" if upnl >= 0 else ""
                        upnl_col   = "#3fb950" if upnl >= 0 else "#f85149"
                        dir_badge  = ("🟢 LONG" if pos.direction == "LONG"
                                      else "🔴 SHORT")
                        elapsed    = datetime.now(timezone.utc) - pos.opened_at
                        mins       = int(elapsed.total_seconds() / 60)
                        time_str   = (f"{mins}m ago" if mins < 60
                                      else f"{mins//60}h {mins%60}m ago")

                        pc1, pc2, pc3, pc4, pc5, pc6, pc7, pc8 = st.columns(
                            [1.1, 1.1, 0.7, 1.1, 1.2, 1.4, 1.8, 0.9]
                        )
                        pc1.markdown(
                            f'<span style="font-weight:700;color:#e2e8f0;'
                            f'font-family:monospace;">{pos.symbol}</span>',
                            unsafe_allow_html=True,
                        )
                        pc2.markdown(dir_badge)
                        pc3.markdown(
                            f'<span style="font-family:monospace;color:#768390;">'
                            f'{pos.qty}</span>',
                            unsafe_allow_html=True,
                        )
                        pc4.markdown(
                            f'<span style="font-family:monospace;color:#768390;">'
                            f'@ {pos.entry_price:.2f}</span>',
                            unsafe_allow_html=True,
                        )
                        pc5.markdown(
                            f'<span style="font-family:monospace;font-weight:700;'
                            f'color:{upnl_col};">{upnl_sign}${upnl:,.2f}</span>',
                            unsafe_allow_html=True,
                        )
                        pc6.markdown(
                            f'<span style="font-size:0.75rem;color:#5a6a7a;">'
                            f'SL {pos.stop_loss:.2f} / TP {pos.take_profit:.2f}'
                            f'</span>',
                            unsafe_allow_html=True,
                        )
                        pc7.markdown(
                            f'<span style="font-size:0.72rem;color:#484f58;">'
                            f'{pos.strategy_used[:30] if pos.strategy_used else "—"}'
                            f'  ·  {time_str}</span>',
                            unsafe_allow_html=True,
                        )
                        if pc8.button("Close", key=f"close_{pos.id}",
                                      use_container_width=True):
                            ok, msg, _ = broker.close_position(
                                pos.id, display_price, reason="manual"
                            )
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.rerun()

    else:
        # Empty state
        st.markdown("""
<div style="text-align:center;padding:100px 0;">
  <div style="font-size:4rem;margin-bottom:20px;opacity:0.3;">📈</div>
  <div style="font-size:1rem;font-weight:600;color:#484f58;letter-spacing:0.5px;">
    Enter a symbol above and click <span style="color:#388bfd;">▶ Run Analysis</span>
  </div>
  <div style="font-size:0.78rem;color:#30363d;margin-top:8px;">
    Runs 10 strategies + Claude AI simultaneously
  </div>
</div>""", unsafe_allow_html=True)

    # ── Chat ─────────────────────────────────────────────────────── #
    st.markdown("---")
    _section_label("💬 Ask the AI")
    chat_in = st.chat_input("Ask about this setup, risk, when to exit…")
    if chat_in:
        agent = st.session_state.agent
        if not agent.conversation_history:
            st.warning("Run an analysis first so Claude has context.")
        else:
            with st.spinner("Thinking…"):
                resp = agent.ask(chat_in)
            st.session_state.chat_history.append(("user",      chat_in))
            st.session_state.chat_history.append(("assistant", resp))
            st.session_state.agent = agent

    for role, msg in st.session_state.chat_history[-8:]:
        with st.chat_message(role):
            st.markdown(msg)


# ================================================================== #
# TAB: SCANNER                                                         #
# ================================================================== #
with tab_scanner:
    _section_label("Watchlist Scanner")

    s1, s2, s3, s4, s5 = st.columns([3.5, 0.9, 0.9, 1.1, 1.1])
    with s1:
        watchlist = st.text_input("Symbols", value="ES, NQ, CL, GC, RTY, YM",
                                  placeholder="ES, NQ, CL…", label_visibility="collapsed")
    with s2:
        scan_tf = st.selectbox("TF", ["1m","5m","15m","30m","1h"], index=1,
                               key="scan_tf", label_visibility="collapsed")
    with s3:
        scan_mkt = st.selectbox("Market", ["futures","equities"],
                                key="scan_market", label_visibility="collapsed")
    with s4:
        scan_btn = st.button("🔍 Scan Now", type="primary", use_container_width=True)
    with s5:
        live_scan = st.toggle("Auto 60s", value=False, key="live_scan")

    if live_scan:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60_000, key="scanner_refresh")
        except ImportError:
            pass

    def _scan_one(sym, tf, mkt):
        try:
            sym = sym.strip().upper()
            df  = fetch_historical(sym, tf, days_back=10)
            agg = SignalAggregator(market=mkt).run(df, symbol=sym, timeframe=tf)
            return dict(Symbol=sym, Rec=agg.recommendation, Direction=agg.direction,
                        Score=agg.composite_score, Conf=agg.confidence,
                        Agreeing=", ".join(agg.agreeing_strategies) or "—",
                        Entry=agg.suggested_entry, Stop=agg.suggested_stop,
                        Target=agg.suggested_target, _agg=agg)
        except Exception as e:
            return dict(Symbol=sym.strip().upper(), Rec="ERROR", Direction="ERROR",
                        Score=0.0, Conf=0.0, Agreeing=str(e)[:60],
                        Entry=None, Stop=None, Target=None, _agg=None)

    if scan_btn or (live_scan and not st.session_state.scanner_results):
        syms = [s.strip() for s in watchlist.split(",") if s.strip()]
        prog = st.progress(0, text=f"Scanning {len(syms)} symbols…")
        results, done = [], 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            fmap = {pool.submit(_scan_one, s, scan_tf, scan_mkt): s for s in syms}
            for f in as_completed(fmap):
                results.append(f.result())
                done += 1
                prog.progress(int(done/len(syms)*100), text=f"Scanned {done}/{len(syms)}…")
        prog.empty()
        results.sort(key=lambda x: abs(x["Score"]), reverse=True)
        st.session_state.scanner_results = results

        for r in results:
            if abs(r["Score"]) >= 0.5 and r["Symbol"] not in st.session_state.alerted_symbols:
                st.session_state.alerted_symbols.add(r["Symbol"])
                em = "🟢" if r["Direction"]=="BUY" else "🔴"
                st.toast(f"{em} {r['Rec']} on **{r['Symbol']}**", icon="🔔")
                st.components.v1.html(ALERT_JS, height=0)

    if st.session_state.scanner_results:
        res = st.session_state.scanner_results
        table = []
        for r in res:
            if r["Direction"] == "ERROR":
                table.append({"Symbol":r["Symbol"],"Recommendation":"⚠️ ERROR",
                              "Score":"—","Conf":"—","Entry":"—","Stop":"—","Target":"—",
                              "Strategies":r["Agreeing"]})
                continue
            em = "🟢" if r["Direction"]=="BUY" else "🔴" if r["Direction"]=="SELL" else "⚪"
            table.append({
                "Symbol":         r["Symbol"],
                "Recommendation": f"{em} {r['Rec']}",
                "Score":          f"{r['Score']:+.3f}",
                "Conf":           f"{r['Conf']:.0%}",
                "Entry":          f"{r['Entry']:.2f}"  if r["Entry"]  else "—",
                "Stop":           f"{r['Stop']:.2f}"   if r["Stop"]   else "—",
                "Target":         f"{r['Target']:.2f}" if r["Target"] else "—",
                "Strategies":     r["Agreeing"],
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        _section_label("Open in Analysis")
        cols = st.columns(min(len(res), 8))
        for i, r in enumerate(res[:8]):
            if r["Direction"] == "ERROR": continue
            em = "🟢" if r["Direction"]=="BUY" else "🔴" if r["Direction"]=="SELL" else "⚪"
            if cols[i].button(f"{em} {r['Symbol']}", key=f"open_{r['Symbol']}",
                              use_container_width=True):
                st.session_state.current_symbol    = r["Symbol"]
                st.session_state.current_timeframe = scan_tf
                st.session_state.current_market    = scan_mkt
                st.session_state.df                = None
                st.session_state.agg_signal        = None
                st.session_state.last_decision     = None
                st.rerun()
    else:
        st.markdown("""
<div style="text-align:center;padding:70px 0;">
  <div style="font-size:3rem;opacity:0.2;margin-bottom:14px;">🔍</div>
  <div style="font-size:0.9rem;color:#484f58;">Enter symbols above and click <strong style="color:#388bfd;">Scan Now</strong></div>
</div>""", unsafe_allow_html=True)


# ================================================================== #
# TAB: BACKTEST                                                        #
# ================================================================== #
with tab_backtest:
    _section_label("Strategy Backtest")

    # ── Engine mode selector ──────────────────────────────────────── #
    bt_engine_mode = st.radio(
        "Engine",
        ["1h System  (EMA55/EMA21 Bounce — 2-4 trades/month, swing intraday)",
         "5m System  (ORB + VWAP Pullback + EMA Stack — 2-3 trades/day)",
         "A+ Institutional  (IB Breakout + VWAP-85 Retest — 1-2 setups/day)"],
        key="bt_engine_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    _bt_is_5m    = bt_engine_mode.startswith("5m")
    _bt_is_aplus = bt_engine_mode.startswith("A+")

    if _bt_is_5m:
        st.markdown(
            "<div style='font-size:0.78rem;color:#d29922;margin-bottom:12px;'>"
            "⚠️ <b>5m mode</b> — yfinance limits 5m data to ~60 days. "
            "Results are directionally useful but statistically limited (fewer trades). "
            "Use for strategy validation, not final sizing decisions."
            "</div>",
            unsafe_allow_html=True,
        )
    elif _bt_is_aplus:
        st.markdown(
            "<div style='font-size:0.78rem;color:#d29922;margin-bottom:12px;'>"
            "⚠️ <b>A+ Institutional mode</b> — Precision IB-breakout + VWAP-85 retest strategy. "
            "Longs-only by default (shorts enabled via Advanced settings). "
            "Recommended markets: ES, NQ, GC, CL, RTY. Best run on 5m bars, 58-day window."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Controls ─────────────────────────────────────────────────── #
    _default_tf  = "5m" if (_bt_is_5m or _bt_is_aplus) else "1h"
    _default_sym = "ES=F"
    _tf_choices  = ["5m","15m","30m","1h"] if (_bt_is_5m or _bt_is_aplus) else ["1m","5m","15m","30m","1h"]
    _tf_idx      = _tf_choices.index(_default_tf)

    b1, b2, b3, b4, b5, b6 = st.columns([2, 1, 1, 1, 1.2, 1.4])
    with b1: bt_sym  = st.text_input("Symbol", value=_default_sym, key="bt_sym")
    with b2: bt_tf   = st.selectbox("Timeframe", _tf_choices, index=_tf_idx, key="bt_tf")
    with b3: bt_mkt  = st.selectbox("Market", ["futures","equities"], key="bt_mkt")
    _tf_max = {"1m": 7, "5m": 58, "15m": 58, "30m": 58, "1h": 365}.get(bt_tf, 58)
    with b4: bt_days = st.number_input("Days", min_value=10, max_value=_tf_max, value=min(58 if (_bt_is_5m or _bt_is_aplus) else 365, _tf_max),
                                       key="bt_days",
                                       help=f"Max for {bt_tf}: {_tf_max}d (yfinance limit)")
    with b5: bt_rr   = st.select_slider("R:R Target",
                                        options=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0], value=2.0,
                                        key="bt_rr")
    with b6: bt_run  = st.button("⚡ Run Backtest", type="primary",
                                  use_container_width=True, key="bt_run")

    with st.expander("Advanced settings", expanded=False):
        ac1, ac2, ac3, ac4, ac5, ac6, ac7 = st.columns(7)
        with ac1: bt_warmup   = st.number_input("Warmup bars",     30,  300,  200,  key="bt_warmup")
        with ac2: bt_maxbars  = st.number_input("Max hold (bars)", 5,   100,  24,   key="bt_maxbars")
        with ac3: bt_atr_stop = st.slider("ATR stop mult",         0.5, 2.5,  1.0,  step=0.1,
                                           key="bt_atr_stop",
                                           help="Stop = ATR × this multiplier (1.0 = 1 ATR stop)")
        with ac4: bt_min_adx  = st.number_input("Min ADX",         15,  40,   25,   key="bt_min_adx",
                                                 help="ADX threshold for trend confirmation")
        with ac5: bt_min_score= st.slider("Min signal score",      0.40, 0.85, 0.55, step=0.05,
                                           key="bt_min_score",
                                           help="Minimum score to trigger a trade (ema55=0.80, ema21=0.75)")
        with ac6: bt_short    = st.toggle("Allow shorts",  value=True, key="bt_short",
                                           help="All 3 strategies have short-side gates — recommended ON for regime-complete system")
        with ac7: bt_rth_only = st.toggle("RTH only",  value=True, key="bt_rth_only",
                                           help="Only trade during Regular Trading Hours (9am-4pm ET) — critical for 1h futures")

    if bt_run:
        bt_sym_u = bt_sym.strip().upper()
        prog = st.progress(0, text="Fetching historical data…")
        try:
            df_bt = fetch_historical(bt_sym_u, bt_tf, bt_days)
            prog.progress(15, text=f"Running backtest on {len(df_bt)} bars…")
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            prog.empty()
            st.stop()

        if _bt_is_aplus:
            engine_ap = BacktestEngineAPlus(
                max_bars    = st.session_state.get("bt_maxbars",   36),
                min_adx     = st.session_state.get("bt_min_adx",   18),
                min_score   = st.session_state.get("bt_min_score", 0.65),
                allow_short = st.session_state.get("bt_short",     False),
                require_macro_confirm = False,
            )
            result = engine_ap.run(df_bt, symbol=bt_sym_u, timeframe=bt_tf)
            prog.progress(100, text="Done")
        elif _bt_is_5m:
            engine_5m = BacktestEngine5m(
                warmup      = st.session_state.get("bt_warmup",    50),
                max_bars    = st.session_state.get("bt_maxbars",   24),
                rr_ratio    = bt_rr,
                atr_stop    = st.session_state.get("bt_atr_stop",  1.5),
                min_adx     = st.session_state.get("bt_min_adx",   18),
                min_score   = st.session_state.get("bt_min_score", 0.50),
                allow_short = st.session_state.get("bt_short",     True),
                rth_only    = st.session_state.get("bt_rth_only",  True),
            )
            result = engine_5m.run(df_bt, symbol=bt_sym_u, timeframe=bt_tf)
            prog.progress(100, text="Done")
        else:
            engine = BacktestEngine(
                warmup      = st.session_state.get("bt_warmup",    200),
                max_bars    = st.session_state.get("bt_maxbars",   24),
                rr_ratio    = bt_rr,
                atr_stop    = st.session_state.get("bt_atr_stop",  1.0),
                min_adx     = st.session_state.get("bt_min_adx",   20),
                min_score   = st.session_state.get("bt_min_score", 0.55),
                allow_short = st.session_state.get("bt_short",     False),
                be_trail    = False,
                rth_only    = st.session_state.get("bt_rth_only",  True),
            )
            total_steps = [0]
            def _prog_cb(done, total):
                pct = 15 + int(done / max(total, 1) * 75)
                prog.progress(pct, text=f"Simulating bar {done}/{total}…")
            result = engine.run(
                df_bt, symbol=bt_sym_u, timeframe=bt_tf, market=bt_mkt,
                progress_callback=_prog_cb,
            )
            prog.progress(100, text="Done")

        time.sleep(0.15)
        prog.empty()
        st.session_state["bt_result"]       = result
        st.session_state["bt_result_5m"]    = _bt_is_5m
        st.session_state["bt_result_aplus"] = _bt_is_aplus

    # ── Output ───────────────────────────────────────────────────── #
    result = st.session_state.get("bt_result")
    _result_is_5m    = st.session_state.get("bt_result_5m", False)
    _result_is_aplus = st.session_state.get("bt_result_aplus", False)
    _result_no_be    = _result_is_5m or _result_is_aplus  # no BE trail column for these engines

    if result and result.total_trades > 0:
        # ── Summary metric row ─────────────────────────────────── #
        pf_val = result.profit_factor
        pf_str = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"
        exp_str = f"{result.expectancy:+.2f}R"
        dd_str  = f"{result.max_drawdown_pct:.1%}"
        wr_col  = "#3fb950" if result.win_rate >= 0.5 else "#f85149"
        pf_col  = "#3fb950" if pf_val >= 1.5 else ("#d29922" if pf_val >= 1.0 else "#f85149")
        ex_col  = "#3fb950" if result.expectancy > 0 else "#f85149"

        if _result_no_be:
            # 5m / A+ engines — no BE trail column
            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:18px;">
  {_metric_box("Trades",        str(result.total_trades),               "388bfd")}
  {_metric_box("Win Rate",      f"{result.win_rate:.0%}",               wr_col.lstrip('#'))}
  {_metric_box("Profit Factor", pf_str,                                 pf_col.lstrip('#'))}
  {_metric_box("Expectancy",    exp_str,                                ex_col.lstrip('#'))}
  {_metric_box("Max Drawdown",  dd_str,                                 "f85149")}
  {_metric_box("Avg Hold",      f"{result.avg_bars_held:.1f} bars",     "768390")}
  {_metric_box("Total Bars",    str(result.total_bars),                  "30363d")}
</div>""", unsafe_allow_html=True)
        else:
            be_col = "#3fb950" if result.be_trail_rate >= 0.25 else "#768390"
            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-bottom:18px;">
  {_metric_box("Trades",        str(result.total_trades),               "388bfd")}
  {_metric_box("Win Rate",      f"{result.win_rate:.0%}",               wr_col.lstrip('#'))}
  {_metric_box("Profit Factor", pf_str,                                 pf_col.lstrip('#'))}
  {_metric_box("Expectancy",    exp_str,                                ex_col.lstrip('#'))}
  {_metric_box("Max Drawdown",  dd_str,                                 "f85149")}
  {_metric_box("Avg Hold",      f"{result.avg_bars_held:.1f} bars",     "768390")}
  {_metric_box("BE Trail Rate", f"{result.be_trail_rate:.0%}",          be_col.lstrip('#'))}
  {_metric_box("Total Bars",    str(result.total_bars),                  "30363d")}
</div>""", unsafe_allow_html=True)

        # ── Equity curve ───────────────────────────────────────── #
        _section_label("Equity Curve  (price points)")
        eq_color = "#3fb950" if result.equity_curve[-1] >= 0 else "#f85149"
        fill_col = f"rgba({'63,185,80' if result.equity_curve[-1]>=0 else '248,81,73'},0.08)"
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=result.equity_curve,
            fill="tozeroy", fillcolor=fill_col,
            line=dict(color=eq_color, width=2),
            name="P&L (pts)",
        ))
        fig_eq.add_hline(y=0, line_dash="dot", line_color="#21262d", line_width=1)
        fig_eq.update_layout(
            height=220, template="plotly_dark",
            paper_bgcolor="#080b12", plot_bgcolor="#0d1117",
            showlegend=False, margin=dict(l=50, r=20, t=8, b=20),
            font=dict(size=9, color="#5a6a7a"),
            xaxis=dict(gridcolor="#161b22", title="Trade #"),
            yaxis=dict(gridcolor="#161b22", title="Pts"),
        )
        st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

        # ── Per-strategy breakdown ─────────────────────────────── #
        _section_label("By Strategy")
        if _result_no_be:
            # Compute manually from trades list for 5m engine
            _strat_map: dict = {}
            for t in result.trades:
                s = t.strategy
                if s not in _strat_map:
                    _strat_map[s] = {"trades": 0, "wins": 0, "pnl": 0.0, "r": 0.0}
                _strat_map[s]["trades"] += 1
                _strat_map[s]["pnl"]    += t.pnl_pts
                _strat_map[s]["r"]      += t.r_multiple
                if t.pnl_pts > 0:
                    _strat_map[s]["wins"] += 1
            strat_rows = []
            for s, st_stats in sorted(_strat_map.items(), key=lambda x: x[1]["pnl"], reverse=True):
                n = st_stats["trades"]
                wr = st_stats["wins"] / n if n > 0 else 0
                strat_rows.append({
                    "Strategy":   s,
                    "Trades":     n,
                    "Win Rate":   f"{wr:.0%}",
                    "P&L (pts)":  f"{st_stats['pnl']:+.1f}",
                    "Expectancy": f"{st_stats['r']/n:+.2f}R" if n > 0 else "—",
                })
        else:
            by_strat = result.by_strategy()
            strat_rows = []
            for s, stats in sorted(by_strat.items(),
                                   key=lambda x: x[1]["total_pnl_pts"], reverse=True):
                pf = stats["profit_factor"]
                strat_rows.append({
                    "Strategy":      s,
                    "Trades":        stats["trades"],
                    "Win Rate":      f"{stats['win_rate']:.0%}",
                    "Profit Factor": f"{pf:.2f}" if pf != float("inf") else "∞",
                    "Expectancy":    f"{stats['expectancy']:+.2f}R",
                    "P&L (pts)":     f"{stats['total_pnl_pts']:+.1f}",
                })
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

        # ── By regime ─────────────────────────────────────────── #
        if not _result_no_be:
            _section_label("By Regime")
            by_reg = result.by_regime()
            reg_rows = []
            for reg, stats in sorted(by_reg.items(), key=lambda x: x[1]["total_pnl_pts"], reverse=True):
                pf = stats["profit_factor"]
                reg_rows.append({
                    "Regime":        reg,
                    "Trades":        stats["trades"],
                    "Win Rate":      f"{stats['win_rate']:.0%}",
                    "Profit Factor": f"{pf:.2f}" if pf != float("inf") else "∞",
                    "P&L (pts)":     f"{stats['total_pnl_pts']:+.1f}",
                })
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

        # ── Trade log ─────────────────────────────────────────── #
        _section_label(f"Trade Log  ({result.total_trades} trades)")
        log_rows = []
        for t in result.trades:
            em = "🟢" if t.direction == "BUY" else "🔴"
            _er_lc = t.exit_reason.lower() if t.exit_reason else ""
            ex = "🎯" if "target" in _er_lc else ("🛑" if "stop" in _er_lc else ("🌙" if "eod" in _er_lc else "⏱"))
            row = {
                "":         f"{em} {t.direction}",
                "Strategy": t.strategy,
                "Regime":   t.regime,
                "Entry":    f"{t.entry_price:.2f}",
                "Exit":     f"{t.exit_price:.2f}",
                "Reason":   f"{ex} {t.exit_reason}",
                "P&L pts":  f"{t.pnl_pts:+.1f}",
                "R":        f"{t.r_multiple:+.2f}",
                "Bars":     t.bars_held,
            }
            if not _result_no_be:
                row["BE"] = "✓" if t.be_moved else ""
            log_rows.append(row)
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        # ── Claude AI Analysis ─────────────────────────────────── #
        _section_label("🤖 Claude AI Analysis")
        ai_col1, ai_col2 = st.columns([5, 1])
        with ai_col2:
            run_claude_analysis = st.button("Analyse with Claude", type="primary",
                                            use_container_width=True, key="bt_claude_btn")
        with ai_col1:
            st.markdown(
                "<div style='font-size:0.82rem;color:#484f58;padding-top:6px;'>"
                "Claude reviews every philosophy's performance and tells you what's working, "
                "what isn't, and what to adjust.</div>",
                unsafe_allow_html=True,
            )

        if run_claude_analysis:
            if not settings.anthropic_api_key:
                st.error("ANTHROPIC_API_KEY not set in .env — add it to enable Claude analysis.")
            else:
                with st.spinner("Claude is analysing the backtest…"):
                    try:
                        import anthropic as _ant
                        _client = _ant.Anthropic(api_key=settings.anthropic_api_key)
                        _summary = result.summary_for_claude()
                        _analysis_prompt = f"""You are an expert quantitative trading analyst reviewing a backtest of an evidence-based EMA bounce trading system.

The system uses three strategies for futures (primarily ES/NQ 1h RTH session):
1. ema55_bounce — Deep pullback to EMA55 in an uptrend (EMA55 > EMA200). Binary entry: all conditions must be true (ADX > 20 rising, DI+ > DI-, low touched EMA55, close above EMA55 + EMA21). Score: 0.80. Historically: 3yr PF 1.84, WR 47.6%, 42 trades.
2. ema21_bounce — Shallow pullback to EMA21 (only when EMA21/EMA55 spread ≥ 0.3%). Requires EMA21 > EMA55, close > EMA200, same ADX/DI gates. Score: 0.75. Historically: 3yr PF 1.46, WR 50%, 36 trades.
3. vwap_reversion — Rare mean reversion at extreme VWAP deviation (ADX < 18, deviation > 2 ATR, RSI < 28). Rarely fires.

Entry: next bar open. Stop: ATR × multiplier. Target: RR × risk. Max hold: 24 bars.
RTH filter: only enter during 9am-4pm ET (confirmed 50%+ PF improvement vs 24hr trading).

Here are the backtest results:

{_summary}

Please provide:
1. **Overall verdict** — is the system viable? Compare PF to the expected 1.5+ based on historical research.
2. **Strategy breakdown** — which strategies are contributing and which aren't?
3. **Regime analysis** — are trades occurring in the expected trending regime?
4. **Key risk metrics** — max drawdown assessment, trade frequency adequacy
5. **Parameter recommendations** — should the user adjust ADX threshold, RR ratio, or ATR stop?
6. **Market suitability** — is this symbol/timeframe combination appropriate for this strategy?
7. **Action items** — top 2-3 specific things to do to improve results

Be direct, specific, and quantitative. The target is PF ≥ 1.5 consistently."""

                        _resp = _client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1800,
                            messages=[{"role": "user", "content": _analysis_prompt}],
                        )
                        _analysis_text = _resp.content[0].text
                        st.session_state["bt_claude_analysis"] = _analysis_text
                    except Exception as e:
                        st.error(f"Claude analysis failed: {e}")

        _claude_analysis = st.session_state.get("bt_claude_analysis", "")
        if _claude_analysis:
            st.markdown(
                f"""<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                padding:18px 22px;font-size:0.88rem;line-height:1.7;color:#c9d1d9;
                white-space:pre-wrap;">{_claude_analysis}</div>""",
                unsafe_allow_html=True,
            )

    elif result and result.total_trades == 0:
        if _result_is_5m:
            st.warning(
                "No trades triggered. Try a different symbol or increase the days range. "
                "5m data is limited to ~60 days — ensure you have enough bars after warmup."
            )
        else:
            st.warning(
                "No trades triggered. The EMA55/EMA21 bounce strategies are selective by design. "
                "Try: ES=F or NQ=F on 1h timeframe with 365 days, RTH only enabled. "
                "Requires warmup of 200 bars — ensure enough data is fetched."
            )

    else:
        st.markdown("""
<div style="text-align:center;padding:80px 0;">
  <div style="font-size:3.5rem;margin-bottom:18px;opacity:0.3;">⚡</div>
  <div style="font-size:1rem;font-weight:600;color:#484f58;">
    Configure settings above and click <span style="color:#388bfd;">⚡ Run Backtest</span>
  </div>
  <div style="font-size:0.78rem;color:#30363d;margin-top:8px;">
    Vectorized simulation — typically runs in under 1 second
  </div>
</div>""", unsafe_allow_html=True)

    # ── Multi-market comparison (5m engine only) ──────────────── #
    if _bt_is_5m:
        st.divider()
        _section_label("🌐 Multi-Market Comparison  (all 6 futures, 5m, ~60 days)")
        st.markdown(
            "<div style='font-size:0.78rem;color:#484f58;margin-bottom:12px;'>"
            "Runs the 5m engine on ES, NQ, Gold, Crude Oil, Russell 2000, and Dow Jones simultaneously. "
            "Identifies which markets are cleanest for intraday setups right now."
            "</div>",
            unsafe_allow_html=True,
        )
        mm_col1, mm_col2 = st.columns([1, 5])
        with mm_col1:
            mm_run = st.button("🔄 Scan All Markets", type="secondary",
                               use_container_width=True, key="mm_run")
        if mm_run:
            with st.spinner("Running 5m backtest on 6 markets in parallel…"):
                try:
                    _mm_results = run_multi_market(
                        days_back=55,
                        engine_kwargs=dict(
                            warmup=50, max_bars=24, rr_ratio=2.0, atr_stop=1.5,
                            min_score=0.50, allow_short=True, min_adx=18.0,
                        ),
                    )
                    st.session_state["mm_results"] = _mm_results
                except Exception as _mm_err:
                    st.error(f"Multi-market scan failed: {_mm_err}")

        _mm_results = st.session_state.get("mm_results", {})
        if _mm_results:
            _mm_rows = []
            for sym, res in sorted(_mm_results.items()):
                market_name = FUTURES_UNIVERSE.get(sym, sym)
                if isinstance(res, Exception):
                    _mm_rows.append({
                        "Symbol": sym, "Market": market_name,
                        "Trades": "—", "Win Rate": "—",
                        "Profit Factor": "ERROR", "Expectancy": str(res)[:50],
                        "P&L (pts)": "—", "Best Strategy": "—",
                    })
                    continue

                pf = res.profit_factor
                pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
                pf_flag = "🟢" if pf >= 1.4 else ("🟡" if pf >= 1.0 else "🔴")

                # Find best strategy
                _strat_pnl: dict = {}
                for t in res.trades:
                    _strat_pnl[t.strategy] = _strat_pnl.get(t.strategy, 0) + t.pnl_pts
                best_strat = max(_strat_pnl, key=_strat_pnl.get) if _strat_pnl else "—"

                total_pnl = sum(t.pnl_pts for t in res.trades)

                _mm_rows.append({
                    "Symbol":         sym,
                    "Market":         market_name,
                    "Trades":         res.total_trades,
                    "Win Rate":       f"{res.win_rate:.0%}" if res.total_trades > 0 else "—",
                    "Profit Factor":  f"{pf_flag} {pf_str}",
                    "Expectancy":     f"{res.expectancy:+.2f}R" if res.total_trades > 0 else "—",
                    "P&L (pts)":      f"{total_pnl:+.1f}" if res.total_trades > 0 else "—",
                    "Best Strategy":  best_strat,
                })

            if _mm_rows:
                st.dataframe(pd.DataFrame(_mm_rows), use_container_width=True, hide_index=True)
                # Highlight the best market
                _best_markets = [r for r in _mm_rows if isinstance(r["Trades"], int) and r["Trades"] > 0]
                if _best_markets:
                    _best = max(_best_markets, key=lambda x: float(x["P&L (pts)"].replace("+","")) if x["P&L (pts)"] != "—" else -999)
                    st.success(
                        f"🏆 Best 5m market right now: **{_best['Symbol']}** ({_best['Market']}) — "
                        f"{_best['Trades']} trades | PF {_best['Profit Factor']} | {_best['P&L (pts)']} pts"
                    )


# ================================================================== #
# TAB: JOURNAL                                                         #
# ================================================================== #
with tab_journal:
    _section_label("Trade Journal")
    try:
        conn = duckdb.connect(DB_PATH)
        jdf  = conn.execute("""
            SELECT symbol, direction, entry_price, exit_price, qty,
                   pnl, r_multiple, strategy_used, opened_at, closed_at
            FROM trade_journal ORDER BY opened_at DESC LIMIT 200
        """).df()
        conn.close()

        if len(jdf) > 0:
            wins   = jdf[jdf["pnl"] > 0]
            losses = jdf[jdf["pnl"] <= 0]
            j1, j2, j3, j4, j5 = st.columns(5)
            j1.metric("Total Trades", len(jdf))
            j2.metric("Win Rate",     f"{len(wins)/len(jdf):.0%}")
            j3.metric("Total P&L",    f"${jdf['pnl'].sum():+,.2f}")
            j4.metric("Avg Win",      f"${wins['pnl'].mean():+,.2f}"   if len(wins)   else "—")
            j5.metric("Avg Loss",     f"${losses['pnl'].mean():+,.2f}" if len(losses) else "—")

            jdf["cum_pnl"] = jdf["pnl"].iloc[::-1].cumsum().iloc[::-1]
            pnl_vals   = jdf["cum_pnl"].iloc[::-1].values
            pnl_color  = "#3fb950" if pnl_vals[-1] >= 0 else "#f85149"
            fill_color = f"rgba({'63,185,80' if pnl_vals[-1]>=0 else '248,81,73'},0.1)"

            fig_j = go.Figure()
            fig_j.add_trace(go.Scatter(
                x=list(range(len(jdf))), y=pnl_vals,
                fill="tozeroy", fillcolor=fill_color,
                line=dict(color=pnl_color, width=2), name="P&L",
            ))
            fig_j.add_hline(y=0, line_dash="dot", line_color="#21262d", line_width=1)
            fig_j.update_layout(
                height=210, template="plotly_dark",
                paper_bgcolor="#080b12", plot_bgcolor="#0d1117",
                showlegend=False, margin=dict(l=40, r=20, t=8, b=20),
                font=dict(size=9, color="#5a6a7a"),
                xaxis=dict(gridcolor="#161b22"),
                yaxis=dict(gridcolor="#161b22"),
            )
            st.plotly_chart(fig_j, use_container_width=True, config={"displayModeBar": False})

            st.dataframe(jdf.drop(columns=["cum_pnl"]),
                         use_container_width=True, hide_index=True)
        else:
            st.markdown("""
<div style="text-align:center;padding:70px 0;">
  <div style="font-size:3rem;opacity:0.2;margin-bottom:14px;">📓</div>
  <div style="font-size:0.9rem;color:#484f58;">No completed trades yet.<br>
  Execute paper trades in the Analysis tab.</div>
</div>""", unsafe_allow_html=True)

    except Exception as e:
        st.caption(f"Journal unavailable: {e}")
