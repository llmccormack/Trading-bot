"""
Rival Automations — Trading Terminal FastAPI Backend
Run with: uvicorn dashboard.api:app --reload --port 8000
  (from the trading_bot directory)
"""
import sys, time, traceback, threading, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from data.fetcher import fetch_historical, get_live_price
from strategies.aggregator import SignalAggregator
from risk.manager import RiskManager
from execution.paper import PaperBroker
from execution.tradovate import TradovateBroker
from ai.agent import TradingAgent
from data.store import init_db, DB_PATH, get_open_positions
from backtesting.engine import BacktestEngine
from backtesting.engine_5m import BacktestEngine5m, run_multi_market, FUTURES_UNIVERSE
from backtesting.engine_aplus import BacktestEngineAPlus, APLUS_UNIVERSE
from utils.indicators import add_all, get_key_levels
import duckdb

init_db()

# Seed historical trades if DB is empty (survives Railway redeploys)
try:
    from seed_trades import seed as _seed_trades
    _seed_trades()
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────
# PER-SYMBOL ENGINE ROUTING
# Backtest data showed:
#   ES=F, NQ=F  → A+ IB Retest 5m  (PF 1.34 / 1.83)
#   GC=F, CL=F  → ORB+VWAP+EMA 5m  (PF 5.50 / 1.31)
#   RTY=F, YM=F → A+ IB Retest 5m  (index futures, same structure as ES/NQ)
# ─────────────────────────────────────────────────────────────────────
_COMMODITY_SYMBOLS = {"GC=F"}   # CL dropped — no edge confirmed over 365d 1h backtest

def _make_live_engine(sym: str):
    """Return the correct live-signal engine for this symbol."""
    if sym in _COMMODITY_SYMBOLS:
        return BacktestEngine5m(
            warmup=50, max_bars=24, rr_ratio=2.0, atr_stop=1.5,
            min_adx=18.0, min_score=0.50, allow_short=False, rth_only=True,
        )
    return BacktestEngineAPlus(
        min_adx=18.0, min_score=0.65, allow_short=False, require_macro_confirm=False,
    )

# ─────────────────────────────────────────────────────────────────────
# SERVER-SIDE AUTOPILOT (runs in background thread, no browser needed)
# ─────────────────────────────────────────────────────────────────────
_AP_ET        = pytz.timezone("America/New_York")
_AP_SYMBOLS   = ["NQ=F", "ES=F", "GC=F"]
_AP_INTERVAL  = 5 * 60          # check every 5 minutes
_AP_LOG       = Path(__file__).resolve().parent.parent / "autopilot.log"
_AP_STATE_FILE = Path(__file__).resolve().parent.parent / "autopilot_state.json"
_AP_ARMED     = True            # server-side autopilot starts armed by default
_AP_LOCK      = threading.Lock()

# ── Persistent state (survives server restarts) ───────────────────── #
def _load_ap_state() -> dict:
    """Load autopilot state from disk, return defaults if missing/corrupt."""
    try:
        if _AP_STATE_FILE.exists():
            import json as _json
            return _json.loads(_AP_STATE_FILE.read_text())
    except Exception:
        pass
    return {"health_sent": [], "eod_sent": [], "reset_sent": [], "summary_sent": [], "traded_today": {}}

def _save_ap_state() -> None:
    """Persist autopilot state to disk (called after every mutation)."""
    try:
        import json as _json
        _AP_STATE_FILE.write_text(_json.dumps({
            "health_sent":  list(_AP_HEALTH_SENT),
            "eod_sent":     list(_AP_EOD_SENT),
            "reset_sent":   list(_AP_RESET_SENT),
            "summary_sent": list(_AP_SUMMARY_SENT),
            "traded_today": _AP_TRADED_TODAY,
        }))
    except Exception as e:
        _aplog.warning(f"State save failed: {e}")

_ap_state_init = _load_ap_state()
_AP_TRADED_TODAY: dict[str, dict] = _ap_state_init.get("traded_today", {})
# symbol → {"date": str, "outcome": "open"|"win"|"loss"}
# "open"  — position still live, no new entry
# "win"   — hit target, don't re-enter same name today
# "loss"  — stopped out, allow one more entry (2nd chance in PM window)

_NTFY_TOPIC   = "rival-automation-tradez-wLuke"
_NTFY_URL     = f"https://ntfy.sh/{_NTFY_TOPIC}"

def _phone(title: str, body: str, tags: str = "chart_with_upwards_trend", priority: str = "default") -> None:
    """Send a push notification to phone via ntfy.sh (fire-and-forget)."""
    try:
        import urllib.request
        data = body.encode("utf-8")
        req  = urllib.request.Request(_NTFY_URL, data=data, method="POST")
        # ntfy headers must be latin-1 safe — encode non-ASCII as UTF-8 percent-escape
        def _h(s): return s.encode("utf-8").decode("latin-1", errors="replace")
        req.add_header("Title",    _h(title))
        req.add_header("Tags",     tags)      # tags are always ASCII emoji names
        req.add_header("Priority", priority)
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        _aplog.warning(f"ntfy push failed: {e}")

logging.basicConfig(
    filename=str(_AP_LOG),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
_aplog = logging.getLogger("autopilot")


def _ap_is_market_hours() -> bool:
    now = datetime.now(_AP_ET)
    return now.weekday() < 5 and 9 <= now.hour < 16


def _ap_run_cycle(broker: "PaperBroker") -> None:
    """One scan cycle: check all symbols, execute paper trades on signals."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    today_str = datetime.now(_AP_ET).strftime("%Y-%m-%d")

    def _check(sym: str) -> dict:
        try:
            root   = sym.replace("=F", "")
            df     = fetch_historical(sym, "5m", days_back=58)
            snap   = get_live_price(root)
            cur_px = snap.get("last_price") if snap else None
            engine = _make_live_engine(sym)
            sig = engine.live_signal(df, current_price=cur_px)
            # Auto-close stops/targets
            closed = broker.check_stops_and_targets(root, cur_px) if cur_px else []
            for msg in closed:
                _aplog.info(f"AUTO-CLOSE {sym}: {msg}")
                pnl_m   = __import__("re").search(r"PnL=\$([+-]?[\d.]+)", msg)
                pnl_str = f" | PnL: ${float(pnl_m.group(1)):+.2f}" if pnl_m else ""
                if "[PARTIAL T1]" in msg:
                    # position still open — outcome stays "open"
                    _phone(
                        title    = f"Half Out + BE Stop — {root}{pnl_str}",
                        body     = msg.replace("[PARTIAL T1] ", "")[:200],
                        tags     = "money_with_wings",
                        priority = "high",
                    )
                elif "[TARGET HIT]" in msg:
                    with _AP_LOCK:
                        if sym in _AP_TRADED_TODAY:
                            _AP_TRADED_TODAY[sym]["outcome"] = "win"
                            _save_ap_state()
                    _phone(
                        title    = f"Target Hit — {root}{pnl_str}",
                        body     = msg.replace("[TARGET HIT] ", "")[:200],
                        tags     = "white_check_mark",
                        priority = "high",
                    )
                elif "[STOP HIT]" in msg:
                    with _AP_LOCK:
                        if sym in _AP_TRADED_TODAY:
                            _AP_TRADED_TODAY[sym]["outcome"] = "loss"
                            _save_ap_state()
                    _phone(
                        title    = f"Stop Hit — {root}{pnl_str}",
                        body     = msg.replace("[STOP HIT] ", "")[:200],
                        tags     = "stop_sign",
                        priority = "high",
                    )
            return {"sym": sym, "sig": sig, "root": root, "cur_px": cur_px}
        except Exception as e:
            _aplog.warning(f"{sym} check error: {e}")
            return {"sym": sym, "sig": None, "root": sym, "cur_px": None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_check, s): s for s in _AP_SYMBOLS}
        for f in as_completed(futures):
            r = f.result()
            sym, sig, root, cur_px = r["sym"], r["sig"], r["root"], r["cur_px"]

            if not sig:
                _aplog.info(f"SKIP {sym}: no A+ setup (score below threshold or outside entry window)")
                continue

            # Re-entry logic: allow 2nd entry only if first trade was stopped out
            with _AP_LOCK:
                traded = _AP_TRADED_TODAY.get(sym, {})
            if traded.get("date") == today_str:
                outcome = traded.get("outcome", "open")
                if outcome in ("win", "open"):
                    label = "trade still open" if outcome == "open" else "target hit earlier today"
                    _aplog.info(f"SKIP {sym}: {label} — no re-entry")
                    continue
                # outcome == "loss" → allow 2nd entry
                _aplog.info(f"ALLOW re-entry {sym}: earlier trade stopped out, taking 2nd setup")

            _aplog.info(
                f"SIGNAL {sig['direction']} {sym} | entry={sig['entry']} "
                f"stop={sig['stop']} target={sig['target']} score={sig['score']}"
            )

            # Correlation block: only one index future open at a time
            if root in _INDEX_ROOTS:
                open_index = {p.symbol for p in broker.open_positions if p.symbol in _INDEX_ROOTS}
                if open_index:
                    _aplog.info(f"SKIP {sym}: correlation block — {open_index} index position already open")
                    continue

            # Execute paper trade
            ok, msg, pos = broker.open_position(
                symbol=root,
                direction=sig["direction"],
                entry_price=sig["entry"],
                stop_loss=sig["stop"],
                take_profit=sig["target"],
                target_1=sig.get("target_1", 0.0),   # T1 triggers break-even stop
                strategy_used=sig.get("strategy", "aplus"),
                ai_reasoning=(
                    f"Server autopilot | score={sig['score']} | "
                    f"regime={sig.get('regime','')} | bar={sig.get('bar_time','')}"
                ),
            )
            if ok:
                with _AP_LOCK:
                    _AP_TRADED_TODAY[sym] = {"date": today_str, "outcome": "open"}
                    _save_ap_state()
                _aplog.info(f"TRADE OPENED {sym}: {msg} (id={pos.id if pos else '?'})")
                _phone(
                    title    = f"🤖 Trade Entered — {root}",
                    body     = (
                        f"{sig['direction']} @ {sig['entry']:.2f}\n"
                        f"Stop: {sig['stop']:.2f}  |  Target: {sig['target']:.2f}\n"
                        f"Score: {sig['score']:.2f}  |  {sig.get('regime','')}"
                    ),
                    tags     = "robot",
                    priority = "high",
                )
            else:
                _aplog.warning(f"TRADE REJECTED {sym}: {msg}")


_AP_SUMMARY_SENT: set[str] = set(_ap_state_init.get("summary_sent", []))
_AP_HEALTH_SENT:  set[str] = set(_ap_state_init.get("health_sent",  []))
_AP_EOD_SENT:     set[str] = set(_ap_state_init.get("eod_sent",     []))
_AP_RESET_SENT:   set[str] = set(_ap_state_init.get("reset_sent",   []))

# Index futures — only one open at a time (correlated, no point doubling up)
_INDEX_ROOTS = {"ES", "NQ", "RTY", "YM"}

def _send_daily_summary(broker: "PaperBroker") -> None:
    """Send end-of-day P&L summary to phone."""
    try:
        import duckdb
        today = datetime.now(_AP_ET).strftime("%Y-%m-%d")
        conn  = duckdb.connect(str(Path(__file__).resolve().parent.parent / "trading_bot.duckdb"))
        rows  = conn.execute("""
            SELECT pnl, direction, symbol FROM trade_journal
            WHERE closed_at::DATE = ?
        """, [today]).fetchall()
        conn.close()

        n_trades = len(rows)
        total_pnl = sum(r[0] or 0 for r in rows)
        wins      = sum(1 for r in rows if (r[0] or 0) > 0)
        bal       = broker.account_balance

        if n_trades == 0:
            body = "No trades taken today."
        else:
            wr = wins / n_trades * 100
            body = (
                f"{n_trades} trade{'s' if n_trades>1 else ''} | "
                f"Win rate: {wr:.0f}%\n"
                f"Day P&L: ${total_pnl:+.2f}\n"
                f"Balance: ${bal:,.2f}"
            )

        _phone(
            title    = f"📊 Daily Summary — {today}",
            body     = body,
            tags     = "bar_chart",
            priority = "default",
        )
        _aplog.info(f"Daily summary sent: {body}")
    except Exception as e:
        _aplog.warning(f"Daily summary failed: {e}")


def _ap_background_loop(broker_ref_fn) -> None:
    """Runs forever in a daemon thread. broker_ref_fn() returns the PaperBroker."""
    _aplog.info("Server-side autopilot background loop started")
    while True:
        try:
            broker = broker_ref_fn()
            now_et = datetime.now(_AP_ET)
            today_str = now_et.strftime("%Y-%m-%d")

            if _AP_ARMED and _ap_is_market_hours():
                _ap_run_cycle(broker)

            # ── Daily risk reset + traded-today cleanup at 9:25 AM ───── #
            if (now_et.weekday() < 5 and now_et.hour == 9 and now_et.minute >= 25):
                if today_str not in _AP_RESET_SENT:
                    _AP_RESET_SENT.add(today_str)
                    broker.risk_manager.reset_daily()
                    # Purge stale re-entry records from prior days
                    with _AP_LOCK:
                        stale = [s for s, v in _AP_TRADED_TODAY.items()
                                 if v.get("date") != today_str]
                        for s in stale:
                            del _AP_TRADED_TODAY[s]
                    _save_ap_state()
                    _aplog.info("Daily risk reset complete — daily P&L counter and halt flag cleared")

            # ── Morning health ping at 9:25 AM ET ─────────────────────── #
            if (now_et.weekday() < 5 and now_et.hour == 9 and now_et.minute >= 25):
                if today_str not in _AP_HEALTH_SENT:
                    _AP_HEALTH_SENT.add(today_str)
                    _save_ap_state()
                    open_pos = broker.open_positions
                    _phone(
                        title    = "Rival Automations — Server Alive",
                        body     = (
                            f"Watching NQ, ES, GC | Armed: {_AP_ARMED}\n"
                            f"Open positions: {len(open_pos)} | "
                            f"Balance: ${broker.account_balance:,.0f}\n"
                            f"Entry windows: 10:15-11:30 and 14:00-15:55 ET"
                        ),
                        tags     = "eyes",
                        priority = "default",
                    )
                    _aplog.info("Morning health ping sent")

            # ── EOD force-close at 15:55 ET ───────────────────────────── #
            if (now_et.weekday() < 5 and now_et.hour == 15 and now_et.minute >= 55):
                if today_str not in _AP_EOD_SENT:
                    _AP_EOD_SENT.add(today_str)
                    _save_ap_state()
                    open_pos = broker.open_positions
                    if open_pos:
                        _aplog.info(f"EOD force-close: {len(open_pos)} open position(s)")
                        for pos in open_pos:
                            snap    = get_live_price(pos.symbol)
                            exit_px = snap.get("last_price") if snap else None
                            if not exit_px:
                                exit_px = pos.entry_price   # fallback: close at entry (flat)
                            ok, msg, pnl = broker.close_position(pos.id, exit_px, reason="eod")
                            if ok:
                                _aplog.info(f"EOD CLOSE {pos.symbol}: {msg}")
                                _phone(
                                    title    = f"EOD Close — {pos.symbol}",
                                    body     = msg[:200],
                                    tags     = "bell",
                                    priority = "default",
                                )
                    else:
                        _aplog.info("EOD check: no open positions to close")

            # ── Daily summary at 4:15 PM ET ───────────────────────────── #
            if (now_et.weekday() < 5 and now_et.hour == 16 and now_et.minute >= 15):
                if today_str not in _AP_SUMMARY_SENT:
                    _AP_SUMMARY_SENT.add(today_str)
                    _save_ap_state()
                    _send_daily_summary(broker)

            time.sleep(_AP_INTERVAL)
        except Exception as e:
            _aplog.error(f"Autopilot loop error: {e}")
            time.sleep(60)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Start the server-side autopilot in a background daemon thread
    t = threading.Thread(
        target=_ap_background_loop,
        args=(lambda: _broker,),
        daemon=True,
        name="server-autopilot",
    )
    t.start()
    _aplog.info("Server autopilot thread launched")
    yield


app = FastAPI(title="Rival Automations — Trading Terminal", version="2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the frontend HTML ───────────────────────────────────────── #
_DASH_DIR = Path(__file__).parent

@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse(_DASH_DIR / "index.html")

@app.get("/manifest.json", response_class=FileResponse)
def serve_manifest():
    return FileResponse(_DASH_DIR / "manifest.json", media_type="application/manifest+json")

@app.get("/icon-192.png", response_class=FileResponse)
def serve_icon192():
    return FileResponse(_DASH_DIR / "icon-192.png", media_type="image/png")

@app.get("/icon-512.png", response_class=FileResponse)
def serve_icon512():
    return FileResponse(_DASH_DIR / "icon-512.png", media_type="image/png")

# ── Broker factory — swap paper ↔ Tradovate via TRADING_MODE env var ── #
def _make_broker():
    rm = RiskManager()
    if settings.is_tradovate:
        try:
            broker = TradovateBroker(risk_manager=rm)
            _aplog.info("Broker: Tradovate (%s)", "DEMO" if settings.tradovate_is_demo == "true" else "LIVE")
            return broker
        except Exception as e:
            _aplog.error("Tradovate init failed (%s) — falling back to paper", e)
    _aplog.info("Broker: Paper trading")
    return PaperBroker(risk_manager=rm)

# ── Shared in-memory state (single-process, dev mode) ──────────────── #
_broker  = _make_broker()
_agent   = TradingAgent()
_last_agg: dict = {}

# ─────────────────────────────────────────────────────────────────────
# HEALTH / INFO
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "paper": settings.is_paper}


# ─────────────────────────────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/portfolio")
def portfolio():
    p = _broker.portfolio_summary()
    p["is_paper"]      = settings.is_paper
    p["is_tradovate"]  = settings.is_tradovate
    p["tradovate_demo"] = getattr(settings, "tradovate_is_demo", "true") == "true"
    p["is_halted"]     = _broker.risk_manager.is_halted

    # Inject live price + unrealized P&L for each open position
    total_unrealized = 0.0
    for pos in p.get("positions", []):
        sym  = pos["symbol"]
        snap = get_live_price(sym)
        cur  = snap.get("last_price") if snap else None
        pos["current_price"] = cur
        if cur is not None:
            direction = pos["direction"]
            entry     = pos["entry"]
            qty       = pos["qty"]
            upnl      = (cur - entry) * qty if direction == "LONG" else (entry - cur) * qty
            pos["unrealized_pnl"] = round(upnl, 2)
            # Progress 0..1 from entry toward target (negative = moving toward stop)
            risk   = abs(entry - pos["sl"])
            if risk > 0:
                if direction == "LONG":
                    pos["progress"] = round((cur - entry) / risk, 3)
                else:
                    pos["progress"] = round((entry - cur) / risk, 3)
            else:
                pos["progress"] = 0.0
            total_unrealized += upnl
        else:
            pos["unrealized_pnl"] = None
            pos["progress"]       = 0.0

    p["total_unrealized_pnl"] = round(total_unrealized, 2)
    return p


# ─────────────────────────────────────────────────────────────────────
# LIVE PRICE
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/price/{symbol}")
def live_price(symbol: str):
    try:
        snap = get_live_price(symbol.upper())
        return snap or {}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/prices")
def live_prices_batch():
    """Fetch live prices for all top-traded symbols in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Root symbol → display ticker
    WATCH = {
        "NQ":  "NQ=F",
        "ES":  "ES=F",
        "GC":  "GC=F",
        "CL":  "CL=F",
        "RTY": "RTY=F",
        "YM":  "YM=F",
    }

    def _fetch(root, full):
        try:
            snap = get_live_price(root) or {}
            snap["_symbol"] = full
            return snap
        except Exception:
            return {"_symbol": full, "last_price": None}

    out = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        fmap = {pool.submit(_fetch, root, full): full for root, full in WATCH.items()}
        for f in as_completed(fmap):
            r = f.result()
            out[r["_symbol"]] = r
    return out


# ─────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    symbol:    str = "ES=F"
    timeframe: str = "5m"
    market:    str = "futures"
    days_back: int = 20
    run_claude: bool = True

@app.post("/api/analysis")
def run_analysis(req: AnalysisRequest):
    global _agent
    sym = req.symbol.strip().upper()

    # A+ engine always uses 5m data (same as autopilot)
    tf       = "5m"
    days_back = 58

    try:
        df_raw = fetch_historical(sym, tf, days_back)
        df     = add_all(df_raw.copy())
    except Exception as e:
        raise HTTPException(400, f"Data fetch failed: {e}")

    # Live price
    root   = sym.replace("=F", "")
    snap   = get_live_price(root)
    cur_px = snap.get("last_price") if snap else None

    # Use the same per-symbol engine as the autopilot
    engine = _make_live_engine(sym)
    sig = engine.live_signal(df_raw, current_price=cur_px)

    # Build rejection reason when no signal
    rejection_reason = None
    if sig is None:
        now_et = datetime.now(_AP_ET)
        h, m   = now_et.hour, now_et.minute
        if now_et.weekday() >= 5:
            rejection_reason = "Weekend — markets closed"
        elif not (9 <= h < 16):
            rejection_reason = f"Outside market hours ({h:02d}:{m:02d} ET)"
        elif not (
            (h == 10 and m >= 15) or (h == 11 and m <= 30) or
            (14 <= h < 16)
        ):
            rejection_reason = f"Outside entry windows (10:15–11:30 or 14:00–15:55 ET, now {h:02d}:{m:02d})"
        else:
            rejection_reason = "No A+ setup — score below threshold or conditions not met"

    # Map to frontend format
    if sig:
        recommendation    = sig["direction"]
        direction         = sig["direction"]
        composite_score   = sig["score"]
        confidence        = sig["score"]
        agreeing          = [sig["strategy"]]
        disagreeing: list = []
        neutral:     list = []
        individual_signals = [{
            "strategy":   sig["strategy"],
            "direction":  sig["direction"],
            "confidence": sig["score"],
            "reasoning":  f"{sig.get('regime','')} | ADX={sig.get('adx','')} | bar={sig.get('bar_time','')}",
        }]
        suggested_entry  = sig["entry"]
        suggested_stop   = sig["stop"]
        suggested_target = sig["target"]
    else:
        recommendation    = "NEUTRAL"
        direction         = "NEUTRAL"
        composite_score   = 0.0
        confidence        = 0.0
        agreeing          = []
        disagreeing       = []
        neutral           = ["aplus"]
        individual_signals = [{
            "strategy":   "aplus",
            "direction":  "NEUTRAL",
            "confidence": 0.0,
            "reasoning":  rejection_reason or "No signal",
        }]
        suggested_entry  = cur_px
        suggested_stop   = None
        suggested_target = None

    # Key levels
    try:
        lvls       = get_key_levels(df)
        resistance = lvls.get("resistance", [])[:3]
        support    = lvls.get("support",    [])[:3]
    except Exception:
        resistance, support = [], []

    # Chart data (last 200 bars)
    df_chart = df.tail(200)

    def _col(series, decimals=4):
        import math
        return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(v, decimals)
                for v in series.tolist()]

    chart = {
        "timestamps": df_chart["timestamp"].astype(str).tolist(),
        "open":    _col(df_chart["open"]),
        "high":    _col(df_chart["high"]),
        "low":     _col(df_chart["low"]),
        "close":   _col(df_chart["close"]),
        "volume":  _col(df_chart["volume"], 0),
        "vwap":    (_col(df_chart["vwap"])   if "vwap"  in df_chart.columns else []),
        "ema8":    (_col(df_chart["ema_8"])  if "ema_8" in df_chart.columns else []),
        "ema21":   (_col(df_chart["ema_21"]) if "ema_21" in df_chart.columns else []),
        "ema55":   (_col(df_chart["ema_55"]) if "ema_55" in df_chart.columns else []),
        "rsi":     (_col(df_chart["rsi"], 2) if "rsi"   in df_chart.columns else []),
    }

    # Claude AI (optional)
    decision_data = None
    if req.run_claude:
        try:
            portfolio = _broker.portfolio_summary()
            decision  = _agent.analyze(
                symbol=sym, timeframe=tf,
                aggregated_signal={
                    "direction":              direction,
                    "composite_score":        composite_score,
                    "confidence":             confidence,
                    "recommendation":         recommendation,
                    "agreeing_strategies":    agreeing,
                    "disagreeing_strategies": disagreeing,
                    "neutral_strategies":     neutral,
                    "individual_signals":     individual_signals,
                    "suggested_entry":        suggested_entry,
                    "suggested_stop":         suggested_stop,
                    "suggested_target":       suggested_target,
                },
                portfolio_summary=portfolio,
            )
            decision_data = {
                "action":        decision.action,
                "entry":         decision.entry,
                "stop_loss":     decision.stop_loss,
                "target":        decision.target,
                "r_ratio":       decision.r_ratio,
                "lead_strategy": decision.lead_strategy,
                "reasoning":     decision.reasoning,
                "raw_response":  decision.raw_response,
            }
        except Exception as e:
            decision_data = {"error": str(e)}

    return {
        "symbol":                 sym,
        "timeframe":              tf,
        "recommendation":         recommendation,
        "direction":              direction,
        "composite_score":        composite_score,
        "confidence":             confidence,
        "agreeing_strategies":    agreeing,
        "disagreeing_strategies": disagreeing,
        "neutral_strategies":     neutral,
        "suggested_entry":        suggested_entry,
        "suggested_stop":         suggested_stop,
        "suggested_target":       suggested_target,
        "individual_signals":     individual_signals,
        "resistance":             resistance,
        "support":                support,
        "chart":                  chart,
        "decision":               decision_data,
        "aplus_signal":           sig,
        "rejection_reason":       rejection_reason,
    }


@app.post("/api/chat")
def chat(payload: dict = Body(...)):
    global _agent
    msg = payload.get("message", "")
    if not msg:
        raise HTTPException(400, "message required")
    if not _agent.conversation_history:
        return {"reply": "Run an analysis first so I have market context."}
    try:
        reply = _agent.ask(msg)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────
# SCANNER  — uses same aplus engine as autopilot so results match
# ─────────────────────────────────────────────────────────────────────
_SCANNER_SYMBOLS = ["NQ=F", "ES=F", "GC=F", "RTY=F", "YM=F"]

class ScanRequest(BaseModel):
    symbols:   list[str] = _SCANNER_SYMBOLS
    timeframe: str = "5m"
    market:    str = "futures"

@app.post("/api/scanner")
def run_scanner(req: ScanRequest):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _scan_one(sym):
        sym = sym.strip().upper()
        # Normalise: accept both "NQ" and "NQ=F"
        yf_sym = sym if sym.endswith("=F") else sym + "=F"
        root   = yf_sym.replace("=F", "")
        try:
            df     = fetch_historical(yf_sym, "5m", days_back=58)
            snap   = get_live_price(root)
            cur_px = snap.get("last_price") if snap else None
            engine = _make_live_engine(yf_sym)
            sig = engine.live_signal(df, current_price=cur_px)
            if sig:
                return {
                    "symbol":    root,
                    "rec":       sig["direction"],   # BUY or SELL
                    "direction": sig["direction"],
                    "score":     round(sig["score"], 3),
                    "conf":      round(sig["score"], 3),
                    "agreeing":  [sig.get("strategy", "aplus")],
                    "entry":     sig["entry"],
                    "stop":      sig["stop"],
                    "target":    sig["target"],
                    "regime":    sig.get("regime", ""),
                    "error":     None,
                }
            else:
                return {
                    "symbol": root, "rec": "NEUTRAL", "direction": "NEUTRAL",
                    "score": 0, "conf": 0, "agreeing": [],
                    "entry": cur_px, "stop": None, "target": None,
                    "regime": "", "error": None,
                }
        except Exception as e:
            return {"symbol": root, "rec": "ERROR", "direction": "ERROR",
                    "score": 0, "conf": 0, "agreeing": [], "entry": None,
                    "stop": None, "target": None, "regime": "", "error": str(e)[:100]}

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        fmap = {pool.submit(_scan_one, s): s for s in req.symbols}
        for f in as_completed(fmap):
            results.append(f.result())

    # Signals first, then neutrals, sorted by score
    results.sort(key=lambda x: (x["rec"] != "BUY", -abs(x["score"])))
    return {"results": results}


# ─────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbol:    str   = "ES=F"
    timeframe: str   = "5m"
    market:    str   = "futures"
    days_back: int   = 58
    engine:    str   = "5m"   # "1h", "5m", "aplus"
    rr_ratio:  float = 2.0
    warmup:    int   = 50
    max_bars:  int   = 24
    atr_stop:  float = 1.5
    min_adx:   float = 18.0
    min_score: float = 0.50
    allow_short: bool = False
    rth_only:    bool = True

@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    sym = req.symbol.strip().upper()
    try:
        df_bt = fetch_historical(sym, req.timeframe, req.days_back)
    except Exception as e:
        raise HTTPException(400, f"Data fetch failed: {e}")

    try:
        if req.engine == "aplus":
            # aplus engine has its own well-tuned defaults — don't let the generic
            # BacktestRequest fields (allow_short=True, min_score=0.50) override them.
            # Only honour overrides if the user explicitly tightened the settings.
            engine = BacktestEngineAPlus(
                max_bars=req.max_bars,
                min_adx=req.min_adx,
                min_score=max(req.min_score, 0.65),   # enforce minimum quality floor
                allow_short=req.allow_short,
                require_macro_confirm=False,
            )
        elif req.engine == "5m":
            engine = BacktestEngine5m(
                warmup=req.warmup, max_bars=req.max_bars,
                rr_ratio=req.rr_ratio, atr_stop=req.atr_stop,
                min_adx=req.min_adx, min_score=req.min_score,
                allow_short=req.allow_short, rth_only=req.rth_only,
            )
        else:
            engine = BacktestEngine(
                warmup=req.warmup, max_bars=req.max_bars,
                rr_ratio=req.rr_ratio, atr_stop=req.atr_stop,
                min_adx=req.min_adx, min_score=req.min_score,
                allow_short=req.allow_short, rth_only=req.rth_only,
            )

        result = engine.run(df_bt, symbol=sym, timeframe=req.timeframe,
                            **({} if req.engine in ("5m","aplus") else {"market": req.market}))
    except Exception as e:
        raise HTTPException(500, f"Backtest failed: {e}\n{traceback.format_exc()}")

    if result.total_trades == 0:
        return {"total_trades": 0, "message": "No trades triggered with current settings."}

    pf = result.profit_factor
    trades_out = []
    for t in result.trades:
        trades_out.append({
            "direction":   t.direction,
            "strategy":    t.strategy,
            "regime":      t.regime,
            "entry_price": round(t.entry_price, 2),
            "exit_price":  round(t.exit_price, 2),
            "exit_reason": t.exit_reason,
            "pnl_pts":     round(t.pnl_pts, 2),
            "r_multiple":  round(t.r_multiple, 3),
            "bars_held":   t.bars_held,
            "be_moved":    getattr(t, "be_moved", False),
        })

    # Strategy breakdown
    strat_map: dict = {}
    for t in result.trades:
        s = t.strategy
        if s not in strat_map:
            strat_map[s] = {"trades": 0, "wins": 0, "pnl": 0.0, "r": 0.0}
        strat_map[s]["trades"] += 1
        strat_map[s]["pnl"]    += t.pnl_pts
        strat_map[s]["r"]      += t.r_multiple
        if t.pnl_pts > 0:
            strat_map[s]["wins"] += 1
    by_strategy = []
    for s, st in sorted(strat_map.items(), key=lambda x: x[1]["pnl"], reverse=True):
        n = st["trades"]
        by_strategy.append({
            "strategy":   s,
            "trades":     n,
            "win_rate":   round(st["wins"]/n, 3) if n else 0,
            "pnl_pts":    round(st["pnl"], 2),
            "expectancy": round(st["r"]/n, 3) if n else 0,
        })

    return {
        "symbol":         sym,
        "engine":         req.engine,
        "total_trades":   result.total_trades,
        "win_rate":       round(result.win_rate, 3),
        "profit_factor":  round(pf, 3) if pf != float("inf") else 9999,
        "expectancy":     round(result.expectancy, 3),
        "max_drawdown":   round(result.max_drawdown_pct, 4),
        "avg_bars_held":  round(result.avg_bars_held, 1),
        "total_bars":     result.total_bars,
        "be_trail_rate":  round(getattr(result, "be_trail_rate", 0), 3),
        "equity_curve":   [round(v, 2) for v in result.equity_curve],
        "by_strategy":    by_strategy,
        "trades":         trades_out,
    }


@app.post("/api/backtest/multi")
def run_multi_backtest():
    try:
        results = run_multi_market(
            days_back=55,
            engine_kwargs=dict(
                warmup=50, max_bars=24, rr_ratio=2.0, atr_stop=1.5,
                min_score=0.50, allow_short=False, min_adx=18.0,
            ),
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    out = []
    for sym, res in sorted(results.items()):
        market_name = FUTURES_UNIVERSE.get(sym, sym)
        if isinstance(res, Exception):
            out.append({"symbol": sym, "market": market_name, "error": str(res)[:80]})
            continue
        pf = res.profit_factor
        strat_pnl: dict = {}
        for t in res.trades:
            strat_pnl[t.strategy] = strat_pnl.get(t.strategy, 0) + t.pnl_pts
        best_strat = max(strat_pnl, key=strat_pnl.get) if strat_pnl else "—"
        total_pnl  = sum(t.pnl_pts for t in res.trades)
        out.append({
            "symbol":        sym,
            "market":        market_name,
            "trades":        res.total_trades,
            "win_rate":      round(res.win_rate, 3),
            "profit_factor": round(pf, 3) if pf != float("inf") else 9999,
            "expectancy":    round(res.expectancy, 3),
            "total_pnl":     round(total_pnl, 2),
            "best_strategy": best_strat,
        })
    return {"results": out}


# ─────────────────────────────────────────────────────────────────────
# AUTOPILOT  — multi-symbol, parallel
# ─────────────────────────────────────────────────────────────────────

# Profit-factor ranked defaults (backtest-confirmed edge):
#   NQ=F  PF 1.83 (A+ 5m)  |  ES=F  PF 1.34 (A+ 5m)  |  GC=F  PF 5.50/1.58 (ORB 5m / 1h)
#   CL=F removed — PF 0.88 over 365d 1h, no confirmed edge
AP_DEFAULT_SYMBOLS = ["NQ=F", "ES=F", "GC=F"]

# Ticker roots for get_live_price (strip =F suffix)
_TICKER_ROOT = {"NQ=F": "NQ", "ES=F": "ES", "GC=F": "GC", "CL=F": "CL",
                "RTY=F": "RTY", "YM=F": "YM"}

class AutopilotRequest(BaseModel):
    mode:    str       = "aplus"   # "1h", "5m", "aplus"
    symbols: list[str] = AP_DEFAULT_SYMBOLS

@app.post("/api/autopilot/check")
def autopilot_check(req: AutopilotRequest):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tf       = "5m" if req.mode in ("5m", "aplus") else "1h"
    days_back = 58 if tf == "5m" else 365

    def _check_one(sym: str) -> dict:
        sym = sym.strip().upper()
        try:
            df      = fetch_historical(sym, tf, days_back=days_back)
            root    = _TICKER_ROOT.get(sym, sym.replace("=F", ""))
            snap    = get_live_price(root)
            cur_px  = snap.get("last_price") if snap else None
            # Per-symbol engine routing (overridden by manual mode selection)
            if req.mode in ("aplus", "5m"):
                engine = _make_live_engine(sym)
            elif req.mode == "1h":
                engine = BacktestEngine(warmup=200, max_bars=24, rr_ratio=2.0, atr_stop=1.0,
                                        min_score=0.55, allow_short=False, min_adx=25.0, rth_only=True)
            else:
                engine = _make_live_engine(sym)
            sig     = engine.live_signal(df, current_price=cur_px)

            # Auto-close any open paper positions that have hit stop or target
            closed_msgs = []
            if cur_px:
                closed_msgs = _broker.check_stops_and_targets(root, cur_px)

            return {"symbol": sym, "signal": sig, "error": None, "closed": closed_msgs}
        except Exception as e:
            return {"symbol": sym, "signal": None, "error": str(e)[:120], "closed": []}

    results = {}
    all_closed: list[str] = []
    with ThreadPoolExecutor(max_workers=min(len(req.symbols), 6)) as pool:
        fmap = {pool.submit(_check_one, s): s for s in req.symbols}
        for f in as_completed(fmap):
            r = f.result()
            results[r["symbol"]] = {"signal": r["signal"], "error": r["error"]}
            all_closed.extend(r.get("closed", []))

    # Correlation warning: flag if 2+ correlated index futures fire the same direction
    _index_futures = {"ES=F", "NQ=F", "RTY=F", "YM=F"}
    firing_index = [
        sym for sym, r in results.items()
        if r["signal"] and sym in _index_futures
    ]
    corr_warning = None
    if len(firing_index) >= 2:
        dirs = set(results[s]["signal"]["direction"] for s in firing_index)
        if len(dirs) == 1:
            corr_warning = (
                f"⚠️ {', '.join(firing_index)} all firing {list(dirs)[0]} — "
                "these are highly correlated. Trading both doubles index exposure."
            )

    return {
        "results":      results,
        "checked_at":   datetime.now().isoformat(),
        "corr_warning": corr_warning,
        "closed_trades": all_closed,   # positions auto-closed at stop/target this cycle
    }


# ─────────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ─────────────────────────────────────────────────────────────────────
class TradeRequest(BaseModel):
    symbol:       str
    direction:    str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    target_1:     float = 0.0
    strategy_used: str = ""
    ai_reasoning:  str = ""

@app.post("/api/trade/open")
def open_trade(req: TradeRequest):
    ok, msg, pos = _broker.open_position(
        symbol=req.symbol, direction=req.direction,
        entry_price=req.entry_price, stop_loss=req.stop_loss,
        take_profit=req.take_profit, target_1=req.target_1,
        strategy_used=req.strategy_used,
        ai_reasoning=req.ai_reasoning,
    )
    return {"ok": ok, "message": msg, "position_id": str(pos.id) if pos else None}


@app.post("/api/trade/close/{position_id}")
def close_trade(position_id: str, payload: dict = Body(...)):
    price = payload.get("price", 0.0)
    ok, msg, _ = _broker.close_position(position_id, price, reason="manual")
    return {"ok": ok, "message": msg}


@app.get("/api/positions")
def get_positions():
    try:
        df = get_open_positions()
        return {"positions": df.to_dict("records") if not df.empty else []}
    except Exception:
        return {"positions": [p.__dict__ for p in _broker.open_positions]}


# ─────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# SETTINGS  (read / write .env)
# ─────────────────────────────────────────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

def _read_env() -> dict:
    """Parse .env file into a dict (ignores comments / blank lines, strips inline comments)."""
    pairs = {}
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                # Strip any trailing inline comment (e.g.  value  # comment)
                if " #" in v:
                    v = v[:v.index(" #")].strip()
                pairs[k.strip()] = v
    return pairs

def _write_env(pairs: dict) -> None:
    """Write a flat key=value .env file (no comments preserved)."""
    lines = [f'{k}="{v}"' for k, v in sorted(pairs.items())]
    _ENV_PATH.write_text("\n".join(lines) + "\n")

@app.get("/api/autopilot/log")
def ap_log(lines: int = 50):
    """Return recent lines from the server-side autopilot log."""
    try:
        if _AP_LOG.exists():
            all_lines = _AP_LOG.read_text().splitlines()
            return {"lines": all_lines[-lines:]}
        return {"lines": []}
    except Exception:
        return {"lines": []}


@app.get("/api/settings")
def get_settings():
    """Return current settings (API keys masked)."""
    return {
        "trading_mode":             settings.trading_mode,
        "paper_account_size":       settings.paper_account_size,
        "max_risk_per_trade_pct":   settings.max_risk_per_trade_pct,
        "max_daily_loss_pct":       settings.max_daily_loss_pct,
        "max_concurrent_positions": settings.max_concurrent_positions,
        "ibkr_host":                settings.ibkr_host,
        "ibkr_port":                settings.ibkr_port,
        # True/False tells the frontend whether a key is set (value masked)
        "anthropic_api_key": bool(settings.anthropic_api_key),
        "polygon_api_key":   bool(settings.polygon_api_key),
    }

class SettingsUpdate(BaseModel):
    trading_mode:             Optional[str]   = None
    paper_account_size:       Optional[float] = None
    max_risk_per_trade_pct:   Optional[float] = None
    max_daily_loss_pct:       Optional[float] = None
    max_concurrent_positions: Optional[int]   = None
    ibkr_host:                Optional[str]   = None
    ibkr_port:                Optional[int]   = None
    anthropic_api_key:        Optional[str]   = None
    polygon_api_key:          Optional[str]   = None

@app.post("/api/settings")
def save_settings(req: SettingsUpdate):
    """Persist settings to .env and reload the in-process settings object."""
    try:
        pairs = _read_env()

        # Map field → env var name (matches pydantic-settings convention)
        field_map = {
            "trading_mode":             "TRADING_MODE",
            "paper_account_size":       "PAPER_ACCOUNT_SIZE",
            "max_risk_per_trade_pct":   "MAX_RISK_PER_TRADE_PCT",
            "max_daily_loss_pct":       "MAX_DAILY_LOSS_PCT",
            "max_concurrent_positions": "MAX_CONCURRENT_POSITIONS",
            "ibkr_host":                "IBKR_HOST",
            "ibkr_port":                "IBKR_PORT",
            "anthropic_api_key":        "ANTHROPIC_API_KEY",
            "polygon_api_key":          "POLYGON_API_KEY",
        }

        data = req.model_dump(exclude_none=True)
        for field, value in data.items():
            env_key = field_map.get(field)
            if env_key:
                pairs[env_key] = str(value)

        _write_env(pairs)

        # Apply non-sensitive fields immediately to the in-process settings object
        # (API keys require a full server restart to take effect)
        _LIVE_FIELDS = {
            "trading_mode":             "trading_mode",
            "paper_account_size":       "paper_account_size",
            "max_risk_per_trade_pct":   "max_risk_per_trade_pct",
            "max_daily_loss_pct":       "max_daily_loss_pct",
            "max_concurrent_positions": "max_concurrent_positions",
            "ibkr_host":                "ibkr_host",
            "ibkr_port":                "ibkr_port",
        }
        for field, attr in _LIVE_FIELDS.items():
            if field in data:
                try:
                    object.__setattr__(settings, attr, data[field])
                except Exception:
                    pass

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/autopilot/state")
def ap_state():
    return {
        "armed":       _AP_ARMED,
        "symbols":     _AP_SYMBOLS,
        "interval_s":  _AP_INTERVAL,
        "market_hours": _ap_is_market_hours(),
        "traded_today": _AP_TRADED_TODAY,
    }

@app.post("/api/autopilot/arm")
def ap_arm(body: dict = Body(...)):
    global _AP_ARMED
    _AP_ARMED = bool(body.get("armed", True))
    _aplog.info(f"Server autopilot {'ARMED' if _AP_ARMED else 'DISARMED'} via API")
    return {"ok": True, "armed": _AP_ARMED}


@app.get("/api/journal")
def get_journal(limit: int = 200):
    try:
        conn = duckdb.connect(DB_PATH)
        jdf  = conn.execute(f"""
            SELECT symbol, direction, entry_price, exit_price, qty,
                   pnl, r_multiple, strategy_used, opened_at, closed_at
            FROM trade_journal ORDER BY opened_at DESC LIMIT {limit}
        """).df()
        conn.close()

        if len(jdf) == 0:
            return {"trades": [], "stats": None}

        wins   = jdf[jdf["pnl"] > 0]
        losses = jdf[jdf["pnl"] <= 0]
        cum_pnl = jdf["pnl"].iloc[::-1].cumsum().iloc[::-1]

        return {
            "trades": jdf.to_dict("records"),
            "equity_curve": [round(v, 2) for v in cum_pnl.iloc[::-1].values],
            "stats": {
                "total_trades": len(jdf),
                "win_rate":     round(len(wins)/len(jdf), 3),
                "total_pnl":    round(float(jdf["pnl"].sum()), 2),
                "avg_win":      round(float(wins["pnl"].mean()), 2) if len(wins)  else 0,
                "avg_loss":     round(float(losses["pnl"].mean()), 2) if len(losses) else 0,
                "profit_factor": (
                    round(float(wins["pnl"].sum()) / abs(float(losses["pnl"].sum())), 3)
                    if losses["pnl"].sum() != 0 else 9999
                ),
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))
