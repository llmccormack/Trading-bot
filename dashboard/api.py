"""
Rival Automations — Trading Terminal FastAPI Backend
Run with: uvicorn dashboard.api:app --reload --port 8000
  (from the trading_bot directory)
"""
import sys, time, traceback, threading, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
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
from backtesting.trade_sim import TradeSimulator
from backtesting.engine import BacktestEngine
from backtesting.engine_5m import BacktestEngine5m, run_multi_market, FUTURES_UNIVERSE
from backtesting.engine_aplus import BacktestEngineAPlus, APLUS_UNIVERSE
from backtesting.engine_fade_rip import FadeTheRipEngine
from backtesting.engine_orb import ORBEngine
from backtesting.engine_vwap_pullback import VWAPPullbackEngine
from utils.indicators import add_all, get_key_levels
import duckdb

# DB init and seeding are deferred to lifespan startup (not import time)
# so that importing this module in tests has no side effects.

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
        min_adx=18.0, min_score=0.75, allow_short=False, require_macro_confirm=False,
        skip_monday=True, skip_power_hour=True,    # skip ALL power hour — IB is stale by 2 PM
        # skip_lunch and require_vwap_slope available but backtested neutral at current sample size
        # re-evaluate after 60+ live trades
    )

# ─────────────────────────────────────────────────────────────────────
# SERVER-SIDE AUTOPILOT (runs in background thread, no browser needed)
# ─────────────────────────────────────────────────────────────────────
_AP_ET        = pytz.timezone("America/New_York")
_AP_SYMBOLS   = ["NQ=F", "ES=F"]  # GC removed — no edge confirmed in current regime
_AP_INTERVAL  = 5 * 60          # check every 5 minutes

# ── VIX filter ────────────────────────────────────────────────────── #
# If VIX >= this threshold at market open, skip all entries for the day.
# Protects against high-volatility event days (e.g. tariff announcements).
# TopStep daily loss limit is ~$2k — extreme VIX days blow through that.
VIX_BLOCK_THRESHOLD = 22.0
_VIX_CACHE: dict = {}   # {"date": str, "vix": float, "blocked": bool}

def _get_vix_today() -> tuple[float, bool]:
    """
    Fetch today's VIX level once and cache it for the day.
    Returns (vix_level, is_blocked).
    """
    today_str = datetime.now(_AP_ET).strftime("%Y-%m-%d")
    if _VIX_CACHE.get("date") == today_str:
        return _VIX_CACHE["vix"], _VIX_CACHE["blocked"]
    try:
        import yfinance as yf
        vix_df = yf.download("^VIX", period="2d", interval="1d", progress=False)
        vix = float(vix_df["Close"].iloc[-1])
    except Exception as e:
        _aplog.warning(f"VIX fetch failed: {e} — allowing trades")
        return 0.0, False
    blocked = vix >= VIX_BLOCK_THRESHOLD
    _VIX_CACHE.update({"date": today_str, "vix": round(vix, 2), "blocked": blocked})
    if blocked:
        _aplog.warning(
            f"VIX BLOCK: VIX={vix:.1f} >= {VIX_BLOCK_THRESHOLD} — "
            f"no new entries today (extreme volatility)"
        )
        # Log only — no push. Trade notifications cover what matters.
    else:
        _aplog.info(f"VIX check OK: {vix:.1f} (threshold {VIX_BLOCK_THRESHOLD})")
    return vix, blocked
_AP_LOG        = Path(__file__).resolve().parent.parent / "autopilot.log"
_AP_STATE_FILE = Path(__file__).resolve().parent.parent / "autopilot_state.json"
_AP_PID_FILE   = Path(__file__).resolve().parent.parent / "autopilot.pid"
_AP_ARMED      = True            # server-side autopilot starts armed by default
_AP_LOCK       = threading.Lock()

# ── PID lock — kill stale server if another instance is already running ─ #
def _enforce_single_instance() -> None:
    """
    Write our PID to autopilot.pid. If the file already exists with a live PID,
    kill that process first so only one autopilot loop is ever running.
    """
    import os, signal
    if _AP_PID_FILE.exists():
        try:
            old_pid = int(_AP_PID_FILE.read_text().strip())
            if old_pid != os.getpid():
                try:
                    os.kill(old_pid, signal.SIGKILL)
                    import time as _t; _t.sleep(0.5)
                    logging.getLogger("autopilot").warning(
                        f"Killed stale autopilot process PID {old_pid} — only one instance allowed"
                    )
                except ProcessLookupError:
                    pass   # already gone
        except (ValueError, OSError):
            pass
    _AP_PID_FILE.write_text(str(os.getpid()))

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
_NOTIFY_ENABLED = True   # PID lock prevents duplicate processes — notifications are safe again

def _phone(title: str, body: str, tags: str = "chart_with_upwards_trend", priority: str = "default") -> None:
    """Send a push notification to phone via ntfy.sh (fire-and-forget)."""
    if not _NOTIFY_ENABLED:
        _aplog.info(f"[notify off] {title} — {body[:80]}")
        return
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



def _get_today_pnl() -> float:
    """Return today's total realized P&L from the journal (negative = loss).

    Excludes partial_exit_t1 rows — the final close row stores total_pnl
    (partial + runner) so including partials would double-count T1 exits.
    """
    try:
        today = datetime.now(_AP_ET).strftime("%Y-%m-%d")
        conn  = duckdb.connect(DB_PATH)
        row   = conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trade_journal "
            "WHERE closed_at::DATE = ? AND ai_reasoning != 'partial_exit_t1'",
            [today],
        ).fetchone()
        conn.close()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────
# STRATEGY STATE + HEALTH HELPERS
# ─────────────────────────────────────────────────────────────────────

def _strat_state(name: str) -> str:
    """Return configured state for a strategy: 'paper' | 'shadow' | 'disabled'."""
    _map = {
        "aplus":          settings.strategy_aplus,
        "orb":            settings.strategy_orb,
        "fade_the_rip":   settings.strategy_fade_rip,
        "vwap_pullback":  settings.strategy_vwap_pb,
    }
    return _map.get(name, "paper")


def _score_to_risk_mult(sig: dict, regime: dict | None = None) -> float:
    """
    Convert strategy + composite score → position-size multiplier.

    Applied on top of max_risk_per_trade_pct so the base config stays
    unchanged — we just press harder on high-confidence setups.

    Tiers
    -----
    A+   score ≥ 0.85 + regime green → 1.25× (conviction + structure aligned)
    A+   score ≥ 0.85, regime not green → 1.00× (score good, conditions not ideal)
    A+   score ≥ 0.80 → 1.00× (solid, standard size)
    A+   score < 0.80 → 0.50× (borderline pass, half-size)

    Fade score ≥ 0.80 → 1.00× (premium: strong rejection candle)
    Fade score < 0.80 → 0.75× (normal: still profitable but less edge)

    ORB  any score     → 0.50× (unproven sample — half until PF > 1.8)

    Others             → 0.50× (conservative default)

    Regime gate (A+ 1.25× only):
      tod     ∈ {open, primary}   — not lunch, power_hour, or eod
      adx     ≥ 20                — trending market, not ranging chop
      atr_pct ∈ [70, 130]         — normal volatility (not flat or spike day)
    All three must be green; otherwise the 0.85+ setup still gets 1.00×.
    """
    strategy = sig.get("strategy", "")
    score    = float(sig.get("score", 0.75))
    reg      = regime or {}

    if strategy == "aplus":
        if score >= 0.85:
            tod     = reg.get("tod", "")
            adx     = float(reg.get("adx", 0))
            atr_pct = int(reg.get("atr_pct", 100))
            regime_green = (
                tod in ("open", "primary")
                and adx >= 20
                and 70 <= atr_pct <= 130
            )
            return 1.25 if regime_green else 1.0
        if score >= 0.80:
            return 1.0
        return 0.5                  # 0.75–0.80: marginal pass, half-size

    if strategy == "fade_the_rip":
        return 1.0 if score >= 0.80 else 0.75

    if strategy == "orb":
        return 0.5                  # half until 25+ shadow trades confirm edge

    return 0.5                      # default: conservative


def _check_strategy_health(strategy: str, min_trades: int = 15) -> tuple[str, float, int]:
    """
    Assess recent performance for a strategy using its last 20 closed trades.

    Returns (status, avg_r, trade_count) where:
      "ok"               — positive expectancy, trade normally
      "degraded"         — avg R < 0 over last N trades (auto-shadow until it recovers)
      "insufficient_data" — fewer than min_trades available (no action taken)
    """
    try:
        conn = duckdb.connect(DB_PATH)
        rows = conn.execute(
            """SELECT r_multiple FROM trade_journal
               WHERE strategy_used = ?
                 AND r_multiple IS NOT NULL
                 AND ai_reasoning != 'partial_exit_t1'
               ORDER BY closed_at DESC LIMIT 20""",
            [strategy],
        ).fetchall()
        conn.close()
    except Exception:
        return "ok", 0.0, 0

    if not rows:
        return "insufficient_data", 0.0, 0

    r_vals = [float(r[0]) for r in rows]
    n      = len(r_vals)
    avg_r  = sum(r_vals) / n

    if n < min_trades:
        return "insufficient_data", avg_r, n

    return ("degraded" if avg_r < 0.0 else "ok"), avg_r, n


def _ap_run_cycle(broker: "PaperBroker") -> None:
    """One scan cycle: check all symbols, execute paper trades on signals."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from utils.event_calendar import is_news_blackout_today

    today_str = datetime.now(_AP_ET).strftime("%Y-%m-%d")

    # ── News/event blackout ──────────────────────────────────────────── #
    _blackout, _blackout_reason = is_news_blackout_today()
    if _blackout:
        _aplog.info(f"CYCLE SKIP: {_blackout_reason}")
        # Still run stop/target checks on open positions — just no new entries.
        for sym in _AP_SYMBOLS:
            try:
                root   = sym.replace("=F", "")
                snap   = get_live_price(root)
                cur_px = snap.get("last_price") if snap else None
                if cur_px:
                    for msg in broker.check_stops_and_targets(root, cur_px):
                        _aplog.info(f"AUTO-CLOSE {sym}: {msg}")
            except Exception:
                pass
        return

    def _check(sym: str) -> dict:
        try:
            root   = sym.replace("=F", "")
            df     = fetch_historical(sym, "5m", days_back=58)
            snap   = get_live_price(root)
            cur_px = snap.get("last_price") if snap else None

            # Ensure ATR is present — fetch_historical returns raw OHLCV.
            # add_atr() here guarantees shadow ticking always has a valid ATR.
            # The engine's live_signal() will also call add_all() internally —
            # that's fine, the extra pass is idempotent and cheap.
            if not df.empty and "atr" not in df.columns:
                from utils.indicators import add_atr as _add_atr
                df = _add_atr(df)

            _now_et             = datetime.now(ZoneInfo("America/New_York"))
            _is_monday          = _now_et.weekday() == 0

            # ── Tick open shadow positions for this symbol ───────────── #
            if cur_px and "atr" in df.columns:
                _atr_now = float(df["atr"].iloc[-1])
                _h, _m   = _now_et.hour, _now_et.minute
                _tick_shadow_positions(root, cur_px, _atr_now, eod=(_h >= 15 and _m >= 55))
            _is_power_hour_open = _now_et.hour == 14 and _now_et.minute < 30
            _index_sym          = sym not in _COMMODITY_SYMBOLS
            _es_sym             = sym == "ES=F"

            sig: dict | None = None

            # ── A+ / commodity engine (primary) ─────────────────────── #
            _aplus_st = _strat_state("aplus")
            if _aplus_st != "disabled":
                _aplus_raw = _make_live_engine(sym).live_signal(df, current_price=cur_px)
                if _aplus_raw:
                    if _aplus_st == "shadow":
                        _aplog.info(
                            f"[SHADOW A+] {sym} {_aplus_raw['direction']} "
                            f"entry={_aplus_raw['entry']:.2f} score={_aplus_raw['score']:.2f} — not trading"
                        )
                    else:
                        sig = _aplus_raw

            # ── ORB: 9:45–10:15 AM, skips Monday ───────────────────── #
            _orb_st = _strat_state("orb")
            if sig is None and _index_sym and not _is_monday and _orb_st != "disabled":
                _orb_raw = ORBEngine(skip_monday=True).live_signal(df, current_price=cur_px)
                if _orb_raw:
                    if _orb_st == "shadow":
                        _aplog.info(
                            f"[SHADOW ORB] {sym} {_orb_raw['direction']} "
                            f"entry={_orb_raw['entry']:.2f} score={_orb_raw['score']:.2f} — not trading"
                        )
                        atr_now = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
                        _open_shadow_position("orb", root, _orb_raw, atr_now)
                    else:
                        sig = _orb_raw

            # ── VWAP PB: ES only, outside power hour ────────────────── #
            _vwap_st = _strat_state("vwap_pullback")
            if _es_sym and not _is_monday and not _is_power_hour_open and _vwap_st != "disabled":
                _vwap_raw = VWAPPullbackEngine(skip_monday=True).live_signal(df, current_price=cur_px)
                if _vwap_raw:
                    if _vwap_st == "shadow":
                        _aplog.info(
                            f"[SHADOW VWAP PB] {sym} {_vwap_raw['direction']} "
                            f"entry={_vwap_raw['entry']:.2f} stop={_vwap_raw['stop']:.2f} "
                            f"score={_vwap_raw['score']:.2f} — not trading (shadow mode)"
                        )
                    elif sig is None:
                        sig = _vwap_raw

            # ── Fade the Rip: short fallback, disabled in TopStep ────── #
            _fade_st = _strat_state("fade_the_rip")
            if (sig is None and _index_sym and not _is_monday and not _is_power_hour_open
                    and not settings.topstep_mode and _fade_st != "disabled"):
                _fade_raw = FadeTheRipEngine().live_signal(df, current_price=cur_px)
                if _fade_raw:
                    if _fade_st == "shadow":
                        _aplog.info(
                            f"[SHADOW FADE RIP] {sym} {_fade_raw['direction']} "
                            f"entry={_fade_raw['entry']:.2f} score={_fade_raw['score']:.2f} — not trading"
                        )
                    else:
                        sig = _fade_raw

            # ── Degradation guard: auto-shadow if last 20 trades are negative ──
            if sig is not None:
                _strat_name = sig.get("strategy", "")
                _dstatus, _davg_r, _dn = _check_strategy_health(_strat_name)
                if _dstatus == "degraded":
                    _aplog.warning(
                        f"[DEGRADED] {_strat_name}: last {_dn} trades avg R={_davg_r:.3f} < 0 "
                        f"— auto-shadow until expectancy recovers"
                    )
                    sig = None

            # ── TopStep mode: enforce tighter score threshold ────────── #
            if sig and settings.topstep_mode and sig.get("score", 1.0) < 0.75:
                _aplog.info(
                    f"SKIP {sym}: TopStep mode — score {sig['score']:.2f} below 0.75 threshold"
                )
                sig = None

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
            # ── Regime tags: logged with every executed trade ──────────── #
            # Also returned as a structured dict so the outer loop can use
            # individual fields for sizing gates and entry blocks.
            _rtags       = ""
            _regime_dict: dict = {}
            try:
                _atr     = float(df["atr"].iloc[-1])
                _atr_avg = float(df["atr_avg20"].iloc[-1]) if "atr_avg20" in df.columns else _atr
                _atr_pct = int(_atr / _atr_avg * 100) if _atr_avg > 0 else 100
                _adx_val = float(df["adx"].iloc[-1]) if "adx" in df.columns else 0.0
                _vix_now = _VIX_CACHE.get("vix", 0.0)
                _h, _m   = _now_et.hour, _now_et.minute
                _tod = (
                    "open"       if _h < 10 or (_h == 10 and _m < 15) else
                    "primary"    if _h < 11 or (_h == 11 and _m < 30) else
                    "lunch"      if _h < 14 else
                    "power_hour" if _h < 15 or (_h == 15 and _m < 30) else
                    "eod"
                )
                _trend = "trending" if _adx_val >= 20 else "ranging"
                _rtags = (
                    f"vix={_vix_now:.1f} atr_pct={_atr_pct} "
                    f"adx={_adx_val:.0f} tod={_tod} market={_trend}"
                )
                _regime_dict = {
                    "vix": _vix_now, "atr_pct": _atr_pct,
                    "adx": _adx_val, "tod": _tod, "market": _trend,
                }
            except Exception:
                pass

            return {
                "sym": sym, "sig": sig, "root": root, "cur_px": cur_px,
                "regime_tags": _rtags, "regime": _regime_dict,
            }
        except Exception as e:
            _aplog.warning(f"{sym} check error: {e}")
            return {"sym": sym, "sig": None, "root": sym, "cur_px": None,
                    "regime_tags": "", "regime": {}}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_check, s): s for s in _AP_SYMBOLS}
        for f in as_completed(futures):
            r = f.result()
            sym, sig, root, cur_px = r["sym"], r["sig"], r["root"], r["cur_px"]
            _rtags  = r.get("regime_tags", "")
            _regime = r.get("regime", {})

            if not sig:
                _aplog.info(f"SKIP {sym}: no setup (A+ long or Fade short — score below threshold or outside window)")
                continue

            # ── Lunch-hour block (12:00–14:00 ET) ───────────────────── #
            # Structural setups (IB retest, ORB, Fade) all degrade in the
            # low-volume, mean-reverting chop of the midday session.
            # Block new entries rather than letting score carry the weight.
            if _regime.get("tod") == "lunch":
                _aplog.info(
                    f"SKIP {sym}: lunch window (12–2pm ET) — "
                    f"structural setups degrade in low-volume chop"
                )
                continue

            # ── TopStep combine: hard daily loss limit ──────────────── #
            if settings.topstep_mode:
                today_pnl = _get_today_pnl()
                if today_pnl <= -settings.topstep_daily_loss_limit:
                    _aplog.warning(
                        f"TOPSTEP HALT {sym}: daily loss ${today_pnl:.2f} >= "
                        f"limit ${settings.topstep_daily_loss_limit:.0f} — no more entries today"
                    )
                    continue
                # Warn at 70% of limit
                warn_threshold = settings.topstep_daily_loss_limit * 0.70
                if today_pnl <= -warn_threshold:
                    _aplog.warning(
                        f"TOPSTEP CAUTION {sym}: daily loss ${today_pnl:.2f} "
                        f"({abs(today_pnl)/settings.topstep_daily_loss_limit*100:.0f}% of limit)"
                    )

            # ── VIX block: skip entries on extreme volatility days ─── #
            _vix_threshold_live = 20.0 if settings.topstep_mode else VIX_BLOCK_THRESHOLD
            _, vix_blocked = _get_vix_today()
            # Re-check with TopStep's tighter VIX threshold
            if settings.topstep_mode and _VIX_CACHE.get("vix", 0.0) >= _vix_threshold_live:
                vix_blocked = True
            if vix_blocked:
                _aplog.info(f"SKIP {sym}: VIX block active — too volatile to trade today")
                continue

            # Re-entry logic:
            #   NQ: up to 2 trades/day regardless of outcome (backtested — improves PF)
            #   All others: 1 trade/day, re-entry only after a loss
            max_trades_today = 1 if settings.topstep_mode else (2 if root == "NQ" else 1)
            with _AP_LOCK:
                traded = _AP_TRADED_TODAY.get(sym, {})
            if traded.get("date") == today_str:
                trade_count = traded.get("count", 1)
                outcome     = traded.get("outcome", "open")
                if outcome == "open":
                    _aplog.info(f"SKIP {sym}: trade still open — no re-entry")
                    continue
                if trade_count >= max_trades_today:
                    _aplog.info(f"SKIP {sym}: {trade_count} trade(s) already taken today (max {max_trades_today})")
                    continue
                _aplog.info(f"ALLOW re-entry {sym}: {trade_count} trade(s) done, taking setup #{trade_count + 1}")

            _aplog.info(
                f"SIGNAL {sig['direction']} {sym} | entry={sig['entry']} "
                f"stop={sig['stop']} target={sig['target']} score={sig['score']}"
            )

            # ── Dollar risk cap — final backstop against extreme ATR days ──
            # Even if the engine ATR-spike filter passes, cap absolute dollar risk
            # per contract so a single stop can never blow the daily limit.
            _CONTRACT_MULT = {"NQ": 20, "ES": 50, "GC": 10, "CL": 100}
            _MAX_DOLLAR_RISK = {"NQ": 1000, "ES": 800}   # per contract per trade
            _mult = _CONTRACT_MULT.get(root, 20)
            _risk_pts   = abs(sig["entry"] - sig["stop"])
            _dollar_risk = _risk_pts * _mult
            _max_risk = _MAX_DOLLAR_RISK.get(root, 1200)
            if _dollar_risk > _max_risk:
                _aplog.warning(
                    f"SKIP {sym}: dollar risk ${_dollar_risk:.0f} > cap ${_max_risk} "
                    f"(stop={_risk_pts:.1f}pts × ${_mult}/pt) — ATR too wide today"
                )
                continue

            # ── TopStep: 2 consecutive losses today = stop trading ────── #
            # Protects the trailing max loss limit from being blown in a single
            # bad day. Reset each morning with the daily reset.
            if settings.topstep_mode:
                with _AP_LOCK:
                    _all_today = [
                        v for v in _AP_TRADED_TODAY.values()
                        if v.get("date") == today_str
                    ]
                consec_losses = sum(1 for v in _all_today if v.get("outcome") == "loss")
                if consec_losses >= 2:
                    _aplog.warning(
                        f"TOPSTEP SKIP {sym}: {consec_losses} losses today — "
                        f"stopping entries to protect trailing max loss"
                    )
                    continue

            # Correlation block: only one index future open at a time
            if root in _INDEX_ROOTS:
                open_index = {p.symbol for p in broker.open_positions if p.symbol in _INDEX_ROOTS}
                if open_index:
                    _aplog.info(f"SKIP {sym}: correlation block — {open_index} index position already open")
                    continue

            # ── Profit-protection guard ──────────────────────────────── #
            # If we're up 3R+ on the day (account-wide), protect the gains.
            # Per-symbol 2R guard: if that symbol made 2R+ today, skip it.
            _base_1r = broker.account_balance * (settings.max_risk_per_trade_pct / 100)
            _daily_pnl = broker.risk_manager.daily_pnl
            if _daily_pnl >= 3.0 * _base_1r:
                _aplog.info(
                    f"SKIP {sym}: daily PnL +${_daily_pnl:.0f} ≥ 3R — protecting profits, done for today"
                )
                continue
            # Per-symbol: query today's realised PnL for this root
            try:
                import duckdb as _ddb
                _today_iso = datetime.now(_AP_ET).strftime("%Y-%m-%d")
                _sym_pnl_row = _ddb.connect(DB_PATH).execute(
                    "SELECT COALESCE(SUM(pnl),0) FROM trade_journal "
                    "WHERE symbol=? AND closed_at::DATE=? AND ai_reasoning!='partial_exit_t1'",
                    [root, _today_iso],
                ).fetchone()
                _sym_pnl = float(_sym_pnl_row[0]) if _sym_pnl_row else 0.0
                if _sym_pnl >= 2.0 * _base_1r:
                    _aplog.info(
                        f"SKIP {sym}: symbol PnL +${_sym_pnl:.0f} ≥ 2R — done on {root} today"
                    )
                    continue
            except Exception:
                pass   # DB not ready — skip guard, don't block trade

            # ── Dynamic sizing: score-tier + regime-gated risk multiplier ── #
            _risk_mult = _score_to_risk_mult(sig, _regime)
            _aplog.info(
                f"SIZE {sym}: strategy={sig.get('strategy')} score={sig['score']:.2f} "
                f"tod={_regime.get('tod','?')} adx={_regime.get('adx',0):.0f} "
                f"atr_pct={_regime.get('atr_pct',100)} → risk_mult={_risk_mult:.2f}×"
            )

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
                    + (f" | {_rtags}" if _rtags else "")
                ),
                risk_multiplier=_risk_mult,
            )
            if ok:
                with _AP_LOCK:
                    prev_count = _AP_TRADED_TODAY.get(sym, {}).get("count", 0)
                    _AP_TRADED_TODAY[sym] = {"date": today_str, "outcome": "open", "count": prev_count + 1}
                    _save_ap_state()
                _aplog.info(f"TRADE OPENED {sym}: {msg} (id={pos.id if pos else '?'})")
                _phone(
                    title    = f"🤖 Trade Entered — {root}",
                    body     = (
                        f"{sig['direction']} @ {sig['entry']:.2f}\n"
                        f"Stop: {sig['stop']:.2f}  |  Target: {sig['target']:.2f}\n"
                        f"Risk: ${_dollar_risk:.0f} (1R)  |  Score: {sig['score']:.2f}\n"
                        f"{sig.get('regime','')}"
                    ),
                    tags     = "robot",
                    priority = "high",
                )
            else:
                _aplog.warning(f"TRADE REJECTED {sym}: {msg}")


def _send_weekly_summary(broker: "PaperBroker") -> None:
    """Send Friday end-of-week P&L summary (Mon–Fri) to phone."""
    try:
        import duckdb
        now_et = datetime.now(_AP_ET)
        days_since_mon = now_et.weekday()  # 0=Mon … 4=Fri
        week_start = (now_et.replace(hour=0, minute=0, second=0, microsecond=0)
                      - timedelta(days=days_since_mon)).strftime("%Y-%m-%d")
        conn = duckdb.connect(DB_PATH)
        rows = conn.execute(
            "SELECT pnl, r_multiple, symbol FROM trade_journal "
            "WHERE closed_at::DATE >= ? AND ai_reasoning != 'partial_exit_t1' "
            "ORDER BY closed_at",
            [week_start],
        ).fetchall()
        conn.close()

        n = len(rows)
        if n == 0:
            _aplog.info("Weekly wrap: no trades this week — skipping push")
            return

        total_pnl  = sum(r[0] or 0 for r in rows)
        wins       = sum(1 for r in rows if (r[0] or 0) > 0)
        gross_win  = sum(r[0] for r in rows if (r[0] or 0) > 0)
        gross_loss = abs(sum(r[0] for r in rows if (r[0] or 0) <= 0))
        pf         = round(gross_win / gross_loss, 2) if gross_loss > 0 else 9.99
        wr         = wins / n * 100
        avg_r      = sum(r[1] or 0 for r in rows) / n
        bal        = broker.account_balance

        emoji = "✅" if total_pnl > 0 else "❌"
        title = f"{emoji} Week — ${total_pnl:+.2f} | {wr:.0f}% WR"
        body  = (
            f"{n} trade{'s' if n>1 else ''} | {wins}W {n-wins}L | PF {pf:.2f} | Avg {avg_r:+.2f}R\n"
            f"Balance: ${bal:,.2f}"
        )
        if settings.topstep_mode:
            combine_profit = bal - settings.paper_account_size
            body += (
                f"\nCombine P&L: ${combine_profit:+,.0f} / ${settings.topstep_profit_target:,.0f} target"
            )

        # Weekly wrap — log only. Daily report covers day-to-day results.
        _aplog.info(f"Weekly wrap: {title} | {body[:200]}")
    except Exception as e:
        _aplog.warning(f"Weekly summary failed: {e}")


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
        conn  = duckdb.connect(DB_PATH)
        rows  = conn.execute("""
            SELECT pnl, direction, symbol, r_multiple FROM trade_journal
            WHERE closed_at::DATE = ?
              AND ai_reasoning != 'partial_exit_t1'
            ORDER BY closed_at
        """, [today]).fetchall()
        # All-time stats for streak — exclude partial rows so each trade counts once
        all_rows = conn.execute("""
            SELECT pnl FROM trade_journal
            WHERE ai_reasoning != 'partial_exit_t1'
            ORDER BY closed_at DESC LIMIT 20
        """).fetchall()
        conn.close()

        n_trades  = len(rows)
        total_pnl = sum(r[0] or 0 for r in rows)
        wins      = sum(1 for r in rows if (r[0] or 0) > 0)
        bal       = broker.account_balance

        # Current win/loss streak
        streak = 0
        streak_word = ""
        if all_rows:
            stype = 1 if (all_rows[0][0] or 0) > 0 else -1
            for row in all_rows:
                if ((row[0] or 0) > 0) == (stype == 1):
                    streak += 1
                else:
                    break
            streak_word = f"{streak}W streak" if stype == 1 else f"{streak}L streak"

        if n_trades == 0:
            title = "📊 No Trades Today"
            body  = f"Market closed flat. Balance: ${bal:,.2f}"
        else:
            wr = wins / n_trades * 100
            emoji = "✅" if total_pnl > 0 else "❌"
            trade_lines = "\n".join(
                f"  {'✅' if (r[0] or 0)>0 else '❌'} {r[2]} {r[1]} ${r[0]:+.2f} ({r[3]:+.2f}R)"
                for r in rows
            )
            title = f"{emoji} {today} — ${total_pnl:+.2f}"
            body  = (
                f"{n_trades} trade{'s' if n_trades>1 else ''} | {wins}W {n_trades-wins}L | {wr:.0f}% WR\n"
                f"Balance: ${bal:,.2f}"
                + (f" | {streak_word}" if streak_word else "") + "\n"
                + trade_lines
            )

        _phone(
            title    = title,
            body     = body[:500],
            tags     = "bar_chart",
            priority = "default",
        )
        _aplog.info(f"Daily summary sent: {title}")
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

            # ── Morning session open — log only, no push notification ─── #
            # Startup sends the one-and-only "server is alive" push.
            # A 9:25 AM ping on top of that creates duplicate alerts every
            # time the server restarts during market hours.
            if (now_et.weekday() < 5 and now_et.hour == 9 and now_et.minute >= 25):
                if today_str not in _AP_HEALTH_SENT:
                    _AP_HEALTH_SENT.add(today_str)
                    _save_ap_state()
                    open_pos = broker.open_positions
                    _aplog.info(
                        f"Session open | Armed: {_AP_ARMED} | "
                        f"Open: {len(open_pos)} | Balance: ${broker.account_balance:,.0f}"
                    )

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

            # ── Weekly summary on Friday at 4:15 PM ET ────────────────── #
            if (now_et.weekday() == 4 and now_et.hour == 16 and now_et.minute >= 15):
                week_key = f"weekly-{today_str}"
                if week_key not in _AP_SUMMARY_SENT:
                    _AP_SUMMARY_SENT.add(week_key)
                    _save_ap_state()
                    _send_weekly_summary(broker)

            time.sleep(_AP_INTERVAL)
        except Exception as e:
            _aplog.error(f"Autopilot loop error: {e}")
            time.sleep(60)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _broker, _agent
    # ── DB init + seed (moved from module level to avoid import side-effects) ──
    init_db()
    try:
        from seed_trades import seed as _seed_trades
        _seed_trades()
    except Exception:
        pass
    # ── Broker + agent (moved from module level for same reason) ──────────────
    _broker = _make_broker()
    _agent  = TradingAgent()
    # ── Enforce single autopilot instance, then start background loop ─────────
    _enforce_single_instance()
    t = threading.Thread(
        target=_ap_background_loop,
        args=(lambda: _broker,),
        daemon=True,
        name="server-autopilot",
    )
    t.start()
    _aplog.info("Server autopilot thread launched")

    # ── Single startup notification — one ping per deploy/restart ──────
    # All other "server alive" health pings have been removed so this is
    # the only unsolicited push you'll receive about the app itself.
    try:
        open_pos = _broker.open_positions if _broker else []
        _phone(
            title    = "Rival Automations — Online",
            body     = (
                f"Balance: ${_broker.account_balance:,.0f} | "
                f"Armed: {_AP_ARMED} | "
                f"Open positions: {len(open_pos)}"
            ),
            tags     = "rocket",
            priority = "default",
        )
    except Exception:
        pass

    yield
    # Clean up PID file on graceful shutdown
    try:
        if _AP_PID_FILE.exists():
            _AP_PID_FILE.unlink()
    except Exception:
        pass


app = FastAPI(title="Rival Automations — Trading Terminal", version="2.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session-cookie auth ───────────────────────────────────────────── #
# Set DASHBOARD_PASS in Railway environment variables.
# If not set the app runs open (local dev unchanged).
# Users log in once via the dashboard overlay; a 30-day cookie keeps them in.
import os, secrets as _secrets
from fastapi.responses import JSONResponse as _JSONResponse
_DASH_PASS    = os.environ.get("DASHBOARD_PASS", "")
_AUTH_ENABLED = bool(_DASH_PASS)
_SESSIONS: set[str] = set()   # in-memory tokens; cleared on redeploy (that's fine)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _Request
from starlette.responses import Response as _Response

_AUTH_BYPASS = {"/api/health", "/api/login"}

class _SessionAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: _Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        # Static files and the root HTML page are always public —
        # the login overlay inside the HTML handles the auth gate.
        if not path.startswith("/api"):
            return await call_next(request)
        # Public API endpoints (login + health check)
        if path in _AUTH_BYPASS:
            return await call_next(request)
        # All other /api/* routes require a valid session cookie
        if request.cookies.get("ra_session") in _SESSIONS:
            return await call_next(request)
        return _Response("Unauthorized", status_code=401)

if _AUTH_ENABLED:
    app.add_middleware(_SessionAuthMiddleware)

# ── Login endpoint ────────────────────────────────────────────────── #
@app.post("/api/login")
def login(body: dict = Body(...)):
    if not _AUTH_ENABLED:
        return {"ok": True}
    if body.get("password") != _DASH_PASS:
        raise HTTPException(status_code=401, detail="Wrong password")
    token = _secrets.token_hex(32)
    _SESSIONS.add(token)
    resp = _JSONResponse({"ok": True})
    resp.set_cookie(
        "ra_session", token,
        max_age=30 * 24 * 3600,   # 30 days
        httponly=True,
        samesite="strict",
    )
    return resp

@app.post("/api/logout")
def logout(request: _Request):
    token = request.cookies.get("ra_session", "")
    _SESSIONS.discard(token)
    resp = _JSONResponse({"ok": True})
    resp.delete_cookie("ra_session")
    return resp

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

# ── Shared in-memory state — initialised in lifespan, not at import ── #
_broker: "PaperBroker | None" = None
_agent:  "TradingAgent | None" = None
_last_agg: dict = {}

# ─────────────────────────────────────────────────────────────────────
# SHADOW POSITION TRACKING
# Tracks ORB (and any future shadow-mode strategy) signals to completion
# using TradeSimulator so we can evaluate promotion readiness.
# ─────────────────────────────────────────────────────────────────────
_SHADOW_SIMS: dict[str, tuple[TradeSimulator, dict]] = {}
# key   = shadow position id (uuid)
# value = (TradeSimulator instance, metadata dict with symbol/strategy)

_SHADOW_LOCK = threading.Lock()


def _open_shadow_position(strategy: str, symbol: str, sig: dict, atr: float) -> None:
    """
    Create a TradeSimulator for a shadow signal and persist it to shadow_signals.

    sig keys: direction, entry, stop, target_1, target (=T2), score, strategy, regime.
    """
    import uuid as _uuid
    try:
        sid = str(_uuid.uuid4())
        direction = sig["direction"]
        d_int = 1 if direction == "BUY" else -1
        entry = float(sig["entry"])
        stop  = float(sig["stop"])
        t1    = float(sig.get("target_1") or sig.get("t1") or (entry + d_int * abs(entry - stop)))
        t2    = float(sig.get("target") or sig.get("t2") or t1)

        sim = TradeSimulator(
            symbol=symbol,
            direction=d_int,
            entry=entry,
            stop=stop,
            t1=t1,
            t2=t2,
            qty=1.0,
            apply_costs=False,
        )

        meta = {"symbol": symbol, "strategy": strategy}

        with _SHADOW_LOCK:
            _SHADOW_SIMS[sid] = (sim, meta)

        now_ts = datetime.now(timezone.utc).isoformat()
        conn = duckdb.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO shadow_signals
                (id, strategy, symbol, direction, entry, original_stop, current_stop,
                 t1, t2, score, signal_time, status, t1_exited, bars_managed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', FALSE, 0)
            """,
            [sid, strategy, symbol, direction, entry, stop, stop, t1, t2,
             float(sig.get("score", 0.0)), now_ts],
        )
        conn.close()
        _aplog.info(
            f"[SHADOW OPEN] id={sid} {strategy} {symbol} {direction} "
            f"entry={entry:.2f} stop={stop:.2f} t1={t1:.2f} t2={t2:.2f} "
            f"score={sig.get('score', 0.0):.2f}"
        )
    except Exception as e:
        _aplog.warning(f"_open_shadow_position failed: {e}")


def _tick_shadow_positions(symbol: str, current_price: float, atr: float, eod: bool) -> None:
    """
    Advance all open shadow sims for this symbol by one price tick.
    Handles partial (T1) and close events, updating shadow_signals accordingly.
    Thread-safe via _SHADOW_LOCK.
    """
    try:
        with _SHADOW_LOCK:
            matching_ids = [
                sid for sid, (_, meta) in _SHADOW_SIMS.items()
                if meta["symbol"] == symbol
            ]

        for sid in matching_ids:
            try:
                with _SHADOW_LOCK:
                    if sid not in _SHADOW_SIMS:
                        continue
                    sim, meta = _SHADOW_SIMS[sid]

                tick = sim.tick(
                    hi=current_price,
                    lo=current_price,
                    cl=current_price,
                    atr=atr,
                    eod=eod,
                    timeout=False,
                )

                conn = duckdb.connect(DB_PATH)
                if tick is None:
                    conn.execute(
                        "UPDATE shadow_signals SET bars_managed = bars_managed + 1 WHERE id = ?",
                        [sid],
                    )
                elif tick.kind == "partial":
                    conn.execute(
                        """
                        UPDATE shadow_signals
                        SET t1_exited    = TRUE,
                            current_stop = entry,
                            bars_managed = bars_managed + 1
                        WHERE id = ?
                        """,
                        [sid],
                    )
                    _aplog.info(
                        f"[SHADOW T1] id={sid} {meta['symbol']} partial exit "
                        f"@ {tick.exit_price:.2f} (stop → BE)"
                    )
                elif tick.kind == "close":
                    # Use tick.total_pnl — it includes the T1 partial exit PnL
                    # so pnl_pts and r_multiple are consistent with each other.
                    # (Computing from exit_price alone ignores the partial.)
                    pnl_pts = tick.total_pnl
                    r_mult  = tick.r_multiple
                    closed_ts = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        UPDATE shadow_signals
                        SET status      = 'closed',
                            exit_price  = ?,
                            exit_reason = ?,
                            pnl_pts     = ?,
                            r_multiple  = ?,
                            closed_at   = ?,
                            bars_managed = bars_managed + 1
                        WHERE id = ?
                        """,
                        [tick.exit_price, tick.reason, round(pnl_pts, 4),
                         round(r_mult, 3), closed_ts, sid],
                    )
                    with _SHADOW_LOCK:
                        _SHADOW_SIMS.pop(sid, None)
                    _aplog.info(
                        f"[SHADOW CLOSE] id={sid} {meta['symbol']} {tick.reason} "
                        f"@ {tick.exit_price:.2f} pnl={pnl_pts:.2f}pts R={r_mult:.3f}"
                    )
                conn.close()
            except Exception as e:
                _aplog.warning(f"_tick_shadow_positions inner error sid={sid}: {e}")
    except Exception as e:
        _aplog.warning(f"_tick_shadow_positions outer error symbol={symbol}: {e}")


# ─────────────────────────────────────────────────────────────────────
# HEALTH / INFO
# ─────────────────────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok"}

@app.api_route("/api/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat(), "paper": settings.is_paper}


@app.get("/api/strategy-health")
def strategy_health():
    """
    Return per-strategy state + recent performance health.
    Used by the dashboard to display strategy status chips.
    """
    strategies = [
        ("aplus",        "A+ IB Retest",   settings.strategy_aplus),
        ("orb",          "ORB",             settings.strategy_orb),
        ("fade_the_rip", "Fade the Rip",    settings.strategy_fade_rip),
        ("vwap_pullback", "VWAP Pullback",  settings.strategy_vwap_pb),
    ]
    out = []
    for name, label, state in strategies:
        dstatus, avg_r, n_trades = _check_strategy_health(name)
        effective = "shadow" if dstatus == "degraded" and state == "paper" else state
        out.append({
            "name":      name,
            "label":     label,
            "state":     state,           # config state
            "effective": effective,       # may be demoted to shadow by degradation
            "health":    dstatus,         # "ok" | "degraded" | "insufficient_data"
            "avg_r":     round(avg_r, 3),
            "n_trades":  n_trades,
        })
    return {"strategies": out}


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


@app.get("/api/vix")
def get_vix_level():
    """Return today's cached VIX level and block status."""
    vix, blocked = _get_vix_today()
    return {
        "vix":        round(vix, 2),
        "blocked":    blocked,
        "threshold":  20.0 if settings.topstep_mode else VIX_BLOCK_THRESHOLD,
        "topstep_mode": settings.topstep_mode,
    }


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
                min_score=max(req.min_score, 0.75),   # enforce minimum quality floor
                allow_short=req.allow_short,
                require_macro_confirm=False,
                skip_monday=True,
                skip_power_hour=True,      # IB is stale by 2 PM — no power hour entries
                max_atr_multiple=1.5,      # filter extreme volatility spike days
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

# Backtest results (60d Mar-Apr 2026, high-volatility tariff regime, VIX>=22 blocked live):
#   NQ=F  A+ | ORB PF 1.06 (44% WR) | VWAP PB: no edge in current regime (disabled)
#   ES=F  A+ | ORB PF 0.89 (40% WR) | VWAP PB PF 3.12 (ES only, T1=1.5R, trail after T2)
#   GC=F  ORB 5m / 1h
#   Signal cascade: A+ → ORB (9:45-10:15, NQ+ES) → VWAP PB (ES only) → Fade the Rip
AP_DEFAULT_SYMBOLS = ["NQ=F", "ES=F"]

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

            # Cascade: ORB → VWAP PB → Fade the Rip (index futures only)
            if req.mode in ("aplus", "5m") and sig is None and sym not in _COMMODITY_SYMBOLS:
                _net = datetime.now(ZoneInfo("America/New_York"))
                _mon = _net.weekday() == 0
                _pho = _net.hour == 14 and _net.minute < 30
                if not _mon:
                    sig = ORBEngine(skip_monday=True).live_signal(df, current_price=cur_px)
                if sig is None and sym == "ES=F" and not _mon and not _pho:
                    sig = VWAPPullbackEngine(skip_monday=True).live_signal(df, current_price=cur_px)
                if sig is None and not _mon and not _pho:
                    sig = FadeTheRipEngine().live_signal(df, current_price=cur_px)

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
# QUICK PERFORMANCE SNAPSHOT  (used by dashboard stats panel)
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/performance/snapshot")
def performance_snapshot():
    """Return 30-day PF/WR for each active strategy across NQ and ES."""
    results = []
    try:
        for sym in ["NQ=F", "ES=F"]:
            df = fetch_historical(sym, "5m", days_back=30)

            # A+
            try:
                r = BacktestEngineAPlus(min_adx=18.0, min_score=0.75, allow_short=False,
                                        require_macro_confirm=False, skip_monday=True,
                                        skip_power_hour=True, max_atr_multiple=1.5).run(df, symbol=sym)
                if r.trades:
                    wins = sum(1 for t in r.trades if t.r_multiple >= 1.0)
                    gw = sum(t.pnl_pts for t in r.trades if t.pnl_pts > 0)
                    gl = abs(sum(t.pnl_pts for t in r.trades if t.pnl_pts < 0))
                    results.append({
                        "symbol": sym, "strategy": "A+ IB Retest",
                        "trades": len(r.trades),
                        "wr": round(wins / len(r.trades) * 100, 1),
                        "pf": round(gw / gl, 2) if gl > 0 else 99.0,
                        "total_r": round(sum(t.r_multiple for t in r.trades), 1),
                    })
            except Exception:
                pass

            # ORB
            try:
                r = ORBEngine(allow_short=False).run(df, symbol=sym)
                if r.trades:
                    wins = sum(1 for t in r.trades if t.r_multiple >= 1.0)
                    gw = sum(t.pnl_pts for t in r.trades if t.pnl_pts > 0)
                    gl = abs(sum(t.pnl_pts for t in r.trades if t.pnl_pts < 0))
                    results.append({
                        "symbol": sym, "strategy": "ORB",
                        "trades": len(r.trades),
                        "wr": round(wins / len(r.trades) * 100, 1),
                        "pf": round(gw / gl, 2) if gl > 0 else 99.0,
                        "total_r": round(sum(t.r_multiple for t in r.trades), 1),
                    })
            except Exception:
                pass

            # VWAP PB (ES only)
            if sym == "ES=F":
                try:
                    r = VWAPPullbackEngine().run(df, symbol=sym)
                    if r.trades:
                        wins = sum(1 for t in r.trades if t.r_multiple >= 1.0)
                        gw = sum(t.pnl_pts for t in r.trades if t.pnl_pts > 0)
                        gl = abs(sum(t.pnl_pts for t in r.trades if t.pnl_pts < 0))
                        results.append({
                            "symbol": sym, "strategy": "VWAP PB",
                            "trades": len(r.trades),
                            "wr": round(wins / len(r.trades) * 100, 1),
                            "pf": round(gw / gl, 2) if gl > 0 else 99.0,
                            "total_r": round(sum(t.r_multiple for t in r.trades), 1),
                        })
                except Exception:
                    pass

    except Exception as e:
        return {"results": [], "error": str(e)}

    return {"results": results, "days": 30}


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
        "ibkr_host":                getattr(settings, "ibkr_host", None),
        "ibkr_port":                getattr(settings, "ibkr_port", None),
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
    vix_level, vix_blocked = _VIX_CACHE.get("vix", 0.0), _VIX_CACHE.get("blocked", False)
    return {
        "armed":        _AP_ARMED,
        "symbols":      _AP_SYMBOLS,
        "interval_s":   _AP_INTERVAL,
        "market_hours": _ap_is_market_hours(),
        "traded_today": _AP_TRADED_TODAY,
        "vix":          vix_level,
        "vix_blocked":  vix_blocked,
        "vix_threshold": VIX_BLOCK_THRESHOLD,
    }

@app.post("/api/autopilot/arm")
def ap_arm(body: dict = Body(...)):
    global _AP_ARMED
    _AP_ARMED = bool(body.get("armed", True))
    _aplog.info(f"Server autopilot {'ARMED' if _AP_ARMED else 'DISARMED'} via API")
    return {"ok": True, "armed": _AP_ARMED}




# ─────────────────────────────────────────────────────────────────────
# TOPSTEP COMBINE MODE
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/topstep/state")
def topstep_state():
    """Return TopStep combine mode state and progress metrics."""
    today_pnl    = _get_today_pnl()
    limit        = settings.topstep_daily_loss_limit
    target       = settings.topstep_profit_target
    bal          = _broker.account_balance
    baseline     = settings.paper_account_size
    total_profit = bal - baseline

    # How close to daily limit (0.0 = fine, 1.0 = at limit)
    daily_loss_pct = max(0.0, -today_pnl / limit) if limit > 0 else 0.0
    # Progress toward profit target (0.0 → 1.0)
    profit_progress = max(0.0, min(1.0, total_profit / target)) if target > 0 else 0.0

    return {
        "enabled":            settings.topstep_mode,
        "daily_loss_limit":   limit,
        "profit_target":      target,
        "today_pnl":          round(today_pnl, 2),
        "total_profit":       round(total_profit, 2),
        "daily_loss_pct":     round(daily_loss_pct, 3),
        "profit_progress":    round(profit_progress, 3),
        "halted_today":       today_pnl <= -limit,
        "account_balance":    round(bal, 2),
        "score_threshold":    0.75,
        "vix_block_threshold": 20.0 if settings.topstep_mode else VIX_BLOCK_THRESHOLD,
    }


@app.post("/api/topstep/toggle")
def topstep_toggle(body: dict = Body(...)):
    """Enable or disable TopStep combine mode."""
    enabled = bool(body.get("enabled", True))
    try:
        object.__setattr__(settings, "topstep_mode", enabled)
    except Exception:
        pass
    # Persist to .env
    pairs = _read_env()
    pairs["TOPSTEP_MODE"] = "true" if enabled else "false"
    _write_env(pairs)
    _aplog.info(f"TopStep mode {'ENABLED' if enabled else 'DISABLED'} via API")
    return {"ok": True, "enabled": enabled}

# ─────────────────────────────────────────────────────────────────────
# PAPER ACCOUNT RESET
# ─────────────────────────────────────────────────────────────────────
@app.post("/api/admin/reset-paper")
def reset_paper_account():
    """
    Wipe all journal and position history and reset the paper account to
    the configured starting balance ($100K by default).

    This is a hard reset — all trade history is permanently deleted.
    Use only when you want a clean slate (e.g. after fixing accounting bugs).
    """
    if _broker is None:
        raise HTTPException(503, "Broker not initialised")

    try:
        # 1. Clear all in-memory positions (no DB update — we're deleting the table)
        with _broker._close_lock:
            _broker._positions.clear()

        # 2. Wipe DB — positions first (trade_journal may reference them)
        conn = duckdb.connect(DB_PATH)
        conn.execute("DELETE FROM trade_journal")
        conn.execute("DELETE FROM positions")
        conn.close()

        # 3. Reset broker financial state
        _broker.account_balance = settings.paper_account_size
        _broker.risk_manager._daily_realized_pnl = 0.0
        _broker.risk_manager._open_positions     = 0
        _broker.risk_manager._halted             = False

        # 4. Clear autopilot operational state
        with _AP_LOCK:
            _AP_TRADED_TODAY.clear()
        today_str = datetime.now(_AP_ET).strftime("%Y-%m-%d")
        _AP_SUMMARY_SENT.discard(today_str)
        _AP_HEALTH_SENT.discard(today_str)
        _AP_EOD_SENT.discard(today_str)
        _AP_RESET_SENT.discard(today_str)
        _save_ap_state()

        _aplog.info(
            f"Paper account reset — journal cleared, "
            f"balance reset to ${settings.paper_account_size:,.0f}"
        )
        _phone(
            title = "Paper Account Reset ✓",
            body  = f"Fresh start at ${settings.paper_account_size:,.0f}. All history cleared.",
            tags  = "white_check_mark",
        )
        return {
            "ok":              True,
            "account_balance": settings.paper_account_size,
            "message":         "Paper account reset. All journal and position data deleted.",
        }
    except Exception as e:
        raise HTTPException(500, f"Reset failed: {e}")


@app.get("/api/journal")
def get_journal(limit: int = 200):
    try:
        conn = duckdb.connect(DB_PATH)
        jdf  = conn.execute(f"""
            SELECT symbol, direction, entry_price, exit_price, qty,
                   pnl, r_multiple, strategy_used, opened_at, closed_at,
                   ai_reasoning,
                   tod, market, atr_pct, adx, vix
            FROM trade_journal ORDER BY opened_at DESC LIMIT {limit}
        """).df()
        conn.close()

        # Extract composite score from ai_reasoning (e.g. "score=0.76")
        import re as _re
        def _parse_score(text):
            if not isinstance(text, str):
                return None
            m = _re.search(r"score=([0-9.]+)", text)
            return round(float(m.group(1)), 2) if m else None
        jdf["composite_score"] = jdf["ai_reasoning"].apply(_parse_score)

        if len(jdf) == 0:
            return {"trades": [], "stats": None}

        wins   = jdf[jdf["pnl"] > 0]
        losses = jdf[jdf["pnl"] <= 0]

        # Equity curve (oldest → newest)
        cum_pnl = jdf["pnl"].iloc[::-1].cumsum()

        # ── Extra stats ──────────────────────────────────────────────
        # Win/loss streak
        results = [1 if p > 0 else -1 for p in jdf["pnl"].iloc[::-1]]
        cur_streak = win_streak = loss_streak = 0
        streak_type = results[-1] if results else 1
        for r in reversed(results):
            if r == streak_type:
                cur_streak += 1
            else:
                break
        for r in results:
            if r == 1:
                win_streak += 1
            else:
                break
        loss_run = 0
        for r in results:
            if r == -1:
                loss_run += 1
            else:
                break

        # Best / worst single trade
        best_trade  = round(float(jdf["pnl"].max()), 2)
        worst_trade = round(float(jdf["pnl"].min()), 2)

        # Best / worst day + daily breakdown for bar chart
        import pandas as pd
        jdf["_date"] = pd.to_datetime(jdf["opened_at"]).dt.date
        by_day = jdf.groupby("_date")["pnl"].sum().sort_index()   # oldest → newest
        best_day  = round(float(by_day.max()), 2) if len(by_day) else 0
        worst_day = round(float(by_day.min()), 2) if len(by_day) else 0
        # daily_pnl: [{date, pnl, trades, wins}] for bar chart
        by_day_trades = jdf.groupby("_date").agg(
            pnl   = ("pnl", "sum"),
            trades= ("pnl", "count"),
            wins  = ("pnl", lambda x: (x > 0).sum()),
        ).sort_index()
        daily_pnl = [
            {
                "date":   str(d),
                "pnl":    round(float(row["pnl"]), 2),
                "trades": int(row["trades"]),
                "wins":   int(row["wins"]),
            }
            for d, row in by_day_trades.iterrows()
        ]

        # Avg R
        avg_r = round(float(jdf["r_multiple"].mean()), 2) if len(jdf) else 0

        overall_wr = round(len(wins)/len(jdf), 3)

        # ── Per-strategy breakdown ────────────────────────────────────
        strategy_stats = []
        for strat, grp in jdf.groupby("strategy_used"):
            s_wins   = grp[grp["pnl"] > 0]
            s_losses = grp[grp["pnl"] <= 0]
            s_gw     = float(s_wins["pnl"].sum())
            s_gl     = abs(float(s_losses["pnl"].sum()))
            strategy_stats.append({
                "strategy":      strat or "unknown",
                "trades":        len(grp),
                "win_rate":      round(len(s_wins) / len(grp), 3),
                "total_pnl":     round(float(grp["pnl"].sum()), 2),
                "avg_r":         round(float(grp["r_multiple"].mean()), 2),
                "profit_factor": round(s_gw / s_gl, 2) if s_gl > 0 else 9.99,
            })
        strategy_stats.sort(key=lambda x: x["total_pnl"], reverse=True)

        # ── TopStep combine readiness (only computed when mode is on) ──
        topstep_readiness = None
        if settings.topstep_mode:
            unique_days    = int(jdf["_date"].nunique())
            bal            = _broker.account_balance
            combine_profit = round(bal - settings.paper_account_size, 2)
            worst_day_val  = round(float(by_day.min()), 2)
            topstep_readiness = {
                "days_traded":       unique_days,
                "combine_profit":    combine_profit,
                "profit_target":     settings.topstep_profit_target,
                "daily_loss_limit":  2000.0,   # TopStep official limit; bot halts at 1000
                "worst_day":         worst_day_val,
                "win_rate":          overall_wr,
                "balance":           round(bal, 2),
                # Requirement checklist
                "profit_met":        combine_profit >= settings.topstep_profit_target,
                "days_met":          unique_days >= 10,
                "wr_met":            overall_wr >= 0.55,
                "daily_loss_met":    worst_day_val >= -2000.0,
            }

        # NaN → None: DuckDB returns NULL numerics as float NaN in pandas.
        # NaN is not JSON-serialisable; None serialises as null.
        # This affects the new regime columns (tod/market/atr_pct/adx/vix)
        # for older journal rows that pre-date the schema migration.
        import math as _math
        _trades_df   = jdf.drop(columns=["_date", "ai_reasoning"])
        _trades_list = [
            {k: (None if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)) else v)
             for k, v in row.items()}
            for row in _trades_df.to_dict("records")
        ]

        return {
            "trades": _trades_list,
            "equity_curve": [round(v, 2) for v in cum_pnl.values],
            "daily_pnl": daily_pnl,
            "strategy_stats": strategy_stats,
            "topstep_readiness": topstep_readiness,
            "stats": {
                "total_trades":  len(jdf),
                "win_rate":      overall_wr,
                "total_pnl":     round(float(jdf["pnl"].sum()), 2),
                "avg_win":       round(float(wins["pnl"].mean()), 2)   if len(wins)   else 0,
                "avg_loss":      round(float(losses["pnl"].mean()), 2) if len(losses) else 0,
                "profit_factor": (
                    round(float(wins["pnl"].sum()) / abs(float(losses["pnl"].sum())), 3)
                    if losses["pnl"].sum() != 0 else 9999
                ),
                "avg_r":         avg_r,
                "best_trade":    best_trade,
                "worst_trade":   worst_trade,
                "best_day":      best_day,
                "worst_day":     worst_day,
                "current_streak": cur_streak,
                "streak_type":   "win" if streak_type == 1 else "loss",
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────
# SHADOW PERFORMANCE  — per-strategy stats for promotion decisions
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/shadow-performance")
def shadow_performance():
    """
    Return shadow trade stats per strategy.

    Promotion criteria: total >= 25 AND profit_factor >= 1.8 AND avg_r >= 0.25.
    """
    try:
        conn = duckdb.connect(DB_PATH)
        rows = conn.execute(
            """
            SELECT strategy, pnl_pts, r_multiple
            FROM shadow_signals
            WHERE status = 'closed'
            """
        ).fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"shadow_performance DB error: {e}")

    # Aggregate per strategy
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for strategy, pnl_pts, r_mult in rows:
        buckets[strategy].append((pnl_pts or 0.0, r_mult or 0.0))

    out: dict = {}
    for strategy, trades in buckets.items():
        total  = len(trades)
        wins   = sum(1 for p, _ in trades if p > 0)
        losses = total - wins
        gross_win  = sum(p for p, _ in trades if p > 0)
        gross_loss = abs(sum(p for p, _ in trades if p <= 0))
        avg_r  = round(sum(r for _, r in trades) / total, 3) if total else 0.0
        pf     = round(gross_win / gross_loss, 3) if gross_loss > 0 else 9.999
        wr     = round(wins / total, 3) if total else 0.0

        promote_ready = (total >= 25 and pf >= 1.8 and avg_r >= 0.25)
        out[strategy] = {
            "total":            total,
            "wins":             wins,
            "losses":           losses,
            "win_rate":         wr,
            "avg_r":            avg_r,
            "profit_factor":    pf,
            "promote_ready":    promote_ready,
            "promote_criteria": "need 25+ trades, PF > 1.8, avg_r > 0.25",
        }

    return out


# ─────────────────────────────────────────────────────────────────────
# REGIME STATS  — per-bucket performance from trade_journal
# ─────────────────────────────────────────────────────────────────────
@app.get("/api/regime-stats")
def regime_stats(days: int = 60, strategy: str = ""):
    """
    Return trade performance sliced by regime bucket.

    Query params:
      days     — look-back window (default 60)
      strategy — filter to one strategy slug (default: all)

    Response shape:
      {
        "by_tod":       [{tod, trades, win_rate, avg_r, profit_factor}, ...],
        "by_market":    [{market, trades, win_rate, avg_r, profit_factor}, ...],
        "by_atr_bucket":[{bucket, atr_pct_range, trades, win_rate, avg_r, profit_factor}, ...],
        "note": "Only trades with regime columns populated are included."
      }

    Older journal rows (pre-migration) will have NULL regime columns and are
    excluded from this endpoint; they still appear in /api/journal.
    """
    try:
        conn = duckdb.connect(DB_PATH)
        strat_filter = "AND strategy_used = ?" if strategy else ""
        params: list = [days]
        if strategy:
            params.append(strategy)

        rows = conn.execute(f"""
            SELECT pnl, r_multiple, tod, market, atr_pct
            FROM trade_journal
            WHERE closed_at >= NOW() - INTERVAL '{days}' DAY
              AND ai_reasoning != 'partial_exit_t1'
              AND tod IS NOT NULL
              {strat_filter}
        """, params).fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"regime_stats DB error: {e}")

    if not rows:
        return {"by_tod": [], "by_market": [], "by_atr_bucket": [],
                "note": "No trades with regime data yet."}

    def _bucket_stats(groups: dict) -> list[dict]:
        out = []
        for label, trades in sorted(groups.items()):
            total = len(trades)
            wins  = sum(1 for p, _ in trades if p > 0)
            gw    = sum(p for p, _ in trades if p > 0)
            gl    = abs(sum(p for p, _ in trades if p <= 0))
            avg_r = round(sum(r for _, r in trades) / total, 3) if total else 0.0
            pf    = round(gw / gl, 3) if gl > 0 else 9.999
            out.append({
                "bucket":        label,
                "trades":        total,
                "win_rate":      round(wins / total, 3),
                "avg_r":         avg_r,
                "profit_factor": pf,
            })
        return out

    # ── Group by tod ─────────────────────────────────────────────────
    tod_groups: dict = {}
    market_groups: dict = {}
    atr_groups: dict = {}
    ATR_BUCKETS = [("low (<70)", lambda p: p < 70),
                   ("normal (70–130)", lambda p: 70 <= p <= 130),
                   ("spike (>130)", lambda p: p > 130)]

    for pnl, r_mult, tod, market, atr_pct in rows:
        pnl    = pnl    or 0.0
        r_mult = r_mult or 0.0

        if tod:
            tod_groups.setdefault(tod, []).append((pnl, r_mult))
        if market:
            market_groups.setdefault(market, []).append((pnl, r_mult))
        if atr_pct is not None:
            for label, test in ATR_BUCKETS:
                if test(atr_pct):
                    atr_groups.setdefault(label, []).append((pnl, r_mult))
                    break

    # Rename tod keys for readability and sort by session order
    TOD_ORDER = ["open", "primary", "lunch", "power_hour", "eod"]
    by_tod = sorted(
        _bucket_stats(tod_groups),
        key=lambda x: TOD_ORDER.index(x["bucket"]) if x["bucket"] in TOD_ORDER else 99,
    )
    # Rename bucket → tod/market for frontend clarity
    for row in by_tod:
        row["tod"] = row.pop("bucket")
    by_market = _bucket_stats(market_groups)
    for row in by_market:
        row["market"] = row.pop("bucket")
    by_atr = _bucket_stats(atr_groups)
    for row in by_atr:
        row["atr_bucket"] = row.pop("bucket")

    return {
        "by_tod":        by_tod,
        "by_market":     by_market,
        "by_atr_bucket": by_atr,
        "note": f"Includes {len(rows)} trades with regime data from last {days} days.",
    }
