"""
All-Strategy Backtest — Combined revenue projection
─────────────────────────────────────────────────────
Tests all 5 strategies head-to-head:
  1. A+            (10:15 AM – 3:30 PM, long-biased)
  2. Fade the Rip  (10:00 AM – 3:30 PM, short-only)
  3. ORB           (9:35 AM – 10:15 AM, both directions)
  4. VWAP Pullback (10:00 AM – 3:30 PM, both directions)
  5. EMA8 Cont.    (10:00 AM – 3:30 PM, both directions)

Run: cd trading_bot && .venv/bin/python backtesting/test_all_strategies.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import pandas as pd
import yfinance as yf

try:
    yf.set_tz_cache_location(tempfile.gettempdir())
except Exception:
    pass

from backtesting.engine_aplus         import BacktestEngineAPlus
from backtesting.engine_fade_rip      import FadeTheRipEngine
from backtesting.engine_orb           import ORBEngine
from backtesting.engine_vwap_pullback import VWAPPullbackEngine
from backtesting.models import BacktestResult

SYMBOLS = ["NQ=F", "ES=F"]
PERIOD  = "60d"
TF      = "5m"
R_VALUE = 500   # $ per R per account (conservative NQ/ES estimate)

# VWAP Pullback performs better on ES (mean-reverting) than NQ (trendy).
# Restrict VWAP PB to ES=F only — NQ has shown no edge (PF 0.47 historically).
VWAP_PB_SYMBOLS = {"ES=F"}


def fetch(symbol: str) -> pd.DataFrame:
    raw = yf.download(symbol, period=PERIOD, interval=TF, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {symbol}")
    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    for ts_col in ["datetime", "date", "timestamp"]:
        if ts_col in raw.columns:
            raw = raw.rename(columns={ts_col: "timestamp"})
            break
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    return raw[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def stats(result: BacktestResult) -> dict:
    trades = result.trades
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "avg_r": 0.0, "total_r": 0.0,
            "gross_win": 0.0, "gross_loss": 0.0,
        }
    wins   = [t for t in trades if t.r_multiple > 0]
    losses = [t for t in trades if t.r_multiple <= 0]
    gw = sum(t.r_multiple for t in wins)
    gl = abs(sum(t.r_multiple for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    ar = sum(t.r_multiple for t in trades) / len(trades)
    tr = sum(t.r_multiple for t in trades)
    return {
        "trades":        len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(pf, 2),
        "avg_r":         round(ar, 2),
        "total_r":       round(tr, 2),
        "gross_win":     gw,
        "gross_loss":    gl,
    }


def exit_breakdown(trades, label=""):
    from collections import Counter
    if not trades:
        return
    reasons = Counter(t.exit_reason for t in trades)
    total   = len(trades)
    print(f"      {'Exit':<12} {'Count':>6}  {'%':>5}  {'Avg R':>7}")
    for reason, count in sorted(reasons.items()):
        avg_r = sum(t.r_multiple for t in trades if t.exit_reason == reason) / count
        print(f"      {reason:<12} {count:>6}  {count/total*100:>4.0f}%  {avg_r:>+7.2f}R")


def combine_stats(stat_list: list[dict]) -> dict:
    if not stat_list:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_r": 0.0, "total_r": 0.0,
                "gross_win": 0.0, "gross_loss": 0.0}
    tot = {"trades": 0, "wins": 0, "losses": 0, "gross_win": 0.0, "gross_loss": 0.0, "total_r": 0.0}
    for s in stat_list:
        tot["trades"]     += s["trades"]
        tot["wins"]       += s["wins"]
        tot["losses"]     += s["losses"]
        tot["gross_win"]  += s.get("gross_win", 0.0)
        tot["gross_loss"] += s.get("gross_loss", 0.0)
        tot["total_r"]    += s["total_r"]
    pf = tot["gross_win"] / tot["gross_loss"] if tot["gross_loss"] > 0 else float("inf")
    wr = tot["wins"] / tot["trades"] * 100 if tot["trades"] > 0 else 0.0
    ar = tot["total_r"] / tot["trades"] if tot["trades"] > 0 else 0.0
    return {
        "trades":        tot["trades"],
        "wins":          tot["wins"],
        "losses":        tot["losses"],
        "win_rate":      round(wr, 1),
        "profit_factor": round(pf, 2),
        "avg_r":         round(ar, 2),
        "total_r":       round(tot["total_r"], 2),
        "gross_win":     tot["gross_win"],
        "gross_loss":    tot["gross_loss"],
    }


def verdict(name: str, pf: float, n: int) -> str:
    if pf >= 1.8 and n >= 10:
        return f"STRONG EDGE — DEPLOY alongside existing strategies"
    elif pf >= 1.4 and n >= 6:
        return f"MARGINAL EDGE — Paper trade 2 more weeks before deploying"
    elif pf >= 1.1 and n >= 4:
        return f"WEAK EDGE — Needs more data / tuning"
    else:
        return f"NO EDGE — Do not deploy (PF {pf:.2f}, N={n})"


def main():
    engines = {
        "A+":           BacktestEngineAPlus(min_score=0.75, skip_monday=True, skip_power_hour=True),
        "Fade Rip":     FadeTheRipEngine(),
        "ORB":          ORBEngine(min_score=0.70, skip_monday=True),
        "VWAP PB":      VWAPPullbackEngine(min_score=0.65, skip_monday=True, skip_power_hour=True),
    }
    engine_run = {
        "A+":       lambda e, df, sym: e.run(df, symbol=sym),
        "Fade Rip": lambda e, df, sym: e.run(df, symbol=sym),
        "ORB":      lambda e, df, sym: e.run(df, symbol=sym),
        # VWAP PB: skip NQ — no edge confirmed (PF 0.47 historically); ES only
        "VWAP PB":  lambda e, df, sym: e.run(df, symbol=sym) if sym in VWAP_PB_SYMBOLS else None,
    }

    print(f"\n{'='*80}")
    print(f"  ALL-STRATEGY BACKTEST  |  {PERIOD} of {TF} bars  |  {', '.join(SYMBOLS)}")
    print(f"{'='*80}")

    # Collect per-strategy stats across all symbols
    all_stats: dict[str, list[dict]] = {name: [] for name in engines}

    for sym in SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  {sym}")
        print(f"{'─'*80}")

        try:
            df = fetch(sym)
            print(f"  Loaded {len(df):,} bars ({PERIOD})\n")
        except Exception as e:
            print(f"  SKIP — {e}\n")
            continue

        sym_results: dict[str, dict] = {}
        sym_trades:  dict[str, list] = {}

        for name, engine in engines.items():
            try:
                result = engine_run[name](engine, df.copy(), sym)
                if result is None:
                    # Strategy skipped for this symbol (e.g. VWAP PB skips NQ)
                    empty = stats(BacktestResult(symbol=sym, timeframe=TF, market="futures", total_bars=0))
                    sym_results[name] = empty
                    sym_trades[name]  = []
                    # Don't add to all_stats — excluded from grand combined
                    continue
                s = stats(result)
                sym_results[name] = s
                sym_trades[name]  = result.trades
                all_stats[name].append(s)
            except Exception as ex:
                import traceback; traceback.print_exc()
                print(f"  {name}: ERROR — {ex}")
                sym_results[name] = stats(BacktestResult(symbol=sym, timeframe=TF, market="futures", total_bars=0))
                sym_trades[name]  = []
                all_stats[name].append(sym_results[name])

        # Per-symbol table
        col = 11
        names = list(engines.keys())
        header = f"  {'Metric':<16}" + "".join(f"{n:>{col}}" for n in names)
        print(header)
        print(f"  {'-'*(16 + col * len(names))}")
        for metric, label in [
            ("trades",        "Trades"),
            ("win_rate",      "Win Rate %"),
            ("profit_factor", "PF"),
            ("avg_r",         "Avg R"),
            ("total_r",       "Total R"),
        ]:
            row = f"  {label:<16}"
            for name in names:
                v = sym_results[name].get(metric, 0)
                row += f"{str(v):>{col}}"
            print(row)

        # Exit breakdowns (condensed)
        print()
        for name in names:
            trades = sym_trades[name]
            if trades:
                wr    = sym_results[name]["win_rate"]
                pf    = sym_results[name]["profit_factor"]
                total = sym_results[name]["total_r"]
                print(f"  {name} — {len(trades)} trades  WR {wr}%  PF {pf}  Total {total:+.1f}R")
                exit_breakdown(trades)

    # ── Grand Combined ─────────────────────────────────────────────── #
    print(f"\n{'='*80}")
    print(f"  GRAND COMBINED ({', '.join(SYMBOLS)})  —  {PERIOD}")
    print(f"{'='*80}\n")

    combined: dict[str, dict] = {name: combine_stats(all_stats[name]) for name in engines}
    total_all_r = sum(combined[n]["total_r"] for n in engines)
    total_all_trades = sum(combined[n]["trades"] for n in engines)

    col = 11
    names = list(engines.keys())
    header = f"  {'Metric':<16}" + "".join(f"{n:>{col}}" for n in names) + f"{'TOTAL':>{col}}"
    print(header)
    print(f"  {'-'*(16 + col * (len(names) + 1))}")

    for metric, label in [
        ("trades",        "Trades"),
        ("win_rate",      "Win Rate %"),
        ("profit_factor", "PF"),
        ("avg_r",         "Avg R"),
        ("total_r",       "Total R"),
    ]:
        row = f"  {label:<16}"
        for name in names:
            v = combined[name].get(metric, 0)
            row += f"{str(v):>{col}}"
        if metric == "total_r":
            row += f"{total_all_r:>{col}.1f}"
        elif metric == "trades":
            row += f"{total_all_trades:>{col}}"
        else:
            row += f"{'—':>{col}}"
        print(row)

    # ── Revenue Impact ─────────────────────────────────────────────── #
    print(f"\n{'─'*80}")
    print(f"  Revenue Impact  (${R_VALUE}/R per account, single account)")
    print(f"{'─'*80}")

    existing_r   = combined["A+"]["total_r"] + combined["Fade Rip"]["total_r"]
    new_r        = combined["ORB"]["total_r"] + combined["VWAP PB"]["total_r"]
    existing_usd = existing_r * R_VALUE
    new_usd      = new_r * R_VALUE
    total_usd    = total_all_r * R_VALUE

    print(f"  Existing (A+ + Fade Rip):        {existing_r:>+6.1f}R  →  ${existing_usd:>+8,.0f}")
    for name in ["ORB", "VWAP PB"]:
        r   = combined[name]["total_r"]
        usd = r * R_VALUE
        n   = combined[name]["trades"]
        pf  = combined[name]["profit_factor"]
        tag = " ✓ DEPLOY" if pf >= 1.5 and n >= 8 else (" ~ paper" if pf >= 1.2 else " ✗ no edge")
        note = "  (ES only)" if name == "VWAP PB" else ""
        print(f"  + {name:<22}:  {r:>+6.1f}R  →  ${usd:>+8,.0f}  PF={pf:<5}{tag}{note}")

    uplift_pct = (new_r / existing_r * 100) if existing_r > 0 else 0
    print(f"  {'─'*60}")
    print(f"  New strategies add:              {new_r:>+6.1f}R  →  ${new_usd:>+8,.0f}  ({uplift_pct:+.0f}%)")
    print(f"  ALL STRATEGIES COMBINED:         {total_all_r:>+6.1f}R  →  ${total_usd:>+8,.0f}")

    print(f"\n  At 90% TopStep payout: ${total_usd * 0.9:,.0f}/month (single account)")
    print(f"  Two accounts:          ${total_usd * 0.9 * 2:,.0f}/month")

    # ── Strategy verdicts ──────────────────────────────────────────── #
    print(f"\n{'─'*80}")
    print(f"  VERDICTS")
    print(f"{'─'*80}")
    for name in ["ORB", "VWAP PB"]:
        pf = combined[name]["profit_factor"]
        n  = combined[name]["trades"]
        note = " (ES-only test)" if name == "VWAP PB" else ""
        print(f"  {name:<12}: {verdict(name, pf, n)}{note}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
