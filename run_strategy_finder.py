"""
Strategy Finder — finds the best engine/timeframe combo for each symbol.
Tests 4 engines against ES, NQ, GC, CL and ranks by profit factor.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
from data.fetcher import fetch_historical
from backtesting.engine_aplus import BacktestEngineAPlus
from backtesting.engine_5m   import BacktestEngine5m
from backtesting.engine      import BacktestEngine

SYMBOLS = ["ES=F", "NQ=F", "GC=F", "CL=F"]
NAMES   = {"ES=F": "ES", "NQ=F": "NQ", "GC=F": "GC", "CL=F": "CL"}

ENGINES = [
    {
        "label": "A+ IB Retest (5m)",
        "tf": "5m", "days": 59,
        "build": lambda: BacktestEngineAPlus(
            min_adx=18.0, min_score=0.65,
            allow_short=False, require_macro_confirm=False,
        ),
    },
    {
        "label": "ORB+VWAP+EMA Stack (5m)",
        "tf": "5m", "days": 59,
        "build": lambda: BacktestEngine5m(
            warmup=50, max_bars=24, rr_ratio=2.0,
            atr_stop=1.5, min_adx=18.0, min_score=0.50,
            allow_short=False, rth_only=True,
        ),
    },
    {
        "label": "EMA55+BB+PDHL (1h longs)",
        "tf": "1h", "days": 365,
        "build": lambda: BacktestEngine(
            warmup=200, max_bars=24, rr_ratio=2.0, atr_stop=1.0,
            min_adx=18.0, min_score=0.50,
            allow_short=False, rth_only=True,
        ),
    },
    {
        "label": "EMA55+BB+PDHL (1h both)",
        "tf": "1h", "days": 365,
        "build": lambda: BacktestEngine(
            warmup=200, max_bars=24, rr_ratio=2.0, atr_stop=1.0,
            min_adx=18.0, min_score=0.50,
            allow_short=True, rth_only=True,
        ),
    },
]


def pf(trades):
    wins  = sum(t.pnl_pts for t in trades if t.pnl_pts > 0)
    loss  = abs(sum(t.pnl_pts for t in trades if t.pnl_pts <= 0))
    return wins / loss if loss > 0 else (math.inf if wins > 0 else 0.0)

def wr(trades):
    return len([t for t in trades if t.pnl_pts > 0]) / len(trades) if trades else 0.0

def exp(trades):
    return sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0


def main():
    # Cache fetched data to avoid duplicate downloads
    cache: dict[tuple, object] = {}

    print("\nFetching data and running all combinations...\n")

    # results[sym][engine_label] = (n, wr, pf, exp, total_pts)
    results = {s: {} for s in SYMBOLS}

    for sym in SYMBOLS:
        print(f"  {NAMES[sym]}:", end=" ", flush=True)
        for eng_cfg in ENGINES:
            key = (sym, eng_cfg["tf"], eng_cfg["days"])
            if key not in cache:
                try:
                    cache[key] = fetch_historical(sym, eng_cfg["tf"], eng_cfg["days"])
                except Exception as e:
                    cache[key] = None
                    print(f"[fetch err: {e}]", end=" ")
            df = cache[key]
            if df is None:
                results[sym][eng_cfg["label"]] = None
                continue
            try:
                engine = eng_cfg["build"]()
                if eng_cfg["tf"] in ("5m", "1h") and hasattr(engine, "run"):
                    if eng_cfg["tf"] == "1h":
                        r = engine.run(df, symbol=sym, timeframe="1h", market="futures")
                    else:
                        r = engine.run(df, symbol=sym, timeframe="5m")
                trades = r.trades
                results[sym][eng_cfg["label"]] = (
                    len(trades),
                    wr(trades),
                    pf(trades),
                    exp(trades),
                    round(sum(t.pnl_pts for t in trades), 2),
                )
            except Exception as e:
                results[sym][eng_cfg["label"]] = None
                print(f"[run err: {e}]", end=" ")
            print(".", end="", flush=True)
        print()

    # ── Print results table ──────────────────────────────────────────── #
    print()
    print("=" * 80)
    print("  RESULTS BY SYMBOL")
    print("=" * 80)

    best_per_sym = {}

    for sym in SYMBOLS:
        print(f"\n  {NAMES[sym]}")
        print(f"  {'Engine':<30}  {'n':>4}  {'WR':>6}  {'PF':>5}  {'Expect':>7}  {'pts':>10}")
        print(f"  {'-'*30}  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*10}")

        best_pf, best_label = -1, None
        for eng_cfg in ENGINES:
            label = eng_cfg["label"]
            v = results[sym].get(label)
            if v is None:
                print(f"  {label:<30}  {'ERROR':>37}")
                continue
            n, w, p, e, pts = v
            marker = ""
            if n > 0 and p > best_pf:
                best_pf = p
                best_label = label
            flag = " ◀ BEST" if False else ""  # marked after loop
            print(f"  {label:<30}  {n:>4}  {w:>6.1%}  {p:>5.2f}  {e:>+7.2f}R  {pts:>+10.2f}")

        if best_label:
            best_per_sym[sym] = (best_label, best_pf)
            print(f"  ✅ Best: {best_label} (PF {best_pf:.2f})")

    # ── Summary ──────────────────────────────────────────────────────── #
    print()
    print("=" * 80)
    print("  RECOMMENDED STRATEGY PER SYMBOL")
    print("=" * 80)
    print()
    for sym in SYMBOLS:
        if sym in best_per_sym:
            label, best = best_per_sym[sym]
            print(f"  {NAMES[sym]:<5}  →  {label}  (PF {best:.2f})")
    print()


if __name__ == "__main__":
    main()
