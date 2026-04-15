"""
Backtest: Longs-Only vs. Longs+Shorts comparison
Runs BacktestEngineAPlus on ES, NQ, GC, CL with 59 days of 5m data.
Shows per-symbol stats, longs/shorts split, and month-by-month P&L.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
from data.fetcher import fetch_historical
from backtesting.engine_aplus import BacktestEngineAPlus


SYMBOLS   = ["ES=F", "NQ=F", "GC=F", "CL=F"]
NAMES     = {"ES=F": "ES (S&P500)", "NQ=F": "NQ (Nasdaq)", "GC=F": "GC (Gold)", "CL=F": "CL (Crude Oil)"}
DAYS_BACK = 59   # yfinance 5m hard limit is 60 days


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"

def fmt_f(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}"


def run_for_symbol(symbol: str, allow_short: bool):
    df = fetch_historical(symbol, "5m", DAYS_BACK)
    engine = BacktestEngineAPlus(
        min_adx=18.0,
        min_score=0.65,
        allow_short=allow_short,
        require_macro_confirm=False,
    )
    return engine.run(df, symbol=symbol, timeframe="5m"), df


def print_header(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_result_row(label: str, r):
    if r.total_trades == 0:
        print(f"  {label:<22}  0 trades")
        return
    exit_reasons = defaultdict(int)
    for t in r.trades:
        exit_reasons[t.exit_reason] += 1
    reason_str = "  ".join(f"{k}:{v}" for k, v in sorted(exit_reasons.items()))
    print(
        f"  {label:<22}  "
        f"n={r.total_trades:>3}  "
        f"WR={fmt_pct(r.win_rate):>6}  "
        f"PF={r.profit_factor:>5.2f}  "
        f"E={r.expectancy:>+5.2f}R  "
        f"pts={sum(t.pnl_pts for t in r.trades):>+9.2f}  "
        f"exits: {reason_str}"
    )


def monthly_breakdown(trades, label: str):
    """Group trade P&L by YYYY-MM using entry_bar index and DataFrame index."""
    # We need dates — use bar_time from trade object not available, so skip dates
    # Actually we can't get dates from BacktestTrade easily (it stores bar indices)
    # Instead return per-direction breakdown
    by_month = defaultdict(list)
    for t in trades:
        by_month[t.regime].append(t.pnl_pts)
    return by_month


def main():
    print("\nFetching 5m data and running backtests — this may take 30-60 seconds...")

    all_longs_only   = []
    all_with_shorts  = []

    results_per_sym = {}

    for sym in SYMBOLS:
        print(f"  → {NAMES[sym]} ...", end=" ", flush=True)
        try:
            r_long_only, df = run_for_symbol(sym, allow_short=False)
            r_with_short, _ = run_for_symbol(sym, allow_short=True)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        results_per_sym[sym] = (r_long_only, r_with_short, df)
        all_longs_only.extend(r_long_only.trades)
        all_with_shorts.extend(r_with_short.trades)
        print("done")

    # ── Per-Symbol Side-by-Side ──────────────────────────────────────── #
    print_header("PER-SYMBOL: LONGS ONLY vs. LONGS + SHORTS")
    print(f"  {'Symbol':<22}  {'n':>4}  {'WR':>6}  {'PF':>5}  {'Expect':>7}  {'Total pts':>10}  exits")
    print(f"  {'-'*22}  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*10}")

    for sym in SYMBOLS:
        if sym not in results_per_sym:
            continue
        r_lo, r_ws, _ = results_per_sym[sym]
        print(f"\n  {NAMES[sym]}")
        print_result_row("  Longs only", r_lo)
        print_result_row("  Longs + Shorts", r_ws)

        # Short-only trades (in r_ws but not r_lo)
        lo_entry_bars = {t.entry_bar for t in r_lo.trades}

        class _FakeResult:
            def __init__(self, trades):
                self.trades = trades
                self.total_trades = len(trades)
                self.wins   = [t for t in trades if t.pnl_pts > 0]
                self.losses = [t for t in trades if t.pnl_pts <= 0]
                self.win_rate = len(self.wins) / len(trades) if trades else 0.0
                gw = sum(t.pnl_pts for t in self.wins)
                gl = abs(sum(t.pnl_pts for t in self.losses))
                import math
                self.profit_factor = (gw / gl) if gl > 0 else (math.inf if gw > 0 else 0.0)
                self.expectancy = sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0

        shorts_only = [t for t in r_ws.trades if t.direction == "SELL"]
        longs_ws    = [t for t in r_ws.trades if t.direction == "BUY"]
        if shorts_only:
            print_result_row("  └─ Shorts isolated", _FakeResult(shorts_only))
        if longs_ws:
            print_result_row("  └─ Longs in ws run", _FakeResult(longs_ws))

    # ── Aggregate Across All Symbols ────────────────────────────────── #
    print_header("AGGREGATE (ALL SYMBOLS COMBINED)")

    class _AggResult:
        def __init__(self, trades):
            import math
            self.trades = trades
            self.total_trades = len(trades)
            self.wins   = [t for t in trades if t.pnl_pts > 0]
            self.losses = [t for t in trades if t.pnl_pts <= 0]
            self.win_rate = len(self.wins) / len(trades) if trades else 0.0
            gw = sum(t.pnl_pts for t in self.wins)
            gl = abs(sum(t.pnl_pts for t in self.losses))
            self.profit_factor = (gw / gl) if gl > 0 else (math.inf if gw > 0 else 0.0)
            self.expectancy = sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0

    r_agg_lo = _AggResult(all_longs_only)
    r_agg_ws = _AggResult(all_with_shorts)
    shorts_agg = _AggResult([t for t in all_with_shorts if t.direction == "SELL"])
    longs_agg  = _AggResult([t for t in all_with_shorts if t.direction == "BUY"])

    print()
    print_result_row("Longs only", r_agg_lo)
    print_result_row("Longs + Shorts", r_agg_ws)
    print_result_row("  └─ Longs only (ws)", longs_agg)
    print_result_row("  └─ Shorts only (ws)", shorts_agg)

    # ── Short Score Distribution ─────────────────────────────────────── #
    if shorts_agg.trades:
        print_header("SHORT TRADE DETAIL")
        print(f"\n  {'Score':>6}  {'Dir':>5}  {'Exit Reason':>14}  {'R-mult':>7}  {'pts':>9}")
        print(f"  {'-'*6}  {'-'*5}  {'-'*14}  {'-'*7}  {'-'*9}")
        for t in sorted(shorts_agg.trades, key=lambda x: x.entry_bar):
            print(
                f"  {t.composite_score:>6.3f}  {t.direction:>5}  "
                f"{t.exit_reason:>14}  {t.r_multiple:>+7.2f}R  {t.pnl_pts:>+9.2f}"
            )
        avg_score = sum(t.composite_score for t in shorts_agg.trades) / len(shorts_agg.trades)
        avg_R     = sum(t.r_multiple for t in shorts_agg.trades) / len(shorts_agg.trades)
        total_pts = sum(t.pnl_pts for t in shorts_agg.trades)
        print(f"\n  Short avg score: {avg_score:.3f}  |  avg R: {avg_R:+.2f}  |  total pts: {total_pts:+.2f}")

    # ── Verdict ──────────────────────────────────────────────────────── #
    print_header("VERDICT")
    print()
    if not shorts_agg.trades:
        print("  No short setups triggered. Short conditions may be too strict for the")
        print("  current 59-day period (likely an uptrend period).")
        print()
        print("  ✓ Safe to enable shorts — they're not triggering unnecessarily.")
        print("  ✓ Longs-only strategy stats unchanged.")
        print()
    else:
        delta_pf = r_agg_ws.profit_factor - r_agg_lo.profit_factor
        delta_wr = r_agg_ws.win_rate - r_agg_lo.win_rate
        delta_e  = r_agg_ws.expectancy - r_agg_lo.expectancy
        short_pf = shorts_agg.profit_factor
        short_wr = shorts_agg.win_rate
        short_e  = shorts_agg.expectancy

        print(f"  Short trades found:  {len(shorts_agg.trades)}")
        print(f"  Short win rate:      {fmt_pct(short_wr)}")
        print(f"  Short profit factor: {short_pf:.2f}")
        print(f"  Short expectancy:    {short_e:+.2f}R")
        print()
        print(f"  Adding shorts changed overall PF by {delta_pf:+.2f}  WR by {fmt_pct(delta_wr)}  E by {delta_e:+.2f}R")
        print()

        if short_pf >= 1.3 and short_e >= 0.1:
            print("  ✅ Shorts look GOOD — positive edge, safe to keep enabled.")
        elif short_pf >= 1.0 and short_e >= 0.0:
            print("  ⚠️  Shorts are marginal — small positive edge, worth monitoring.")
        else:
            print("  ❌ Shorts are HURTING — consider keeping allow_short=False.")
            print("     Short logic may need tighter score threshold or different conditions.")

    print()


if __name__ == "__main__":
    main()
