"""
NQ=F Backtest Diagnostic Script
Fetches 5m data, runs both BacktestEngine5m and BacktestEngineAPlus, prints all trades and stats.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Make sure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import pytz

ET = pytz.timezone("America/New_York")

print("=" * 80)
print("NQ=F BACKTEST DIAGNOSTIC")
print("=" * 80)

# ── 1. Fetch data ────────────────────────────────────────────────────────── #
print("\n[1] FETCHING NQ=F DATA (5m, 58 days back) ...")
from data.fetcher import fetch_historical

try:
    df = fetch_historical("NQ=F", "5m", 58)
    print(f"    OK — {len(df)} rows returned")
except Exception as e:
    print(f"    ERROR fetching data: {e}")
    sys.exit(1)

# ── 2. Data sanity check ─────────────────────────────────────────────────── #
print("\n[2] DATA SANITY CHECK")
print(f"    Rows       : {len(df)}")
print(f"    Columns    : {list(df.columns)}")
print(f"    Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")

# Check for nulls
nulls = df.isnull().sum()
if nulls.any():
    print(f"    NULL counts: {nulls[nulls > 0].to_dict()}")
else:
    print(f"    NULL counts: none")

# Check for duplicate timestamps
dupes = df["timestamp"].duplicated().sum()
print(f"    Duplicate timestamps: {dupes}")

# Price sanity (NQ should be in rough range 15000-25000 in 2024-2025)
print(f"    Close range: {df['close'].min():.2f} – {df['close'].max():.2f}")
print(f"    Volume range: {df['volume'].min():.0f} – {df['volume'].max():.0f}")

# Check for zero-volume bars
zero_vol = (df["volume"] == 0).sum()
print(f"    Zero-volume bars: {zero_vol}")

# Check for large gaps (> 10 minutes between consecutive bars during RTH)
df_sorted = df.sort_values("timestamp").reset_index(drop=True)
df_et = df_sorted.copy()
df_et["ts_et"] = df_et["timestamp"].dt.tz_convert(ET)
# Only check RTH
rth_mask = (
    (df_et["ts_et"].dt.hour >= 9) &
    (df_et["ts_et"].dt.hour <= 16) &
    ~((df_et["ts_et"].dt.hour == 9) & (df_et["ts_et"].dt.minute < 30))
)
df_rth = df_et[rth_mask].copy()
df_rth["gap_min"] = df_rth["timestamp"].diff().dt.total_seconds() / 60
big_gaps = df_rth[df_rth["gap_min"] > 10]
if len(big_gaps) > 0:
    print(f"    RTH gaps > 10min: {len(big_gaps)}")
    for _, g in big_gaps.head(5).iterrows():
        print(f"      {g['ts_et']}  gap={g['gap_min']:.0f}min")
else:
    print(f"    RTH gaps > 10min: none")

print("\n    FIRST 5 ROWS:")
for _, row in df.head(5).iterrows():
    ts_et = row["timestamp"].astimezone(ET)
    print(f"      {ts_et.strftime('%Y-%m-%d %H:%M ET')}  O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:.0f}")

print("\n    LAST 5 ROWS:")
for _, row in df.tail(5).iterrows():
    ts_et = row["timestamp"].astimezone(ET)
    print(f"      {ts_et.strftime('%Y-%m-%d %H:%M ET')}  O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:.0f}")

# ── 3. Run BacktestEngine5m ──────────────────────────────────────────────── #
print("\n" + "=" * 80)
print("[3] RUNNING BacktestEngine5m  (enable_ema_stack=False, allow_short=True)")
print("=" * 80)

from backtesting.engine_5m import BacktestEngine5m

try:
    engine5m = BacktestEngine5m(enable_ema_stack=False)
    result5m = engine5m.run(df.copy(), symbol="NQ=F", timeframe="5m")
    print(f"\n    Total bars processed : {result5m.total_bars}")
    print(f"    Total trades         : {result5m.total_trades}")

    if result5m.trades:
        print(f"    Win rate             : {result5m.win_rate:.1%}")
        print(f"    Profit factor        : {result5m.profit_factor:.2f}")
        print(f"    Total PnL (pts)      : {sum(t.pnl_pts for t in result5m.trades):.2f}")
        print(f"    Avg winner (pts)     : {result5m.avg_winner:.2f}")
        print(f"    Avg loser (pts)      : {result5m.avg_loser:.2f}")
        print(f"    Avg bars held        : {result5m.avg_bars_held:.1f}")

        print(f"\n    ALL 5m ENGINE TRADES ({result5m.total_trades} total):")
        print(f"    {'#':<4} {'bar_time_et':<22} {'dir':<5} {'strategy':<18} {'entry':>9} {'stop':>9} {'target':>9} {'exit':>9} {'result':<8} {'pnl_pts':>9} {'R':>6} {'reason':<12}")
        print(f"    {'-'*4} {'-'*22} {'-'*5} {'-'*18} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*6} {'-'*12}")

        df_prepared = engine5m._prepare(df.copy())
        for n, t in enumerate(result5m.trades, 1):
            # Get entry bar time
            if t.entry_bar < len(df_prepared):
                ts = df_prepared["ts_et"].iloc[t.entry_bar]
                bar_time_str = ts.strftime("%Y-%m-%d %H:%M")
            else:
                bar_time_str = f"bar#{t.entry_bar}"
            result_str = "WIN" if t.pnl_pts > 0 else "LOSS"
            print(f"    {n:<4} {bar_time_str:<22} {t.direction:<5} {t.strategy:<18} {t.entry_price:>9.2f} {t.stop_loss:>9.2f} {t.take_profit:>9.2f} {t.exit_price:>9.2f} {result_str:<8} {t.pnl_pts:>9.2f} {t.r_multiple:>6.2f} {t.exit_reason:<12}")

        # Per-strategy breakdown
        strats = {}
        for t in result5m.trades:
            strats.setdefault(t.strategy, []).append(t)
        print(f"\n    PER-STRATEGY BREAKDOWN:")
        for strat, trades in strats.items():
            wins = [t for t in trades if t.pnl_pts > 0]
            total_pnl = sum(t.pnl_pts for t in trades)
            wr = len(wins)/len(trades) if trades else 0
            print(f"      {strat:<18}: {len(trades):>3} trades, WR={wr:.0%}, total_pnl={total_pnl:.2f}pts")
    else:
        print("\n    *** NO TRADES GENERATED by BacktestEngine5m ***")
        print("    Possible reasons:")
        print("      - ADX below min_adx=18 most of the time")
        print("      - ORB conditions not met (tight ORB required)")
        print("      - VWAP pullback streak condition not met (need 5+ bars above VWAP)")
        # Diagnose why no trades
        df_prep = engine5m._prepare(df.copy())
        print(f"\n    DIAGNOSTIC: Checking indicator values on first 10 RTH bars ...")
        rth_bars = df_prep[
            (df_prep["ts_et"].dt.hour >= 10) &
            (df_prep["ts_et"].dt.hour <= 14)
        ].head(20)
        print(f"    {'ts_et':<22} {'close':>9} {'adx':>6} {'orb_hi':>9} {'orb_lo':>9} {'tight':>6} {'streak':>7}")
        for _, r in rth_bars.head(10).iterrows():
            ts_str = r["ts_et"].strftime("%Y-%m-%d %H:%M")
            orb_h = r.get("orb_high", float("nan"))
            orb_l = r.get("orb_low", float("nan"))
            tight = r.get("orb_tight", False)
            streak = r.get("vwap_above_streak", 0)
            print(f"    {ts_str:<22} {r['close']:>9.2f} {r['adx']:>6.1f} {orb_h:>9.2f} {orb_l:>9.2f} {str(tight):>6} {streak:>7}")

except Exception as e:
    import traceback
    print(f"\n    ERROR running BacktestEngine5m: {e}")
    traceback.print_exc()

# ── 4. Run BacktestEngineAPlus ───────────────────────────────────────────── #
print("\n" + "=" * 80)
print("[4] RUNNING BacktestEngineAPlus  (allow_short=False)")
print("=" * 80)

from backtesting.engine_aplus import BacktestEngineAPlus

try:
    engine_ap = BacktestEngineAPlus(allow_short=False)
    result_ap = engine_ap.run(df.copy(), symbol="NQ=F", timeframe="5m")
    print(f"\n    Total bars processed : {result_ap.total_bars}")
    print(f"    Total trades         : {result_ap.total_trades}")

    if result_ap.trades:
        print(f"    Win rate             : {result_ap.win_rate:.1%}")
        print(f"    Profit factor        : {result_ap.profit_factor:.2f}")
        print(f"    Total PnL (pts)      : {sum(t.pnl_pts for t in result_ap.trades):.2f}")
        print(f"    Avg winner (pts)     : {result_ap.avg_winner:.2f}")
        print(f"    Avg loser (pts)      : {result_ap.avg_loser:.2f}")
        print(f"    Avg bars held        : {result_ap.avg_bars_held:.1f}")

        print(f"\n    ALL A+ ENGINE TRADES ({result_ap.total_trades} total):")
        print(f"    {'#':<4} {'bar_time_et':<22} {'dir':<5} {'strategy':<20} {'entry':>9} {'stop':>9} {'t1':>9} {'t2':>9} {'exit':>9} {'result':<8} {'pnl_pts':>9} {'R':>6} {'reason':<12} {'score':>6}")
        print(f"    {'-'*4} {'-'*22} {'-'*5} {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*6} {'-'*12} {'-'*6}")

        df_ap_prepared = engine_ap._prepare(df.copy())
        for n, t in enumerate(result_ap.trades, 1):
            if t.entry_bar < len(df_ap_prepared):
                ts = df_ap_prepared["ts_et"].iloc[t.entry_bar]
                bar_time_str = ts.strftime("%Y-%m-%d %H:%M")
            else:
                bar_time_str = f"bar#{t.entry_bar}"
            result_str = "WIN" if t.pnl_pts > 0 else "LOSS"
            # take_profit is T2 in aplus
            print(f"    {n:<4} {bar_time_str:<22} {t.direction:<5} {t.regime:<20} {t.entry_price:>9.2f} {t.stop_loss:>9.2f} {'N/A':>9} {t.take_profit:>9.2f} {t.exit_price:>9.2f} {result_str:<8} {t.pnl_pts:>9.2f} {t.r_multiple:>6.2f} {t.exit_reason:<12} {t.composite_score:>6.3f}")

    else:
        print("\n    *** NO TRADES GENERATED by BacktestEngineAPlus ***")
        print("    Diagnosing A+ filter chain on a sample of bars ...")

        df_ap = engine_ap._prepare(df.copy())
        n_total = len(df_ap)

        # Check how many bars pass the entry window
        in_window = df_ap.apply(
            lambda r: engine_ap._is_entry_window(r["ts_et"].hour, r["ts_et"].minute), axis=1
        )
        print(f"\n    Bars in entry window      : {in_window.sum()} / {n_total}")

        # Check warmup constraint
        warmup_pass = df_ap.index >= 100
        print(f"    Bars past warmup (>=100)  : {warmup_pass.sum()}")
        combined = in_window & warmup_pass

        # Check IB broken up
        ib_up_pass = combined & df_ap["ib_broken_up"].astype(bool)
        print(f"    + ib_broken_up=True       : {ib_up_pass.sum()}")

        # Check above EMA200
        above_e200 = ib_up_pass & (df_ap["close"] > df_ap["ema_200"])
        print(f"    + close > EMA200          : {above_e200.sum()}")

        # Check ADX
        adx_pass = above_e200 & (df_ap["adx"] >= engine_ap.min_adx)
        print(f"    + ADX >= {engine_ap.min_adx}            : {adx_pass.sum()}")

        # Check +DI > -DI
        di_pass = adx_pass & (df_ap["dmp"] > df_ap["dmn"])
        print(f"    + +DI > -DI               : {di_pass.sum()}")

        # IB High and VWAP-85
        if di_pass.sum() > 0:
            sub = df_ap[di_pass]
            above_ibh_vwap = (sub["close"] > sub[["ib_high", "vwap_85"]].min(axis=1))
            print(f"    + close > min(IB_H,VWAP85): {above_ibh_vwap.sum()}")

        # Print some IB stats for the first 3 days
        print(f"\n    SAMPLE IB VALUES BY DAY (first 5 trading days):")
        print(f"    {'date':<12} {'ib_high':>10} {'ib_low':>10} {'ib_range':>10} {'ib_broken_up':>14}")
        daily = df_ap.groupby("date_et")
        day_count = 0
        for date, gidx in daily.groups.items():
            if day_count >= 5:
                break
            gidx = sorted(gidx)
            day_df = df_ap.iloc[gidx]
            # first post-IB bar
            post_ib = day_df[~((day_df["ts_et"].dt.hour == 9) & (day_df["ts_et"].dt.minute >= 30) & (day_df["ts_et"].dt.minute <= 40))]
            if len(post_ib) > 0:
                first_post = post_ib.iloc[0]
                ihi = first_post.get("ib_high", float("nan"))
                ilo = first_post.get("ib_low", float("nan"))
                rng = (ihi - ilo) if not (np.isnan(ihi) or np.isnan(ilo)) else float("nan")
                any_broken_up = day_df["ib_broken_up"].any()
                print(f"    {str(date):<12} {ihi:>10.2f} {ilo:>10.2f} {rng:>10.2f} {str(any_broken_up):>14}")
            day_count += 1

        # Show ADX distribution
        adx_vals = df_ap[in_window & warmup_pass]["adx"].dropna()
        if len(adx_vals) > 0:
            print(f"\n    ADX distribution in entry window:")
            print(f"      min={adx_vals.min():.1f}  median={adx_vals.median():.1f}  max={adx_vals.max():.1f}  mean={adx_vals.mean():.1f}")
            print(f"      bars with ADX < {engine_ap.min_adx}: {(adx_vals < engine_ap.min_adx).sum()} / {len(adx_vals)} ({(adx_vals < engine_ap.min_adx).mean():.0%})")

except Exception as e:
    import traceback
    print(f"\n    ERROR running BacktestEngineAPlus: {e}")
    traceback.print_exc()

# ── 5. Summary comparison ────────────────────────────────────────────────── #
print("\n" + "=" * 80)
print("[5] SUMMARY COMPARISON")
print("=" * 80)

try:
    print(f"\n  Engine5m  : {result5m.summary()}")
except:
    print(f"\n  Engine5m  : ERROR (see above)")

try:
    print(f"  EnginAPlus: {result_ap.summary()}")
except:
    print(f"  EngineAPlus: ERROR (see above)")

print("\nDiagnostic complete.")
