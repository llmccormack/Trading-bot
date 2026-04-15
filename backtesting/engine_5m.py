"""
BacktestEngine5m — 5-Minute Bar Day Trading Engine
────────────────────────────────────────────────────
Three strategies designed for high-frequency intraday trading on 5m bars,
targeting 2-3 trades/day across ES, NQ, GC, CL, RTY, YM.

STRATEGY MAP:
  1. ORB (Opening Range Breakout)
     Win rate target: 52-58% | RR 2:1
     Captures the 10am-noon breakout after the first 30-min opening range.
     Requires tight range (< 1× ATR) and ADX < 25 (coiling, not already trending).

  2. VWAP Pullback
     Win rate target: 55-62% | RR 1.5:1
     Price trending above VWAP, pulls back into VWAP, first close back above.
     Requires 3+ consecutive bars above VWAP before the pullback.

  3. EMA21 Stack Pullback
     Win rate target: 50-56% | RR 2:1
     EMA8 > EMA21 > EMA55 bullish stack. Price pulls back to EMA21,
     closes back above. Momentum continuation in clear intraday trend.

KEY SETTINGS:
  - RTH only: 9:30 AM – 3:55 PM ET (5m bars, no overnight)
  - EOD force-close: any open position closed on last bar of RTH (15:55 ET)
  - ADX filter: min_adx=18 (lower bar than 1h since intraday is noisier)
  - max_bars: 24 (2 hours max hold on 5m bars)
  - yfinance 5m data limit: ~60 days (statistically limited, directional only)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

from utils.indicators import add_all
from backtesting.models import BacktestTrade, BacktestResult


ET = pytz.timezone("America/New_York")

# RTH on 5m: 9:30 AM – 3:55 PM ET (last complete bar opens at 15:55)
# We treat 9:30-15:55 as valid entry bars
RTH_START_H, RTH_START_M = 9, 30
RTH_END_H,   RTH_END_M   = 15, 50

# ORB window: first 30 minutes = 9:30-9:55 ET (6 bars: :30 :35 :40 :45 :50 :55)
ORB_END_H, ORB_END_M = 9, 55

# ORB signal window: 10:00 AM – 12:00 PM ET
ORB_SIGNAL_START_H, ORB_SIGNAL_START_M = 10, 0
ORB_SIGNAL_END_H,   ORB_SIGNAL_END_M   = 12, 0

STRATEGIES = ["orb", "vwap_pullback", "ema_stack_5m"]

STRATEGY_LABELS = {
    "orb":          "ORB Breakout",
    "vwap_pullback": "VWAP Pullback",
    "ema_stack_5m": "EMA21 Stack",
}


class BacktestEngine5m:
    """
    5-minute bar backtest and live-signal engine.
    Runs ORB, VWAP Pullback, and EMA Stack strategies.
    """

    def __init__(
        self,
        warmup:    int   = 50,
        max_bars:  int   = 24,    # 2 hours on 5m
        rr_ratio:  float = 2.0,
        atr_stop:  float = 1.5,   # wider than 1h — 5m bars are noisier
        min_score: float = 0.50,
        allow_short:    bool = True,
        min_adx:        float = 18.0,  # lower than 1h
        rth_only:       bool  = True,
        enable_ema_stack: bool = False,  # EMA stack disabled by default — underperforms in volatile regimes
    ):
        self.warmup           = warmup
        self.max_bars         = max_bars
        self.rr_ratio         = rr_ratio
        self.atr_stop         = atr_stop
        self.min_score        = min_score
        self.allow_short      = allow_short
        self.min_adx          = min_adx
        self.rth_only         = rth_only
        self.enable_ema_stack = enable_ema_stack

    # ─────────────────────────────────────────────────────────── #
    # Public API                                                   #
    # ─────────────────────────────────────────────────────────── #

    def run(self, df_raw: pd.DataFrame, symbol: str = "ES=F", timeframe: str = "5m") -> BacktestResult:
        """Full historical backtest. Returns BacktestResult."""
        df = self._prepare(df_raw)
        n  = len(df)
        result = BacktestResult(symbol=symbol, timeframe=timeframe, market="futures", total_bars=n)

        open_trade: Optional[dict] = None
        equity = 0.0

        for i in range(self.warmup, n):
            bar_time_et = df["ts_et"].iloc[i]
            h, m = bar_time_et.hour, bar_time_et.minute

            # ── Check stops / targets on open trade ─────────────── #
            if open_trade is not None:
                lo = df["low"].iloc[i]
                hi = df["high"].iloc[i]
                cl = df["close"].iloc[i]

                hit_stop   = (open_trade["dir"] ==  1 and lo <= open_trade["sl"]) or \
                             (open_trade["dir"] == -1 and hi >= open_trade["sl"])
                hit_target = (open_trade["dir"] ==  1 and hi >= open_trade["tp"]) or \
                             (open_trade["dir"] == -1 and lo <= open_trade["tp"])

                # EOD force-close at 15:55 ET
                eod_close  = (h == RTH_END_H and m >= RTH_END_M) or (h > RTH_END_H)

                if hit_stop or hit_target or eod_close or (i - open_trade["entry_bar"] >= self.max_bars):
                    if hit_target and not hit_stop:
                        exit_px, reason = open_trade["tp"], "target"
                    elif hit_stop:
                        exit_px, reason = open_trade["sl"], "stop"
                    elif eod_close:
                        exit_px, reason = cl, "eod"
                    else:
                        exit_px, reason = cl, "timeout"

                    pnl = (exit_px - open_trade["entry"]) * open_trade["dir"]
                    risk = abs(open_trade["entry"] - open_trade["sl"])
                    r_mult = pnl / risk if risk > 0 else 0.0
                    equity += pnl

                    t = BacktestTrade(
                        strategy    = open_trade["strategy"],
                        direction   = "BUY" if open_trade["dir"] == 1 else "SELL",
                        entry_bar   = open_trade["entry_bar"],
                        exit_bar    = i,
                        entry_price = open_trade["entry"],
                        exit_price  = round(exit_px, 4),
                        stop_loss   = open_trade["sl"],
                        take_profit = open_trade["tp"],
                        exit_reason = reason,
                        pnl_pts     = round(pnl, 4),
                        r_multiple  = round(r_mult, 2),
                        bars_held   = i - open_trade["entry_bar"],
                        composite_score = open_trade["score"],
                        regime      = open_trade["regime"],
                    )
                    result.trades.append(t)
                    result.equity_curve.append(round(equity, 4))
                    open_trade = None

            # ── Skip entry if position open or outside RTH ───────── #
            if open_trade is not None:
                continue
            if self.rth_only and not self._is_rth(h, m):
                continue

            # ── Score all strategies ─────────────────────────────── #
            sig = self._score(df, i)
            if sig is None:
                continue

            direction, strategy, score, regime = sig

            # ── Build trade parameters ───────────────────────────── #
            cl    = df["close"].iloc[i]
            atr   = df["atr"].iloc[i]
            entry = cl
            if direction == "BUY":
                sl = entry - self.atr_stop * atr
                tp = entry + self.atr_stop * self.rr_ratio * atr
            else:
                sl = entry + self.atr_stop * atr
                tp = entry - self.atr_stop * self.rr_ratio * atr

            # Override RR for VWAP pullback (1.5:1 = higher hit rate)
            if strategy == "vwap_pullback":
                if direction == "BUY":
                    tp = entry + self.atr_stop * 1.5 * atr
                else:
                    tp = entry - self.atr_stop * 1.5 * atr

            open_trade = {
                "entry_bar": i,
                "dir":       1 if direction == "BUY" else -1,
                "entry":     entry,
                "sl":        round(sl, 4),
                "tp":        round(tp, 4),
                "strategy":  strategy,
                "score":     score,
                "regime":    regime,
            }

        return result

    def live_signal(self, df_raw: pd.DataFrame, current_price: float | None = None) -> dict | None:
        """
        Evaluate the most recently completed 5m bar for a live signal.
        Returns a signal dict or None.
        """
        df = self._prepare(df_raw)
        n  = len(df)
        if n < self.warmup + 2:
            return None

        i = n - 1  # most recently completed bar (n-1)

        bar_time_utc = df["timestamp"].iloc[i]
        bar_time_et  = bar_time_utc.astimezone(ET)
        h, m = bar_time_et.hour, bar_time_et.minute

        if self.rth_only and not self._is_rth(h, m):
            return None

        sig = self._score(df, i)
        if sig is None:
            return None

        direction, strategy, score, regime = sig

        cl    = current_price if current_price else df["close"].iloc[i]
        atr   = df["atr"].iloc[i]
        entry = cl
        rr_use = 1.5 if strategy == "vwap_pullback" else self.rr_ratio

        if direction == "BUY":
            sl = entry - self.atr_stop * atr
            tp = entry + self.atr_stop * rr_use * atr
            # T1 = halfway to T2 (1R from entry), for partial-exit + BE-stop logic
            t1 = entry + self.atr_stop * atr
        else:
            sl = entry + self.atr_stop * atr
            tp = entry - self.atr_stop * rr_use * atr
            t1 = entry - self.atr_stop * atr

        risk_pts = abs(entry - sl)

        return {
            "direction":      direction,
            "strategy":       strategy,
            "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
            "regime":         regime,
            "entry":          round(entry, 2),
            "stop":           round(sl, 2),
            "target":         round(tp, 2),
            "target_1":       round(t1, 2),   # partial exit / BE trigger at 1R
            "target_2":       round(tp, 2),
            "risk_pts":       round(risk_pts, 2),
            "rr":             rr_use,
            "score":          round(score, 3),
            "adx":            round(float(df["adx"].iloc[i]), 1),
            "bar_time":       bar_time_et.strftime("%Y-%m-%d %H:%M ET"),
            "timeframe":      "5m",
        }

    # ─────────────────────────────────────────────────────────── #
    # Internal helpers                                             #
    # ─────────────────────────────────────────────────────────── #

    def _prepare(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Add indicators, ET timestamps, ORB arrays, and day context."""
        df = df_raw.copy()
        df = add_all(df)  # adds ema8, ema21, ema55, ema200, atr, adx, bb_upper, bb_lower, rsi, vwap

        # ET timestamps (needed for RTH gating and ORB detection)
        df["ts_et"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        df["date_et"] = df["ts_et"].dt.date
        df["hour_et"] = df["ts_et"].dt.hour
        df["min_et"]  = df["ts_et"].dt.minute

        # ── ORB precompute ───────────────────────────────────────── #
        # ORB = high/low of first 30 min (9:30-9:55 ET, 6 five-min bars)
        orb_high = np.full(len(df), np.nan)
        orb_low  = np.full(len(df), np.nan)
        orb_tight = np.zeros(len(df), dtype=bool)
        orb_atr  = np.full(len(df), np.nan)

        # Group by trading date to compute ORB per day
        for date, group_idx in df.groupby("date_et").groups.items():
            group_idx = sorted(group_idx)
            day_df = df.iloc[group_idx]

            # Find the opening range bars (9:30-9:55 ET)
            orb_mask = (
                ((day_df["hour_et"] == 9) & (day_df["min_et"] >= 30)) |
                ((day_df["hour_et"] == 9) & (day_df["min_et"] == 55))
            )
            # More precisely: hour==9 and minute in [30,35,40,45,50,55]
            orb_mask = (day_df["hour_et"] == 9) & (day_df["min_et"] >= 30)
            orb_bars = day_df[orb_mask]

            if len(orb_bars) < 3:  # need at least 3 bars to be valid
                continue

            oh = float(orb_bars["high"].max())
            ol = float(orb_bars["low"].min())
            # Use average ATR from ORB bars
            oa = float(orb_bars["atr"].mean()) if "atr" in orb_bars.columns else (oh - ol)

            tight = (oh - ol) < 2.0 * oa  # tight ORB = range < 2.0× ATR (relaxed from 1.5)

            # Apply to all bars in this day AFTER the ORB window
            for idx in group_idx:
                row = df.iloc[idx]
                if not (row["hour_et"] == 9 and row["min_et"] >= 30):
                    # Post-ORB bars get the day's ORB values
                    orb_high[idx]  = oh
                    orb_low[idx]   = ol
                    orb_tight[idx] = tight
                    orb_atr[idx]   = oa

        df["orb_high"]  = orb_high
        df["orb_low"]   = orb_low
        df["orb_tight"] = orb_tight
        df["orb_atr"]   = orb_atr

        # ── VWAP above/below streak ──────────────────────────────── #
        # Count consecutive RTH bars close was above VWAP.
        # Rules:
        #   - Non-RTH bars (overnight/pre-market) always get streak=0
        #   - The first RTH bar of each session (9:30) always starts at 0
        #     (prev bar is pre-market, so RTH context is fresh)
        #   - Streak only accumulates within RTH hours
        # This prevents overnight price drift from pre-loading a false streak.
        is_rth_bar = np.array([
            self._is_rth(int(df["hour_et"].iloc[j]), int(df["min_et"].iloc[j]))
            for j in range(len(df))
        ], dtype=bool)

        above_vwap = (df["close"] > df["vwap"]).astype(int)
        streak = np.zeros(len(df), dtype=int)
        for j in range(1, len(df)):
            if not is_rth_bar[j]:
                # Non-RTH bar: no streak
                streak[j] = 0
            elif not is_rth_bar[j - 1]:
                # First RTH bar of the session (prev was pre-market) — fresh start
                streak[j] = 0
            elif above_vwap.iloc[j - 1]:
                streak[j] = streak[j - 1] + 1
            else:
                streak[j] = 0
        df["vwap_above_streak"] = streak

        # ── Last RTH bar per day ─────────────────────────────────── #
        is_last_rth = np.zeros(len(df), dtype=bool)
        for date, group_idx in df.groupby("date_et").groups.items():
            group_idx = sorted(group_idx)
            # Find last bar in RTH for this day
            for idx in reversed(group_idx):
                row = df.iloc[idx]
                if self._is_rth(row["hour_et"], row["min_et"]):
                    is_last_rth[idx] = True
                    break
        df["is_last_rth"] = is_last_rth

        return df

    def _is_rth(self, h: int, m: int) -> bool:
        """True if hour/minute is within RTH session (9:30 AM - 3:55 PM ET)."""
        if h < RTH_START_H or h > RTH_END_H:
            return False
        if h == RTH_START_H and m < RTH_START_M:
            return False
        if h == RTH_END_H and m > RTH_END_M:
            return False
        return True

    def _is_orb_window(self, h: int, m: int) -> bool:
        """True if current bar is in the ORB signal window (10:00-11:55 ET)."""
        if h < ORB_SIGNAL_START_H:
            return False
        if h > ORB_SIGNAL_END_H:
            return False
        if h == ORB_SIGNAL_END_H and m >= ORB_SIGNAL_END_M:
            return False
        return True

    def _score(self, df: pd.DataFrame, i: int) -> tuple[str, str, float, str] | None:
        """
        Score bar i across all strategies.
        Returns (direction, strategy, score, regime) or None.
        """
        adx = float(df["adx"].iloc[i])
        if adx < self.min_adx:
            # ADX too weak — market not moving enough for intraday setups
            # But still allow VWAP pullback if we can find some trend
            pass  # We'll gate individually per strategy

        results: list[tuple[float, str, str, str]] = []  # score, direction, strategy, regime

        # ── Strategy 1: ORB ──────────────────────────────────────── #
        orb = self._orb(df, i)
        if orb:
            results.append(orb)

        # ── Strategy 2: VWAP Pullback ────────────────────────────── #
        vp = self._vwap_pullback(df, i)
        if vp:
            results.append(vp)

        # ── Strategy 3: EMA21 Stack Pullback ─────────────────────── #
        # Disabled by default — performs poorly in high-volatility/selloff regimes.
        # Enable with enable_ema_stack=True when market structure is clean/trending.
        if self.enable_ema_stack:
            es = self._ema_stack_5m(df, i)
            if es:
                results.append(es)

        if not results:
            return None

        # Pick highest score
        best = max(results, key=lambda x: x[0])
        score, direction, strategy, regime = best

        if score < self.min_score:
            return None

        # Apply short filter
        if direction == "SELL" and not self.allow_short:
            return None

        return direction, strategy, score, regime

    # ─────────────────────────────────────────────────────────── #
    # Strategies                                                   #
    # ─────────────────────────────────────────────────────────── #

    def _orb(self, df: pd.DataFrame, i: int) -> tuple[float, str, str, str] | None:
        """
        Opening Range Breakout.
        Fires when price breaks out of the first 30-min range (9:30-9:55 ET).
        Only during 10:00 AM - 12:00 PM ET. Requires tight ORB and ADX < 30.
        """
        h, m   = int(df["hour_et"].iloc[i]), int(df["min_et"].iloc[i])
        if not self._is_orb_window(h, m):
            return None

        orb_h  = df["orb_high"].iloc[i]
        orb_l  = df["orb_low"].iloc[i]
        tight  = bool(df["orb_tight"].iloc[i])

        if np.isnan(orb_h) or np.isnan(orb_l):
            return None

        # Only fire on tight ORB (compressed, coiling price action)
        if not tight:
            return None

        cl   = df["close"].iloc[i]
        prev_cl = df["close"].iloc[i - 1]
        adx  = float(df["adx"].iloc[i])

        # Don't trade ORB if already in strong trend (ADX > 30 = already moving)
        if adx > 30:
            return None

        # Bullish breakout: close above ORB high on this bar (prev was below or at)
        if cl > orb_h and prev_cl <= orb_h:
            score = 0.55
            if adx > 20: score += 0.05     # mild trend boost
            if adx > 25: score += 0.05
            return score, "BUY", "orb", "orb_breakout"

        # Bearish breakdown: close below ORB low
        if cl < orb_l and prev_cl >= orb_l:
            score = 0.55
            if adx > 20: score += 0.05
            if adx > 25: score += 0.05
            return score, "SELL", "orb", "orb_breakdown"

        return None

    def _vwap_pullback(self, df: pd.DataFrame, i: int) -> tuple[float, str, str, str] | None:
        """
        VWAP Pullback.
        Price has been trending above VWAP (5+ RTH bars), pulls back to VWAP,
        then closes back above. Requires macro regime alignment (close > EMA200).

        Short mirror: trending below VWAP, rallies to VWAP, closes back below.
        Requires close < EMA200.

        NOT allowed in first 30 min of RTH (before 10:00 AM ET) — opening range
        is too noisy to reliably determine VWAP direction.
        """
        if i < 5:
            return None

        # Require at least 10:00 AM ET before taking VWAP pullback setups
        h = int(df["hour_et"].iloc[i])
        m = int(df["min_et"].iloc[i])
        if h < 10:
            return None

        vwap    = df["vwap"].iloc[i]
        cl      = df["close"].iloc[i]
        prev_cl = df["close"].iloc[i - 1]
        streak  = int(df["vwap_above_streak"].iloc[i - 1])  # RTH-only streak

        adx   = float(df["adx"].iloc[i])
        ema21 = float(df["ema_21"].iloc[i])
        ema55 = float(df["ema_55"].iloc[i])
        e200  = float(df["ema_200"].iloc[i])

        # ADX must show momentum
        if adx < self.min_adx:
            return None

        # ── Long: was above VWAP 8+ RTH bars, dipped to VWAP, closed back above ─ #
        # 8 bars = 40 min above VWAP — enough to confirm a real intraday uptrend.
        # VWAP must also be rising (slope > 0 over last 3 bars) to ensure we're
        # riding an established intraday uptrend, not a random bounce.
        vwap_slope_up = (i >= 3) and (float(df["vwap"].iloc[i]) > float(df["vwap"].iloc[i - 3]))
        if streak >= 8 and vwap_slope_up and prev_cl <= vwap * 1.001 and cl > vwap:
            # Macro regime: close must be above EMA200 (uptrend)
            if cl < e200:
                return None
            # EMA21 should be above VWAP (confirming intraday uptrend)
            if ema21 < vwap:
                return None
            score = 0.55
            if adx > 22:  score += 0.05
            if adx > 28:  score += 0.05
            if ema55 > e200 * 1.001: score += 0.05  # EMA55 > EMA200 = healthy uptrend
            return score, "BUY", "vwap_pullback", "vwap_trend"

        # ── Short: was below VWAP 5+ RTH bars, rallied to VWAP, closed back below ─ #
        # Count ONLY RTH bars in the below-streak (skip pre-market / overnight)
        below_streak = 0
        for k in range(i - 1, max(i - 20, -1), -1):
            kh = int(df["hour_et"].iloc[k])
            km = int(df["min_et"].iloc[k])
            if not self._is_rth(kh, km):
                break  # hit a non-RTH bar — stop counting (don't jump across sessions)
            if df["close"].iloc[k] < df["vwap"].iloc[k]:
                below_streak += 1
            else:
                break

        # VWAP slope must be falling (slope < 0 over last 3 bars)
        vwap_slope_dn = (i >= 3) and (float(df["vwap"].iloc[i]) < float(df["vwap"].iloc[i - 3]))
        if below_streak >= 8 and vwap_slope_dn and prev_cl >= vwap * 0.999 and cl < vwap:
            # Macro regime: close must be below EMA200 (downtrend)
            if cl > e200:
                return None
            if ema21 > vwap:
                return None
            score = 0.55
            if adx > 22:  score += 0.05
            if adx > 28:  score += 0.05
            if ema55 < e200 * 0.999: score += 0.05
            return score, "SELL", "vwap_pullback", "vwap_trend"

        return None

    def _ema_stack_5m(self, df: pd.DataFrame, i: int) -> tuple[float, str, str, str] | None:
        """
        EMA21 Stack Pullback.
        EMA8 > EMA21 > EMA55 (bullish stack). Price pulls back to EMA21,
        low touches EMA21, closes back above. Requires macro alignment (close > EMA200).

        Short mirror: EMA8 < EMA21 < EMA55. Requires close < EMA200.
        """
        e8   = float(df["ema_8"].iloc[i])
        e21  = float(df["ema_21"].iloc[i])
        e55  = float(df["ema_55"].iloc[i])
        e200 = float(df["ema_200"].iloc[i])
        cl   = float(df["close"].iloc[i])
        lo   = float(df["low"].iloc[i])
        hi   = float(df["high"].iloc[i])
        prev_cl = float(df["close"].iloc[i - 1])
        adx  = float(df["adx"].iloc[i])
        dmp  = float(df["dmp"].iloc[i])   # +DI
        dmn  = float(df["dmn"].iloc[i])   # -DI

        # EMA stack requires STRONG trend (ADX > 25) to avoid choppy-market noise
        # This is intentionally stricter than the engine's global min_adx
        EMA_STACK_MIN_ADX = max(self.min_adx, 25.0)
        if adx < EMA_STACK_MIN_ADX:
            return None

        STACK_GAP = 0.001  # 0.1% gap — tighter than before to avoid false stacks

        # ── Bullish stack ────────────────────────────────────────── #
        bullish_stack = (e8 > e21 * (1 + STACK_GAP)) and (e21 > e55 * (1 + STACK_GAP))
        if bullish_stack:
            # Macro regime: close must be above EMA200
            if cl < e200:
                return None
            # DI alignment: +DI must dominate (buying pressure confirmed)
            if np.isnan(dmp) or np.isnan(dmn) or dmp <= dmn:
                return None
            # Pullback: previous close was at or below EMA21, current close above EMA21
            if prev_cl <= e21 * 1.001 and cl > e21:
                # Low must have tagged EMA21 zone (real pullback)
                if lo <= e21 * 1.003:
                    score = 0.55
                    if adx > 22: score += 0.05
                    if adx > 28: score += 0.05
                    if e8 > e21 * (1 + 2 * STACK_GAP): score += 0.05  # strong stack
                    if e55 > e200 * 1.001: score += 0.05  # healthy longer-term trend
                    return score, "BUY", "ema_stack_5m", "ema_stack_bull"

        # ── Bearish stack ────────────────────────────────────────── #
        bearish_stack = (e8 < e21 * (1 - STACK_GAP)) and (e21 < e55 * (1 - STACK_GAP))
        if bearish_stack:
            # Macro regime: close must be below EMA200
            if cl > e200:
                return None
            # DI alignment: -DI must dominate (selling pressure confirmed)
            if np.isnan(dmp) or np.isnan(dmn) or dmn <= dmp:
                return None
            if prev_cl >= e21 * 0.999 and cl < e21:
                if hi >= e21 * 0.997:
                    score = 0.55
                    if adx > 22: score += 0.05
                    if adx > 28: score += 0.05
                    if e8 < e21 * (1 - 2 * STACK_GAP): score += 0.05
                    if e55 < e200 * 0.999: score += 0.05
                    return score, "SELL", "ema_stack_5m", "ema_stack_bear"

        return None


# ─────────────────────────────────────────────────────────────────── #
# Multi-market batch runner                                            #
# ─────────────────────────────────────────────────────────────────── #

FUTURES_UNIVERSE = {
    "ES=F":  "S&P 500",
    "NQ=F":  "Nasdaq 100",
    "GC=F":  "Gold",
    "CL=F":  "Crude Oil",
    "RTY=F": "Russell 2000",
    "YM=F":  "Dow Jones",
}


def run_multi_market(
    days_back: int = 60,
    engine_kwargs: dict | None = None,
) -> dict[str, BacktestResult | Exception]:
    """
    Run BacktestEngine5m on all futures in FUTURES_UNIVERSE in parallel.
    Returns dict of symbol → BacktestResult (or Exception on failure).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data.fetcher import fetch_historical

    kwargs = engine_kwargs or {}
    results: dict[str, BacktestResult | Exception] = {}

    def _fetch_and_run(symbol: str) -> tuple[str, BacktestResult | Exception]:
        try:
            df = fetch_historical(symbol, "5m", days_back)
            engine = BacktestEngine5m(**kwargs)
            result = engine.run(df, symbol=symbol, timeframe="5m")
            return symbol, result
        except Exception as e:
            return symbol, e

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_and_run, sym): sym for sym in FUTURES_UNIVERSE}
        for fut in as_completed(futures):
            sym, res = fut.result()
            results[sym] = res

    return results
