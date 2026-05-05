"""
VWAPPullbackEngine — VWAP Pullback / VWAP Rejection (5-minute bars)
────────────────────────────────────────────────────────────────────
Concept: In a trending market, institutional order flow clusters around
VWAP. Price pulls back to VWAP (in an uptrend) or bounces to VWAP (in a
downtrend), and smart money steps in to defend the level. Fade the move
away from VWAP in the direction of the primary trend.

STRATEGY LOGIC:
  LONG (uptrend pullback to VWAP):
    1. Trend filter: EMA8 > EMA21 > EMA55 (stacked bull) and price above EMA21
    2. +DI > -DI and ADX >= min_adx
    3. Recent bars show at least 6-of-10 closing ABOVE VWAP (bias confirmed)
    4. Current bar touches or crosses VWAP from above (pullback to support)
    5. Reversal candle at VWAP: hammer, bullish engulfing, or strong bull close
    6. RSI in recovery zone (38–62) — not oversold and not overbought

  SHORT (downtrend bounce to VWAP):
    1. Trend filter: EMA8 < EMA21 < EMA55 (stacked bear) and price below EMA21
    2. -DI > +DI and ADX >= min_adx
    3. Recent bars show at least 6-of-10 closing BELOW VWAP (bias confirmed)
    4. Current bar touches or bounces to VWAP from below (rejection at resistance)
    5. Reversal candle at VWAP: shooting star, bearish engulfing, or strong bear close
    6. RSI in rejection zone (38–62) — not overbought and not oversold

TIMING:
  - Entry window: 10:00 AM – 3:30 PM ET (fills the gap after ORB closes at 10:15)
  - Complements A+ (which requires Initial Balance context)
  - Force-close: 3:55 PM ET
  - Max hold: 30 bars (~2.5 hours on 5m)
  - Max 2 trades per day

STOPS & TARGETS:
  - Stop:     VWAP ± 0.8 ATR (beyond key level)
  - Target 1: 2.0R  (trim, move stop to breakeven)
  - Target 2: 3.0R  (runner — VWAP pullbacks can run hard)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytz

from utils.indicators import add_all
from backtesting.models import BacktestTrade, BacktestResult

ET = pytz.timezone("America/New_York")

STRATEGY_LABEL = "VWAP Pullback"

# Timing
ENTRY_START_H, ENTRY_START_M = 10, 0
ENTRY_END_H,   ENTRY_END_M   = 15, 30
EOD_H,         EOD_M         = 15, 55
MAX_BARS       = 30
MAX_TRADES_DAY = 2

# Signal params
MIN_BOUNCE_BARS   = 7    # of last 10 bars on correct side of VWAP (relaxed from 8 — 8 was too strict)
VWAP_TOL_ATR      = 0.40 # proximity to VWAP — slightly widened from 0.35 to catch more genuine touches
STOP_ATR_MULT     = 0.8  # stop beyond VWAP
T1_R              = 1.0  # take profits sooner — choppy markets rarely sustain 1.5R+ moves
T2_R              = 2.0  # runner target — lowered from 3.0, more realistic on VWAP plays


class VWAPPullbackEngine:
    """VWAP Pullback / Rejection — 5-minute bars. Long and short capable."""

    def __init__(
        self,
        min_adx:          float = 20.0,  # raised from 18 — need genuine trend, not just drift
        min_score:        float = 0.65,
        allow_long:       bool  = True,
        allow_short:      bool  = True,
        skip_monday:      bool  = False,
        skip_power_hour:  bool  = False, # skip 14:00-15:30 ET entries (IB stale, chop increases)
        t1_r:             float = T1_R,
        t2_r:             float = T2_R,
        stop_atr_mult:    float = STOP_ATR_MULT,
        max_trades_day:   int   = 1,     # one high-quality VWAP touch per day
        max_atr_multiple: float = 1.5,   # skip entries when ATR > N× its own 20-bar rolling avg
    ):
        self.min_adx          = min_adx
        self.min_score        = min_score
        self.allow_long       = allow_long
        self.allow_short      = allow_short
        self.skip_monday      = skip_monday
        self.skip_power_hour  = skip_power_hour
        self.t1_r             = t1_r
        self.t2_r             = t2_r
        self.stop_atr_mult    = stop_atr_mult
        self.max_trades_day   = max_trades_day
        self.max_atr_multiple = max_atr_multiple

    # ─────────────────────────────────────────────────────────────── #
    # Public API                                                       #
    # ─────────────────────────────────────────────────────────────── #

    def run(self, df_raw: pd.DataFrame, symbol: str = "NQ=F", timeframe: str = "5m") -> BacktestResult:
        df = self._prepare(df_raw)
        n  = len(df)
        result = BacktestResult(symbol=symbol, timeframe=timeframe, market="futures", total_bars=n)

        open_trade: dict | None = None
        equity   = 0.0
        be_moved = False
        day_counts: dict[str, int] = {}

        for i in range(50, n):
            bar_time_et = df["ts_et"].iloc[i]
            h, m = bar_time_et.hour, bar_time_et.minute
            day_key = bar_time_et.strftime("%Y-%m-%d")

            # ── Manage open trade ─────────────────────────────────── #
            if open_trade is not None:
                lo      = float(df["low"].iloc[i])
                hi      = float(df["high"].iloc[i])
                cl      = float(df["close"].iloc[i])
                atr_now = float(df["atr"].iloc[i])
                d       = open_trade["dir"]

                eod     = (h == EOD_H and m >= EOD_M) or h > EOD_H
                timeout = (i - open_trade["entry_bar"]) >= MAX_BARS

                # ── Trailing stop mode (after T2 broken through) ──── #
                if open_trade.get("t2_hit", False):
                    trail_sl = open_trade["trail_sl"]
                    if d == 1:
                        open_trade["trail_sl"] = max(trail_sl, hi - 0.8 * atr_now)
                    else:
                        open_trade["trail_sl"] = min(trail_sl, lo + 0.8 * atr_now)
                    trail_sl = open_trade["trail_sl"]
                    hit_trail = (d == 1 and lo <= trail_sl) or (d == -1 and hi >= trail_sl)
                    if hit_trail:
                        exit_px, reason = trail_sl, "trail_stop"
                    elif eod:
                        exit_px, reason = cl, "eod"
                    elif timeout:
                        exit_px, reason = cl, "timeout"
                    else:
                        continue

                # ── Normal management ─────────────────────────────── #
                else:
                    if not be_moved:
                        t1 = open_trade["t1"]
                        if (d == 1 and hi >= t1) or (d == -1 and lo <= t1):
                            open_trade["sl"] = open_trade["entry"]
                            be_moved = True

                    hit_t2   = (d ==  1 and hi >= open_trade["t2"]) or (d == -1 and lo <= open_trade["t2"])
                    hit_t1   = (d ==  1 and hi >= open_trade["t1"]) or (d == -1 and lo <= open_trade["t1"])
                    hit_stop = (d ==  1 and lo <= open_trade["sl"]) or (d == -1 and hi >= open_trade["sl"])

                    # Priority: stop > T2 > T1 > eod > timeout
                    if hit_stop:
                        exit_px, reason = open_trade["sl"], "stop"
                    elif hit_t2:
                        _clear_break = (d == 1 and hi >= open_trade["t2"] + 1.0 * atr_now) or \
                                       (d == -1 and lo <= open_trade["t2"] - 1.0 * atr_now)
                        if _clear_break:
                            open_trade["t2_hit"] = True
                            open_trade["trail_sl"] = open_trade["t2"] - 0.8 * atr_now if d == 1 else open_trade["t2"] + 0.8 * atr_now
                            continue
                        else:
                            exit_px, reason = open_trade["t2"], "target2"
                    elif hit_t1 and be_moved:
                        exit_px, reason = open_trade["t1"], "target1"
                    elif eod:
                        exit_px, reason = cl, "eod"
                    elif timeout:
                        exit_px, reason = cl, "timeout"
                    else:
                        continue

                pnl    = (exit_px - open_trade["entry"]) * d
                risk   = abs(open_trade["entry"] - open_trade["sl_orig"])
                r_mult = pnl / risk if risk > 0 else 0.0
                equity += pnl
                be_moved = False

                result.trades.append(BacktestTrade(
                    strategy    = "vwap_pullback",
                    direction   = "BUY" if d == 1 else "SELL",
                    entry_bar   = open_trade["entry_bar"],
                    exit_bar    = i,
                    entry_price = open_trade["entry"],
                    exit_price  = round(exit_px, 4),
                    stop_loss   = open_trade["sl_orig"],
                    take_profit = open_trade["t2"],
                    exit_reason = reason,
                    pnl_pts     = round(pnl, 4),
                    r_multiple  = round(r_mult, 2),
                    bars_held   = i - open_trade["entry_bar"],
                    composite_score = open_trade["score"],
                    regime      = open_trade["regime"],
                    be_moved    = be_moved,
                ))
                result.equity_curve.append(round(equity, 4))
                open_trade = None
                continue

            # ── Filters ───────────────────────────────────────────── #
            if self.skip_monday and bar_time_et.weekday() == 0:
                continue
            if not self._in_window(h, m):
                continue
            if day_counts.get(day_key, 0) >= self.max_trades_day:
                continue

            # ── Check signal ──────────────────────────────────────── #
            sig = self._check_signal(df, i)
            if sig is None:
                continue

            direction, score, regime, sl, t1, t2 = sig
            if score < self.min_score:
                continue
            if direction == "BUY" and not self.allow_long:
                continue
            if direction == "SELL" and not self.allow_short:
                continue

            entry    = float(df["close"].iloc[i])
            risk_pts = abs(entry - sl)
            if risk_pts <= 0:
                continue

            open_trade = {
                "entry_bar": i,
                "dir":       1 if direction == "BUY" else -1,
                "entry":     entry,
                "sl":        round(sl, 4),
                "sl_orig":   round(sl, 4),
                "t1":        round(t1, 4),
                "t2":        round(t2, 4),
                "score":     score,
                "regime":    regime,
                "t2_hit":    False,
                "trail_sl":  None,
            }
            day_counts[day_key] = day_counts.get(day_key, 0) + 1
            be_moved = False

        return result

    def live_signal(self, df_raw: pd.DataFrame, current_price: float | None = None) -> dict | None:
        df = self._prepare(df_raw)
        n  = len(df)
        if n < 55:
            return None

        i = n - 1
        bar_time_et = df["ts_et"].iloc[i]
        h, m = bar_time_et.hour, bar_time_et.minute

        if self.skip_monday and bar_time_et.weekday() == 0:
            return None
        if not self._in_window(h, m):
            return None

        sig = self._check_signal(df, i)
        if sig is None:
            return None

        direction, score, regime, sl, t1, t2 = sig
        if score < self.min_score:
            return None
        if direction == "BUY" and not self.allow_long:
            return None
        if direction == "SELL" and not self.allow_short:
            return None

        entry    = current_price if current_price else float(df["close"].iloc[i])
        risk_pts = abs(entry - sl)
        if risk_pts <= 0:
            return None

        rr = abs(entry - t2) / risk_pts if risk_pts > 0 else 0.0
        return {
            "direction":      direction,
            "strategy":       "vwap_pullback",
            "strategy_label": STRATEGY_LABEL,
            "regime":         regime,
            "entry":          round(entry, 2),
            "stop":           round(sl, 2),
            "target":         round(t2, 2),
            "target_1":       round(t1, 2),
            "target_2":       round(t2, 2),
            "risk_pts":       round(risk_pts, 2),
            "rr":             round(rr, 2),
            "score":          round(score, 3),
            "adx":            round(float(df["adx"].iloc[i]), 1),
            "bar_time":       bar_time_et.strftime("%Y-%m-%d %H:%M ET"),
            "timeframe":      "5m",
        }

    # ─────────────────────────────────────────────────────────────── #
    # Signal Detection                                                 #
    # ─────────────────────────────────────────────────────────────── #

    def _check_signal(self, df: pd.DataFrame, i: int) -> tuple | None:
        long_sig  = self._long_signal(df, i)
        short_sig = self._short_signal(df, i)
        if long_sig and short_sig:
            return long_sig if long_sig[1] >= short_sig[1] else short_sig
        return long_sig or short_sig

    def _long_signal(self, df: pd.DataFrame, i: int) -> tuple | None:
        """VWAP pullback long: uptrend, price dips to VWAP, reversal candle."""
        if i < 12:
            return None

        cl   = float(df["close"].iloc[i])
        op   = float(df["open"].iloc[i])
        hi   = float(df["high"].iloc[i])
        lo   = float(df["low"].iloc[i])
        e8   = float(df["ema_8"].iloc[i])
        e21  = float(df["ema_21"].iloc[i])
        e55  = float(df["ema_55"].iloc[i])
        atr  = float(df["atr"].iloc[i])
        adx  = float(df["adx"].iloc[i])
        dmp  = float(df["dmp"].iloc[i])
        dmn  = float(df["dmn"].iloc[i])
        rsi  = float(df["rsi"].iloc[i])
        vwap = float(df["vwap"].iloc[i])

        if any(np.isnan(v) for v in [cl, e8, e21, e55, atr, adx, dmp, dmn, rsi, vwap]):
            return None
        if atr <= 0 or adx < self.min_adx:
            return None

        # ATR spike filter — skip entries on extreme volatility days
        # (tariff shock / FOMC surprise — structural stop levels not trustworthy)
        atr_avg20 = float(df["atr_avg20"].iloc[i])
        if not np.isnan(atr_avg20) and atr_avg20 > 0:
            if atr > self.max_atr_multiple * atr_avg20:
                return None

        # 1. Bullish EMA stack
        if not (e8 > e21 > e55):
            return None

        # 2. Price above EMA21 overall (we're in a dip, not a reversal)
        if cl < e21 * 0.997:   # allow tiny breach
            return None

        # 3. DI alignment
        if dmp <= dmn:
            return None

        # 4. Recent bars mostly above VWAP
        recent = df.iloc[max(0, i - 10): i]
        above_vwap = sum(
            1 for _, b in recent.iterrows()
            if not np.isnan(float(b.get("vwap", np.nan)))
            and float(b["close"]) > float(b.get("vwap", np.nan))
        )
        if above_vwap < MIN_BOUNCE_BARS:
            return None

        # 5. Price touched VWAP (low dipped into VWAP zone)
        at_vwap = lo <= vwap + VWAP_TOL_ATR * atr
        if not at_vwap:
            return None

        # 5b. Close must be BACK ABOVE VWAP (genuine support, not still below)
        if cl <= vwap:
            return None

        # 6. RSI in recovery zone
        if not (40 <= rsi <= 63):
            return None

        # 7. Reversal candle
        bullish_eng  = self._is_bullish_engulfing(df, i)
        hammer       = self._is_hammer(op, hi, lo, cl)
        strong_close = cl > op and (cl - lo) / (hi - lo + 1e-8) >= 0.70
        if not (bullish_eng or hammer or strong_close):
            return None

        # 8. MACD turning up (momentum confirmation)
        mhist      = float(df["macd_hist"].iloc[i])
        mhist_prev = float(df["macd_hist"].iloc[i - 1]) if i >= 1 else np.nan
        if np.isnan(mhist) or np.isnan(mhist_prev) or mhist <= mhist_prev:
            return None

        # 9. Volume confirmation
        vol = float(df["volume"].iloc[i])
        va  = float(df["vol_avg"].iloc[i])
        if np.isnan(va) or va <= 0 or vol < 1.1 * va:
            return None

        # ── Levels ────────────────────────────────────────────────── #
        sl   = vwap - self.stop_atr_mult * atr
        risk = cl - sl
        if risk <= 0 or risk > 3 * atr:
            return None
        t1 = cl + self.t1_r * risk
        t2 = cl + self.t2_r * risk

        # ── Score ─────────────────────────────────────────────────── #
        score = 0.58
        if bullish_eng:    score += 0.10
        elif hammer:       score += 0.08
        elif strong_close: score += 0.05
        if above_vwap >= 8:   score += 0.04   # very consistent above VWAP
        if adx >= 25:         score += 0.04
        vol    = float(df["volume"].iloc[i])
        va     = float(df["vol_avg"].iloc[i])
        if not np.isnan(va) and va > 0 and vol >= va:
            score += 0.04
        if 45 <= rsi <= 58:   score += 0.03   # ideal RSI sweet spot
        e200 = float(df["ema_200"].iloc[i])
        if not np.isnan(e200) and cl > e200:
            score += 0.02

        return ("BUY", round(score, 3), "vwap_pullback_long", round(sl, 4), round(t1, 4), round(t2, 4))

    def _short_signal(self, df: pd.DataFrame, i: int) -> tuple | None:
        """VWAP rejection short: downtrend, price bounces to VWAP, reversal candle."""
        if i < 12:
            return None

        cl   = float(df["close"].iloc[i])
        op   = float(df["open"].iloc[i])
        hi   = float(df["high"].iloc[i])
        lo   = float(df["low"].iloc[i])
        e8   = float(df["ema_8"].iloc[i])
        e21  = float(df["ema_21"].iloc[i])
        e55  = float(df["ema_55"].iloc[i])
        atr  = float(df["atr"].iloc[i])
        adx  = float(df["adx"].iloc[i])
        dmp  = float(df["dmp"].iloc[i])
        dmn  = float(df["dmn"].iloc[i])
        rsi  = float(df["rsi"].iloc[i])
        vwap = float(df["vwap"].iloc[i])

        if any(np.isnan(v) for v in [cl, e8, e21, e55, atr, adx, dmp, dmn, rsi, vwap]):
            return None
        if atr <= 0 or adx < self.min_adx:
            return None

        # ATR spike filter — skip entries on extreme volatility days
        atr_avg20 = float(df["atr_avg20"].iloc[i])
        if not np.isnan(atr_avg20) and atr_avg20 > 0:
            if atr > self.max_atr_multiple * atr_avg20:
                return None

        # 1. Bearish EMA stack
        if not (e8 < e21 < e55):
            return None

        # 2. Price below EMA21 overall
        if cl > e21 * 1.003:   # allow tiny breach
            return None

        # 3. DI alignment
        if dmn <= dmp:
            return None

        # 4. Recent bars mostly below VWAP
        recent = df.iloc[max(0, i - 10): i]
        below_vwap = sum(
            1 for _, b in recent.iterrows()
            if not np.isnan(float(b.get("vwap", np.nan)))
            and float(b["close"]) < float(b.get("vwap", np.nan))
        )
        if below_vwap < MIN_BOUNCE_BARS:
            return None

        # 5. Price touched VWAP (high bounced into VWAP zone)
        at_vwap = hi >= vwap - VWAP_TOL_ATR * atr
        if not at_vwap:
            return None

        # 5b. Close must be BACK BELOW VWAP (genuine resistance, not still above)
        if cl >= vwap:
            return None

        # 6. RSI in rejection zone
        if not (37 <= rsi <= 60):
            return None

        # 7. Reversal candle
        bearish_eng   = self._is_bearish_engulfing(df, i)
        shooting_star = self._is_shooting_star(op, hi, lo, cl)
        strong_close  = cl < op and (cl - lo) / (hi - lo + 1e-8) <= 0.30
        if not (bearish_eng or shooting_star or strong_close):
            return None

        # 8. MACD turning down (momentum confirmation)
        mhist      = float(df["macd_hist"].iloc[i])
        mhist_prev = float(df["macd_hist"].iloc[i - 1]) if i >= 1 else np.nan
        if np.isnan(mhist) or np.isnan(mhist_prev) or mhist >= mhist_prev:
            return None

        # 9. Volume confirmation
        vol = float(df["volume"].iloc[i])
        va  = float(df["vol_avg"].iloc[i])
        if np.isnan(va) or va <= 0 or vol < 1.1 * va:
            return None

        # ── Levels ────────────────────────────────────────────────── #
        sl   = vwap + self.stop_atr_mult * atr
        risk = sl - cl
        if risk <= 0 or risk > 3 * atr:
            return None
        t1 = cl - self.t1_r * risk
        t2 = cl - self.t2_r * risk

        # ── Score ─────────────────────────────────────────────────── #
        score = 0.58
        if bearish_eng:      score += 0.10
        elif shooting_star:  score += 0.08
        elif strong_close:   score += 0.05
        if below_vwap >= 8:  score += 0.04
        if adx >= 25:        score += 0.04
        vol    = float(df["volume"].iloc[i])
        va     = float(df["vol_avg"].iloc[i])
        if not np.isnan(va) and va > 0 and vol >= va:
            score += 0.04
        if 38 <= rsi <= 55:  score += 0.03
        e200 = float(df["ema_200"].iloc[i])
        if not np.isnan(e200) and cl < e200:
            score += 0.02

        return ("SELL", round(score, 3), "vwap_rejection_short", round(sl, 4), round(t1, 4), round(t2, 4))

    # ─────────────────────────────────────────────────────────────── #
    # Candle Patterns                                                  #
    # ─────────────────────────────────────────────────────────────── #

    def _is_bullish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        c_op = float(df["open"].iloc[i]);   c_cl = float(df["close"].iloc[i])
        p_op = float(df["open"].iloc[i-1]); p_cl = float(df["close"].iloc[i-1])
        return p_cl < p_op and c_cl > c_op and c_cl >= p_op and c_op <= p_cl

    def _is_bearish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        c_op = float(df["open"].iloc[i]);   c_cl = float(df["close"].iloc[i])
        p_op = float(df["open"].iloc[i-1]); p_cl = float(df["close"].iloc[i-1])
        return p_cl > p_op and c_cl < c_op and c_op >= p_cl and c_cl <= p_op

    def _is_hammer(self, op: float, hi: float, lo: float, cl: float) -> bool:
        body = abs(cl - op)
        rng  = hi - lo
        if rng < 1e-8:
            return False
        lower_shadow = min(op, cl) - lo
        upper_shadow = hi - max(op, cl)
        return (lower_shadow >= 2.0 * body and
                upper_shadow <= 0.5 * max(body, 1e-8) and
                body >= rng * 0.05)

    def _is_shooting_star(self, op: float, hi: float, lo: float, cl: float) -> bool:
        body = abs(cl - op)
        rng  = hi - lo
        if rng < 1e-8:
            return False
        upper_shadow = hi - max(op, cl)
        lower_shadow = min(op, cl) - lo
        return (upper_shadow >= 2.0 * body and
                lower_shadow <= 0.5 * max(body, 1e-8) and
                body >= rng * 0.05)

    # ─────────────────────────────────────────────────────────────── #
    # Helpers                                                          #
    # ─────────────────────────────────────────────────────────────── #

    def _in_window(self, h: int, m: int) -> bool:
        if h < ENTRY_START_H:
            return False
        if h == ENTRY_START_H and m < ENTRY_START_M:
            return False
        if h > ENTRY_END_H:
            return False
        if h == ENTRY_END_H and m > ENTRY_END_M:
            return False
        # Optional: skip power hour (14:00 – 15:30 ET) — IB stale, institutions unwind
        if self.skip_power_hour and (h == 14 or (h == 15 and m <= 30)):
            return False
        return True

    def _prepare(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df = add_all(df)
        if "timestamp" in df.columns:
            df["ts_et"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        else:
            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                df["ts_et"] = idx.tz_convert(ET)
            else:
                df["ts_et"] = idx.tz_localize("UTC").tz_convert(ET)
        # Rolling ATR baseline — used by ATR spike filter
        df["atr_avg20"] = df["atr"].rolling(20).mean()
        return df.reset_index(drop=True)
