"""
BacktestEngine5mIBRetest — 5-Minute IB Retest System
─────────────────────────────────────────────────────
5-minute institutional structure strategy built around the Initial Balance.

STRATEGY LOGIC:
  1. Wait for the 15-minute Initial Balance (IB) to form (9:30–9:44 ET, 3 bars on 5m).
  2. Identify breakout direction: close above IB High (long) or below IB Low (short).
  3. Wait for a RETEST of the IB level or the 85-period rolling VWAP (within 0.75 ATR).
  4. Enter on a reversal candle: Engulfing, Hammer/Shooting Star, or Strong Close.
  5. Confirm with MACD (9,21,16) + RSI direction + EMA8/21 alignment.
  6. Macro filter: the 10:15 AM confirmation candle must validate the bias (optional).

TIMING:
  - Primary entry window: 10:15 AM – 11:30 AM ET
  - Power Hour continuation: 2:00 PM – 3:55 PM ET
  - Lunch Lull (no entries): 11:30 AM – 2:00 PM ET

STOPS (structural):
  - Long: below IB Low (clamped 0.75–1.5 ATR from entry)
  - Short: above IB High (clamped 0.75–1.5 ATR from entry)

EXITS (two-tiered):
  - Target 1: 1σ VWAP-85 band (trim here, move stop to BE)
  - Target 2: 2σ VWAP-85 band (runner — 2.5 R minimum fallback)
  - Breakeven: stop moves to entry once T1 is touched
  - EOD: force-close at 3:55 PM ET
  - Max hold: 36 bars (3 hours on 5m)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytz
import ta

from utils.indicators import add_all
from backtesting.models import BacktestTrade, BacktestResult


ET = pytz.timezone("America/New_York")

# ── Session timing constants ────────────────────────────────────── #
IB_START_H, IB_START_M = 9, 30     # Initial Balance starts (9:30 ET)
IB_END_H,   IB_END_M   = 9, 44     # IB ends after 3 five-min bars (9:44 included)

PRIMARY_START_H, PRIMARY_START_M = 10, 15   # Entry window opens
LUNCH_START_H,   LUNCH_START_M   = 11, 30   # Lunch lull starts (no entries)
POWER_HOUR_H,    POWER_HOUR_M    = 14, 0    # Power Hour (2 PM ET)
EOD_H,           EOD_M           = 15, 55   # Force-close

STRATEGY_LABEL = "5m IB Retest"

# Recommended markets for the A+ strategy.
# YM=F (Dow Jones) excluded — poor IB-retest structure and oversized point-value stops.
APLUS_UNIVERSE = {
    "ES=F":  "S&P 500",
    "NQ=F":  "Nasdaq 100",
    "GC=F":  "Gold",
    "CL=F":  "Crude Oil",
    "RTY=F": "Russell 2000",
}


class BacktestEngineAPlus:
    """
    A+ Institutional Alignment Strategy — 5-minute bars.
    """

    def __init__(
        self,
        max_bars:    int   = 36,      # 3 hours on 5m
        atr_stop_cap: float = 1.25,   # cap stop ATR multiplier (was 2.0 — now enforced)
        min_rr:      float = 1.8,     # skip trade if RR < this (T1 should be ~2R)
        min_adx:     float = 18.0,    # require trending market (ADX threshold)
        min_score:   float = 0.65,    # minimum quality score
        allow_short: bool  = False,   # longs-only default (trend-following bias on index futures)
        require_macro_confirm: bool = False,  # 10:15 candle confirms direction (strict — off by default)
        retest_tolerance: float = 0.5,        # ATR multiplier for retest zone (tight = high quality)
        max_trades_per_day: int = 0,          # 0 = unlimited; 1 = one trade/day; 2 = two/day etc.
        skip_monday: bool = False,            # skip all entries on Monday (weak day historically)
        skip_power_hour_open: bool = False,   # skip 14:00-14:30 (first 30 min of power hour)
        skip_power_hour: bool = False,        # skip ALL of power hour (14:00-15:55) — IB is stale by then
        max_atr_multiple: float = 1.5,        # skip if current ATR > this × 20-bar avg ATR (spike filter)
    ):
        self.max_bars              = max_bars
        self.atr_stop_cap          = atr_stop_cap
        self.min_rr                = min_rr
        self.min_adx               = min_adx
        self.min_score             = min_score
        self.allow_short           = allow_short
        self.require_macro_confirm = require_macro_confirm
        self.retest_tolerance      = retest_tolerance
        self.max_trades_per_day    = max_trades_per_day
        self.skip_monday           = skip_monday
        self.skip_power_hour_open  = skip_power_hour_open
        self.skip_power_hour       = skip_power_hour
        self.max_atr_multiple      = max_atr_multiple

    # ─────────────────────────────────────────────────────────────── #
    # Public API                                                       #
    # ─────────────────────────────────────────────────────────────── #

    def run(self, df_raw: pd.DataFrame, symbol: str = "ES=F", timeframe: str = "5m") -> BacktestResult:
        """Full historical backtest. Returns BacktestResult."""
        df = self._prepare(df_raw)
        n  = len(df)
        result = BacktestResult(symbol=symbol, timeframe=timeframe, market="futures", total_bars=n)

        open_trade: dict | None = None
        equity = 0.0
        be_moved = False
        _day_trade_count: dict[str, int] = {}   # date → trades taken that day

        for i in range(100, n):  # warmup=100 for rolling VWAP-85 + indicators
            bar_time_et = df["ts_et"].iloc[i]
            h, m = bar_time_et.hour, bar_time_et.minute
            _day_key = bar_time_et.strftime("%Y-%m-%d")

            # ── Manage open trade ─────────────────────────────────── #
            if open_trade is not None:
                lo      = float(df["low"].iloc[i])
                hi      = float(df["high"].iloc[i])
                cl      = float(df["close"].iloc[i])
                atr_now = float(df["atr"].iloc[i])

                eod     = (h == EOD_H and m >= EOD_M) or (h > EOD_H)
                timeout = (i - open_trade["entry_bar"]) >= self.max_bars

                # ── Trailing stop mode (after T2 broken through) ──── #
                if open_trade.get("t2_hit", False):
                    trail_sl = open_trade["trail_sl"]
                    if open_trade["dir"] == 1:
                        open_trade["trail_sl"] = max(trail_sl, hi - 0.8 * atr_now)
                    else:
                        open_trade["trail_sl"] = min(trail_sl, lo + 0.8 * atr_now)
                    trail_sl = open_trade["trail_sl"]
                    hit_trail = (open_trade["dir"] == 1 and lo <= trail_sl) or \
                                (open_trade["dir"] == -1 and hi >= trail_sl)
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
                    # Breakeven: move stop to entry once T1 is touched
                    if not be_moved:
                        t1 = open_trade["t1"]
                        if (open_trade["dir"] == 1 and hi >= t1) or \
                           (open_trade["dir"] == -1 and lo <= t1):
                            open_trade["sl"] = open_trade["entry"]
                            be_moved = True

                    hit_t2   = (open_trade["dir"] ==  1 and hi >= open_trade["t2"]) or \
                               (open_trade["dir"] == -1 and lo <= open_trade["t2"])
                    hit_t1   = (open_trade["dir"] ==  1 and hi >= open_trade["t1"]) or \
                               (open_trade["dir"] == -1 and lo <= open_trade["t1"])
                    hit_stop = (open_trade["dir"] ==  1 and lo <= open_trade["sl"]) or \
                               (open_trade["dir"] == -1 and hi >= open_trade["sl"])

                    # Priority: stop > T2 > T1 > eod > timeout
                    if hit_stop:
                        exit_px, reason = open_trade["sl"], "stop"
                    elif hit_t2:
                        # Only trail if price clearly breaks +1 ATR beyond T2
                        _clear_break = (open_trade["dir"] == 1 and hi >= open_trade["t2"] + 1.0 * atr_now) or \
                                       (open_trade["dir"] == -1 and lo <= open_trade["t2"] - 1.0 * atr_now)
                        if _clear_break:
                            open_trade["t2_hit"] = True
                            open_trade["trail_sl"] = open_trade["t2"] - 0.8 * atr_now \
                                if open_trade["dir"] == 1 else open_trade["t2"] + 0.8 * atr_now
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
                        continue  # stay in trade

                pnl  = (exit_px - open_trade["entry"]) * open_trade["dir"]
                risk = abs(open_trade["entry"] - open_trade["sl_orig"])
                r_mult = pnl / risk if risk > 0 else 0.0
                equity += pnl
                be_moved = False

                t = BacktestTrade(
                    strategy    = "aplus",
                    direction   = "BUY" if open_trade["dir"] == 1 else "SELL",
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
                )
                result.trades.append(t)
                result.equity_curve.append(round(equity, 4))
                open_trade = None
                continue

            # ── Day-of-week and time-of-day filters ───────────────── #
            if self.skip_monday and bar_time_et.weekday() == 0:  # 0 = Monday
                continue
            if self.skip_power_hour_open and h == 14 and m < 30:
                continue

            # ── Skip if outside entry windows ─────────────────────── #
            if not self._is_entry_window(h, m):
                continue

            # ── Check for A+ setup ────────────────────────────────── #
            sig = self._score(df, i)
            if sig is None:
                continue

            direction, score, regime, sl, t1, t2 = sig

            # Minimum R:R check
            entry = float(df["close"].iloc[i])
            risk_pts = abs(entry - sl)
            reward_t1 = abs(entry - t1)
            if risk_pts <= 0 or (reward_t1 / risk_pts) < self.min_rr:
                continue

            # Minimum quality score
            if score < self.min_score:
                continue

            if direction == "SELL" and not self.allow_short:
                continue

            # Per-day trade limit (0 = unlimited)
            if self.max_trades_per_day > 0:
                if _day_trade_count.get(_day_key, 0) >= self.max_trades_per_day:
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
            _day_trade_count[_day_key] = _day_trade_count.get(_day_key, 0) + 1
            be_moved = False

        return result

    def live_signal(self, df_raw: pd.DataFrame, current_price: float | None = None) -> dict | None:
        """Evaluate most recently completed bar for a live signal."""
        df = self._prepare(df_raw)
        n  = len(df)
        if n < 105:
            return None

        i = n - 1
        bar_time_utc = df["timestamp"].iloc[i]
        bar_time_et  = bar_time_utc.astimezone(ET)
        h, m = bar_time_et.hour, bar_time_et.minute

        if self.skip_monday and bar_time_et.weekday() == 0:
            return None
        if self.skip_power_hour_open and h == 14 and m < 30:
            return None

        if not self._is_entry_window(h, m):
            return None

        sig = self._score(df, i)
        if sig is None:
            return None

        direction, score, regime, sl, t1, t2 = sig
        entry = current_price if current_price else float(df["close"].iloc[i])
        risk_pts = abs(entry - sl)
        reward_t1 = abs(entry - t1)

        if risk_pts <= 0 or (reward_t1 / risk_pts) < self.min_rr:
            return None

        if direction == "SELL" and not self.allow_short:
            return None

        rr = abs(entry - t2) / risk_pts if risk_pts > 0 else 0.0

        return {
            "direction":      direction,
            "strategy":       "aplus",
            "strategy_label": STRATEGY_LABEL,
            "regime":         regime,
            "entry":          round(entry, 2),
            "stop":           round(sl, 2),
            "target":         round(t2, 2),   # T2 as primary target
            "target_1":       round(t1, 2),   # T1 (trim here, move BE)
            "target_2":       round(t2, 2),
            "risk_pts":       round(risk_pts, 2),
            "rr":             round(rr, 2),
            "score":          round(score, 3),
            "adx":            round(float(df["adx"].iloc[i]), 1),
            "bar_time":       bar_time_et.strftime("%Y-%m-%d %H:%M ET"),
            "timeframe":      "5m",
        }

    # ─────────────────────────────────────────────────────────────── #
    # Preparation                                                      #
    # ─────────────────────────────────────────────────────────────── #

    def _prepare(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df = add_all(df)  # ema_8/21/55/200, atr, adx, vwap (session), etc.

        # ── EMA 9 (trigger line) ──────────────────────────────────── #
        df["ema_9"] = ta.trend.EMAIndicator(close=df["close"], window=9).ema_indicator()

        # ── 85-period Rolling VWAP (not session reset) ────────────── #
        typical  = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol   = (typical * df["volume"]).rolling(85).sum()
        vol_sum  = df["volume"].rolling(85).sum()
        df["vwap_85"] = tp_vol / vol_sum.replace(0, np.nan)

        # VWAP-85 standard deviation bands (1σ and 2σ)
        dev = typical - df["vwap_85"]
        vwap_std = dev.rolling(85).std()
        df["vwap_85_u1"] = df["vwap_85"] + 1.0 * vwap_std
        df["vwap_85_l1"] = df["vwap_85"] - 1.0 * vwap_std
        df["vwap_85_u2"] = df["vwap_85"] + 2.0 * vwap_std
        df["vwap_85_l2"] = df["vwap_85"] - 2.0 * vwap_std

        # ── Custom MACD (9, 21, 16) ───────────────────────────────── #
        _macd = ta.trend.MACD(close=df["close"], window_fast=9, window_slow=21, window_sign=16)
        df["macd_ap"]     = _macd.macd()
        df["macd_ap_sig"] = _macd.macd_signal()
        df["macd_ap_hist"]= _macd.macd_diff()

        # ── RSI (14) ──────────────────────────────────────────────── #
        df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

        # ── 20-bar average volume (for volume confirmation bonus) ──── #
        df["vol_avg20"] = df["volume"].rolling(20).mean()

        # ── 20-bar rolling average ATR (volatility baseline for spike filter) ── #
        df["atr_avg20"] = df["atr"].rolling(20).mean()

        # ── ET timestamps ─────────────────────────────────────────── #
        df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        df["date_et"] = df["ts_et"].dt.date
        df["hour_et"] = df["ts_et"].dt.hour
        df["min_et"]  = df["ts_et"].dt.minute

        # ── 15-min Initial Balance (IB) per day ───────────────────── #
        # First 3 bars: 9:30, 9:35, 9:40 = 15 minutes of initial balance
        ib_high  = np.full(len(df), np.nan)
        ib_low   = np.full(len(df), np.nan)

        # Macro range (9:50–10:10 AM) and 10:15 confirmation per day
        macro_high  = np.full(len(df), np.nan)
        macro_low   = np.full(len(df), np.nan)
        macro_conf_long  = np.zeros(len(df), dtype=bool)   # 10:15 bar closed above macro_high
        macro_conf_short = np.zeros(len(df), dtype=bool)   # 10:15 bar closed below macro_low

        # IB broken tracker (has ORB been breached today before bar i)
        ib_broken_up   = np.zeros(len(df), dtype=bool)
        ib_broken_down = np.zeros(len(df), dtype=bool)

        # Previous day high/low
        pdh = np.full(len(df), np.nan)
        pdl = np.full(len(df), np.nan)

        for date, group_idx in df.groupby("date_et").groups.items():
            group_idx = sorted(group_idx)
            day_df    = df.iloc[group_idx]

            # ── IB: first 3 bars (9:30, 9:35, 9:40) ─────────────── #
            ib_mask = (day_df["hour_et"] == 9) & (day_df["min_et"] >= 30) & (day_df["min_et"] <= 40)
            ib_bars = day_df[ib_mask]
            if len(ib_bars) >= 1:
                ihi = float(ib_bars["high"].max())
                ilo = float(ib_bars["low"].min())
                for idx in group_idx:
                    r = df.iloc[idx]
                    if not (r["hour_et"] == 9 and r["min_et"] >= 30 and r["min_et"] <= 40):
                        ib_high[idx] = ihi
                        ib_low[idx]  = ilo

            # ── Macro range: 9:50–10:10 AM ───────────────────────── #
            macro_mask = (
                ((day_df["hour_et"] == 9)  & (day_df["min_et"] >= 50)) |
                ((day_df["hour_et"] == 10) & (day_df["min_et"] <= 10))
            )
            macro_bars = day_df[macro_mask]
            mhi, mlo = np.nan, np.nan
            if len(macro_bars) >= 1:
                mhi = float(macro_bars["high"].max())
                mlo = float(macro_bars["low"].min())
                for idx in group_idx:
                    macro_high[idx] = mhi
                    macro_low[idx]  = mlo

            # ── 10:15 macro confirmation ──────────────────────────── #
            conf_bar_mask = (day_df["hour_et"] == 10) & (day_df["min_et"] == 15)
            conf_bars = day_df[conf_bar_mask]
            conf_long_day  = False
            conf_short_day = False
            if len(conf_bars) == 1 and not np.isnan(mhi) and not np.isnan(mlo):
                conf_close = float(conf_bars["close"].iloc[0])
                conf_long_day  = conf_close > mhi
                conf_short_day = conf_close < mlo
            # Apply from 10:15 onward
            for idx in group_idx:
                r = df.iloc[idx]
                if r["hour_et"] > 10 or (r["hour_et"] == 10 and r["min_et"] >= 15):
                    macro_conf_long[idx]  = conf_long_day
                    macro_conf_short[idx] = conf_short_day

            # ── IB breakout tracking ──────────────────────────────── #
            broke_up   = False
            broke_down = False
            for idx in group_idx:
                ib_broken_up[idx]   = broke_up
                ib_broken_down[idx] = broke_down
                cl = df["close"].iloc[idx]
                if not np.isnan(ib_high[idx]) and cl > ib_high[idx]:
                    broke_up = True
                if not np.isnan(ib_low[idx]) and cl < ib_low[idx]:
                    broke_down = True

        df["ib_high"]         = ib_high
        df["ib_low"]          = ib_low
        df["macro_high"]      = macro_high
        df["macro_low"]       = macro_low
        df["macro_conf_long"] = macro_conf_long
        df["macro_conf_short"]= macro_conf_short
        df["ib_broken_up"]    = ib_broken_up
        df["ib_broken_down"]  = ib_broken_down

        # Previous day high/low (simple approach: shift daily high/low by 1)
        df_daily = df.groupby("date_et").agg(
            day_high=("high", "max"),
            day_low =("low",  "min"),
        ).reset_index()
        df_daily["pdh"] = df_daily["day_high"].shift(1)
        df_daily["pdl"] = df_daily["day_low"].shift(1)
        pdhl_map = df_daily.set_index("date_et")[["pdh", "pdl"]].to_dict(orient="index")

        pdh_arr = np.full(len(df), np.nan)
        pdl_arr = np.full(len(df), np.nan)
        for i_idx, row in df.iterrows():
            d = row["date_et"]
            if d in pdhl_map:
                pdh_arr[df.index.get_loc(i_idx)] = pdhl_map[d]["pdh"]
                pdl_arr[df.index.get_loc(i_idx)] = pdhl_map[d]["pdl"]

        df["pdh"] = pdh_arr
        df["pdl"] = pdl_arr

        # ── Previous day close and direction ─────────────────────── #
        # pdc = last bar's close from the prior trading day
        # prev_day_bullish = prev day closed above its open (up day)
        pdc_arr = np.full(len(df), np.nan)
        prev_bull_arr = np.zeros(len(df), dtype=bool)

        sorted_dates = sorted(df["date_et"].unique())
        day_open_close = {}
        for date in sorted_dates:
            day_rows = df[df["date_et"] == date]
            # First RTH bar (9:30 open) for opening price
            open_bar = day_rows[(day_rows["hour_et"] == 9) & (day_rows["min_et"] == 30)]
            # Last bar of the day for closing price
            close_bar = day_rows.sort_values("ts_et").iloc[-1]
            day_open_close[date] = {
                "open": float(open_bar["open"].iloc[0]) if len(open_bar) > 0 else float(day_rows["open"].iloc[0]),
                "close": float(close_bar["close"]),
            }

        for i_idx, row in df.iterrows():
            d = row["date_et"]
            date_pos = sorted_dates.index(d) if d in sorted_dates else -1
            if date_pos > 0:
                prev_date = sorted_dates[date_pos - 1]
                if prev_date in day_open_close:
                    pdc_arr[df.index.get_loc(i_idx)]      = day_open_close[prev_date]["close"]
                    prev_bull_arr[df.index.get_loc(i_idx)] = (
                        day_open_close[prev_date]["close"] > day_open_close[prev_date]["open"]
                    )

        df["pdc"]           = pdc_arr
        df["prev_day_bull"] = prev_bull_arr

        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────── #
    # Timing                                                           #
    # ─────────────────────────────────────────────────────────────── #

    def _is_entry_window(self, h: int, m: int) -> bool:
        """True if hour/minute is in a valid entry window."""
        # Primary: 10:15 AM – 11:30 AM
        in_primary = (
            (h == 10 and m >= 15) or
            (h == 11 and m < 30)
        )
        # Lunch continuation: 11:30 AM – 2:00 PM
        in_lunch = (h == 11 and m >= 30) or (h == 12) or (h == 13)
        # Power Hour: 2:00 PM – 3:55 PM (disabled by skip_power_hour)
        if self.skip_power_hour:
            return in_primary or in_lunch
        in_power = (
            (h == 14) or
            (h == 15 and m < 55)
        )
        return in_primary or in_lunch or in_power

    # ─────────────────────────────────────────────────────────────── #
    # Scoring                                                          #
    # ─────────────────────────────────────────────────────────────── #

    def _score(self, df: pd.DataFrame, i: int) -> tuple | None:
        """
        Returns (direction, score, regime, stop_price, target1, target2) or None.
        """
        long_sig  = self._aplus_long(df, i)
        short_sig = self._aplus_short(df, i)

        if long_sig and short_sig:
            return long_sig if long_sig[1] >= short_sig[1] else short_sig
        return long_sig or short_sig

    # ─────────────────────────────────────────────────────────────── #
    # A+ Long Setup                                                    #
    # ─────────────────────────────────────────────────────────────── #

    def _aplus_long(self, df: pd.DataFrame, i: int) -> tuple | None:
        """
        Long entry rules:
        1. close > EMA200 (macro uptrend) + EMA8 > EMA21 (short-term momentum up)
        2. IB already broken to the upside (ib_broken_up = True)
        3. close > IB High AND close > VWAP-85
        4. Retest: within last 1-5 bars, price pulled back into the IB High / VWAP-85 zone
           (within self.retest_tolerance × ATR — default 0.75)
        5. Reversal candle: engulfing, hammer, OR strong close (closes top 30% of bar)
        6. MACD(9,21,16) histogram rising or signal above 0
        7. RSI > 45 (not in oversold reversal — confirms momentum)
        8. Macro confirmation (10:15 bar) if required
        """
        if i < 3:
            return None

        cl    = float(df["close"].iloc[i])
        e200  = float(df["ema_200"].iloc[i])
        e8    = float(df["ema_8"].iloc[i])
        e21   = float(df["ema_21"].iloc[i])
        vwap85= float(df["vwap_85"].iloc[i])
        ib_hi = float(df["ib_high"].iloc[i])
        atr   = float(df["atr"].iloc[i])
        adx   = float(df["adx"].iloc[i])
        dmp   = float(df["dmp"].iloc[i])   # +DI
        dmn   = float(df["dmn"].iloc[i])   # -DI
        rsi   = float(df["rsi_14"].iloc[i])

        # Guard NaN
        if any(np.isnan(v) for v in [cl, e200, vwap85, ib_hi, atr]):
            return None

        # ADX trend filter — only trade in trending markets
        if np.isnan(adx) or adx < self.min_adx:
            return None

        # Volatility spike filter: skip if ATR is unusually high vs its own 20-bar baseline.
        # On extreme news/event days (tariff shocks, FOMC surprises), ATR can be 2-3× normal,
        # which inflates stops to $1,800+ and makes the setup structurally unreliable.
        atr_avg20 = float(df["atr_avg20"].iloc[i])
        if not np.isnan(atr_avg20) and atr_avg20 > 0:
            if atr > self.max_atr_multiple * atr_avg20:
                return None   # volatility spike — structural stops not trustworthy

        # IB range quality filter:
        # Too narrow → opening had no conviction (choppy, no directional bias formed)
        # Too wide   → extreme open (partially caught by ATR spike filter, double-check here)
        ib_lo = float(df["ib_low"].iloc[i])
        if not np.isnan(ib_lo) and not np.isnan(ib_hi):
            ib_range = ib_hi - ib_lo
            if ib_range < 0.25 * atr:    # doji-like open — no clear bias formed
                return None
            if ib_range > 3.0 * atr:     # extreme open — too much noise
                return None

        # 1. Macro bias: price above EMA200 + EMA8 > EMA21 (short-term uptrend)
        #    + EMA55 rising (medium-term trend up: ~4.5 hrs of context)
        #    + DI alignment (+DI dominates, confirming directional momentum)
        if cl < e200:
            return None
        if not np.isnan(e8) and not np.isnan(e21) and e8 <= e21:
            return None   # short-term momentum must be up
        # prev_day_bull: used as score penalty below (not a hard block)

        # EMA55: price must be above the 4.5-hour average AND trend rising intraday
        e55 = float(df["ema_55"].iloc[i])
        if not np.isnan(e55) and cl <= e55:
            return None   # price below EMA55 = no long
        e55_prev = float(df["ema_55"].iloc[i - 15]) if i >= 15 else np.nan
        if not np.isnan(e55) and not np.isnan(e55_prev) and e55 <= e55_prev:
            return None   # EMA55 must be rising (75 min momentum)
        # EMA55 daily slope: must be higher than 1 full trading day ago (78 bars)
        e55_1day = float(df["ema_55"].iloc[i - 78]) if i >= 78 else np.nan
        if not np.isnan(e55) and not np.isnan(e55_1day) and e55 <= e55_1day:
            return None   # filters multi-day downtrend environments
        if np.isnan(dmp) or np.isnan(dmn) or dmp <= dmn:
            return None

        # 2. IB already broken upward
        if not bool(df["ib_broken_up"].iloc[i]):
            return None

        # 3. Close confirms: above IB High AND above VWAP-85
        support_level = min(ib_hi, vwap85)
        if cl <= support_level:
            return None

        # 4. Retest: within the previous 1-5 bars, at least one bar's low
        #    touched the IB High or VWAP-85 zone (within retest_tolerance × ATR).
        tol = self.retest_tolerance * atr
        retest_bar = None
        for k in range(max(3, i - 5), i):
            bar_lo = float(df["low"].iloc[k])
            if bar_lo <= ib_hi + tol:
                retest_bar = k
                break
            if bar_lo <= vwap85 + tol:
                retest_bar = k
                break
        if retest_bar is None:
            return None

        # 5. Reversal candle: engulfing, hammer, OR strong close
        #    Strong close only qualifies when price retested very close to the IB
        #    (within 0.5 ATR) — prevents random green bars far above IB counting
        bullish_eng  = self._is_bullish_engulfing(df, i)
        hammer       = self._is_hammer(df, i)
        _deep_retest = float(df["low"].iloc[i]) <= ib_hi + 0.5 * atr
        strong_close = self._is_strong_close_bull(df, i) and _deep_retest
        if not (bullish_eng or hammer or strong_close):
            return None

        # 6. MACD (9,21,16) momentum — signal above zero or histogram rising
        hist      = df["macd_ap_hist"].iloc[i]
        hist_prev = df["macd_ap_hist"].iloc[i - 1]
        sig_line  = df["macd_ap_sig"].iloc[i]
        macd_ok = (
            (not np.isnan(hist) and not np.isnan(hist_prev) and hist > hist_prev) or
            (not np.isnan(sig_line) and sig_line > 0)
        )
        if not macd_ok:
            return None

        # 7. RSI gate: require momentum (RSI > 45) and avoid overbought (RSI < 75)
        #    RSI 45–75 is the sweet spot: momentum intact but not exhausted
        if not np.isnan(rsi) and (rsi < 45 or rsi > 75):
            return None

        # 8. Macro confirmation
        if self.require_macro_confirm and not bool(df["macro_conf_long"].iloc[i]):
            return None

        # ── Build trade levels ──────────────────────────────────── #
        # ATR-anchored stop — capped at atr_stop_cap × ATR to limit dollar risk on
        # moderately elevated volatility days that pass the spike filter.
        stop_atr = min(atr, self.atr_stop_cap * (atr_avg20 if not np.isnan(atr_avg20) else atr))
        sl = cl - 1.0 * stop_atr

        u1 = float(df["vwap_85_u1"].iloc[i])
        u2 = float(df["vwap_85_u2"].iloc[i])
        risk = cl - sl
        if np.isnan(u1) or u1 <= cl or (u1 - cl) < 1.5 * risk:
            u1 = cl + 2.0 * risk   # T1 = 2R fallback (improved from 1.5R)
        if np.isnan(u2) or u2 <= u1:
            u2 = cl + 3.0 * risk   # T2 = 3R fallback

        # ── Score ──────────────────────────────────────────────── #
        score = 0.60
        if bullish_eng:    score += 0.08
        elif hammer:       score += 0.06
        elif strong_close: score += 0.04
        near_ib = float(df["low"].iloc[i]) <= ib_hi + 1.0 * atr
        if near_ib:   score += 0.05
        if not np.isnan(hist) and not np.isnan(hist_prev) and hist > 0 and hist > hist_prev:
            score += 0.05
        # Volume conviction bonus
        vol     = float(df["volume"].iloc[i])
        vol_avg = float(df["vol_avg20"].iloc[i])
        if not np.isnan(vol_avg) and vol_avg > 0 and vol >= 1.2 * vol_avg:
            score += 0.03
        # EMA alignment bonus: EMA8 > EMA21 = short-term momentum recovering
        if not np.isnan(e8) and not np.isnan(e21) and e8 > e21:
            score += 0.02
        # RSI recovery bonus: RSI in 45-65 zone = pulled back cleanly, not extreme
        if not np.isnan(rsi) and 45 <= rsi <= 65:
            score += 0.02
        # Previous day direction: bonus if prior day was bullish, small penalty if bearish
        # (not a hard block — today can still rally after a red day)
        if bool(df["prev_day_bull"].iloc[i]):
            score += 0.03
        else:
            score -= 0.03

        return ("BUY", round(score, 3), "ib_retest_long", round(sl, 4), round(u1, 4), round(u2, 4))

    # ─────────────────────────────────────────────────────────────── #
    # A+ Short Setup                                                   #
    # ─────────────────────────────────────────────────────────────── #

    def _aplus_short(self, df: pd.DataFrame, i: int) -> tuple | None:
        """
        Short entry rules (mirror of long):
        1. close < EMA200 + EMA8 < EMA21 (short-term momentum down)
        2. IB already broken to the downside
        3. close < IB Low AND close < VWAP-85
        4. Retest: within last 1-5 bars, price rallied back into IB Low / VWAP-85 zone
        5. Reversal candle: bearish engulfing, shooting star, OR strong close bear
        6. MACD(9,21,16) histogram falling or signal below 0
        7. RSI < 55 (not overbought — confirms downward momentum)
        8. Macro confirmation if required
        """
        if i < 3:
            return None

        cl    = float(df["close"].iloc[i])
        e200  = float(df["ema_200"].iloc[i])
        e8    = float(df["ema_8"].iloc[i])
        e21   = float(df["ema_21"].iloc[i])
        e55   = float(df["ema_55"].iloc[i])
        vwap85= float(df["vwap_85"].iloc[i])
        ib_lo = float(df["ib_low"].iloc[i])
        atr   = float(df["atr"].iloc[i])
        adx   = float(df["adx"].iloc[i])
        dmp   = float(df["dmp"].iloc[i])
        dmn   = float(df["dmn"].iloc[i])
        rsi   = float(df["rsi_14"].iloc[i])

        if any(np.isnan(v) for v in [cl, e200, vwap85, ib_lo, atr]):
            return None

        # ADX trend filter
        if np.isnan(adx) or adx < self.min_adx:
            return None

        # Volatility spike filter (same as long side)
        atr_avg20 = float(df["atr_avg20"].iloc[i])
        if not np.isnan(atr_avg20) and atr_avg20 > 0:
            if atr > self.max_atr_multiple * atr_avg20:
                return None

        # IB range quality filter (same as long side)
        ib_hi_s = float(df["ib_high"].iloc[i])
        ib_lo   = float(df["ib_low"].iloc[i])
        if not np.isnan(ib_lo) and not np.isnan(ib_hi_s):
            ib_range = ib_hi_s - ib_lo
            if ib_range < 0.25 * atr:
                return None
            if ib_range > 3.0 * atr:
                return None

        # 1. Macro bias: price below EMA200 + DI alignment (-DI dominates)
        #    + short-term EMA8 < EMA21 (momentum is down)
        if cl > e200:
            return None
        if not np.isnan(e55) and cl > e55:
            return None
        if not np.isnan(e8) and not np.isnan(e21) and e8 > e21:
            return None
        if np.isnan(dmp) or np.isnan(dmn) or dmn <= dmp:
            return None

        # 2. IB already broken downward
        if not bool(df["ib_broken_down"].iloc[i]):
            return None

        # 3. Close confirms: below IB Low AND below VWAP-85
        resistance_level = max(ib_lo, vwap85)
        if cl >= resistance_level:
            return None

        # 4. Retest: within the previous 1-5 bars, at least one bar's high
        #    touched back up into the IB Low or VWAP-85 zone.
        tol = self.retest_tolerance * atr
        retest_bar = None
        for k in range(max(3, i - 5), i):
            bar_hi = float(df["high"].iloc[k])
            if bar_hi >= ib_lo - tol:
                retest_bar = k
                break
            if bar_hi >= vwap85 - tol:
                retest_bar = k
                break
        if retest_bar is None:
            return None

        # 5. Reversal candle: bearish engulfing, shooting star, OR strong close bear
        bearish_eng    = self._is_bearish_engulfing(df, i)
        shooting_star  = self._is_shooting_star(df, i)
        strong_close_b = self._is_strong_close_bear(df, i)
        if not (bearish_eng or shooting_star or strong_close_b):
            return None

        # 6. MACD momentum
        hist      = df["macd_ap_hist"].iloc[i]
        hist_prev = df["macd_ap_hist"].iloc[i - 1]
        sig_line  = df["macd_ap_sig"].iloc[i]
        macd_ok = (
            (not np.isnan(hist) and not np.isnan(hist_prev) and hist < hist_prev) or
            (not np.isnan(sig_line) and sig_line < 0)
        )
        if not macd_ok:
            return None

        # 7. RSI confirmation: must be below 55
        if not np.isnan(rsi) and rsi > 55:
            return None

        # 8. Macro confirmation
        if self.require_macro_confirm and not bool(df["macro_conf_short"].iloc[i]):
            return None

        # ── Build trade levels ──────────────────────────────────── #
        # ATR-anchored stop — capped at atr_stop_cap × avg ATR (same logic as long)
        stop_atr = min(atr, self.atr_stop_cap * (atr_avg20 if not np.isnan(atr_avg20) else atr))
        sl = cl + 1.0 * stop_atr

        l1 = float(df["vwap_85_l1"].iloc[i])
        l2 = float(df["vwap_85_l2"].iloc[i])
        risk = sl - cl
        if np.isnan(l1) or l1 >= cl or (cl - l1) < 1.5 * risk:
            l1 = cl - 2.0 * risk   # T1 = 2R fallback
        if np.isnan(l2) or l2 >= l1:
            l2 = cl - 3.0 * risk   # T2 = 3R fallback

        # ── Score ──────────────────────────────────────────────── #
        score = 0.60
        if bearish_eng:      score += 0.08
        elif shooting_star:  score += 0.06
        elif strong_close_b: score += 0.04
        near_ib = float(df["high"].iloc[i]) >= ib_lo - 1.0 * atr
        if near_ib:  score += 0.05
        if not np.isnan(hist) and not np.isnan(hist_prev) and hist < 0 and hist < hist_prev:
            score += 0.05
        vol     = float(df["volume"].iloc[i])
        vol_avg = float(df["vol_avg20"].iloc[i])
        if not np.isnan(vol_avg) and vol_avg > 0 and vol >= 1.2 * vol_avg:
            score += 0.03
        if not np.isnan(rsi) and rsi < 45:
            score += 0.02

        return ("SELL", round(score, 3), "ib_retest_short", round(sl, 4), round(l1, 4), round(l2, 4))

    # ─────────────────────────────────────────────────────────────── #
    # Candle Pattern Detection                                         #
    # ─────────────────────────────────────────────────────────────── #

    def _is_strong_close_bull(self, df: pd.DataFrame, i: int) -> bool:
        """
        Bar closes in the top 30% of its high-low range AND is bullish (close > open).
        Catches momentum bars that aren't perfect engulfings but show clear conviction.
        Minimum range: 0.3 × ATR (filters out tiny doji-like bars).
        """
        op = float(df["open"].iloc[i]);  cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]);  lo = float(df["low"].iloc[i])
        atr = float(df["atr"].iloc[i])
        full_range = hi - lo
        if full_range < 0.3 * atr:
            return False
        if cl <= op:            # must be bullish
            return False
        close_position = (cl - lo) / full_range   # 0 = closed at low, 1 = at high
        return close_position >= 0.70

    def _is_strong_close_bear(self, df: pd.DataFrame, i: int) -> bool:
        """
        Bar closes in the bottom 30% of its high-low range AND is bearish (close < open).
        """
        op = float(df["open"].iloc[i]);  cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]);  lo = float(df["low"].iloc[i])
        atr = float(df["atr"].iloc[i])
        full_range = hi - lo
        if full_range < 0.3 * atr:
            return False
        if cl >= op:            # must be bearish
            return False
        close_position = (cl - lo) / full_range
        return close_position <= 0.30

    def _is_bullish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        """
        Current candle is bullish and fully engulfs the previous bearish candle.
        Previous candle must be bearish (close < open).
        Current candle body engulfs: current open <= prev close AND current close >= prev open.
        """
        if i < 1:
            return False
        c_op = float(df["open"].iloc[i]);  c_cl = float(df["close"].iloc[i])
        p_op = float(df["open"].iloc[i-1]); p_cl = float(df["close"].iloc[i-1])
        prev_bearish = p_cl < p_op
        curr_bullish = c_cl > c_op
        engulfs      = c_cl >= p_op and c_op <= p_cl
        return prev_bearish and curr_bullish and engulfs

    def _is_hammer(self, df: pd.DataFrame, i: int) -> bool:
        """
        Small body at top of range, long lower shadow.
        Lower shadow >= 2× body length. Upper shadow <= 0.5× body.
        """
        op = float(df["open"].iloc[i]);  cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]);  lo = float(df["low"].iloc[i])
        body         = abs(cl - op)
        full_range   = hi - lo
        if full_range < 1e-8:
            return False
        lower_shadow = min(op, cl) - lo
        upper_shadow = hi - max(op, cl)
        return (
            lower_shadow >= 2.0 * body and
            upper_shadow <= 0.5 * max(body, 1e-8) and
            body >= full_range * 0.05   # not a doji
        )

    def _is_bearish_engulfing(self, df: pd.DataFrame, i: int) -> bool:
        """
        Current candle is bearish and fully engulfs the previous bullish candle.
        """
        if i < 1:
            return False
        c_op = float(df["open"].iloc[i]);  c_cl = float(df["close"].iloc[i])
        p_op = float(df["open"].iloc[i-1]); p_cl = float(df["close"].iloc[i-1])
        prev_bullish = p_cl > p_op
        curr_bearish = c_cl < c_op
        engulfs      = c_op >= p_cl and c_cl <= p_op
        return prev_bullish and curr_bearish and engulfs

    def _is_shooting_star(self, df: pd.DataFrame, i: int) -> bool:
        """
        Small body at bottom of range, long upper shadow.
        Upper shadow >= 2× body. Lower shadow <= 0.5× body.
        """
        op = float(df["open"].iloc[i]);  cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]);  lo = float(df["low"].iloc[i])
        body         = abs(cl - op)
        full_range   = hi - lo
        if full_range < 1e-8:
            return False
        upper_shadow = hi - max(op, cl)
        lower_shadow = min(op, cl) - lo
        return (
            upper_shadow >= 2.0 * body and
            lower_shadow <= 0.5 * max(body, 1e-8) and
            body >= full_range * 0.05
        )
