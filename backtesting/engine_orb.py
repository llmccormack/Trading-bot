"""
ORBEngine — Opening Range Breakout (5-minute bars)
───────────────────────────────────────────────────
Trades the break and retest of the first 5-minute candle of the RTH session.

The 9:30 candle represents the initial battle between buyers and sellers.
When price breaks out and then pulls back to test the broken level, it signals
institutional conviction behind the move.

STRATEGY LOGIC:
  1. Opening Range = high and low of the 9:30–9:35 AM ET candle (ORB range ≥ 0.4 ATR).
  2. Confirmed breakout: close above ORB high (long only — shorts have no edge in testing).
  3. Retest: within last 1–6 bars, price pulled back within 0.40 ATR of ORB high.
  4. Reversal candle: bullish engulfing, hammer, or strong close on the entry bar.
  5. Momentum: MACD(9,21,16) histogram rising AND signal line > 0.
  6. Volume: entry bar ≥ 1.15x 20-bar average.
  7. Trend stack: EMA8 > EMA21, DI+ > DI-, price > EMA200.
  8. RSI: 42–76 (momentum zone, not overbought).
  9. Skip 9:35–9:45 AM — first two bars post-ORB are too volatile (0% WR in backtest).
  10. Gap filter: skip if overnight gap > 1.2 ATR (large gaps often fill) or gap opposes direction.

TIMING:
  - ORB defined: 9:30–9:35 AM ET (first bar closes at 9:35)
  - Entry window: 9:45 AM – 10:15 AM ET (hands off to A+ strategy after)
  - Force-close: 3:55 PM ET
  - Max hold: 36 bars (3 hours on 5m)

STOPS & TARGETS:
  - Stop:     1.0 × ATR below entry
  - Target 1: 2.0R  (stop moves to breakeven)
  - Target 2: 3.0R  (switches to 0.8 ATR trailing stop to let big days run)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytz
import ta

from utils.indicators import add_all
from backtesting.models import BacktestTrade, BacktestResult

ET = pytz.timezone("America/New_York")

STRATEGY_LABEL = "Opening Range Breakout"

# Timing
ENTRY_START_H, ENTRY_START_M = 9, 45   # skip 9:35-9:45 — first 2 bars after ORB are too noisy
ENTRY_END_H,   ENTRY_END_M   = 10, 15  # hand off to A+ strategy
EOD_H,         EOD_M         = 15, 55
MAX_BARS = 36


class ORBEngine:
    """
    Opening Range Breakout — 5-minute bars.
    Designed for NQ=F and ES=F.
    """

    def __init__(
        self,
        min_adx:          float = 16.0,
        min_score:        float = 0.70,   # raised from 0.65 — filters marginal setups (hammer-only)
        min_rr:           float = 1.8,
        allow_long:       bool  = True,
        allow_short:      bool  = False,  # short ORB shows no edge — disabled by default
        retest_tolerance: float = 0.40,   # tight retest — must be a genuine test of the level
        skip_monday:      bool  = False,
        max_gap_atr:      float = 1.2,    # skip if overnight gap > 1.2 ATR — large gaps often fill
    ):
        self.min_adx          = min_adx
        self.min_score        = min_score
        self.min_rr           = min_rr
        self.allow_long       = allow_long
        self.allow_short      = allow_short
        self.retest_tolerance = retest_tolerance
        self.skip_monday      = skip_monday
        self.max_gap_atr      = max_gap_atr

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
        _day_trade_count: dict[str, int] = {}

        for i in range(50, n):
            bar_time_et = df["ts_et"].iloc[i]
            h, m = bar_time_et.hour, bar_time_et.minute
            _day_key = bar_time_et.strftime("%Y-%m-%d")

            # ── Manage open trade ─────────────────────────────────── #
            if open_trade is not None:
                lo  = float(df["low"].iloc[i])
                hi  = float(df["high"].iloc[i])
                cl  = float(df["close"].iloc[i])
                atr_now = float(df["atr"].iloc[i])

                eod     = (h == EOD_H and m >= EOD_M) or (h > EOD_H)
                timeout = (i - open_trade["entry_bar"]) >= MAX_BARS

                # ── Trailing stop mode (after T2 broken through) ──── #
                if open_trade.get("t2_hit", False):
                    trail_sl = open_trade["trail_sl"]
                    if open_trade["dir"] == 1:
                        new_trail = hi - 0.8 * atr_now
                        open_trade["trail_sl"] = max(trail_sl, new_trail)
                    else:
                        new_trail = lo + 0.8 * atr_now
                        open_trade["trail_sl"] = min(trail_sl, new_trail)
                    trail_sl = open_trade["trail_sl"]

                    hit_trail = (open_trade["dir"] ==  1 and lo <= trail_sl) or \
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
                        continue

                pnl    = (exit_px - open_trade["entry"]) * open_trade["dir"]
                risk   = abs(open_trade["entry"] - open_trade["sl_orig"])
                r_mult = pnl / risk if risk > 0 else 0.0
                equity += pnl
                be_moved = False

                result.trades.append(BacktestTrade(
                    strategy    = "orb",
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
                ))
                result.equity_curve.append(round(equity, 4))
                open_trade = None
                continue

            # ── Filters ───────────────────────────────────────────── #
            if self.skip_monday and bar_time_et.weekday() == 0:
                continue

            if not self._is_entry_window(h, m):
                continue

            # One trade per day for ORB
            if _day_trade_count.get(_day_key, 0) >= 1:
                continue

            # ── Check for setup ────────────────────────────────────── #
            sig = self._score(df, i)
            if sig is None:
                continue

            direction, score, regime, sl, t1, t2 = sig

            entry     = float(df["close"].iloc[i])
            risk_pts  = abs(entry - sl)
            reward_t1 = abs(entry - t1)

            if risk_pts <= 0 or (reward_t1 / risk_pts) < self.min_rr:
                continue
            if score < self.min_score:
                continue
            if direction == "BUY" and not self.allow_long:
                continue
            if direction == "SELL" and not self.allow_short:
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
        df = self._prepare(df_raw)
        n  = len(df)
        if n < 55:
            return None

        i = n - 1
        bar_time_et = df["ts_et"].iloc[i]
        h, m = bar_time_et.hour, bar_time_et.minute

        if self.skip_monday and bar_time_et.weekday() == 0:
            return None
        if not self._is_entry_window(h, m):
            return None

        sig = self._score(df, i)
        if sig is None:
            return None

        direction, score, regime, sl, t1, t2 = sig
        entry    = current_price if current_price else float(df["close"].iloc[i])
        risk_pts = abs(entry - sl)
        reward_t1 = abs(entry - t1)

        if risk_pts <= 0 or (reward_t1 / risk_pts) < self.min_rr:
            return None
        if score < self.min_score:
            return None
        if direction == "BUY" and not self.allow_long:
            return None
        if direction == "SELL" and not self.allow_short:
            return None

        rr = abs(entry - t2) / risk_pts if risk_pts > 0 else 0.0
        return {
            "direction":      direction,
            "strategy":       "orb",
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
    # Preparation                                                      #
    # ─────────────────────────────────────────────────────────────── #

    def _prepare(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df = add_all(df)

        # Custom MACD (9, 21, 16)
        _macd = ta.trend.MACD(close=df["close"], window_fast=9, window_slow=21, window_sign=16)
        df["macd_orb"]      = _macd.macd()
        df["macd_orb_sig"]  = _macd.macd_signal()
        df["macd_orb_hist"] = _macd.macd_diff()

        # RSI (14)
        df["rsi_14"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

        # 20-bar average volume
        df["vol_avg20"] = df["volume"].rolling(20).mean()

        # ET timestamps
        df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        df["date_et"] = df["ts_et"].dt.date
        df["hour_et"] = df["ts_et"].dt.hour
        df["min_et"]  = df["ts_et"].dt.minute

        # ── Opening Range per day ─────────────────────────────────── #
        # ORB = the single 9:30 AM candle (first RTH bar)
        orb_high_arr    = np.full(len(df), np.nan)
        orb_low_arr     = np.full(len(df), np.nan)
        orb_broken_up   = np.zeros(len(df), dtype=bool)
        orb_broken_down = np.zeros(len(df), dtype=bool)
        orb_bullish_arr = np.zeros(len(df), dtype=bool)  # ORB candle closed green

        for date, group_idx in df.groupby("date_et").groups.items():
            group_idx = sorted(group_idx)
            day_df    = df.iloc[group_idx]

            # First RTH bar: 9:30 AM
            orb_mask = (day_df["hour_et"] == 9) & (day_df["min_et"] == 30)
            orb_bars = day_df[orb_mask]

            if len(orb_bars) == 0:
                continue

            orb_h        = float(orb_bars["high"].max())
            orb_l        = float(orb_bars["low"].min())
            orb_is_bull  = float(orb_bars["close"].iloc[-1]) >= float(orb_bars["open"].iloc[0])

            # Apply to all bars after the opening candle
            broke_up   = False
            broke_down = False
            for idx in group_idx:
                iloc_pos = df.index.get_loc(idx)
                r = df.iloc[iloc_pos]
                # Only set ORB levels for bars after the 9:30 candle
                if not (r["hour_et"] == 9 and r["min_et"] == 30):
                    orb_high_arr[iloc_pos]    = orb_h
                    orb_low_arr[iloc_pos]     = orb_l
                    orb_broken_up[iloc_pos]   = broke_up
                    orb_broken_down[iloc_pos] = broke_down
                    orb_bullish_arr[iloc_pos] = orb_is_bull

                cl = float(r["close"])
                if cl > orb_h:
                    broke_up = True
                if cl < orb_l:
                    broke_down = True

        df["orb_high"]        = orb_high_arr
        df["orb_low"]         = orb_low_arr
        df["orb_broken_up"]   = orb_broken_up
        df["orb_broken_down"] = orb_broken_down
        df["orb_bullish"]     = orb_bullish_arr

        # ── Previous day's close (gap direction filter) ────────────── #
        prev_close_arr = np.full(len(df), np.nan)
        dates = sorted(df["date_et"].unique())
        for i_d, date in enumerate(dates):
            if i_d == 0:
                continue
            prev_date = dates[i_d - 1]
            prev_mask = df["date_et"] == prev_date
            if prev_mask.any():
                prev_close_val = float(df.loc[prev_mask, "close"].iloc[-1])
                day_mask = (df["date_et"] == date).values
                prev_close_arr[day_mask] = prev_close_val
        df["prev_close"] = prev_close_arr

        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────── #
    # Timing                                                           #
    # ─────────────────────────────────────────────────────────────── #

    def _is_entry_window(self, h: int, m: int) -> bool:
        # 9:35 AM – 10:15 AM ET
        if h == 9 and m >= ENTRY_START_M:
            return True
        if h == 10 and m < ENTRY_END_M:
            return True
        return False

    # ─────────────────────────────────────────────────────────────── #
    # Scoring                                                          #
    # ─────────────────────────────────────────────────────────────── #

    def _score(self, df: pd.DataFrame, i: int) -> tuple | None:
        long_sig  = self._orb_long(df, i)
        short_sig = self._orb_short(df, i)
        if long_sig and short_sig:
            return long_sig if long_sig[1] >= short_sig[1] else short_sig
        return long_sig or short_sig

    # ─────────────────────────────────────────────────────────────── #
    # Long Setup                                                       #
    # ─────────────────────────────────────────────────────────────── #

    def _orb_long(self, df: pd.DataFrame, i: int) -> tuple | None:
        """
        Long: ORB high broken upward → retest → reversal candle + momentum.
        """
        if i < 3:
            return None

        cl    = float(df["close"].iloc[i])
        e8    = float(df["ema_8"].iloc[i])
        e21   = float(df["ema_21"].iloc[i])
        e200  = float(df["ema_200"].iloc[i])
        atr   = float(df["atr"].iloc[i])
        adx   = float(df["adx"].iloc[i])
        dmp   = float(df["dmp"].iloc[i])
        dmn   = float(df["dmn"].iloc[i])
        rsi   = float(df["rsi_14"].iloc[i])
        orb_h = float(df["orb_high"].iloc[i])

        if any(np.isnan(v) for v in [cl, atr, orb_h]):
            return None
        if np.isnan(adx) or adx < self.min_adx:
            return None

        # 0. ORB range must be meaningful (not a doji / tiny range)
        orb_l_today = float(df["orb_low"].iloc[i])
        if not np.isnan(orb_l_today) and (orb_h - orb_l_today) < 0.4 * atr:
            return None

        # 0b. Gap direction filter for longs:
        #   - Don't long if overnight gap down > 0.25 ATR (overhead resistance from gap fill)
        #   - Don't long if overnight gap up > max_gap_atr (large gap-up → often fades back)
        prev_cl = float(df["prev_close"].iloc[i])
        if not np.isnan(prev_cl) and prev_cl > 0:
            day_open = float(df["orb_high"].iloc[i])  # ORB open ≈ first bar high/low midpoint
            # Use the ORB candle open price for gap calculation
            orb_bar_mask = (df["date_et"] == df["date_et"].iloc[i]) & \
                           (df["hour_et"] == 9) & (df["min_et"] == 30)
            if orb_bar_mask.any():
                day_open = float(df.loc[orb_bar_mask, "open"].iloc[0])
            gap = day_open - prev_cl
            if gap < -0.25 * atr:          # gapped down — overhead resistance
                return None
            if gap > self.max_gap_atr * atr:  # gapped up too much — likely to fade
                return None

        # 1. ORB high already broken upward before this bar
        if not bool(df["orb_broken_up"].iloc[i]):
            return None

        # 2. Price is still above ORB high (breakout held)
        if cl <= orb_h:
            return None

        # 3. Macro alignment: price must be above EMA200 (long on uptrend days only)
        if not np.isnan(e200) and cl < e200:
            return None

        # 3b. EMA momentum alignment
        if not np.isnan(e8) and not np.isnan(e21) and e8 <= e21:
            return None

        # 4. DI alignment (+DI > -DI = directional upward pressure)
        if np.isnan(dmp) or np.isnan(dmn) or dmp <= dmn:
            return None

        # 5. Retest: within last 1-6 bars, price pulled back to ORB high zone
        tol = self.retest_tolerance * atr
        retest_bar = None
        for k in range(max(1, i - 6), i):
            if float(df["low"].iloc[k]) <= orb_h + tol:
                retest_bar = k
                break
        if retest_bar is None:
            return None

        # 6. Reversal candle on the current bar
        bullish_eng  = self._is_bullish_engulfing(df, i)
        hammer       = self._is_hammer(df, i)
        deep_retest  = float(df["low"].iloc[i]) <= orb_h + 0.5 * atr
        strong_close = self._is_strong_close_bull(df, i) and deep_retest
        if not (bullish_eng or hammer or strong_close):
            return None

        # 7. MACD momentum — must be turning up AND sig_line confirms upward pressure
        hist      = df["macd_orb_hist"].iloc[i]
        hist_prev = df["macd_orb_hist"].iloc[i - 1]
        sig_line  = df["macd_orb_sig"].iloc[i]
        if np.isnan(hist) or np.isnan(hist_prev) or np.isnan(sig_line):
            return None
        if not (hist > hist_prev and sig_line > 0):
            return None

        # 7b. Volume must be above average on entry bar
        vol     = float(df["volume"].iloc[i])
        vol_avg = float(df["vol_avg20"].iloc[i])
        if np.isnan(vol_avg) or vol_avg <= 0 or vol < 1.15 * vol_avg:
            return None

        # 8. RSI: momentum zone — not oversold, not overbought
        if not np.isnan(rsi) and (rsi < 42 or rsi > 76):
            return None

        # ── Build levels ──────────────────────────────────────────── #
        sl   = cl - 1.0 * atr
        risk = cl - sl
        t1   = cl + 2.0 * risk
        t2   = cl + 3.0 * risk

        # ── Score ──────────────────────────────────────────────────── #
        score = 0.58
        if bullish_eng:    score += 0.10
        elif hammer:       score += 0.07
        elif strong_close: score += 0.05
        if deep_retest:    score += 0.06   # tight retest of ORB = high quality
        if hist > 0 and hist > hist_prev:
            score += 0.05
        if not np.isnan(vol_avg) and vol_avg > 0 and vol >= 1.15 * vol_avg:
            score += 0.04   # volume confirms breakout
        if not np.isnan(e200) and cl > e200:
            score += 0.03   # macro uptrend
        if not np.isnan(rsi) and 45 <= rsi <= 65:
            score += 0.02

        return ("BUY", round(score, 3), "orb_long", round(sl, 4), round(t1, 4), round(t2, 4))

    # ─────────────────────────────────────────────────────────────── #
    # Short Setup                                                      #
    # ─────────────────────────────────────────────────────────────── #

    def _orb_short(self, df: pd.DataFrame, i: int) -> tuple | None:
        """
        Short: ORB low broken downward → retest → reversal candle + momentum.
        """
        if i < 3:
            return None

        cl    = float(df["close"].iloc[i])
        e8    = float(df["ema_8"].iloc[i])
        e21   = float(df["ema_21"].iloc[i])
        e200  = float(df["ema_200"].iloc[i])
        atr   = float(df["atr"].iloc[i])
        adx   = float(df["adx"].iloc[i])
        dmp   = float(df["dmp"].iloc[i])
        dmn   = float(df["dmn"].iloc[i])
        rsi   = float(df["rsi_14"].iloc[i])
        orb_l = float(df["orb_low"].iloc[i])

        if any(np.isnan(v) for v in [cl, atr, orb_l]):
            return None
        if np.isnan(adx) or adx < self.min_adx:
            return None

        # 0. ORB range must be meaningful
        orb_h_today = float(df["orb_high"].iloc[i])
        if not np.isnan(orb_h_today) and (orb_h_today - orb_l) < 0.4 * atr:
            return None

        # 0b. Gap direction filter for shorts:
        #   - Don't short if overnight gap up > 0.25 ATR (support from gap fill demand)
        #   - Don't short if overnight gap down > max_gap_atr (large gap-down → often snaps back)
        prev_cl = float(df["prev_close"].iloc[i])
        if not np.isnan(prev_cl) and prev_cl > 0:
            orb_bar_mask = (df["date_et"] == df["date_et"].iloc[i]) & \
                           (df["hour_et"] == 9) & (df["min_et"] == 30)
            day_open = float(df["orb_low"].iloc[i])
            if orb_bar_mask.any():
                day_open = float(df.loc[orb_bar_mask, "open"].iloc[0])
            gap = day_open - prev_cl
            if gap > 0.25 * atr:                  # gapped up — demand underneath
                return None
            if gap < -self.max_gap_atr * atr:      # gapped down too much — likely to snap back
                return None

        # 1. ORB low broken downward before this bar
        if not bool(df["orb_broken_down"].iloc[i]):
            return None

        # 2. Price is still below ORB low (breakdown held)
        if cl >= orb_l:
            return None

        # 3. Macro alignment: price must be below EMA200 (short on downtrend days only)
        if not np.isnan(e200) and cl > e200:
            return None

        # 3b. EMA momentum alignment (short-term downtrend)
        if not np.isnan(e8) and not np.isnan(e21) and e8 >= e21:
            return None

        # 4. DI alignment (-DI > +DI)
        if np.isnan(dmp) or np.isnan(dmn) or dmn <= dmp:
            return None

        # 5. Retest: within last 1-6 bars, price rallied back to ORB low zone
        tol = self.retest_tolerance * atr
        retest_bar = None
        for k in range(max(1, i - 6), i):
            if float(df["high"].iloc[k]) >= orb_l - tol:
                retest_bar = k
                break
        if retest_bar is None:
            return None

        # 6. Reversal candle
        bearish_eng    = self._is_bearish_engulfing(df, i)
        shooting_star  = self._is_shooting_star(df, i)
        deep_retest    = float(df["high"].iloc[i]) >= orb_l - 0.5 * atr
        strong_close_b = self._is_strong_close_bear(df, i) and deep_retest
        if not (bearish_eng or shooting_star or strong_close_b):
            return None

        # 7. MACD momentum — must be turning down AND sig_line confirms downward pressure
        hist      = df["macd_orb_hist"].iloc[i]
        hist_prev = df["macd_orb_hist"].iloc[i - 1]
        sig_line  = df["macd_orb_sig"].iloc[i]
        if np.isnan(hist) or np.isnan(hist_prev) or np.isnan(sig_line):
            return None
        if not (hist < hist_prev and sig_line < 0):
            return None

        # 7b. Volume must be above average on entry bar
        vol     = float(df["volume"].iloc[i])
        vol_avg = float(df["vol_avg20"].iloc[i])
        if np.isnan(vol_avg) or vol_avg <= 0 or vol < 1.15 * vol_avg:
            return None

        # 8. RSI: downward pressure
        if not np.isnan(rsi) and rsi > 60:
            return None

        # ── Build levels ──────────────────────────────────────────── #
        sl   = cl + 1.0 * atr
        risk = sl - cl
        t1   = cl - 2.0 * risk
        t2   = cl - 3.0 * risk

        # ── Score ──────────────────────────────────────────────────── #
        score = 0.58
        if bearish_eng:      score += 0.10
        elif shooting_star:  score += 0.07
        elif strong_close_b: score += 0.05
        if deep_retest:      score += 0.06
        if hist < 0 and hist < hist_prev:
            score += 0.05
        if not np.isnan(vol_avg) and vol_avg > 0 and vol >= 1.15 * vol_avg:
            score += 0.04
        if not np.isnan(e200) and cl < e200:
            score += 0.03   # macro downtrend
        if not np.isnan(rsi) and 35 <= rsi <= 55:
            score += 0.02

        return ("SELL", round(score, 3), "orb_short", round(sl, 4), round(t1, 4), round(t2, 4))

    # ─────────────────────────────────────────────────────────────── #
    # Candle Patterns                                                   #
    # ─────────────────────────────────────────────────────────────── #

    def _is_strong_close_bull(self, df, i):
        op = float(df["open"].iloc[i]); cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        atr = float(df["atr"].iloc[i])
        full_range = hi - lo
        if full_range < 0.3 * atr or cl <= op:
            return False
        return (cl - lo) / full_range >= 0.70

    def _is_strong_close_bear(self, df, i):
        op = float(df["open"].iloc[i]); cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        atr = float(df["atr"].iloc[i])
        full_range = hi - lo
        if full_range < 0.3 * atr or cl >= op:
            return False
        return (cl - lo) / full_range <= 0.30

    def _is_bullish_engulfing(self, df, i):
        if i < 1:
            return False
        c_op = float(df["open"].iloc[i]);  c_cl = float(df["close"].iloc[i])
        p_op = float(df["open"].iloc[i-1]); p_cl = float(df["close"].iloc[i-1])
        return p_cl < p_op and c_cl > c_op and c_cl >= p_op and c_op <= p_cl

    def _is_hammer(self, df, i):
        op = float(df["open"].iloc[i]); cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        body = abs(cl - op)
        full_range = hi - lo
        if full_range < 1e-8:
            return False
        lower_shadow = min(op, cl) - lo
        upper_shadow = hi - max(op, cl)
        return (lower_shadow >= 2.0 * body and
                upper_shadow <= 0.5 * max(body, 1e-8) and
                body >= full_range * 0.05)

    def _is_bearish_engulfing(self, df, i):
        if i < 1:
            return False
        c_op = float(df["open"].iloc[i]);  c_cl = float(df["close"].iloc[i])
        p_op = float(df["open"].iloc[i-1]); p_cl = float(df["close"].iloc[i-1])
        return p_cl > p_op and c_cl < c_op and c_op >= p_cl and c_cl <= p_op

    def _is_shooting_star(self, df, i):
        op = float(df["open"].iloc[i]); cl = float(df["close"].iloc[i])
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        body = abs(cl - op)
        full_range = hi - lo
        if full_range < 1e-8:
            return False
        upper_shadow = hi - max(op, cl)
        lower_shadow = min(op, cl) - lo
        return (upper_shadow >= 2.0 * body and
                lower_shadow <= 0.5 * max(body, 1e-8) and
                body >= full_range * 0.05)
