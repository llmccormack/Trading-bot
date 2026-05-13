"""
FadeTheRip — Short-Only Mean Reversion Strategy
─────────────────────────────────────────────────
Concept: On a bearish day, price will rip counter-trend into VWAP or EMA21.
         Fade (short) that bounce when it shows signs of failure.

STRATEGY LOGIC:
  1. Day bias must be BEARISH: price below VWAP85 AND EMA8 < EMA21 at 10:15 AM
  2. Wait for a dead-cat bounce: 2+ consecutive green candles, RSI climbs to 52–68
  3. Rejection signal at a key level (VWAP85 or EMA21):
       - Bearish Engulfing candle, OR
       - Shooting Star / Pin Bar, OR
       - Strong bearish close (close in bottom 25% of candle range)
  4. Entry: open of next candle after signal
  5. Stop: above bounce high + 0.4 ATR
  6. Target 1: 1.5R (partial exit, move stop to BE)
  7. Target 2: 2.5R (runner)
  8. Trade window: 10:00 AM – 3:30 PM ET only
  9. Max 2 trades per day
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytz

from utils.indicators import add_all
from backtesting.models import BacktestTrade, BacktestResult

ET = pytz.timezone("America/New_York")

# ── Timing ──────────────────────────────────────────────────────── #
ENTRY_START_H, ENTRY_START_M = 10, 0
ENTRY_END_H,   ENTRY_END_M   = 15, 30
BIAS_CONFIRM_H, BIAS_CONFIRM_M = 10, 15   # earliest bar to read day bias

# ── Strategy params ─────────────────────────────────────────────── #
MIN_BOUNCE_BARS    = 2       # consecutive green bars needed to call it a bounce
RSI_BOUNCE_LOW     = 52      # RSI floor for bounce (must be recovering) — tightened from 50 (backtest: NQ PF 5.05→5.91, ES PF 2.89→4.55)
RSI_BOUNCE_HIGH    = 65      # RSI ceiling — tightened from 70 (avoids overextended bounces)
VWAP_PROXIMITY_ATR = 0.6     # within X ATR of VWAP or EMA21 = "at the level"
STOP_ATR_MULT      = 0.4     # stop above bounce high + X ATR
T1_R               = 1.5     # first target in R
T2_R               = 2.5     # second target (runner)
MAX_TRADES_PER_DAY = 2
MAX_BARS_HELD      = 30      # ~2.5 hours on 5m


class FadeTheRipEngine:
    """Short-only mean reversion backtest on 5-minute bars."""

    def __init__(
        self,
        min_adx: float = 15.0,
        stop_atr_mult: float = STOP_ATR_MULT,
        t1_r: float = T1_R,
        t2_r: float = T2_R,
        max_trades_per_day: int = MAX_TRADES_PER_DAY,
    ):
        self.min_adx            = min_adx
        self.stop_atr_mult      = stop_atr_mult
        self.t1_r               = t1_r
        self.t2_r               = t2_r
        self.max_trades_per_day = max_trades_per_day

    # ─────────────────────────────────────────────────────────────── #

    def run(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "5m") -> BacktestResult:
        df = df.copy()
        df = add_all(df)
        df = self._add_time_cols(df)

        trades: list[BacktestTrade] = []
        open_trade: dict | None = None
        trades_today = 0
        last_trade_day = None
        t1_hit = False

        for i in range(50, len(df)):
            row = df.iloc[i]

            # ── Track new day ──────────────────────────────────── #
            today = (int(row["hour_et"]), int(row["min_et"]))
            trade_day = str(df.index[i])[:10]
            if trade_day != last_trade_day:
                trades_today = 0
                last_trade_day = trade_day

            # ── Manage open trade ──────────────────────────────── #
            if open_trade is not None:
                result = self._manage_trade(df, i, open_trade, t1_hit)
                if result == "t1_hit":
                    t1_hit = True
                elif result is not None:
                    trades.append(result)
                    open_trade = None
                    t1_hit = False
                continue

            # ── Entry window check ─────────────────────────────── #
            h, m = int(row["hour_et"]), int(row["min_et"])
            in_window = (
                (h > ENTRY_START_H or (h == ENTRY_START_H and m >= ENTRY_START_M))
                and (h < ENTRY_END_H or (h == ENTRY_END_H and m <= ENTRY_END_M))
            )
            if not in_window:
                continue

            if trades_today >= self.max_trades_per_day:
                continue

            # ── Look for signal ────────────────────────────────── #
            sig = self._check_signal(df, i)
            if sig is None:
                continue

            entry, stop, t1, t2 = sig
            open_trade = {
                "entry_bar": i,
                "entry_price": entry,
                "stop": stop,
                "t1": t1,
                "t2": t2,
                "risk_pts": stop - entry,
            }
            t1_hit = False
            trades_today += 1

        result = BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            market="futures",
            total_bars=len(df),
            trades=trades,
        )
        return result

    # ─────────────────────────────────────────────────────────────── #
    # Signal Detection                                                 #
    # ─────────────────────────────────────────────────────────────── #

    def _check_signal(self, df: pd.DataFrame, i: int) -> tuple | None:
        """Returns (entry, stop, t1, t2) or None."""

        # Need enough lookback
        if i < 5:
            return None

        row  = df.iloc[i]
        atr  = float(row.get("atr", np.nan))
        if np.isnan(atr) or atr <= 0:
            return None

        # ADX filter — need a trending (not choppy) market
        adx = float(row.get("adx", np.nan))
        if np.isnan(adx) or adx < self.min_adx:
            return None

        vwap = float(row.get("vwap", np.nan))
        e8   = float(row.get("ema_8",   np.nan))
        e21  = float(row.get("ema_21",  np.nan))
        rsi  = float(row.get("rsi",     np.nan))
        cl   = float(row["close"])
        op   = float(row["open"])
        hi   = float(row["high"])
        lo   = float(row["low"])

        if any(np.isnan(v) for v in [vwap, e8, e21, rsi]):
            return None

        # 1. Day bias: bearish context — majority of recent bars below VWAP
        #    AND medium-term trend down (EMA21 < EMA55)
        e55 = float(row.get("ema_55", np.nan))
        recent = df.iloc[max(0, i-10): i]
        bars_below_vwap = sum(
            1 for _, b in recent.iterrows()
            if not np.isnan(float(b.get("vwap", np.nan)))
            and float(b["close"]) < float(b.get("vwap", np.nan))
        )
        if bars_below_vwap < 6:   # at least 6 of last 10 bars below VWAP
            return None
        if not np.isnan(e55) and e21 >= e55:   # medium-term must be bearish
            return None

        # 2. Detect bounce: price is NOW near or above VWAP (the "rip")
        #    and last 2+ bars were green (counter-trend bounce)
        bounce_bars = df.iloc[i - MIN_BOUNCE_BARS: i]
        all_green = all(
            float(b["close"]) > float(b["open"])
            for _, b in bounce_bars.iterrows()
        )
        if not all_green:
            return None

        # RSI recovering but not overbought — typical bounce zone
        if not (RSI_BOUNCE_LOW <= rsi <= RSI_BOUNCE_HIGH):
            return None

        # 3. Price must be near VWAP or EMA21 (within VWAP_PROXIMITY_ATR)
        #    This is where we fade — at the level, not below it
        near_vwap = abs(cl - vwap) <= VWAP_PROXIMITY_ATR * atr
        near_e21  = abs(cl - e21)  <= VWAP_PROXIMITY_ATR * atr
        if not (near_vwap or near_e21):
            return None

        # 4. Rejection candle on this bar
        if not self._is_rejection_candle(op, hi, lo, cl):
            return None

        # 5. Build trade levels
        bounce_high = float(df.iloc[i - MIN_BOUNCE_BARS: i + 1]["high"].max())
        entry = cl                               # enter at close of signal bar
        stop  = bounce_high + self.stop_atr_mult * atr
        risk  = stop - entry
        if risk <= 0:
            return None

        t1 = entry - self.t1_r * risk
        t2 = entry - self.t2_r * risk

        # Sanity: minimum R:R
        if risk > 3 * atr:  # stop too wide
            return None

        return (entry, stop, t1, t2)

    def _is_rejection_candle(self, op: float, hi: float, lo: float, cl: float) -> bool:
        """Bearish engulfing, shooting star, or strong bearish close."""
        rng = hi - lo
        if rng <= 0:
            return False

        body      = abs(cl - op)
        upper_wick = hi - max(cl, op)
        lower_wick = min(cl, op) - lo
        bearish    = cl < op

        # Bearish engulfing: large body, close near low
        if bearish and body / rng >= 0.6 and lower_wick / rng <= 0.2:
            return True

        # Shooting star: long upper wick (≥2× body), small lower wick
        if upper_wick >= 2.0 * body and lower_wick <= 0.3 * rng:
            return True

        # Strong bearish close: close in bottom 25% of range
        if (cl - lo) / rng <= 0.25 and bearish:
            return True

        return False

    # ─────────────────────────────────────────────────────────────── #
    # Trade Management                                                 #
    # ─────────────────────────────────────────────────────────────── #

    def _manage_trade(
        self, df: pd.DataFrame, i: int, trade: dict, t1_hit: bool
    ) -> str | BacktestTrade | None:
        """
        Returns:
          "t1_hit"       — first target hit, move stop to BE
          BacktestTrade  — trade closed
          None           — still open
        """
        row   = df.iloc[i]
        hi    = float(row["high"])
        lo    = float(row["low"])
        cl    = float(row["close"])
        h, m  = int(row["hour_et"]), int(row["min_et"])

        entry  = trade["entry_price"]
        stop   = trade["stop"]
        t1     = trade["t1"]
        t2     = trade["t2"]
        risk   = trade["risk_pts"]
        e_bar  = trade["entry_bar"]

        # After T1, stop moves to breakeven
        effective_stop = entry if t1_hit else stop

        # EOD force close
        eod = h == 15 and m >= 55
        bars_held = i - e_bar
        timeout   = bars_held >= MAX_BARS_HELD

        # Short: profit is downward movement
        # Stop hit (price went up through stop)
        if hi >= effective_stop:
            exit_px    = effective_stop
            runner_pnl = entry - exit_px          # 0 when stopped at BE
            if t1_hit:
                pnl = trade.get("partial_pnl", 0.0) + runner_pnl * 0.5
            else:
                pnl = runner_pnl
            r      = pnl / risk if risk > 0 else 0.0
            reason = "be_stopped" if t1_hit else "stop"
            return self._make_trade(trade, i, exit_px, reason, pnl, r, bars_held, t1_hit)

        # T1 hit — book half position profit, store partial_pnl, signal caller to move stop
        if not t1_hit and lo <= t1:
            trade["partial_pnl"] = (entry - t1) * 0.5   # short: entry > t1, so positive
            return "t1_hit"

        # T2 hit (runner — half position)
        if t1_hit and lo <= t2:
            exit_px    = t2
            runner_pnl = entry - exit_px
            pnl        = trade.get("partial_pnl", 0.0) + runner_pnl * 0.5
            r          = pnl / risk if risk > 0 else 0.0
            return self._make_trade(trade, i, exit_px, "target2", pnl, r, bars_held, t1_hit)

        # EOD or timeout
        if eod or timeout:
            exit_px    = cl
            runner_pnl = entry - exit_px
            if t1_hit:
                pnl = trade.get("partial_pnl", 0.0) + runner_pnl * 0.5
            else:
                pnl = runner_pnl
            r      = pnl / risk if risk > 0 else 0.0
            reason = "eod" if eod else "timeout"
            return self._make_trade(trade, i, exit_px, reason, pnl, r, bars_held, t1_hit)

        return None

    def _make_trade(
        self, trade: dict, exit_bar: int, exit_px: float,
        reason: str, pnl: float, r: float, bars_held: int, t1_hit: bool
    ) -> BacktestTrade:
        return BacktestTrade(
            strategy="fade_the_rip",
            direction="SELL",
            entry_bar=trade["entry_bar"],
            exit_bar=exit_bar,
            entry_price=trade["entry_price"],
            exit_price=exit_px,
            stop_loss=trade["stop"],
            take_profit=trade["t2"],
            exit_reason=reason,
            pnl_pts=round(pnl, 4),
            r_multiple=round(r, 3),
            bars_held=bars_held,
            composite_score=0.0,
            regime="fade_rip",
            be_moved=t1_hit,
        )

    # ─────────────────────────────────────────────────────────────── #
    # Live Signal (mirrors A+ interface for autopilot integration)    #
    # ─────────────────────────────────────────────────────────────── #

    def live_signal(self, df_raw: pd.DataFrame, current_price: float | None = None) -> dict | None:
        """Evaluate most recently completed bar for a live short signal."""
        from utils.indicators import add_all
        df = df_raw.copy()
        df = add_all(df)
        df = self._add_time_cols(df)
        n = len(df)
        if n < 55:
            return None

        i = n - 1
        row = df.iloc[i]
        h, m = int(row["hour_et"]), int(row["min_et"])

        # Entry window: 10:00 AM – 3:30 PM ET
        in_window = (
            (h > ENTRY_START_H or (h == ENTRY_START_H and m >= ENTRY_START_M))
            and (h < ENTRY_END_H or (h == ENTRY_END_H and m <= ENTRY_END_M))
        )
        if not in_window:
            return None

        sig = self._check_signal(df, i)
        if sig is None:
            return None

        entry, stop, t1, t2 = sig
        if current_price:
            entry = current_price
            risk  = stop - entry
            t1    = entry - self.t1_r * risk
            t2    = entry - self.t2_r * risk

        risk_pts = stop - entry
        if risk_pts <= 0:
            return None

        rr = (entry - t2) / risk_pts if risk_pts > 0 else 0.0

        bar_time_utc = df["timestamp"].iloc[i]
        bar_time_et  = bar_time_utc.astimezone(ET)

        return {
            "direction":      "SELL",
            "strategy":       "fade_the_rip",
            "strategy_label": "Fade the Rip",
            "regime":         "vwap_rejection_short",
            "entry":          round(entry, 2),
            "stop":           round(stop, 2),
            "target":         round(t2, 2),
            "target_1":       round(t1, 2),
            "target_2":       round(t2, 2),
            "risk_pts":       round(risk_pts, 2),
            "rr":             round(rr, 2),
            "score":          0.75,
            "adx":            round(float(row.get("adx", 0)), 1),
            "bar_time":       bar_time_et.strftime("%Y-%m-%d %H:%M ET"),
            "timeframe":      "5m",
        }

    # ─────────────────────────────────────────────────────────────── #
    # Helpers                                                          #
    # ─────────────────────────────────────────────────────────────── #

    def _add_time_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if "hour_et" not in df.columns:
            if "timestamp" in df.columns:
                ts_et = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
            else:
                idx = df.index
                if hasattr(idx, "tz") and idx.tz is not None:
                    ts_et = idx.tz_convert(ET)
                else:
                    ts_et = idx.tz_localize("UTC").tz_convert(ET)
            df["hour_et"] = ts_et.dt.hour
            df["min_et"]  = ts_et.dt.minute
        return df
