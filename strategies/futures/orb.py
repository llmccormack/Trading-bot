"""
Opening Range Breakout (ORB) — first 30-minute range, break with volume.
Classic, time-tested futures setup. Only fires during opening session.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_atr, add_volume_metrics, add_rsi, add_ema


class ORBStrategy(BaseStrategy):
    name = "orb"
    market = "futures"

    def __init__(self, vol_min=1.5, min_rr=2.0):
        self.vol_min = vol_min
        self.min_rr  = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 30):
            return self._neutral(symbol, timeframe, "Need 30+ bars")
        df = df.copy()
        df = add_atr(df); df = add_volume_metrics(df); df = add_rsi(df); df = add_ema(df, [21])

        last      = df.iloc[-1]; close = last["close"]
        atr       = last["atr"]; rsi = last.get("rsi", 50)
        vol_ratio = last.get("vol_ratio", 1.0)

        if pd.isna(atr) or atr == 0: return self._neutral(symbol, timeframe, "ATR unavailable")

        # Define opening range from first 6 bars (≈30 min on 5m) of the most recent session
        if "timestamp" in df.columns:
            df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
            today_df    = df[df["_date"] == df["_date"].iloc[-1]]
            if len(today_df) < 7:
                return self._neutral(symbol, timeframe, f"Only {len(today_df)} bars today — need 7+ for ORB")
            opening_bars  = today_df.head(6)
            post_open     = today_df.iloc[6:]
        else:
            opening_bars = df.head(6); post_open = df.iloc[6:]

        orb_high = opening_bars["high"].max()
        orb_low  = opening_bars["low"].min()
        orb_size = orb_high - orb_low

        if orb_size < atr * 0.3:
            return self._neutral(symbol, timeframe,
                f"ORB too narrow ({orb_size:.2f} vs {atr*0.3:.2f} min) — choppy open, skip")

        # Need current bar to be breaking the ORB with volume
        buffer = atr * 0.08

        bull_break = close > orb_high + buffer and vol_ratio >= self.vol_min and rsi < 75
        bear_break = close < orb_low  - buffer and vol_ratio >= self.vol_min and rsi > 25

        if bull_break:
            stop   = orb_high - atr * 0.3; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + orb_size * 1.5  # project 1.5x the ORB size
            rr     = (target - close) / risk
            if rr < self.min_rr: target = close + risk * self.min_rr; rr = self.min_rr
            return Signal(direction="BUY", confidence=round(min(0.65 + vol_ratio * 0.03, 0.82), 2),
                strategy=self.name,
                reasoning=(f"ORB breakout above {orb_high:.2f} (range: {orb_low:.2f}–{orb_high:.2f}, "
                           f"size={orb_size:.2f}). Volume {vol_ratio:.1f}x confirms. RSI={rsi:.0f}. R:R {rr:.1f}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if bear_break:
            stop   = orb_low + atr * 0.3; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close - orb_size * 1.5
            rr     = (close - target) / risk
            if rr < self.min_rr: target = close - risk * self.min_rr; rr = self.min_rr
            return Signal(direction="SELL", confidence=round(min(0.65 + vol_ratio * 0.03, 0.82), 2),
                strategy=self.name,
                reasoning=(f"ORB breakdown below {orb_low:.2f} (range: {orb_low:.2f}–{orb_high:.2f}, "
                           f"size={orb_size:.2f}). Volume {vol_ratio:.1f}x confirms. RSI={rsi:.0f}. R:R {rr:.1f}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        return self._neutral(symbol, timeframe,
            f"ORB range: {orb_low:.2f}–{orb_high:.2f}. Price at {close:.2f} — no breakout yet. Vol={vol_ratio:.1f}x")
