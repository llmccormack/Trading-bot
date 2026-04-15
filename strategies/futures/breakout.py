"""
Breakout — prior day high/low break with strong volume. RSI and trend filters added.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_atr, add_volume_metrics, add_rsi, add_ema, prev_day_levels


class BreakoutStrategy(BaseStrategy):
    name = "breakout"
    market = "futures"

    def __init__(self, vol_multiplier=2.0, min_rr=2.0):
        self.vol_multiplier = vol_multiplier
        self.min_rr         = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 50):
            return self._neutral(symbol, timeframe, "Need 50+ bars")
        df = df.copy()
        df = add_atr(df); df = add_volume_metrics(df); df = add_rsi(df); df = add_ema(df, [21])

        last = df.iloc[-1]; prev = df.iloc[-2]
        close     = last["close"]; atr = last["atr"]
        vol_ratio = last.get("vol_ratio", 1.0); rsi = last.get("rsi", 50)
        ema21     = last.get("ema_21")

        if pd.isna(atr) or atr == 0:
            return self._neutral(symbol, timeframe, "ATR unavailable")

        # Volume MUST confirm — breakouts without volume are traps
        if vol_ratio < self.vol_multiplier:
            return self._neutral(symbol, timeframe,
                f"Vol {vol_ratio:.1f}x avg — need {self.vol_multiplier}x for breakout confirmation")

        pdl = prev_day_levels(df)
        pd_high = pdl.get("prev_high"); pd_low = pdl.get("prev_low")
        if not pd_high or not pd_low:
            return self._neutral(symbol, timeframe, "Prior day levels unavailable")

        buffer = atr * 0.1
        bull_bo = prev["close"] <= pd_high and close > pd_high + buffer
        bear_bo = prev["close"] >= pd_low  and close < pd_low  - buffer

        # Don't enter overbought/oversold breakouts — they often reverse immediately
        if bull_bo and rsi < 75 and (pd.isna(ema21) or close > ema21):
            stop = pd_high - atr * 0.3; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + risk * self.min_rr
            conf   = round(min(0.65 + (vol_ratio - self.vol_multiplier) * 0.04, 0.85), 2)
            return Signal(direction="BUY", confidence=conf, strategy=self.name,
                reasoning=(f"Breakout above prior day high {pd_high:.2f}. "
                           f"Volume {vol_ratio:.1f}x avg confirms. RSI={rsi:.0f}. "
                           f"Price above EMA21. R:R {self.min_rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if bear_bo and rsi > 25 and (pd.isna(ema21) or close < ema21):
            stop = pd_low + atr * 0.3; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close - risk * self.min_rr
            conf   = round(min(0.65 + (vol_ratio - self.vol_multiplier) * 0.04, 0.85), 2)
            return Signal(direction="SELL", confidence=conf, strategy=self.name,
                reasoning=(f"Breakdown below prior day low {pd_low:.2f}. "
                           f"Volume {vol_ratio:.1f}x avg confirms. RSI={rsi:.0f}. "
                           f"Price below EMA21. R:R {self.min_rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        return self._neutral(symbol, timeframe,
            f"No breakout. PD High={pd_high:.2f}, PD Low={pd_low:.2f}, Close={close:.2f}, Vol={vol_ratio:.1f}x")
