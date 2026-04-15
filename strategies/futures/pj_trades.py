"""
PJ Trades — VWAP-anchored trend continuation.
Strict: clear directional move first, then clean pullback to VWAP with declining vol.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_vwap, add_atr, add_volume_metrics, add_rsi


class PJTradesStrategy(BaseStrategy):
    name = "pj_trades"
    market = "futures"

    def __init__(self, min_rr=1.5):
        self.min_rr = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 30):
            return self._neutral(symbol, timeframe, "Need 30+ bars")
        df = df.copy()
        df = add_vwap(df); df = add_atr(df); df = add_volume_metrics(df); df = add_rsi(df)

        last  = df.iloc[-1]; close = last["close"]
        vwap  = last.get("vwap"); atr = last["atr"]
        rsi   = last.get("rsi", 50)

        if pd.isna(vwap) or pd.isna(atr) or atr == 0:
            return self._neutral(symbol, timeframe, "VWAP/ATR unavailable")

        near_vwap = abs(close - vwap) < atr * 0.35
        above_vwap = close > vwap
        below_vwap = close < vwap

        # Need a clear directional move first — check last 10 bars for structure
        recent = df.tail(12)
        highs  = recent["high"].values
        lows   = recent["low"].values

        # Require at least 2 higher highs and 2 higher lows for bullish, opposite for bearish
        bull_hh = sum(highs[i] > highs[i-2] for i in range(2, len(highs))) >= 2
        bull_hl = sum(lows[i]  > lows[i-2]  for i in range(2, len(lows)))  >= 2
        bear_lh = sum(highs[i] < highs[i-2] for i in range(2, len(highs))) >= 2
        bear_ll = sum(lows[i]  < lows[i-2]  for i in range(2, len(lows)))  >= 2

        bull_structure = bull_hh and bull_hl
        bear_structure = bear_lh and bear_ll

        # Volume declining on the pullback (last 3 bars decreasing vol = healthy pullback)
        clean_pullback = df["volume"].tail(3).is_monotonic_decreasing

        if above_vwap and bull_structure and near_vwap and rsi < 65:
            stop = last["low"] - atr * 0.2; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + risk * 2.0; rr = 2.0
            conf = round(0.60 + (0.10 if clean_pullback else 0) + (0.05 if last.get("vol_ratio",1)>1 else 0), 2)
            return Signal(direction="BUY", confidence=conf, strategy=self.name,
                reasoning=(f"Price above VWAP ({vwap:.2f}), HH/HL structure confirmed. "
                           f"Pulling back to VWAP — trend continuation long. "
                           f"{'Clean declining vol on pullback. ' if clean_pullback else ''}"
                           f"RSI={rsi:.0f}. R:R {rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if below_vwap and bear_structure and near_vwap and rsi > 35:
            stop = last["high"] + atr * 0.2; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close - risk * 2.0; rr = 2.0
            conf = round(0.60 + (0.10 if clean_pullback else 0) + (0.05 if last.get("vol_ratio",1)>1 else 0), 2)
            return Signal(direction="SELL", confidence=conf, strategy=self.name,
                reasoning=(f"Price below VWAP ({vwap:.2f}), LL/LH structure confirmed. "
                           f"Bouncing to VWAP — trend continuation short. "
                           f"{'Clean declining vol on bounce. ' if clean_pullback else ''}"
                           f"RSI={rsi:.0f}. R:R {rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        reasons = []
        if not (bull_structure or bear_structure): reasons.append("no clear HH/HL or LL/LH structure")
        if not near_vwap: reasons.append(f"price {abs(close-vwap)/atr:.1f} ATRs from VWAP")
        if rsi >= 65 and above_vwap: reasons.append(f"RSI={rsi:.0f} overbought for long")
        if rsi <= 35 and below_vwap: reasons.append(f"RSI={rsi:.0f} oversold for short")
        return self._neutral(symbol, timeframe, f"No VWAP setup: {', '.join(reasons) or 'mixed conditions'}")
