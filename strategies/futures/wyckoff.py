"""
Wyckoff Method — accumulation/distribution phases, springs, upthrusts.
Looks for climactic volume + failed breakout then reversal.
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_atr, add_volume_metrics, add_rsi, get_key_levels


class WyckoffStrategy(BaseStrategy):
    name = "wyckoff"
    market = "futures"

    def __init__(self, min_rr=2.0):
        self.min_rr = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 60):
            return self._neutral(symbol, timeframe, "Need 60+ bars")
        df = df.copy()
        df = add_atr(df); df = add_volume_metrics(df); df = add_rsi(df)

        last  = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
        close = last["close"]; atr = last["atr"]
        rsi   = last.get("rsi", 50)
        vol_ratio = last.get("vol_ratio", 1.0)

        if pd.isna(atr) or atr == 0: return self._neutral(symbol, timeframe, "ATR unavailable")

        levels = get_key_levels(df, lookback=8)

        # ---- SPRING (Wyckoff bullish) ----
        # Price dips BELOW support on high volume then IMMEDIATELY recovers above support
        # = Wyckoff spring = institutional buying
        support_levels = levels["support"]
        spring = False; spring_level = None
        if support_levels:
            nearest_sup = min(support_levels, key=lambda x: abs(x - close))
            # Previous 2 candles dipped below support; current is recovering
            prev_dipped = prev2["low"] < nearest_sup or prev["low"] < nearest_sup
            recovering  = close > nearest_sup and last["close"] > last["open"]
            high_vol    = prev.get("vol_ratio", 1.0) > 1.8 or prev2.get("vol_ratio", 1.0) > 1.8
            if prev_dipped and recovering and high_vol:
                spring = True; spring_level = nearest_sup

        # ---- UPTHRUST (Wyckoff bearish) ----
        # Price spikes ABOVE resistance on high volume then falls back below = distribution
        resistance_levels = levels["resistance"]
        upthrust = False; upthrust_level = None
        if resistance_levels:
            nearest_res = min(resistance_levels, key=lambda x: abs(x - close))
            prev_spiked = prev2["high"] > nearest_res or prev["high"] > nearest_res
            reversing   = close < nearest_res and last["close"] < last["open"]
            high_vol    = prev.get("vol_ratio", 1.0) > 1.8 or prev2.get("vol_ratio", 1.0) > 1.8
            if prev_spiked and reversing and high_vol:
                upthrust = True; upthrust_level = nearest_res

        # ---- SELLING CLIMAX (potential reversal long) ----
        # Very wide bearish bar + very high volume at a low = exhaustion
        selling_climax = (
            prev["close"] < prev["open"] and
            (prev["high"] - prev["low"]) > atr * 2.0 and
            prev.get("vol_ratio", 1.0) > 2.5 and
            last["close"] > prev["close"] and  # recovery started
            rsi < 35
        )

        # ---- BUYING CLIMAX (potential reversal short) ----
        buying_climax = (
            prev["close"] > prev["open"] and
            (prev["high"] - prev["low"]) > atr * 2.0 and
            prev.get("vol_ratio", 1.0) > 2.5 and
            last["close"] < prev["close"] and  # reversal started
            rsi > 65
        )

        if spring and rsi < 55:
            stop = spring_level - atr * 0.5; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + risk * self.min_rr
            return Signal(direction="BUY", confidence=0.73, strategy=self.name,
                reasoning=(f"Wyckoff Spring at support {spring_level:.2f}: dipped below then recovered "
                           f"on high volume — institutional accumulation. RSI={rsi:.0f}. R:R {self.min_rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if selling_climax:
            stop = df.tail(5)["low"].min() - atr * 0.3; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + risk * self.min_rr
            return Signal(direction="BUY", confidence=0.68, strategy=self.name,
                reasoning=(f"Wyckoff Selling Climax: wide bearish bar ({(prev['high']-prev['low'])/atr:.1f}x ATR) "
                           f"+ {prev.get('vol_ratio',1):.1f}x volume + recovery. RSI={rsi:.0f} oversold. R:R {self.min_rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if upthrust and rsi > 45:
            stop = upthrust_level + atr * 0.5; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close - risk * self.min_rr
            return Signal(direction="SELL", confidence=0.73, strategy=self.name,
                reasoning=(f"Wyckoff Upthrust at resistance {upthrust_level:.2f}: spiked above then reversed "
                           f"on high volume — institutional distribution. RSI={rsi:.0f}. R:R {self.min_rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if buying_climax:
            stop = df.tail(5)["high"].max() + atr * 0.3; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close - risk * self.min_rr
            return Signal(direction="SELL", confidence=0.68, strategy=self.name,
                reasoning=(f"Wyckoff Buying Climax: wide bullish bar ({(prev['high']-prev['low'])/atr:.1f}x ATR) "
                           f"+ {prev.get('vol_ratio',1):.1f}x volume + reversal. RSI={rsi:.0f} overbought. R:R {self.min_rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        return self._neutral(symbol, timeframe,
            f"No Wyckoff spring/upthrust/climax. Vol={vol_ratio:.1f}x, RSI={rsi:.0f}")
