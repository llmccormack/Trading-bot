"""
ICT (Inner Circle Trader) — order blocks, fair value gaps, market structure.
Michael J. Huddleston methodology. Hunts institutional footprints.
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_atr, add_rsi, add_ema


class ICTStrategy(BaseStrategy):
    name = "ict"
    market = "futures"

    def __init__(self, min_rr=2.5):
        self.min_rr = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 50):
            return self._neutral(symbol, timeframe, "Need 50+ bars")
        df = df.copy()
        df = add_atr(df); df = add_rsi(df); df = add_ema(df, [50])

        last  = df.iloc[-1]; close = last["close"]
        atr   = last["atr"]; rsi = last.get("rsi", 50)
        ema50 = last.get("ema_50")
        htf_bias = "bull" if not pd.isna(ema50) and close > ema50 else "bear"

        # --- Fair Value Gap (FVG) detection ---
        # Bullish FVG: candle[i-2].high < candle[i].low (gap between them, candle i-1 was big up)
        # Bearish FVG: candle[i-2].low > candle[i].high
        fvg_bull_zones = []
        fvg_bear_zones = []
        for i in range(2, min(30, len(df))):
            c0 = df.iloc[-i-1]; c1 = df.iloc[-i]; c2 = df.iloc[-i+1]
            # Bullish FVG
            if c0["high"] < c2["low"] and (c1["close"] > c1["open"]):
                fvg_bull_zones.append((c0["high"], c2["low"]))
            # Bearish FVG
            if c0["low"] > c2["high"] and (c1["close"] < c1["open"]):
                fvg_bear_zones.append((c2["high"], c0["low"]))

        # --- Order Block detection ---
        # Bullish OB: last bearish candle before a 3+ candle bullish push
        # Bearish OB: last bullish candle before a 3+ candle bearish push
        bull_ob_zone = None
        bear_ob_zone = None
        for i in range(5, min(40, len(df)-3)):
            candle = df.iloc[-i]
            next3  = df.iloc[-i+1:-i+4]
            if candle["close"] < candle["open"]:  # bearish candle
                if all(next3["close"] > next3["open"]) and next3["close"].iloc[-1] > candle["high"] * 1.003:
                    bull_ob_zone = (candle["low"], candle["high"])
                    break
        for i in range(5, min(40, len(df)-3)):
            candle = df.iloc[-i]
            next3  = df.iloc[-i+1:-i+4]
            if candle["close"] > candle["open"]:  # bullish candle
                if all(next3["close"] < next3["open"]) and next3["close"].iloc[-1] < candle["low"] * 0.997:
                    bear_ob_zone = (candle["low"], candle["high"])
                    break

        # --- Check if price is in a significant zone ---
        in_bull_fvg = any(low <= close <= high for (low, high) in fvg_bull_zones[:3])
        in_bear_fvg = any(low <= close <= high for (low, high) in fvg_bear_zones[:3])
        in_bull_ob  = bull_ob_zone and bull_ob_zone[0] <= close <= bull_ob_zone[1]
        in_bear_ob  = bear_ob_zone and bear_ob_zone[0] <= close <= bear_ob_zone[1]

        # LONG: bullish bias + price in bullish FVG or OB + RSI not overbought
        if htf_bias == "bull" and (in_bull_fvg or in_bull_ob) and rsi < 65:
            zone_type = "Bullish Order Block" if in_bull_ob else "Bullish Fair Value Gap"
            zone      = bull_ob_zone if in_bull_ob else [z for z in fvg_bull_zones if z[0] <= close <= z[1]][0]
            stop      = zone[0] - atr * 0.2; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + risk * self.min_rr; rr = self.min_rr
            return Signal(direction="BUY", confidence=0.70, strategy=self.name,
                reasoning=(f"ICT: Price in {zone_type} ({zone[0]:.2f}–{zone[1]:.2f}). "
                           f"HTF bias bullish (above EMA50). RSI={rsi:.0f}. R:R {rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        # SHORT: bearish bias + price in bearish FVG or OB + RSI not oversold
        if htf_bias == "bear" and (in_bear_fvg or in_bear_ob) and rsi > 35:
            zone_type = "Bearish Order Block" if in_bear_ob else "Bearish Fair Value Gap"
            zone      = bear_ob_zone if in_bear_ob else [z for z in fvg_bear_zones if z[0] <= close <= z[1]][0]
            stop      = zone[1] + atr * 0.2; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            target = close - risk * self.min_rr; rr = self.min_rr
            return Signal(direction="SELL", confidence=0.70, strategy=self.name,
                reasoning=(f"ICT: Price in {zone_type} ({zone[0]:.2f}–{zone[1]:.2f}). "
                           f"HTF bias bearish (below EMA50). RSI={rsi:.0f}. R:R {rr}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        fvg_count = len(fvg_bull_zones) + len(fvg_bear_zones)
        return self._neutral(symbol, timeframe,
            f"ICT: {fvg_count} FVGs found, OB: {'bull' if bull_ob_zone else 'none'}/{'bear' if bear_ob_zone else 'none'}. "
            f"Price not in actionable zone ({htf_bias} bias, RSI={rsi:.0f})")
