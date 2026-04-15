"""
PB Blake — price action at key structural levels.
Requires: HTF alignment + price AT a level + clear rejection candle + RSI not exhausted.
Strict: returns NEUTRAL unless ALL conditions are met.
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_ema, add_atr, add_vwap, add_volume_metrics, add_rsi, get_key_levels


class PBBlakeStrategy(BaseStrategy):
    name = "pb_blake"
    market = "futures"

    def __init__(self, min_rr: float = 2.0, proximity_atr: float = 0.35):
        self.min_rr = min_rr
        self.proximity_atr = proximity_atr

    def generate_signal(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Signal:
        if not self._needs_bars(df, 80):
            return self._neutral(symbol, timeframe, "Need 80+ bars")

        df = df.copy()
        df = add_ema(df, [21, 55, 200])
        df = add_atr(df)
        df = add_vwap(df)
        df = add_volume_metrics(df)
        df = add_rsi(df)

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        prev2 = df.iloc[-3]
        atr   = last["atr"]
        close = last["close"]
        rsi   = last.get("rsi", 50)

        if pd.isna(atr) or atr == 0:
            return self._neutral(symbol, timeframe, "ATR unavailable")

        ema55  = last.get("ema_55")
        ema200 = last.get("ema_200")
        if pd.isna(ema55) or pd.isna(ema200):
            return self._neutral(symbol, timeframe, "EMAs not ready")

        # HTF bias — only trade in direction of 200 EMA
        htf_bullish = close > ema200 and ema55 > ema200
        htf_bearish = close < ema200 and ema55 < ema200

        # Key levels
        levels = get_key_levels(df, lookback=7)
        proximity = atr * self.proximity_atr

        near_support    = [lvl for lvl in levels["support"]    if abs(close - lvl) < proximity]
        near_resistance = [lvl for lvl in levels["resistance"] if abs(close - lvl) < proximity]

        # Rejection candle (last candle must be a clear pin bar / rejection)
        cr = last["high"] - last["low"]
        if cr == 0:
            return self._neutral(symbol, timeframe, "Zero-range candle")

        lower_wick = min(last["open"], last["close"]) - last["low"]
        upper_wick = last["high"] - max(last["open"], last["close"])
        body       = abs(last["close"] - last["open"])

        # Strict: wick must be > 55% of range AND body must be less than 40% of range
        bullish_pin = (lower_wick / cr > 0.55) and (body / cr < 0.40) and last["close"] > last["open"]
        bearish_pin = (upper_wick / cr > 0.55) and (body / cr < 0.40) and last["close"] < last["open"]

        vol_ok = last.get("vol_ratio", 1.0) > 1.1

        # ---- LONG: HTF bullish + at support + bullish pin + RSI not overbought ----
        if htf_bullish and near_support and bullish_pin and rsi < 65:
            support_lvl = min(near_support, key=lambda x: abs(x - close))
            stop        = support_lvl - atr * 0.30
            risk        = close - stop
            if risk <= 0:
                return self._neutral(symbol, timeframe, "Invalid risk")

            above = [r for r in levels["resistance"] if r > close]
            if not above:
                return self._neutral(symbol, timeframe, "No target above")
            target = min(above)
            rr     = (target - close) / risk
            if rr < self.min_rr:
                return self._neutral(symbol, timeframe, f"R:R {rr:.1f} < {self.min_rr} min")

            conf = 0.60 + (0.10 if vol_ok else 0) + (0.05 if close > last.get("vwap", 0) else 0)
            return Signal(
                direction="BUY", confidence=round(conf, 2), strategy=self.name,
                reasoning=(f"Bullish pin bar at support {support_lvl:.2f}. HTF trend bullish (EMA55>{ema200:.0f}). "
                           f"RSI={rsi:.0f} not overbought. R:R {rr:.1f}:1.{' Vol confirmed.' if vol_ok else ''}"),
                symbol=symbol, timeframe=timeframe,
                suggested_entry=close, suggested_stop=round(stop, 2), suggested_target=round(target, 2),
            )

        # ---- SHORT: HTF bearish + at resistance + bearish pin + RSI not oversold ----
        if htf_bearish and near_resistance and bearish_pin and rsi > 35:
            resistance_lvl = min(near_resistance, key=lambda x: abs(x - close))
            stop           = resistance_lvl + atr * 0.30
            risk           = stop - close
            if risk <= 0:
                return self._neutral(symbol, timeframe, "Invalid risk")

            below = [s for s in levels["support"] if s < close]
            if not below:
                return self._neutral(symbol, timeframe, "No target below")
            target = max(below)
            rr     = (close - target) / risk
            if rr < self.min_rr:
                return self._neutral(symbol, timeframe, f"R:R {rr:.1f} < {self.min_rr} min")

            conf = 0.60 + (0.10 if vol_ok else 0) + (0.05 if close < last.get("vwap", float("inf")) else 0)
            return Signal(
                direction="SELL", confidence=round(conf, 2), strategy=self.name,
                reasoning=(f"Bearish pin bar at resistance {resistance_lvl:.2f}. HTF trend bearish (EMA55<{ema200:.0f}). "
                           f"RSI={rsi:.0f} not oversold. R:R {rr:.1f}:1.{' Vol confirmed.' if vol_ok else ''}"),
                symbol=symbol, timeframe=timeframe,
                suggested_entry=close, suggested_stop=round(stop, 2), suggested_target=round(target, 2),
            )

        reasons = []
        if not htf_bullish and not htf_bearish: reasons.append("no HTF trend")
        if not near_support and not near_resistance: reasons.append("price not at key level")
        if not bullish_pin and not bearish_pin: reasons.append("no rejection candle")
        return self._neutral(symbol, timeframe, f"Conditions not met: {', '.join(reasons) or 'mixed signals'}")
