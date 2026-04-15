"""
Trend Following — EMA alignment with ADX + RSI filters.
Fires on:
  1. Fresh EMA 8/21 golden or death cross (high conviction, 0.72)
  2. Established EMA stack (8>21>55 bull / 8<21<55 bear) with price
     pulling back to EMA8 zone and now bouncing (continuation, 0.62)
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_ema, add_atr, add_adx, add_rsi


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"
    market = "futures"

    def __init__(self, adx_min=22.0):
        self.adx_min = adx_min

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 60):
            return self._neutral(symbol, timeframe, "Need 60+ bars")
        df = df.copy()
        df = add_ema(df, [8, 21, 55])
        df = add_atr(df)
        df = add_adx(df)
        df = add_rsi(df)

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        prev2 = df.iloc[-3]

        close = last["close"]
        atr   = last["atr"]
        adx   = last.get("adx", 0)
        rsi   = last.get("rsi", 50)
        ema8  = last.get("ema_8")
        ema21 = last.get("ema_21")
        ema55 = last.get("ema_55")
        p_e8  = prev.get("ema_8")
        p_e21 = prev.get("ema_21")

        if any(pd.isna(v) for v in [atr, adx, ema8, ema21, ema55, p_e8, p_e21]):
            return self._neutral(symbol, timeframe, "Indicators not ready")
        if adx < self.adx_min:
            return self._neutral(symbol, timeframe, f"ADX={adx:.1f} < {self.adx_min} — trend too weak")

        # ── Signal 1: Fresh EMA crossover (high conviction) ──────── #
        golden_cross = p_e8 <= p_e21 and ema8 > ema21
        death_cross  = p_e8 >= p_e21 and ema8 < ema21

        if golden_cross and rsi < 70 and ema21 > ema55:
            stop = close - atr * 2.0
            target = close + atr * 4.0
            return Signal(
                direction="BUY", confidence=0.72, strategy=self.name,
                reasoning=(f"Golden cross: EMA8 just crossed above EMA21. "
                           f"EMA stack bullish (21>{ema55:.1f}). ADX={adx:.1f}. RSI={rsi:.0f}."),
                symbol=symbol, timeframe=timeframe,
                suggested_entry=close, suggested_stop=round(stop, 2), suggested_target=round(target, 2),
            )

        if death_cross and rsi > 30 and ema21 < ema55:
            stop = close + atr * 2.0
            target = close - atr * 4.0
            return Signal(
                direction="SELL", confidence=0.72, strategy=self.name,
                reasoning=(f"Death cross: EMA8 just crossed below EMA21. "
                           f"EMA stack bearish (21<{ema55:.1f}). ADX={adx:.1f}. RSI={rsi:.0f}."),
                symbol=symbol, timeframe=timeframe,
                suggested_entry=close, suggested_stop=round(stop, 2), suggested_target=round(target, 2),
            )

        # ── Signal 2: Established trend + pullback-to-EMA8 bounce ── #
        # Bull: EMA8 > EMA21 > EMA55, price dipped close to EMA8 recently,
        #       last bar closes back above EMA8 with slight uptick
        bull_stack = ema8 > ema21 > ema55
        bear_stack = ema8 < ema21 < ema55

        if bull_stack and rsi < 65:
            # Price touched or crossed below EMA8 in last 3 bars then recovered
            touched_ema8 = any(
                df.iloc[i]["low"] <= df.iloc[i]["ema_8"] * 1.002
                for i in [-3, -2, -1]
            )
            bouncing = close > ema8 and last["close"] > last["open"]
            if touched_ema8 and bouncing:
                stop   = ema8 - atr * 0.5
                target = close + atr * 3.0
                risk   = close - stop
                if risk > 0:
                    return Signal(
                        direction="BUY", confidence=0.62, strategy=self.name,
                        reasoning=(f"Bullish EMA stack (8>21>55). Price pulled back to EMA8 "
                                   f"({ema8:.2f}) and bounced. ADX={adx:.1f}. RSI={rsi:.0f}."),
                        symbol=symbol, timeframe=timeframe,
                        suggested_entry=close, suggested_stop=round(stop, 2), suggested_target=round(target, 2),
                    )

        if bear_stack and rsi > 35:
            touched_ema8 = any(
                df.iloc[i]["high"] >= df.iloc[i]["ema_8"] * 0.998
                for i in [-3, -2, -1]
            )
            rejecting = close < ema8 and last["close"] < last["open"]
            if touched_ema8 and rejecting:
                stop   = ema8 + atr * 0.5
                target = close - atr * 3.0
                risk   = stop - close
                if risk > 0:
                    return Signal(
                        direction="SELL", confidence=0.62, strategy=self.name,
                        reasoning=(f"Bearish EMA stack (8<21<55). Price rallied to EMA8 "
                                   f"({ema8:.2f}) and rejected. ADX={adx:.1f}. RSI={rsi:.0f}."),
                        symbol=symbol, timeframe=timeframe,
                        suggested_entry=close, suggested_stop=round(stop, 2), suggested_target=round(target, 2),
                    )

        stack_desc = ("bullish" if bull_stack else "bearish" if bear_stack else "mixed")
        return self._neutral(symbol, timeframe,
            f"EMA stack {stack_desc}. No fresh cross or clean pullback. ADX={adx:.1f}, RSI={rsi:.0f}.")
