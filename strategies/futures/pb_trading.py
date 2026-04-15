"""
PB Trading — pullback to EMA zone in confirmed trend.
Strict: ADX>25, price in EMA8/21 zone, bounce candle, VWAP side, RSI mid-range.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_ema, add_atr, add_adx, add_vwap, add_rsi, get_key_levels


class PBTradingStrategy(BaseStrategy):
    name = "pb_trading"
    market = "futures"

    def __init__(self, adx_min=25.0, min_rr=2.0):
        self.adx_min = adx_min
        self.min_rr  = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 60):
            return self._neutral(symbol, timeframe, "Need 60+ bars")
        df = df.copy()
        df = add_ema(df, [8, 21, 55]); df = add_atr(df); df = add_adx(df)
        df = add_vwap(df); df = add_rsi(df)

        last  = df.iloc[-1]; prev = df.iloc[-2]
        close = last["close"]; atr = last["atr"]; adx = last.get("adx", 0)
        rsi   = last.get("rsi", 50); ema8 = last.get("ema_8")
        ema21 = last.get("ema_21"); ema55 = last.get("ema_55"); vwap = last.get("vwap")

        if any(pd.isna(v) for v in [atr, adx, ema8, ema21, ema55]) or atr == 0:
            return self._neutral(symbol, timeframe, "Indicators not ready")
        if adx < self.adx_min:
            return self._neutral(symbol, timeframe, f"ADX={adx:.1f} too weak (need {self.adx_min}+)")

        uptrend   = ema8 > ema21 > ema55
        downtrend = ema8 < ema21 < ema55
        if not uptrend and not downtrend:
            return self._neutral(symbol, timeframe, "EMAs not aligned")

        zone_top = max(ema8, ema21) + atr * 0.4
        zone_bot = min(ema8, ema21) - atr * 0.4
        in_zone  = zone_bot <= close <= zone_top
        if not in_zone:
            return self._neutral(symbol, timeframe, f"Price not in EMA pullback zone ({zone_bot:.1f}–{zone_top:.1f})")

        bullish_bounce = last["close"] > last["open"] and last["close"] > prev["close"]
        bearish_bounce = last["close"] < last["open"] and last["close"] < prev["close"]
        above_vwap = not pd.isna(vwap) and close > vwap
        below_vwap = not pd.isna(vwap) and close < vwap
        levels = get_key_levels(df)

        if uptrend and bullish_bounce and above_vwap and 35 < rsi < 65:
            stop = zone_bot - atr * 0.25; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            above  = [r for r in levels["resistance"] if r > close]
            target = min(above) if above else close + risk * self.min_rr
            rr     = (target - close) / risk
            if rr < self.min_rr: target = close + risk * self.min_rr; rr = self.min_rr
            conf = round(0.62 + min(0.18, (adx - self.adx_min) / 60), 2)
            return Signal(direction="BUY", confidence=conf, strategy=self.name,
                reasoning=f"Pullback to EMA zone in uptrend. ADX={adx:.1f}, RSI={rsi:.0f}, above VWAP. R:R {rr:.1f}:1.",
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if downtrend and bearish_bounce and below_vwap and 35 < rsi < 65:
            stop = zone_top + atr * 0.25; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            below  = [s for s in levels["support"] if s < close]
            target = max(below) if below else close - risk * self.min_rr
            rr     = (close - target) / risk
            if rr < self.min_rr: target = close - risk * self.min_rr; rr = self.min_rr
            conf = round(0.62 + min(0.18, (adx - self.adx_min) / 60), 2)
            return Signal(direction="SELL", confidence=conf, strategy=self.name,
                reasoning=f"Bounce to EMA zone in downtrend. ADX={adx:.1f}, RSI={rsi:.0f}, below VWAP. R:R {rr:.1f}:1.",
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        return self._neutral(symbol, timeframe,
            f"In zone but missing: {'bounce ' if not (bullish_bounce or bearish_bounce) else ''}"
            f"{'VWAP ' if not (above_vwap or below_vwap) else ''}{'RSI={:.0f} extended'.format(rsi) if not (35<rsi<65) else ''}")
