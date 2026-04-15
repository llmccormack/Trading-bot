"""
TJR — multi-timeframe structural entries. HTF bias + LTF Break of Structure.
Minimum 3:1 R:R. Quality over quantity — most runs return NEUTRAL.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_ema, add_atr, add_rsi, get_key_levels


class TJRStrategy(BaseStrategy):
    name = "tjr"
    market = "futures"

    def __init__(self, min_rr=3.0):
        self.min_rr = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 100):
            return self._neutral(symbol, timeframe, "Need 100+ bars for MTF analysis")

        df = df.copy()
        df = add_ema(df, [21, 55]); df = add_atr(df); df = add_rsi(df)

        last  = df.iloc[-1]; close = last["close"]
        atr   = last["atr"]; rsi = last.get("rsi", 50)
        ema21 = last.get("ema_21"); ema55 = last.get("ema_55")

        if pd.isna(atr) or atr == 0: return self._neutral(symbol, timeframe, "ATR unavailable")

        # HTF: resample to 1h
        if "timestamp" not in df.columns:
            return self._neutral(symbol, timeframe, "No timestamp column for HTF resample")
        htf = df.set_index("timestamp")[["open","high","low","close","volume"]]\
                .resample("1h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        if len(htf) < 8:
            return self._neutral(symbol, timeframe, f"Only {len(htf)} HTF bars — need 8+")

        hh = htf["high"].values; hl = htf["low"].values
        # HTF must show at least 2 consecutive HH or LL
        htf_bull = all(hh[-i] > hh[-i-1] for i in range(1, 3)) and all(hl[-i] > hl[-i-1] for i in range(1, 3))
        htf_bear = all(hh[-i] < hh[-i-1] for i in range(1, 3)) and all(hl[-i] < hl[-i-1] for i in range(1, 3))

        if not htf_bull and not htf_bear:
            return self._neutral(symbol, timeframe, "No clear HTF structural bias (need 2+ consecutive HH or LL on 1h)")

        # LTF Break of Structure: last 15 bars
        recent = df.tail(15)
        prior_high = recent["high"].iloc[:-3].max()
        prior_low  = recent["low"].iloc[:-3].min()

        bos_bull = close > prior_high and htf_bull and rsi < 70
        bos_bear = close < prior_low  and htf_bear and rsi > 30

        levels = get_key_levels(df)

        if bos_bull:
            stop = prior_low - atr * 0.2; risk = close - stop
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            above  = [r for r in levels["resistance"] if r > close]
            target = min(above) if above else close + risk * self.min_rr
            rr     = (target - close) / risk
            if rr < self.min_rr:
                return self._neutral(symbol, timeframe, f"R:R {rr:.1f} < TJR minimum {self.min_rr}:1")
            return Signal(direction="BUY", confidence=0.72, strategy=self.name,
                reasoning=(f"HTF 1h bullish structure (consecutive HH/HL). "
                           f"LTF BOS above {prior_high:.2f}. RSI={rsi:.0f}. R:R {rr:.1f}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        if bos_bear:
            stop = prior_high + atr * 0.2; risk = stop - close
            if risk <= 0: return self._neutral(symbol, timeframe, "Invalid risk")
            below  = [s for s in levels["support"] if s < close]
            target = max(below) if below else close - risk * self.min_rr
            rr     = (close - target) / risk
            if rr < self.min_rr:
                return self._neutral(symbol, timeframe, f"R:R {rr:.1f} < TJR minimum {self.min_rr}:1")
            return Signal(direction="SELL", confidence=0.72, strategy=self.name,
                reasoning=(f"HTF 1h bearish structure (consecutive LL/LH). "
                           f"LTF BOS below {prior_low:.2f}. RSI={rsi:.0f}. R:R {rr:.1f}:1."),
                symbol=symbol, timeframe=timeframe, suggested_entry=close,
                suggested_stop=round(stop,2), suggested_target=round(target,2))

        return self._neutral(symbol, timeframe,
            f"HTF bias {'bullish' if htf_bull else 'bearish'} but no LTF BOS yet — waiting for confirmation")
