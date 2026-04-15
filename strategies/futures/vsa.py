"""
Volume Spread Analysis (VSA) — Tom Williams / Gavin Holmes methodology.
Reads the story behind price and volume: no demand, no supply, stopping vol, climax.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_atr, add_volume_metrics, add_rsi


class VSAStrategy(BaseStrategy):
    name = "vsa"
    market = "futures"

    def __init__(self, min_rr=2.0):
        self.min_rr = min_rr

    def generate_signal(self, df, symbol="", timeframe=""):
        if not self._needs_bars(df, 40):
            return self._neutral(symbol, timeframe, "Need 40+ bars")
        df = df.copy()
        df = add_atr(df); df = add_volume_metrics(df); df = add_rsi(df)

        last  = df.iloc[-1]; prev = df.iloc[-2]; prev2 = df.iloc[-3]
        close = last["close"]; atr = last["atr"]; rsi = last.get("rsi", 50)
        spread = last["high"] - last["low"]
        vol    = last.get("vol_ratio", 1.0)
        avg_spread = df["high"].tail(20).values - df["low"].tail(20).values
        avg_spread = avg_spread.mean() if len(avg_spread) > 0 else spread

        if pd.isna(atr) or atr == 0: return self._neutral(symbol, timeframe, "ATR unavailable")

        # ---- NO SUPPLY (bullish) ----
        # Down bar with narrow spread + below-average volume = smart money not selling
        no_supply = (
            last["close"] < last["open"] and          # down bar
            spread < avg_spread * 0.75 and            # narrow spread
            vol < 0.75 and                            # low volume
            rsi > 40                                  # not already oversold
        )

        # ---- NO DEMAND (bearish) ----
        # Up bar with narrow spread + below-average volume = smart money not buying
        no_demand = (
            last["close"] > last["open"] and          # up bar
            spread < avg_spread * 0.75 and            # narrow spread
            vol < 0.75 and                            # low volume
            rsi < 60                                  # not already overbought
        )

        # ---- STOPPING VOLUME (bullish) ----
        # High volume on a down bar that closes in the upper half = absorption of selling
        stopping_vol = (
            prev["close"] < prev["open"] and
            vol > 1.8 and
            prev["close"] > (prev["high"] + prev["low"]) / 2 and  # closes upper half
            last["close"] > prev["close"]  # next bar confirms up
        )

        # ---- PROFESSIONAL SELLING (bearish) ----
        # High volume on an up bar that closes in the lower half = distribution
        professional_selling = (
            prev["close"] > prev["open"] and
            prev.get("vol_ratio", 1.0) > 1.8 and
            prev["close"] < (prev["high"] + prev["low"]) / 2 and  # closes lower half
            last["close"] < prev["close"]  # next bar confirms down
        )

        # ---- EFFORT vs RESULT (trend) ----
        # Wide spread up bar + high volume + strong close = genuine buying
        strong_buying = (
            last["close"] > last["open"] and
            spread > avg_spread * 1.3 and
            vol > 1.5 and
            last["close"] > (last["high"] + last["low"]) * 0.6 and
            rsi < 68
        )
        strong_selling = (
            last["close"] < last["open"] and
            spread > avg_spread * 1.3 and
            vol > 1.5 and
            last["close"] < (last["high"] + last["low"]) * 0.4 and
            rsi > 32
        )

        if stopping_vol and rsi < 50:
            stop = df.tail(5)["low"].min() - atr * 0.3; risk = close - stop
            if risk > 0:
                target = close + risk * self.min_rr
                return Signal(direction="BUY", confidence=0.68, strategy=self.name,
                    reasoning=(f"VSA Stopping Volume: high-vol down bar ({prev.get('vol_ratio',1):.1f}x) "
                               f"closing in upper half — smart money absorbing selling. RSI={rsi:.0f}. R:R {self.min_rr}:1."),
                    symbol=symbol, timeframe=timeframe, suggested_entry=close,
                    suggested_stop=round(stop,2), suggested_target=round(target,2))

        if no_supply and rsi < 50:
            stop = last["low"] - atr * 0.4; risk = close - stop
            if risk > 0:
                target = close + risk * self.min_rr
                return Signal(direction="BUY", confidence=0.62, strategy=self.name,
                    reasoning=(f"VSA No Supply: narrow down bar ({spread/avg_spread:.2f}x avg spread) "
                               f"on low volume ({vol:.2f}x) — sellers absent. RSI={rsi:.0f}. R:R {self.min_rr}:1."),
                    symbol=symbol, timeframe=timeframe, suggested_entry=close,
                    suggested_stop=round(stop,2), suggested_target=round(target,2))

        if strong_buying:
            stop = close - atr * 1.5; risk = close - stop
            if risk > 0:
                target = close + risk * self.min_rr
                return Signal(direction="BUY", confidence=0.65, strategy=self.name,
                    reasoning=(f"VSA Strong Buying: wide bar ({spread/avg_spread:.1f}x avg) + {vol:.1f}x vol + strong close. "
                               f"Effort matches result — genuine demand. RSI={rsi:.0f}."),
                    symbol=symbol, timeframe=timeframe, suggested_entry=close,
                    suggested_stop=round(stop,2), suggested_target=round(target,2))

        if professional_selling and rsi > 50:
            stop = df.tail(5)["high"].max() + atr * 0.3; risk = stop - close
            if risk > 0:
                target = close - risk * self.min_rr
                return Signal(direction="SELL", confidence=0.68, strategy=self.name,
                    reasoning=(f"VSA Professional Selling: high-vol up bar ({prev.get('vol_ratio',1):.1f}x) "
                               f"closing in lower half — smart money distributing. RSI={rsi:.0f}. R:R {self.min_rr}:1."),
                    symbol=symbol, timeframe=timeframe, suggested_entry=close,
                    suggested_stop=round(stop,2), suggested_target=round(target,2))

        if no_demand and rsi > 50:
            stop = last["high"] + atr * 0.4; risk = stop - close
            if risk > 0:
                target = close - risk * self.min_rr
                return Signal(direction="SELL", confidence=0.62, strategy=self.name,
                    reasoning=(f"VSA No Demand: narrow up bar ({spread/avg_spread:.2f}x avg spread) "
                               f"on low volume ({vol:.2f}x) — buyers absent. RSI={rsi:.0f}. R:R {self.min_rr}:1."),
                    symbol=symbol, timeframe=timeframe, suggested_entry=close,
                    suggested_stop=round(stop,2), suggested_target=round(target,2))

        if strong_selling:
            stop = close + atr * 1.5; risk = stop - close
            if risk > 0:
                target = close - risk * self.min_rr
                return Signal(direction="SELL", confidence=0.65, strategy=self.name,
                    reasoning=(f"VSA Strong Selling: wide bar ({spread/avg_spread:.1f}x avg) + {vol:.1f}x vol + weak close. "
                               f"Effort matches result — genuine supply. RSI={rsi:.0f}."),
                    symbol=symbol, timeframe=timeframe, suggested_entry=close,
                    suggested_stop=round(stop,2), suggested_target=round(target,2))

        return self._neutral(symbol, timeframe,
            f"VSA: no significant vol/spread pattern. Spread={spread/avg_spread:.2f}x avg, Vol={vol:.2f}x, RSI={rsi:.0f}")
