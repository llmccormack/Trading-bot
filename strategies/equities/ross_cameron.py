"""
Ross Cameron (Warrior Trading) strategy — momentum gap-and-go + bull flag.
See rules/ross_cameron.md for the full philosophy.
"""
import pandas as pd
from strategies.base import BaseStrategy, Signal
from utils.indicators import add_atr, add_vwap, add_volume_metrics


class RossCameronStrategy(BaseStrategy):
    name = "ross_cameron"
    market = "equities"

    def __init__(
        self,
        min_gap_pct: float = 0.10,      # 10% gap up minimum
        min_rel_volume: float = 5.0,     # 5x relative volume
        max_float_m: float = 20.0,       # max 20M float (millions)
        min_rr: float = 2.0,
    ):
        self.min_gap_pct = min_gap_pct
        self.min_rel_volume = min_rel_volume
        self.max_float_m = max_float_m
        self.min_rr = min_rr

    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
        prev_close: float | None = None,
        float_shares: float | None = None,  # float in millions
        has_catalyst: bool = True,
    ) -> Signal:
        if not self._needs_bars(df, 5):
            return self._neutral(symbol, timeframe, "Not enough bars")

        df = df.copy()
        df = add_atr(df)
        df = add_vwap(df)
        df = add_volume_metrics(df)

        last = df.iloc[-1]
        close = last["close"]
        open_price = last["open"]
        atr = last["atr"]
        vwap = last.get("vwap")
        vol_ratio = last.get("vol_ratio", 1.0)

        if pd.isna(atr) or atr == 0:
            return self._neutral(symbol, timeframe, "ATR unavailable")

        # Filter checks
        if not has_catalyst:
            return self._neutral(symbol, timeframe, "No catalyst — Ross Cameron requires news/catalyst")

        if float_shares is not None and float_shares > self.max_float_m:
            return self._neutral(
                symbol, timeframe,
                f"Float {float_shares:.1f}M too large (max {self.max_float_m}M)"
            )

        if vol_ratio < self.min_rel_volume:
            return self._neutral(
                symbol, timeframe,
                f"Relative volume {vol_ratio:.1f}x below minimum {self.min_rel_volume}x"
            )

        # Gap check
        if prev_close is not None:
            gap_pct = (open_price - prev_close) / prev_close
            if gap_pct < self.min_gap_pct:
                return self._neutral(
                    symbol, timeframe,
                    f"Gap {gap_pct:.1%} below minimum {self.min_gap_pct:.0%}"
                )
        else:
            gap_pct = None

        # ---- GAP-AND-GO setup ----
        # Price holding above VWAP and above open — gap holding
        gap_and_go = (
            not pd.isna(vwap)
            and close > vwap
            and close > open_price
            and vol_ratio >= self.min_rel_volume
        )

        if gap_and_go:
            # Stop: below the opening candle low
            stop = df.iloc[0]["low"] - atr * 0.1  # below first candle of the day
            risk = close - stop
            if risk <= 0:
                return self._neutral(symbol, timeframe, "Invalid risk")
            target = close + risk * self.min_rr

            conf = 0.65
            if vol_ratio > self.min_rel_volume * 1.5:
                conf += 0.10
            if float_shares is not None and float_shares < 5.0:
                conf += 0.10  # ultra-low float = more explosive

            return Signal(
                direction="BUY",
                confidence=round(min(conf, 0.90), 2),
                strategy=self.name,
                reasoning=(
                    f"Gap-and-Go setup. "
                    f"{f'Gap: {gap_pct:.1%} from prior close. ' if gap_pct else ''}"
                    f"Relative volume: {vol_ratio:.1f}x. "
                    f"Price above VWAP ({vwap:.2f}) and open — holding the gap. "
                    f"{f'Float: {float_shares:.1f}M. ' if float_shares else ''}"
                    f"Catalyst present. R:R {self.min_rr}:1."
                ),
                symbol=symbol,
                timeframe=timeframe,
                suggested_entry=close,
                suggested_stop=round(stop, 2),
                suggested_target=round(target, 2),
            )

        # ---- BULL FLAG setup ----
        # Detect: price made a strong initial move, now consolidating (declining volume)
        if len(df) >= 10:
            first_move_high = df["high"].iloc[:5].max()
            flag_lows = df["low"].tail(5)
            flag_is_tight = flag_lows.max() - flag_lows.min() < atr * 1.5
            vol_declining = df["volume"].tail(5).is_monotonic_decreasing

            bull_flag_break = (
                close > first_move_high
                and flag_is_tight
                and not pd.isna(vwap)
                and close > vwap
            )

            if bull_flag_break:
                stop = flag_lows.min() - atr * 0.1
                risk = close - stop
                if risk > 0:
                    target = close + (first_move_high - df["low"].iloc[0]) * 1.0  # measured move
                    rr = (target - close) / risk

                    conf = 0.60
                    if vol_declining:
                        conf += 0.10
                    if vol_ratio > 2.0:
                        conf += 0.05

                    return Signal(
                        direction="BUY",
                        confidence=round(conf, 2),
                        strategy=self.name,
                        reasoning=(
                            f"Bull flag breakout. Initial spike to {first_move_high:.2f}, "
                            f"tight flag consolidation {'with declining volume ' if vol_declining else ''}"
                            f"breaking out. Measured move target: {target:.2f}. R:R {rr:.1f}:1."
                        ),
                        symbol=symbol,
                        timeframe=timeframe,
                        suggested_entry=close,
                        suggested_stop=round(stop, 2),
                        suggested_target=round(target, 2),
                    )

        return self._neutral(symbol, timeframe, "No gap-and-go or bull flag setup detected")
