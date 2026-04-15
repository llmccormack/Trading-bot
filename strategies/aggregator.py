"""
Signal aggregator — runs all strategies and combines into a composite score.
Includes 11 total strategy modules across multiple philosophies.
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from strategies.base import Signal, Direction

from strategies.futures.pb_blake      import PBBlakeStrategy
from strategies.futures.pb_trading    import PBTradingStrategy
from strategies.futures.pj_trades     import PJTradesStrategy
from strategies.futures.tjr           import TJRStrategy
from strategies.futures.trend_following import TrendFollowingStrategy
from strategies.futures.breakout      import BreakoutStrategy
from strategies.futures.ict           import ICTStrategy
from strategies.futures.wyckoff       import WyckoffStrategy
from strategies.futures.orb           import ORBStrategy
from strategies.futures.vsa           import VSAStrategy
from strategies.equities.ross_cameron import RossCameronStrategy

# Strategy weights — based on reliability and selectivity
# Higher weight = more influence on the final signal
FUTURES_WEIGHTS = {
    "tjr":             2.0,   # highest bar (3:1 R:R, MTF) — very reliable when it fires
    "ict":             1.8,   # institutional zones — high conviction
    "wyckoff":         1.7,   # climactic events are high conviction
    "pb_blake":        1.5,   # price action at structure
    "pb_trading":      1.4,   # pullback in trend
    "orb":             1.3,   # opening range breakout — time-tested
    "pj_trades":       1.2,   # VWAP continuation
    "vsa":             1.2,   # volume tells the story
    "breakout":        1.0,   # prior day level break
    "trend_following": 0.8,   # confirmation only — fires on crossovers
}
EQUITIES_WEIGHTS = {
    "ross_cameron": 2.0,
}


@dataclass
class AggregatedSignal:
    symbol:               str
    timeframe:            str
    direction:            Direction
    composite_score:      float     # -1.0 (strong sell) to +1.0 (strong buy)
    confidence:           float     # 0.0 to 1.0
    signal_count:         int       # how many strategies agree
    total_strategies:     int       # total strategies run
    agreeing_strategies:  list[str] = field(default_factory=list)
    disagreeing_strategies: list[str] = field(default_factory=list)
    neutral_strategies:   list[str] = field(default_factory=list)
    individual_signals:   list[Signal] = field(default_factory=list)
    suggested_entry:      float | None = None
    suggested_stop:       float | None = None
    suggested_target:     float | None = None

    @property
    def recommendation(self) -> str:
        """Plain-English recommendation label."""
        score = abs(self.composite_score)
        n     = self.signal_count
        if self.direction == "NEUTRAL" or score < 0.15:
            return "WAIT"
        if score >= 0.55 and n >= 3:
            return f"STRONG {'BUY' if self.composite_score > 0 else 'SELL'}"
        if score >= 0.30 and n >= 2:
            return f"{'BUY' if self.composite_score > 0 else 'SELL'}"
        return "WEAK SIGNAL — WAIT"

    @property
    def recommendation_color(self) -> str:
        r = self.recommendation
        if "STRONG BUY"  in r: return "green"
        if "STRONG SELL" in r: return "red"
        if r == "BUY":         return "lightgreen"
        if r == "SELL":        return "salmon"
        return "gray"

    def summary(self) -> str:
        lines = [
            f"[{self.recommendation}] {self.symbol} | score={self.composite_score:+.2f} "
            f"conf={self.confidence:.0%} ({self.signal_count}/{self.total_strategies} agree)",
            f"  Agreeing:    {', '.join(self.agreeing_strategies) or 'none'}",
            f"  Disagreeing: {', '.join(self.disagreeing_strategies) or 'none'}",
        ]
        if self.suggested_entry:
            lines.append(f"  Entry: {self.suggested_entry} | Stop: {self.suggested_stop} | Target: {self.suggested_target}")
        return "\n".join(lines)


class SignalAggregator:
    def __init__(self, market="futures", futures_weights=None, equities_weights=None):
        self.market           = market
        self.futures_weights  = futures_weights  or FUTURES_WEIGHTS
        self.equities_weights = equities_weights or EQUITIES_WEIGHTS

        self.futures_strategies = [
            PBBlakeStrategy(), PBTradingStrategy(), PJTradesStrategy(),
            TJRStrategy(), TrendFollowingStrategy(), BreakoutStrategy(),
            ICTStrategy(), WyckoffStrategy(), ORBStrategy(), VSAStrategy(),
        ]
        self.equities_strategies = [RossCameronStrategy()]

    def run(self, df: pd.DataFrame, symbol="", timeframe="", **equity_kwargs) -> AggregatedSignal:
        strategies = self.futures_strategies if self.market == "futures" else self.equities_strategies
        weights    = self.futures_weights    if self.market == "futures" else self.equities_weights

        signals = []
        for strat in strategies:
            try:
                if self.market == "equities" and strat.name == "ross_cameron":
                    sig = strat.generate_signal(df, symbol, timeframe, **equity_kwargs)
                else:
                    sig = strat.generate_signal(df, symbol, timeframe)
                signals.append(sig)
            except Exception as e:
                from strategies.base import Signal
                signals.append(Signal(direction="NEUTRAL", confidence=0, strategy=strat.name,
                                      reasoning=f"Error: {str(e)[:80]}", symbol=symbol, timeframe=timeframe))

        return self._aggregate(signals, weights, symbol, timeframe)

    def _aggregate(self, signals, weights, symbol, timeframe):
        buy_signals  = [s for s in signals if s.direction == "BUY"]
        sell_signals = [s for s in signals if s.direction == "SELL"]

        buy_score  = sum(weights.get(s.strategy, 1.0) * s.confidence for s in buy_signals)
        sell_score = sum(weights.get(s.strategy, 1.0) * s.confidence for s in sell_signals)

        if buy_score == 0 and sell_score == 0:
            return AggregatedSignal(
                symbol=symbol, timeframe=timeframe, direction="NEUTRAL",
                composite_score=0.0, confidence=0.0, signal_count=0,
                total_strategies=len(signals),
                neutral_strategies=[s.strategy for s in signals],
                individual_signals=signals,
            )

        dominant     = "BUY" if buy_score >= sell_score else "SELL"
        dom_score    = max(buy_score, sell_score)
        opp_score    = min(buy_score, sell_score)
        total        = dom_score + opp_score
        composite    = (dom_score - opp_score) / total if total > 0 else 0
        composite    = composite if dominant == "BUY" else -composite

        agreeing     = [s.strategy for s in signals if s.direction == dominant]
        disagreeing  = [s.strategy for s in signals if s.direction not in (dominant, "NEUTRAL")]
        neutral      = [s.strategy for s in signals if s.direction == "NEUTRAL"]

        total_w  = sum(weights.get(s.strategy, 1.0) for s in signals if s.direction == dominant)
        w_conf   = sum(weights.get(s.strategy, 1.0) * s.confidence for s in signals if s.direction == dominant)
        confidence = w_conf / total_w if total_w > 0 else 0

        # Average trade parameters from agreeing signals
        with_params = [s for s in signals if s.direction == dominant and s.suggested_entry]
        avg_entry  = sum(s.suggested_entry for s in with_params) / len(with_params) if with_params else None
        avg_stop   = sum(s.suggested_stop  for s in with_params if s.suggested_stop) / max(1, len([s for s in with_params if s.suggested_stop])) if with_params else None
        avg_target = sum(s.suggested_target for s in with_params if s.suggested_target) / max(1, len([s for s in with_params if s.suggested_target])) if with_params else None

        return AggregatedSignal(
            symbol=symbol, timeframe=timeframe,
            direction=dominant,
            composite_score=round(composite, 3),
            confidence=round(confidence, 3),
            signal_count=len(agreeing),
            total_strategies=len(signals),
            agreeing_strategies=agreeing,
            disagreeing_strategies=disagreeing,
            neutral_strategies=neutral,
            individual_signals=signals,
            suggested_entry=round(avg_entry, 4) if avg_entry else None,
            suggested_stop=round(avg_stop, 4)   if avg_stop  else None,
            suggested_target=round(avg_target, 4) if avg_target else None,
        )
