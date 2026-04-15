"""Base strategy interface and Signal model."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import pandas as pd


Direction = Literal["BUY", "SELL", "NEUTRAL"]


@dataclass
class Signal:
    direction: Direction
    confidence: float          # 0.0 – 1.0
    strategy: str              # e.g. "pb_blake", "trend_following"
    reasoning: str             # human-readable explanation
    symbol: str = ""
    timeframe: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Optional trade parameters (can be None if strategy doesn't specify)
    suggested_entry: float | None = None
    suggested_stop: float | None = None
    suggested_target: float | None = None

    @property
    def r_ratio(self) -> float | None:
        """Risk:reward ratio if entry/stop/target are provided."""
        if self.suggested_entry and self.suggested_stop and self.suggested_target:
            risk = abs(self.suggested_entry - self.suggested_stop)
            reward = abs(self.suggested_target - self.suggested_entry)
            return round(reward / risk, 2) if risk > 0 else None
        return None

    def __repr__(self) -> str:
        rr = f" | R:R {self.r_ratio}" if self.r_ratio else ""
        return (
            f"[{self.strategy.upper()}] {self.direction} {self.symbol} "
            f"conf={self.confidence:.0%}{rr} | {self.reasoning}"
        )


class BaseStrategy:
    """
    All trading strategies inherit from this.
    Override `generate_signal` with your logic.
    """
    name: str = "base"
    market: Literal["futures", "equities", "all"] = "all"

    def generate_signal(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> Signal:
        raise NotImplementedError

    def _neutral(self, symbol: str = "", timeframe: str = "", reason: str = "No setup") -> Signal:
        return Signal(
            direction="NEUTRAL",
            confidence=0.0,
            strategy=self.name,
            reasoning=reason,
            symbol=symbol,
            timeframe=timeframe,
        )

    def _needs_bars(self, df: pd.DataFrame, minimum: int) -> bool:
        return len(df) >= minimum
