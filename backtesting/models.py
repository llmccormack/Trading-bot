"""
Shared dataclasses used by both BacktestEngine (1h) and BacktestEngine5m (5m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestTrade:
    strategy:        str
    direction:       str
    entry_bar:       int
    exit_bar:        int
    entry_price:     float
    exit_price:      float
    stop_loss:       float
    take_profit:     float
    exit_reason:     str
    pnl_pts:         float
    r_multiple:      float
    bars_held:       int
    composite_score: float = 0.0
    regime:          str   = ""
    be_moved:        bool  = False


@dataclass
class BacktestResult:
    symbol:       str
    timeframe:    str
    market:       str
    total_bars:   int
    trades:       List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float]         = field(default_factory=list)

    @property
    def total_trades(self) -> int:  return len(self.trades)

    @property
    def wins(self):  return [t for t in self.trades if t.pnl_pts > 0]

    @property
    def losses(self): return [t for t in self.trades if t.pnl_pts <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / len(self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        gw = sum(t.pnl_pts for t in self.wins)
        gl = abs(sum(t.pnl_pts for t in self.losses))
        return (gw / gl) if gl > 0 else (math.inf if gw > 0 else 0.0)

    @property
    def expectancy(self) -> float:
        return sum(t.r_multiple for t in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        base = peak = 100_000.0
        equity = base
        max_d = 0.0
        for t in self.trades:
            equity += t.pnl_pts
            peak    = max(peak, equity)
            max_d   = max(max_d, (peak - equity) / peak)
        return max_d

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve: return 0.0
        peak = max_dd = 0.0
        for v in self.equity_curve:
            peak   = max(peak, v)
            max_dd = max(max_dd, peak - v)
        return max_dd

    @property
    def avg_winner(self) -> float:
        return sum(t.pnl_pts for t in self.wins) / len(self.wins) if self.wins else 0.0

    @property
    def avg_loser(self) -> float:
        return sum(t.pnl_pts for t in self.losses) / len(self.losses) if self.losses else 0.0

    @property
    def avg_bars_held(self) -> float:
        return sum(t.bars_held for t in self.trades) / len(self.trades) if self.trades else 0.0

    def summary(self) -> str:
        if not self.trades:
            return f"{self.symbol} {self.timeframe}: 0 trades"
        return (
            f"{self.symbol} {self.timeframe} | "
            f"Trades={self.total_trades} WR={self.win_rate:.1%} "
            f"PF={self.profit_factor:.2f} E={self.expectancy:+.2f}R"
        )
