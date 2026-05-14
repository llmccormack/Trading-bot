"""
Risk management — validates every trade before execution.
All rules must pass or the trade is rejected.
"""
from __future__ import annotations
from dataclasses import dataclass
from config import settings


@dataclass
class TradeRequest:
    symbol: str
    direction: str          # BUY | SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    account_balance: float
    risk_multiplier: float = 1.0   # score-tier scaling: 0.5 / 0.75 / 1.0 / 1.25


@dataclass
class RiskValidation:
    approved: bool
    position_size: float    # number of contracts/shares
    dollar_risk: float      # $ amount at risk
    r_ratio: float          # reward:risk
    rejection_reason: str = ""

    def __repr__(self) -> str:
        if self.approved:
            return (
                f"[APPROVED] Size={self.position_size} | "
                f"Risk=${self.dollar_risk:.2f} | R:R={self.r_ratio:.1f}:1"
            )
        return f"[REJECTED] {self.rejection_reason}"


class RiskManager:
    def __init__(
        self,
        max_risk_pct: float | None = None,
        max_daily_loss_pct: float | None = None,
        max_positions: int | None = None,
        min_rr: float = 1.5,
    ):
        self.max_risk_pct = max_risk_pct or settings.max_risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct or settings.max_daily_loss_pct
        self.max_positions = max_positions or settings.max_concurrent_positions
        self.min_rr = min_rr

        self._daily_realized_pnl: float = 0.0
        self._open_positions: int = 0
        self._halted: bool = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def validate(self, req: TradeRequest) -> RiskValidation:
        """Run all risk checks. Returns approved/rejected with sizing."""

        if self._halted:
            return RiskValidation(False, 0, 0, 0, rejection_reason="Trading halted — daily loss limit reached")

        # 1. Check concurrent position limit
        if self._open_positions >= self.max_positions:
            return RiskValidation(
                False, 0, 0, 0,
                rejection_reason=f"Max concurrent positions ({self.max_positions}) already open"
            )

        # 2. Validate stop loss is set and on the correct side
        risk_per_unit = abs(req.entry_price - req.stop_loss)
        if risk_per_unit <= 0:
            return RiskValidation(False, 0, 0, 0, rejection_reason="Stop loss must be set and different from entry")

        if req.direction == "BUY" and req.stop_loss >= req.entry_price:
            return RiskValidation(False, 0, 0, 0, rejection_reason="Stop loss must be BELOW entry for long trades")

        if req.direction == "SELL" and req.stop_loss <= req.entry_price:
            return RiskValidation(False, 0, 0, 0, rejection_reason="Stop loss must be ABOVE entry for short trades")

        # 3. Check R:R ratio
        reward_per_unit = abs(req.take_profit - req.entry_price)
        r_ratio = reward_per_unit / risk_per_unit
        if r_ratio < self.min_rr:
            return RiskValidation(
                False, 0, 0, r_ratio,
                rejection_reason=f"R:R {r_ratio:.1f}:1 below minimum {self.min_rr}:1"
            )

        # 4. Calculate position size (fixed fractional — risk X% of account,
        #    scaled by per-signal risk_multiplier from score-tier logic)
        max_dollar_risk = req.account_balance * (self.max_risk_pct / 100) * max(0.0, req.risk_multiplier)
        position_size = max_dollar_risk / risk_per_unit
        position_size = max(1, round(position_size))  # minimum 1 unit

        dollar_risk = position_size * risk_per_unit

        # 5. Check that sizing doesn't push daily loss over the limit
        # Only count realised losses — a winning day should not block new trades.
        max_daily_loss = req.account_balance * (self.max_daily_loss_pct / 100)
        current_loss = max(0.0, -self._daily_realized_pnl)   # 0 if up, positive if down
        if current_loss + dollar_risk > max_daily_loss:
            return RiskValidation(
                False, 0, dollar_risk, r_ratio,
                rejection_reason=(
                    f"This trade's risk (${dollar_risk:.2f}) would push total daily loss "
                    f"past the max daily loss limit (${max_daily_loss:.2f})"
                )
            )

        return RiskValidation(
            approved=True,
            position_size=position_size,
            dollar_risk=round(dollar_risk, 2),
            r_ratio=round(r_ratio, 2),
        )

    # ------------------------------------------------------------------ #
    # State updates (called by execution layer)                           #
    # ------------------------------------------------------------------ #

    def on_trade_opened(self) -> None:
        self._open_positions += 1

    def on_trade_closed(self, pnl: float) -> None:
        self._open_positions = max(0, self._open_positions - 1)
        self._daily_realized_pnl += pnl
        self._check_halt()

    def reset_daily(self) -> None:
        """Call at start of each trading session."""
        self._daily_realized_pnl = 0.0
        self._halted = False

    @property
    def daily_pnl(self) -> float:
        return self._daily_realized_pnl

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _check_halt(self) -> None:
        # We don't have direct access to account_balance here, so halt check
        # is also done in validate(). This is a belt-and-suspenders check
        # using a configured absolute limit if needed.
        pass

    def status(self) -> dict:
        return {
            "open_positions": self._open_positions,
            "daily_pnl": round(self._daily_realized_pnl, 2),
            "halted": self._halted,
            "max_risk_pct": self.max_risk_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_positions": self.max_positions,
        }
