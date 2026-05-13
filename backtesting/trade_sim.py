"""
TradeSimulator — shared bar-by-bar trade management logic.
──────────────────────────────────────────────────────────
Single source of truth for:
  • T1 partial exit (half-position)
  • 1R ratchet trailing stop (post-T1), using close-price to match paper broker
  • T2 full exit / "clear-break" switch to ATR trail
  • ATR trailing stop mode (post-T2 clear break)
  • EOD / max-bars force close
  • Tick-based slippage and round-trip commission

Both backtest engines and PaperBroker should delegate to this class so
the stats produced by the backtest are *structurally identical* to what
the live broker executes.

Design principle — matching PaperBroker:
  The live broker is called once per 5-minute bar with a single price
  (current_price).  All checks (T1, trail, stop, target) are evaluated
  against that one price.  This simulator mirrors that: it uses `cl`
  (bar close) as the reference for trail updates and priority checks,
  and hi/lo only for "did price touch the level at all this bar" (intrabar
  stop/target detection).  This keeps backtest numbers honest while still
  catching intrabar stop hits that a paper broker would miss.

Usage (backtest — OHLC bar)
───────────────────────────
    sim = TradeSimulator(symbol="ES", direction=1,
                         entry=5100.0, stop=5085.0, t1=5120.0, t2=5135.0)
    for bar in df.itertuples():
        partial = sim.tick(hi=bar.high, lo=bar.low, cl=bar.close,
                           atr=bar.atr, eod=..., timeout=...)
        if partial and partial.kind == "partial":
            # T1 booked; keep calling tick() for the runner
            continue
        if partial and partial.kind == "close":
            break

Usage (live — single price)
───────────────────────────
    result = sim.tick(hi=px, lo=px, cl=px, atr=atr)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Slippage & commission tables (futures, per contract round-trip) ──── #
SLIPPAGE_PTS: dict[str, float] = {
    "ES": 0.25, "NQ": 0.25, "GC": 0.10,
    "CL": 0.01, "RTY": 0.10, "YM": 1.00,
}
COMMISSION_RT: dict[str, float] = {
    "ES": 5.00, "NQ": 5.00, "GC": 5.00,
    "CL": 7.00, "RTY": 5.00, "YM": 5.00,
}
_DEFAULT_SLIP   = 0.25
_DEFAULT_COMM   = 5.00


@dataclass
class TickResult:
    """Returned by TradeSimulator.tick() when T1 partial fires or the trade closes."""

    kind: Literal["partial", "close"]
    # "partial" → T1 half-exit booked; caller must keep calling tick() for the runner
    # "close"   → trade is over

    reason: str
    # "t1_partial" | "target2" | "trail_stop" | "be_stopped" | "stop" | "eod" | "timeout"

    exit_price: float       # actual fill (after slippage)
    total_pnl:  float       # net PnL for the whole trade so far (grows each tick)
    r_multiple: float       # total_pnl / (original_qty × risk_pts)
    partial_pnl: float      # cumulative PnL booked at T1 exits (0 until T1 hit)


@dataclass
class TradeSimulator:
    """
    Bar-by-bar trade simulator — one instance per trade.

    Parameters
    ----------
    symbol      root symbol ("ES", "NQ", …) — used for slippage/commission lookup
    direction   +1 = long, -1 = short
    entry       fill price (should already include entry slippage from the broker)
    stop        initial stop-loss level
    t1          T1 target (half-position exit, stop → BE)
    t2          T2 target (runner target; after clear-break → ATR trail)
    qty         position size in contracts (default 1.0)
    apply_costs if True, subtract 1-tick slippage + RT commission from each exit
    """

    symbol:       str
    direction:    int
    entry:        float
    stop:         float
    t1:           float
    t2:           float
    qty:          float = 1.0
    apply_costs:  bool  = True

    # ── Mutable state ─────────────────────────────────────────────────── #
    _sl:          float = field(init=False)
    _sl_orig:     float = field(init=False)
    _risk:        float = field(init=False)
    _R:           float = field(init=False)   # 1R for ratchet = abs(T1 − entry) / 2
    _t1_exited:   bool  = field(init=False, default=False)
    _partial_pnl: float = field(init=False, default=0.0)
    _t2_hit:      bool  = field(init=False, default=False)
    _trail_sl:    float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._sl      = self.stop
        self._sl_orig = self.stop
        self._risk    = abs(self.entry - self.stop)
        # R = half the T1 distance.  For a 2R T1 target this equals original risk.
        self._R = abs(self.t1 - self.entry) / 2.0

    # ─────────────────────────────────────────────────────────────────── #
    # Public API                                                           #
    # ─────────────────────────────────────────────────────────────────── #

    def tick(
        self,
        hi:      float,
        lo:      float,
        cl:      float,
        atr:     float,
        eod:     bool = False,
        timeout: bool = False,
    ) -> TickResult | None:
        """
        Process one OHLC bar (or a single live-price update when hi == lo == cl).

        Returns:
          TickResult(kind="partial") — T1 partial booked; keep calling tick()
          TickResult(kind="close")  — trade is over; stop calling tick()
          None                      — trade still open
        """
        d = self.direction

        # ══ Phase A: ATR trailing stop mode (active after T2 clear-break) ══ #
        if self._t2_hit:
            # Ratchet the ATR trail using close price (one-price-per-bar model)
            if d == 1:
                self._trail_sl = max(self._trail_sl, cl - 0.8 * atr)
            else:
                self._trail_sl = min(self._trail_sl, cl + 0.8 * atr)

            hit_trail = (d == 1 and lo <= self._trail_sl) or \
                        (d == -1 and hi >= self._trail_sl)
            if hit_trail:
                return self._close("trail_stop", self._trail_sl)
            if eod:
                return self._close("eod", cl)
            if timeout:
                return self._close("timeout", cl)
            return None

        # ══ Phase B: Normal management ════════════════════════════════════ #

        # ── B1: T1 partial exit ───────────────────────────────────────── #
        if not self._t1_exited:
            t1_hit = (d == 1 and hi >= self.t1) or (d == -1 and lo <= self.t1)
            if t1_hit:
                partial_result = self._partial_exit()
                # After booking, immediately update 1R ratchet from this bar's close
                self._apply_ratchet(cl)
                return partial_result

        # ── B2: 1R ratchet trail update (uses close, one-price model) ─── #
        # Capture the stop BEFORE updating so this bar's stop check uses the
        # pre-ratchet level (trail ratchets take effect from next bar onward).
        _sl_this_bar = self._sl
        if self._t1_exited:
            self._apply_ratchet(cl)

        # ── B3: Stop check (intrabar hi/lo — catches overshot stops) ──── #
        hit_stop = (d == 1 and lo <= _sl_this_bar) or (d == -1 and hi >= _sl_this_bar)
        if hit_stop:
            reason = "be_stopped" if self._t1_exited else "stop"
            return self._close(reason, _sl_this_bar)

        # ── B4: T2 check ─────────────────────────────────────────────── #
        hit_t2 = (d == 1 and hi >= self.t2) or (d == -1 and lo <= self.t2)
        if hit_t2:
            clear_break = (d == 1 and hi >= self.t2 + 1.0 * atr) or \
                          (d == -1 and lo <= self.t2 - 1.0 * atr)
            if clear_break:
                self._t2_hit   = True
                self._trail_sl = (self.t2 - 0.8 * atr) if d == 1 else (self.t2 + 0.8 * atr)
                return None   # still running, check trail next bar
            return self._close("target2", self.t2)

        if eod:
            return self._close("eod", cl)
        if timeout:
            return self._close("timeout", cl)

        return None

    # ─────────────────────────────────────────────────────────────────── #
    # State properties                                                     #
    # ─────────────────────────────────────────────────────────────────── #

    @property
    def current_stop(self) -> float:
        return self._sl

    @property
    def t1_exited(self) -> bool:
        return self._t1_exited

    @property
    def t2_in_trail(self) -> bool:
        return self._t2_hit

    # ─────────────────────────────────────────────────────────────────── #
    # Internal helpers                                                     #
    # ─────────────────────────────────────────────────────────────────── #

    def _slip(self) -> float:
        return SLIPPAGE_PTS.get(self.symbol, _DEFAULT_SLIP) if self.apply_costs else 0.0

    def _comm(self, qty: float) -> float:
        return (COMMISSION_RT.get(self.symbol, _DEFAULT_COMM) * qty) if self.apply_costs else 0.0

    def _apply_ratchet(self, reference_price: float) -> None:
        """Move stop 1R behind reference_price (ratchet: never steps backward)."""
        if not self._t1_exited or self._R <= 0:
            return
        d = self.direction
        if d == 1:
            new_sl = reference_price - self._R
            if new_sl > self._sl:
                self._sl = new_sl
        else:
            new_sl = reference_price + self._R
            if new_sl < self._sl:
                self._sl = new_sl

    def _partial_exit(self) -> TickResult:
        """Close half the position at T1."""
        half_qty  = self.qty / 2.0
        d         = self.direction
        slip      = self._slip()
        # Adverse slippage: fill = T1 - d*slip  (universal: long→fill below T1, short→fill above T1)
        fill      = self.t1 - d * slip
        raw_pnl   = (fill - self.entry) * d * half_qty
        net_pnl   = raw_pnl - self._comm(half_qty)

        self._partial_pnl = net_pnl
        self._t1_exited   = True
        self._sl          = self.entry   # stop → breakeven

        r = net_pnl / (self.qty * self._risk) if (self.qty * self._risk) > 0 else 0.0

        return TickResult(
            kind        = "partial",
            reason      = "t1_partial",
            exit_price  = fill,
            total_pnl   = net_pnl,
            r_multiple  = round(r, 3),
            partial_pnl = net_pnl,
        )

    def _close(self, reason: str, exit_level: float) -> TickResult:
        """Close the remaining position at exit_level (full or runner half)."""
        d          = self.direction
        slip       = self._slip()
        # Universal adverse slippage: fill = level - d*slip
        # Long sell: fill slightly below level; short buy: fill slightly above.
        fill       = exit_level - d * slip
        runner_qty = (self.qty / 2.0) if self._t1_exited else self.qty
        raw_pnl    = (fill - self.entry) * d * runner_qty
        net_runner = raw_pnl - self._comm(runner_qty)
        total_pnl  = self._partial_pnl + net_runner

        r = total_pnl / (self.qty * self._risk) if (self.qty * self._risk) > 0 else 0.0

        return TickResult(
            kind        = "close",
            reason      = reason,
            exit_price  = fill,
            total_pnl   = round(total_pnl, 4),
            r_multiple  = round(r, 3),
            partial_pnl = self._partial_pnl,
        )
