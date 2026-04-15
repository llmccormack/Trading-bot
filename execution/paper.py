"""
Paper trading engine — simulates order execution with realistic fills.
Tracks positions, P&L, and writes to the trade journal.
"""
from __future__ import annotations
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Literal
import duckdb
from config import settings
from data.store import DB_PATH
from risk.manager import RiskManager, TradeRequest, RiskValidation


OrderSide = Literal["BUY", "SELL"]
PositionStatus = Literal["OPEN", "CLOSED"]

SLIPPAGE_PCT = 0.0005  # 0.05% simulated slippage on fills


@dataclass
class Position:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: str = ""      # LONG | SHORT
    qty: float = 0
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    target_1: float = 0.0   # partial exit / break-even trigger (T1); 0 = disabled
    be_moved: bool = False   # True after stop was moved to break-even at T1
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None
    status: PositionStatus = "OPEN"
    strategy_used: str = ""
    ai_reasoning: str = ""

    @property
    def unrealized_pnl(self) -> float | None:
        return None  # calculated externally with current price

    def pnl_at_price(self, current_price: float) -> float:
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - current_price) * self.qty


class PaperBroker:
    """
    Simulated paper trading broker.
    Handles opening/closing positions and logging to DuckDB.
    """

    def __init__(self, risk_manager: RiskManager, db_path: str | None = None):
        self.risk_manager = risk_manager
        self.db_path = db_path or DB_PATH
        self.account_balance = settings.paper_account_size
        self._positions: dict[str, Position] = {}
        self._close_lock = threading.Lock()   # prevents concurrent double-close of same position
        self._reload_open_positions()   # restore positions that survived a server restart

    def _reload_open_positions(self) -> None:
        """Load any OPEN positions and restore running balance from the database on startup."""
        try:
            conn = duckdb.connect(self.db_path)

            # Restore account balance: starting capital + all historical realized P&L
            pnl_row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0.0) FROM positions WHERE status = 'CLOSED'"
            ).fetchone()
            if pnl_row:
                self.account_balance = settings.paper_account_size + float(pnl_row[0])

            # Restore today's daily P&L so the Day P&L display is correct after a restart
            today_pnl_row = conn.execute("""
                SELECT COALESCE(SUM(realized_pnl), 0.0) FROM positions
                WHERE status = 'CLOSED'
                  AND closed_at >= current_date
            """).fetchone()
            if today_pnl_row:
                self.risk_manager._daily_realized_pnl = float(today_pnl_row[0])

            rows = conn.execute("""
                SELECT id, symbol, direction, qty, entry_price,
                       stop_loss, take_profit, opened_at, notes
                FROM positions WHERE status = 'OPEN'
            """).fetchall()
            conn.close()
            for row in rows:
                pos = Position(
                    id          = row[0],
                    symbol      = row[1],
                    direction   = row[2],
                    qty         = row[3],
                    entry_price = row[4],
                    stop_loss   = row[5],
                    take_profit = row[6],
                    opened_at   = row[7],
                    strategy_used = (row[8] or "").split(":")[0].strip(),
                    status      = "OPEN",
                )
                self._positions[pos.id] = pos
        except Exception:
            pass  # DB not ready yet — positions start empty, no problem

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        target_1: float = 0.0,
        strategy_used: str = "",
        ai_reasoning: str = "",
    ) -> tuple[bool, str, Position | None]:
        """
        Attempt to open a position.
        Returns (success, message, position).
        target_1: break-even trigger price (T1 from aplus signal). 0 = disabled.
        """
        req = TradeRequest(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            account_balance=self.account_balance,
        )

        validation: RiskValidation = self.risk_manager.validate(req)

        if not validation.approved:
            return False, f"Risk check failed: {validation.rejection_reason}", None

        # Simulate slippage
        fill_price = entry_price * (1 + SLIPPAGE_PCT) if direction == "BUY" else entry_price * (1 - SLIPPAGE_PCT)

        pos = Position(
            symbol=symbol,
            direction="LONG" if direction == "BUY" else "SHORT",
            qty=validation.position_size,
            entry_price=round(fill_price, 4),
            stop_loss=stop_loss,
            take_profit=take_profit,
            target_1=target_1,
            strategy_used=strategy_used,
            ai_reasoning=ai_reasoning,
        )

        self._positions[pos.id] = pos
        self.risk_manager.on_trade_opened()
        self._save_position(pos)

        msg = (
            f"Opened {pos.direction} {symbol} x{pos.qty} @ {pos.entry_price:.4f} | "
            f"SL={stop_loss:.4f} TP={take_profit:.4f} | Risk=${validation.dollar_risk:.2f}"
        )
        return True, msg, pos

    def close_position(self, position_id: str, exit_price: float, reason: str = "") -> tuple[bool, str, float]:
        """
        Close an open position.
        Returns (success, message, realized_pnl).
        """
        with self._close_lock:
            return self._close_position_locked(position_id, exit_price, reason)

    def _close_position_locked(self, position_id: str, exit_price: float, reason: str = "") -> tuple[bool, str, float]:
        """Internal close — must be called with _close_lock held."""
        if position_id not in self._positions:
            return False, "Position not found", 0.0

        pos = self._positions[position_id]
        if pos.status == "CLOSED":
            return False, "Position already closed", 0.0

        # Simulate slippage on exit
        fill = exit_price * (1 - SLIPPAGE_PCT) if pos.direction == "LONG" else exit_price * (1 + SLIPPAGE_PCT)
        pnl = pos.pnl_at_price(round(fill, 4))

        pos.exit_price = round(fill, 4)
        pos.realized_pnl = round(pnl, 2)
        pos.closed_at = datetime.now(timezone.utc)
        pos.status = "CLOSED"

        self.account_balance += pnl
        self.risk_manager.on_trade_closed(pnl)
        self._update_position(pos)
        self._log_journal(pos, reason)

        msg = (
            f"Closed {pos.direction} {pos.symbol} x{pos.qty} @ {fill:.4f} | "
            f"PnL=${pnl:+.2f} | Balance=${self.account_balance:,.2f}"
        )
        return True, msg, pnl

    def check_stops_and_targets(self, symbol: str, current_price: float) -> list[str]:
        """
        Check open positions for a symbol and auto-close at stop/target.
        Also moves stop to break-even when T1 is hit, then trails 1R behind
        price on the runner until T2.
        Returns list of close messages.
        """
        messages = []
        for pos in list(self._positions.values()):
            if pos.symbol != symbol or pos.status == "CLOSED":
                continue

            # ── Partial exit at T1: close half, move stop to break-even ── #
            if pos.target_1 > 0 and not pos.be_moved:
                hit_t1 = (
                    (pos.direction == "LONG"  and current_price >= pos.target_1) or
                    (pos.direction == "SHORT" and current_price <= pos.target_1)
                )
                if hit_t1:
                    half_qty    = pos.qty / 2.0
                    t1_fill     = pos.target_1
                    partial_pnl = (
                        (t1_fill - pos.entry_price) * half_qty if pos.direction == "LONG"
                        else (pos.entry_price - t1_fill) * half_qty
                    )
                    old_stop      = pos.stop_loss
                    pos.qty       = half_qty          # runner: half position remains
                    pos.stop_loss = pos.entry_price   # move to break-even
                    pos.be_moved  = True

                    self.account_balance                   += partial_pnl
                    self.risk_manager._daily_realized_pnl  += partial_pnl  # credit but keep position open

                    self._update_partial(pos)
                    self._log_partial_journal(pos, t1_fill, half_qty, partial_pnl, old_stop)

                    messages.append(
                        f"[PARTIAL T1] {pos.symbol} {pos.direction} | "
                        f"T1 @ {t1_fill:.2f} | Closed {half_qty} lots | "
                        f"PnL=${partial_pnl:+.2f} | Stop→BE: {old_stop:.2f}→{pos.entry_price:.2f} | "
                        f"Runner targeting {pos.take_profit:.2f}"
                    )

            # ── Trail stop 1R behind price once runner is active (be_moved=True) #
            # T1 ≈ entry + 2R, so R ≈ (T1 - entry) / 2.  The stop moves to
            # (current_price - 1R) for longs, (current_price + 1R) for shorts,
            # but only ever in the favourable direction (ratchet, never steps back).
            if pos.be_moved and pos.target_1 > 0:
                R = abs(pos.target_1 - pos.entry_price) / 2.0
                if R > 0:
                    if pos.direction == "LONG":
                        trail_stop = round(current_price - R, 4)
                        if trail_stop > pos.stop_loss:
                            old_sl        = pos.stop_loss
                            pos.stop_loss = trail_stop
                            self._update_stop(pos)
                            messages.append(
                                f"[TRAIL] {pos.symbol} LONG | "
                                f"Stop {old_sl:.2f} → {trail_stop:.2f} "
                                f"(price={current_price:.2f}, 1R={R:.2f})"
                            )
                    else:  # SHORT
                        trail_stop = round(current_price + R, 4)
                        if trail_stop < pos.stop_loss:
                            old_sl        = pos.stop_loss
                            pos.stop_loss = trail_stop
                            self._update_stop(pos)
                            messages.append(
                                f"[TRAIL] {pos.symbol} SHORT | "
                                f"Stop {old_sl:.2f} → {trail_stop:.2f} "
                                f"(price={current_price:.2f}, 1R={R:.2f})"
                            )

            hit_stop = (
                (pos.direction == "LONG"  and current_price <= pos.stop_loss) or
                (pos.direction == "SHORT" and current_price >= pos.stop_loss)
            )
            hit_target = (
                (pos.direction == "LONG"  and current_price >= pos.take_profit) or
                (pos.direction == "SHORT" and current_price <= pos.take_profit)
            )

            if hit_stop:
                reason = "be_stopped" if pos.be_moved else "stop_loss"
                _, msg, _ = self.close_position(pos.id, pos.stop_loss, reason=reason)
                messages.append(f"[STOP HIT] {msg}")
            elif hit_target:
                _, msg, _ = self.close_position(pos.id, pos.take_profit, reason="take_profit")
                messages.append(f"[TARGET HIT] {msg}")

        return messages

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.status == "OPEN"]

    def portfolio_summary(self) -> dict:
        open_pos = self.open_positions
        return {
            "account_balance": round(self.account_balance, 2),
            "open_positions": len(open_pos),
            "daily_pnl": round(self.risk_manager.daily_pnl, 2),
            "positions": [
                {
                    "id":           p.id,
                    "symbol":       p.symbol,
                    "direction":    p.direction,
                    "qty":          p.qty,
                    "entry":        p.entry_price,
                    "sl":           p.stop_loss,
                    "tp":           p.take_profit,
                    "opened_at":    p.opened_at.isoformat() if p.opened_at else None,
                    "strategy":     p.strategy_used,
                    "risk_pts":     abs(p.entry_price - p.stop_loss),
                    "reward_pts":   abs(p.take_profit - p.entry_price),
                }
                for p in open_pos
            ],
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _save_position(self, pos: Position) -> None:
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            INSERT INTO positions
            (id, symbol, direction, qty, entry_price, stop_loss, take_profit, opened_at, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            pos.id, pos.symbol, pos.direction, pos.qty,
            pos.entry_price, pos.stop_loss, pos.take_profit,
            pos.opened_at.isoformat(), pos.status,
            f"{pos.strategy_used}: {pos.ai_reasoning[:200]}"
        ])
        conn.close()

    def _update_position(self, pos: Position) -> None:
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            UPDATE positions
            SET exit_price=?, realized_pnl=?, closed_at=?, status=?
            WHERE id=?
        """, [pos.exit_price, pos.realized_pnl, pos.closed_at.isoformat(), pos.status, pos.id])
        conn.close()

    def _update_stop(self, pos: Position) -> None:
        """Persist an in-place stop_loss change to DB."""
        conn = duckdb.connect(self.db_path)
        conn.execute("UPDATE positions SET stop_loss=? WHERE id=?",
                     [pos.stop_loss, pos.id])
        conn.close()

    def _update_partial(self, pos: Position) -> None:
        """Persist qty reduction + stop move after a partial exit."""
        conn = duckdb.connect(self.db_path)
        conn.execute("UPDATE positions SET qty=?, stop_loss=? WHERE id=?",
                     [pos.qty, pos.stop_loss, pos.id])
        conn.close()

    def _log_partial_journal(
        self, pos: Position, exit_price: float, closed_qty: float, pnl: float,
        original_stop: float = 0.0,
    ) -> None:
        """Write a journal row for a partial (T1) exit."""
        stop_ref      = original_stop if original_stop > 0 else pos.stop_loss
        original_risk = abs(pos.entry_price - stop_ref) * closed_qty
        r_multiple    = pnl / original_risk if original_risk > 0 else None
        now           = datetime.now(timezone.utc)
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            INSERT INTO trade_journal
            (id, position_id, symbol, direction, entry_price, exit_price, qty,
             pnl, r_multiple, strategy_used, ai_reasoning, opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(uuid.uuid4()), pos.id, pos.symbol, pos.direction,
            pos.entry_price, exit_price, closed_qty,
            round(pnl, 2), r_multiple,
            pos.strategy_used, "partial_exit_t1",
            pos.opened_at.isoformat(), now.isoformat(),
        ])
        conn.close()

    def _log_journal(self, pos: Position, reason: str) -> None:
        initial_risk = abs(pos.entry_price - pos.stop_loss) * pos.qty
        r_multiple = pos.realized_pnl / initial_risk if initial_risk > 0 else None

        conn = duckdb.connect(self.db_path)
        # Guard: skip if a full-close journal entry already exists for this position
        existing = conn.execute(
            "SELECT COUNT(*) FROM trade_journal WHERE position_id = ? AND ai_reasoning != 'partial_exit_t1'",
            [pos.id]
        ).fetchone()[0]
        if existing > 0:
            conn.close()
            return
        conn.execute("""
            INSERT INTO trade_journal
            (id, position_id, symbol, direction, entry_price, exit_price, qty,
             pnl, r_multiple, strategy_used, ai_reasoning, opened_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            str(uuid.uuid4()), pos.id, pos.symbol, pos.direction,
            pos.entry_price, pos.exit_price, pos.qty,
            pos.realized_pnl, r_multiple,
            pos.strategy_used, pos.ai_reasoning[:500],
            pos.opened_at.isoformat(), pos.closed_at.isoformat()
        ])
        conn.close()
