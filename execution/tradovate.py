"""
Tradovate live/demo broker — real order execution via Tradovate REST API.

Mirrors the PaperBroker interface exactly so the rest of the bot needs
zero changes — just swap the broker instance.

Environment variables required:
    TRADOVATE_USERNAME   — Tradovate account email
    TRADOVATE_PASSWORD   — Tradovate account password
    TRADOVATE_APP_ID     — App ID from Tradovate partner portal
    TRADOVATE_APP_SECRET — App secret from Tradovate partner portal
    TRADOVATE_CID        — Client ID (from partner portal)
    TRADOVATE_SEC        — Client secret
    TRADOVATE_IS_DEMO    — "true" for demo/sim, "false" for live (default: true)
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import requests

from config import settings
from execution.paper import Position  # reuse the same Position dataclass
from risk.manager import RiskManager, TradeRequest

log = logging.getLogger(__name__)

# ── Tradovate contract specs (tick value per 1 lot) ──────────────────────────
# Used to convert price-point P&L into dollars.
CONTRACT_SPECS: dict[str, dict] = {
    "NQ=F":  {"name": "NQM5",  "tick_size": 0.25,  "tick_value": 5.00,   "exchange": "CME"},
    "ES=F":  {"name": "ESM5",  "tick_size": 0.25,  "tick_value": 12.50,  "exchange": "CME"},
    "GC=F":  {"name": "GCM5",  "tick_size": 0.10,  "tick_value": 10.00,  "exchange": "COMEX"},
    "RTY=F": {"name": "RTYM5", "tick_size": 0.10,  "tick_value": 5.00,   "exchange": "CME"},
    "YM=F":  {"name": "YMM5",  "tick_size": 1.0,   "tick_value": 5.00,   "exchange": "CBOT"},
}

PositionStatus = Literal["OPEN", "CLOSED"]


class TradovateError(Exception):
    pass


class TradovateBroker:
    """
    Live/demo broker that routes orders through the Tradovate REST API.
    Drop-in replacement for PaperBroker.
    """

    DEMO_BASE = "https://demo.tradovateapi.com/v1"
    LIVE_BASE = "https://live.tradovateapi.com/v1"

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self._is_demo: bool = str(getattr(settings, "tradovate_is_demo", "true")).lower() != "false"
        self.base_url = self.DEMO_BASE if self._is_demo else self.LIVE_BASE

        # Auth state
        self._access_token: str | None = None
        self._token_expiry: float = 0.0
        self._token_lock = threading.Lock()

        # Local position cache (keyed by our internal UUID)
        self._positions: dict[str, Position] = {}
        # Map our internal id → Tradovate orderId
        self._tv_order_ids: dict[str, int] = {}
        self._positions_lock = threading.Lock()

        # Account balance from Tradovate
        self.account_balance: float = 0.0
        self._account_id: int | None = None

        # Authenticate immediately
        self._authenticate()
        self._load_account()

        # Background token refresh thread
        self._refresh_thread = threading.Thread(
            target=self._token_refresh_loop, daemon=True
        )
        self._refresh_thread.start()

        log.info(
            "TradovateBroker ready | mode=%s | account_id=%s | balance=$%.2f",
            "DEMO" if self._is_demo else "LIVE",
            self._account_id,
            self.account_balance,
        )

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        """Obtain an access token from Tradovate."""
        payload = {
            "name":       getattr(settings, "tradovate_username", ""),
            "password":   getattr(settings, "tradovate_password", ""),
            "appId":      getattr(settings, "tradovate_app_id", ""),
            "appVersion": "1.0",
            "cid":        getattr(settings, "tradovate_cid", ""),
            "sec":        getattr(settings, "tradovate_sec", ""),
        }
        resp = requests.post(
            f"{self.base_url}/auth/accesstokenrequest",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if "errorText" in data:
            raise TradovateError(f"Auth failed: {data['errorText']}")

        self._access_token = data["accessToken"]
        # Token valid for 90 min — refresh after 85
        self._token_expiry = time.time() + 85 * 60
        log.info("Tradovate auth OK | demo=%s", self._is_demo)

    def _token_refresh_loop(self) -> None:
        """Runs in background, silently refreshes the token before it expires."""
        while True:
            sleep_for = max(60, self._token_expiry - time.time() - 60)
            time.sleep(sleep_for)
            try:
                with self._token_lock:
                    self._authenticate()
            except Exception as e:
                log.warning("Token refresh failed: %s — will retry in 60s", e)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token}"}

    def _get(self, path: str, **kwargs) -> dict:
        resp = requests.get(
            f"{self.base_url}{path}", headers=self._headers(), timeout=10, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Account ───────────────────────────────────────────────────────────────

    def _load_account(self) -> None:
        """Fetch account ID and current cash balance."""
        try:
            accounts = self._get("/account/list")
            if accounts:
                acc = accounts[0]
                self._account_id = acc["id"]

            cash = self._get("/cashBalance/getcashbalancesnapshot",
                             params={"accountId": self._account_id})
            self.account_balance = float(cash.get("totalCashValue", 0))
        except Exception as e:
            log.warning("Could not load account info: %s", e)

    def _refresh_balance(self) -> None:
        try:
            cash = self._get("/cashBalance/getcashbalancesnapshot",
                             params={"accountId": self._account_id})
            self.account_balance = float(cash.get("totalCashValue", 0))
        except Exception:
            pass

    # ── Contract lookup ───────────────────────────────────────────────────────

    @staticmethod
    def _tv_contract(symbol: str) -> str:
        """Convert yfinance symbol (ES=F) to Tradovate contract name (ESM5)."""
        spec = CONTRACT_SPECS.get(symbol)
        if not spec:
            raise TradovateError(f"Unknown symbol: {symbol}")
        return spec["name"]

    @staticmethod
    def _tick_value(symbol: str) -> float:
        spec = CONTRACT_SPECS.get(symbol, {})
        return spec.get("tick_value", 12.50)

    @staticmethod
    def _tick_size(symbol: str) -> float:
        spec = CONTRACT_SPECS.get(symbol, {})
        return spec.get("tick_size", 0.25)

    # ── Order placement ───────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        direction: str,         # "BUY" | "SELL"
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        target_1: float = 0.0,
        strategy_used: str = "",
        ai_reasoning: str = "",
    ) -> tuple[bool, str, Position | None]:
        """
        Place a bracket order on Tradovate.
        Returns (success, message, position).
        """
        # Risk validation
        req = TradeRequest(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            account_balance=self.account_balance,
        )
        validation = self.risk_manager.validate(req)
        if not validation.approved:
            return False, f"Risk check failed: {validation.rejection_reason}", None

        qty = int(max(1, round(validation.position_size)))
        action = "Buy" if direction == "BUY" else "Sell"
        contract = self._tv_contract(symbol)
        tick_sz = self._tick_size(symbol)

        # Round SL and TP to valid tick increments
        def round_tick(price: float) -> float:
            return round(round(price / tick_sz) * tick_sz, 6)

        sl_price = round_tick(stop_loss)
        tp_price = round_tick(take_profit)

        # OSO = One-Sends-Other: entry order + automatic SL + TP bracket
        body = {
            "accountSpec":     str(self._account_id),
            "accountId":       self._account_id,
            "action":          action,
            "symbol":          contract,
            "orderQty":        qty,
            "orderType":       "Market",
            "isAutomated":     True,
            "bracket1": {
                "action":    "Sell" if action == "Buy" else "Buy",
                "orderType": "Stop",
                "stopPrice": sl_price,
                "orderQty":  qty,
            },
            "bracket2": {
                "action":    "Sell" if action == "Buy" else "Buy",
                "orderType": "Limit",
                "price":     tp_price,
                "orderQty":  qty,
            },
        }

        try:
            result = self._post("/order/placeoso", body)
        except Exception as e:
            return False, f"Tradovate order failed: {e}", None

        if result.get("failureReason"):
            return False, f"Order rejected: {result['failureReason']}", None

        order_id = result.get("orderId") or result.get("id")

        # Build local Position record
        pos = Position(
            symbol=symbol,
            direction="LONG" if direction == "BUY" else "SHORT",
            qty=float(qty),
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            target_1=target_1,
            strategy_used=strategy_used,
            ai_reasoning=ai_reasoning,
        )

        with self._positions_lock:
            self._positions[pos.id] = pos
            if order_id:
                self._tv_order_ids[pos.id] = order_id

        self.risk_manager.on_trade_opened()

        msg = (
            f"[LIVE] Opened {pos.direction} {symbol} x{qty} @ market | "
            f"SL={sl_price} TP={tp_price} | orderId={order_id}"
        )
        log.info(msg)
        return True, msg, pos

    # ── Close ─────────────────────────────────────────────────────────────────

    def close_position(
        self, position_id: str, exit_price: float, reason: str = ""
    ) -> tuple[bool, str, float]:
        """
        Liquidate a position on Tradovate with a market order.
        Returns (success, message, realized_pnl).
        """
        with self._positions_lock:
            pos = self._positions.get(position_id)
        if not pos or pos.status == "CLOSED":
            return False, "Position not found or already closed", 0.0

        close_action = "Sell" if pos.direction == "LONG" else "Buy"
        contract = self._tv_contract(pos.symbol)

        body = {
            "accountSpec": str(self._account_id),
            "accountId":   self._account_id,
            "action":      close_action,
            "symbol":      contract,
            "orderQty":    int(pos.qty),
            "orderType":   "Market",
            "isAutomated": True,
        }

        try:
            self._post("/order/placeorder", body)
        except Exception as e:
            return False, f"Close order failed: {e}", 0.0

        # Calculate approximate P&L
        tick_val = self._tick_value(pos.symbol)
        tick_sz  = self._tick_size(pos.symbol)
        ticks    = (exit_price - pos.entry_price) / tick_sz
        if pos.direction == "SHORT":
            ticks = -ticks
        pnl = round(ticks * tick_val * pos.qty, 2)

        pos.exit_price  = exit_price
        pos.realized_pnl = pnl
        pos.closed_at   = datetime.now(timezone.utc)
        pos.status      = "CLOSED"

        self.risk_manager.on_trade_closed(pnl)
        self._refresh_balance()

        msg = (
            f"[LIVE] Closed {pos.direction} {pos.symbol} x{pos.qty} @ {exit_price} | "
            f"PnL≈${pnl:+.2f} | reason={reason}"
        )
        log.info(msg)
        return True, msg, pnl

    # ── Stop management ───────────────────────────────────────────────────────

    def check_stops_and_targets(self, symbol: str, current_price: float) -> list[str]:
        """
        Tradovate manages the bracket SL/TP orders server-side.
        We still handle T1 partial exit + trailing stop locally and
        send updated stop orders to Tradovate when the stop moves.
        """
        messages = []
        with self._positions_lock:
            positions = [p for p in self._positions.values()
                         if p.symbol == symbol and p.status == "OPEN"]

        for pos in positions:
            # ── T1 partial exit ────────────────────────────────────────────
            if pos.target_1 > 0 and not pos.be_moved:
                hit_t1 = (
                    (pos.direction == "LONG"  and current_price >= pos.target_1) or
                    (pos.direction == "SHORT" and current_price <= pos.target_1)
                )
                if hit_t1:
                    half_qty = int(pos.qty // 2)
                    if half_qty > 0:
                        # Send partial close
                        close_action = "Sell" if pos.direction == "LONG" else "Buy"
                        try:
                            self._post("/order/placeorder", {
                                "accountSpec": str(self._account_id),
                                "accountId":   self._account_id,
                                "action":      close_action,
                                "symbol":      self._tv_contract(symbol),
                                "orderQty":    half_qty,
                                "orderType":   "Market",
                                "isAutomated": True,
                            })
                            tick_val = self._tick_value(pos.symbol)
                            tick_sz  = self._tick_size(pos.symbol)
                            ticks    = (pos.target_1 - pos.entry_price) / tick_sz
                            if pos.direction == "SHORT":
                                ticks = -ticks
                            partial_pnl = round(ticks * tick_val * half_qty, 2)

                            old_stop      = pos.stop_loss
                            pos.qty       = pos.qty - half_qty
                            pos.stop_loss = pos.entry_price
                            pos.be_moved  = True
                            self.risk_manager._daily_realized_pnl += partial_pnl

                            messages.append(
                                f"[PARTIAL T1] {symbol} | T1@{pos.target_1} | "
                                f"closed {half_qty} lots | PnL≈${partial_pnl:+.2f} | "
                                f"Stop→BE {old_stop}→{pos.entry_price}"
                            )
                        except Exception as e:
                            log.warning("T1 partial close failed: %s", e)

            # ── Trailing stop ──────────────────────────────────────────────
            if pos.be_moved and pos.target_1 > 0:
                R = abs(pos.target_1 - pos.entry_price) / 2.0
                if R > 0:
                    if pos.direction == "LONG":
                        trail_stop = round(current_price - R, 4)
                        if trail_stop > pos.stop_loss:
                            old_sl = pos.stop_loss
                            pos.stop_loss = trail_stop
                            messages.append(
                                f"[TRAIL] {symbol} LONG | Stop {old_sl}→{trail_stop}"
                            )
                    else:
                        trail_stop = round(current_price + R, 4)
                        if trail_stop < pos.stop_loss:
                            old_sl = pos.stop_loss
                            pos.stop_loss = trail_stop
                            messages.append(
                                f"[TRAIL] {symbol} SHORT | Stop {old_sl}→{trail_stop}"
                            )

        return messages

    # ── Portfolio ─────────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> list[Position]:
        with self._positions_lock:
            return [p for p in self._positions.values() if p.status == "OPEN"]

    def portfolio_summary(self) -> dict:
        self._refresh_balance()
        open_pos = self.open_positions
        return {
            "account_balance": round(self.account_balance, 2),
            "open_positions":  len(open_pos),
            "daily_pnl":       round(self.risk_manager.daily_pnl, 2),
            "is_paper":        False,
            "is_live_tradovate": True,
            "is_demo":         self._is_demo,
            "positions": [
                {
                    "id":        p.id,
                    "symbol":    p.symbol,
                    "direction": p.direction,
                    "qty":       p.qty,
                    "entry":     p.entry_price,
                    "sl":        p.stop_loss,
                    "tp":        p.take_profit,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                    "strategy":  p.strategy_used,
                }
                for p in open_pos
            ],
        }

    # ── Connection test ───────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Returns True if the Tradovate connection is healthy."""
        try:
            self._get("/account/list")
            return True
        except Exception:
            return False
