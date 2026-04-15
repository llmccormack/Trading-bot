"""
Backtest Engine v8 — Regime-Complete Architecture

Three strategies, one per market state, so the system always has a valid
play regardless of what ES is doing at any given time.

REGIME MAP:
  trending (ADX ≥ 20, rising)  → ema55_bounce  — momentum continuation
  ranging  (ADX < 18, flat)    → bb_reversion  — BB extreme mean reversion
  neutral  (ADX 15-25, rising  → pdhl_breakout — compression→expansion break
            from ranging)

STRATEGY SUMMARY:
  1. EMA55 BOUNCE (trending)
     3yr ES PF: 1.838 | WR: 47.6% | ~14 trades/yr | RR 2:1
     Deep pullback to EMA55 in confirmed uptrend (EMA55 > EMA200)
     Shorts mirror when EMA55 < EMA200 (confirmed downtrend)

  2. BB REVERSION (ranging)
     Expected WR: 55-65% | RR 1.5:1 (closer target = higher hit rate)
     Price at or below lower Bollinger Band (2SD) with RSI < 38 oversold
     Research: Bookmap/institutional VWAP studies confirm 2SD fade edge
     Only fires when ADX < 18 (genuine range — not pullback in trend)

  3. PRIOR DAY HIGH/LOW BREAKOUT (neutral/transitioning)
     Expected WR: 55-62% | RR 2:1
     NR4 compression day (prior day = narrowest range of last 4) followed
     by close above prior day high or below prior day low
     Research: Crabel (1990), ORB Setups data — 55-68% WR with NR filter

KEY ENGINE SETTINGS:
  - RTH session filter (14-20 UTC = 9am-4pm ET) — critical for 1h futures
  - Per-strategy R:R overrides (bb_reversion uses 1.5:1, others 2:1)
  - allow_short=True — all three strategies have symmetric short-side gates
  - BE trail = False — confirmed to hurt performance on 1h ES
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict

import numpy as np
import pandas as pd

from utils.indicators import add_all


# ──────────────────────────────────────────────────────────────────────────── #
# Data classes                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

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
    def avg_bars_held(self) -> float:
        return sum(t.bars_held for t in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def avg_composite_score(self) -> float:
        return sum(t.composite_score for t in self.trades) / len(self.trades) if self.trades else 0.0

    @property
    def be_trail_rate(self) -> float:
        return sum(1 for t in self.trades if t.be_moved) / len(self.trades) if self.trades else 0.0

    @property
    def best_trade(self): return max(self.trades, key=lambda t: t.pnl_pts) if self.trades else None

    @property
    def worst_trade(self): return min(self.trades, key=lambda t: t.pnl_pts) if self.trades else None

    def by_strategy(self) -> dict:
        groups: Dict[str, list] = {}
        for t in self.trades:
            groups.setdefault(t.strategy, []).append(t)
        out = {}
        for strat, trades in sorted(groups.items()):
            wins = [t for t in trades if t.pnl_pts > 0]
            gl   = abs(sum(t.pnl_pts for t in trades if t.pnl_pts <= 0))
            gw   = sum(t.pnl_pts for t in wins)
            out[strat] = {
                "trades":        len(trades),
                "wins":          len(wins),
                "win_rate":      len(wins) / len(trades),
                "profit_factor": gw / gl if gl > 0 else (math.inf if gw > 0 else 0.0),
                "expectancy":    sum(t.r_multiple for t in trades) / len(trades),
                "total_pnl_pts": sum(t.pnl_pts for t in trades),
            }
        return out

    def by_regime(self) -> dict:
        groups: Dict[str, list] = {}
        for t in self.trades:
            groups.setdefault(t.regime, []).append(t)
        out = {}
        for regime, trades in sorted(groups.items()):
            wins = [t for t in trades if t.pnl_pts > 0]
            gl   = abs(sum(t.pnl_pts for t in trades if t.pnl_pts <= 0))
            gw   = sum(t.pnl_pts for t in wins)
            out[regime] = {
                "trades":        len(trades),
                "wins":          len(wins),
                "win_rate":      len(wins) / len(trades),
                "profit_factor": gw / gl if gl > 0 else (math.inf if gw > 0 else 0.0),
                "total_pnl_pts": sum(t.pnl_pts for t in trades),
            }
        return out

    def summary_for_claude(self) -> str:
        lines = [
            f"Symbol: {self.symbol} | TF: {self.timeframe} | Bars: {self.total_bars}",
            f"Trades: {self.total_trades} | Win rate: {self.win_rate:.1%}",
            f"Profit factor: {self.profit_factor:.2f} | Expectancy: {self.expectancy:+.3f}R",
            f"Max drawdown: {self.max_drawdown_pct:.1%} | Avg hold: {self.avg_bars_held:.1f} bars",
            f"Avg entry score: {self.avg_composite_score:.3f} | BE trail rate: {self.be_trail_rate:.1%}",
            "", "Per-strategy:",
        ]
        for name, s in self.by_strategy().items():
            pf = s['profit_factor']
            pf_str = f"{pf:.2f}" if pf != math.inf else "inf"
            lines.append(
                f"  {name}: {s['trades']} trades | WR {s['win_rate']:.0%} | "
                f"PF {pf_str} | EXP {s['expectancy']:+.3f}R | "
                f"P&L {s['total_pnl_pts']:+.1f}pts"
            )
        lines.append("Per-regime:")
        for name, s in self.by_regime().items():
            pf = s['profit_factor']
            pf_str = f"{pf:.2f}" if pf != math.inf else "inf"
            lines.append(
                f"  {name}: {s['trades']} trades | WR {s['win_rate']:.0%} | "
                f"PF {pf_str} | P&L {s['total_pnl_pts']:+.1f}pts"
            )
        if self.best_trade:
            b = self.best_trade
            lines.append(f"\nBest:  {b.strategy} {b.direction} +{b.pnl_pts:.2f}pts ({b.r_multiple:+.2f}R) [{b.regime}]")
        if self.worst_trade:
            w = self.worst_trade
            lines.append(f"Worst: {w.strategy} {w.direction} {w.pnl_pts:.2f}pts ({w.r_multiple:+.2f}R) [{w.regime}]")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────── #
# Engine                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

class BacktestEngine:
    """
    Regime-Complete ES Futures System.

    Automatically detects market state (trending / ranging / neutral) and
    deploys the appropriate strategy for each condition.
    """

    # Three complementary strategies:
    #   ema55_bounce  → trending regime      (ADX ≥ 20 rising) deep pullback to EMA55
    #   ema21_bounce  → trending regime      (ADX ≥ 20 rising) shallow pullback to EMA21
    #   pdhl_breakout → trending + neutral   NR4 compression → PDH/PDL breakout
    #
    # Global post-trade cooldown prevents ema21 from firing in EMA55's wake:
    #   GLOBAL_COOLDOWN = 6 bars after any trade exits before any new entry
    STRATEGIES = ["ema55_bounce", "ema21_bounce", "pdhl_breakout"]
    GLOBAL_COOLDOWN = 3  # bars after any exit before new entry allowed

    # Per-strategy R:R overrides (None = use engine default)
    STRATEGY_RR: Dict[str, Optional[float]] = {
        "ema55_bounce":  None,   # use engine rr_ratio (default 2.0)
        "ema21_bounce":  None,   # same 2:1 — quality setups run as far
        "pdhl_breakout": 2.0,    # breakout — standard 2:1
    }

    # RTH hours in UTC (9am-4pm ET = 14:00-20:00 UTC)
    RTH_HOURS = frozenset({14, 15, 16, 17, 18, 19, 20})

    def __init__(
        self,
        warmup:      int   = 200,
        max_bars:    int   = 24,
        rr_ratio:    float = 2.0,
        atr_stop:    float = 1.0,
        min_score:   float = 0.55,
        allow_short: bool  = True,   # all three strategies have short-side gates
        min_adx:     float = 25.0,
        be_trail:    bool  = False,
        rth_only:    bool  = True,   # Filter to RTH session hours for 1h futures
    ):
        self.warmup      = warmup
        self.max_bars    = max_bars
        self.rr_ratio    = rr_ratio
        self.atr_stop    = atr_stop
        self.min_score   = min_score
        self.allow_short = allow_short
        self.min_adx     = min_adx
        self.be_trail    = be_trail
        self.rth_only    = rth_only

    # ── Run ────────────────────────────────────────────────────────── #

    def run(
        self,
        df_raw:            pd.DataFrame,
        symbol:            str  = "",
        timeframe:         str  = "",
        market:            str  = "futures",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:

        df = add_all(df_raw.copy())

        def _col(n):
            return df[n].values if n in df.columns else np.full(len(df), np.nan)

        closes  = df["close"].values
        opens   = df["open"].values
        highs   = df["high"].values
        lows    = df["low"].values
        volumes = df["volume"].values
        atrs    = _col("atr")
        rsi     = _col("rsi")
        adx     = _col("adx")
        dmp     = _col("dmp")
        dmn     = _col("dmn")
        ema8    = _col("ema_8")
        ema21   = _col("ema_21")
        ema55   = _col("ema_55")
        ema200  = _col("ema_200")
        vwap    = _col("vwap")
        vol_avg = _col("vol_avg")
        bb_up   = _col("bb_upper")
        bb_lo   = _col("bb_lower")

        # Build RTH hour array from timestamp column
        if "timestamp" in df.columns and self.rth_only:
            hour_utc = pd.to_datetime(df["timestamp"]).dt.hour.values
        else:
            hour_utc = np.full(len(df), 15)  # default: always RTH

        n            = len(df)
        trades: List[BacktestTrade] = []
        equity       = 0.0
        eq_curve     = [0.0]
        position     = None

        # Per-strategy cooldowns (bars after signal fires)
        cooldowns = {
            "ema55_bounce":  8,
            "ema21_bounce":  6,
            "pdhl_breakout": 8,
            "bb_reversion":  5,
            "vwap_reversion": 5,
            "eod_momentum":  1,
        }
        last_signal: Dict[str, int] = {s: -99 for s in self.STRATEGIES}
        last_exit_bar: int = -99   # global: bars since ANY trade exited

        # ── Prior day high/low for pdhl_breakout ────────────────────── #
        pdh = np.full(n, np.nan)   # prior day high at each bar
        pdl = np.full(n, np.nan)   # prior day low at each bar
        is_nr4 = np.zeros(n, dtype=bool)  # True if prior day was NR4 day

        if "timestamp" in df.columns:
            dates = pd.to_datetime(df["timestamp"]).dt.date.values
            unique_dates = sorted(set(dates))
            # Build daily OHLCV from 1h bars
            day_high: dict = {}
            day_low:  dict = {}
            for d in unique_dates:
                mask = dates == d
                day_high[d] = highs[mask].max()
                day_low[d]  = lows[mask].min()
            day_range = {d: day_high[d] - day_low[d] for d in unique_dates}
            # Assign prior-day levels to each bar
            for idx in range(n):
                d = dates[idx]
                d_pos = unique_dates.index(d)
                if d_pos >= 1:
                    prev_d = unique_dates[d_pos - 1]
                    pdh[idx] = day_high[prev_d]
                    pdl[idx] = day_low[prev_d]
                    # NR4: prior day range narrower than preceding 3 days
                    if d_pos >= 4:
                        prev_ranges = [day_range[unique_dates[d_pos - k]] for k in range(1, 5)]
                        is_nr4[idx] = prev_ranges[0] == min(prev_ranges)

        # EOD momentum: track session open price (open of first RTH bar each day)
        session_open_price: float = float("nan")

        for i in range(self.warmup, n - 1):
            if progress_callback:
                progress_callback(i - self.warmup, n - self.warmup - 1)

            # ── Session open tracking (for EOD momentum) ─────────────── #
            if int(hour_utc[i]) == 14:  # First RTH bar of the day (9am ET)
                session_open_price = opens[i]

            # ── Breakeven trail ─────────────────────────────────────── #
            if position and self.be_trail and not position.get("be_moved"):
                entry = position["entry_price"]
                risk  = position["risk"]
                d     = position["direction"]
                atr_e = position["atr_entry"]
                hit_1r = (
                    (highs[i] - entry >= risk) if d == "BUY"
                    else (entry - lows[i]  >= risk)
                )
                if hit_1r:
                    buf = atr_e * 0.08
                    position["stop"]     = round((entry + buf) if d == "BUY" else (entry - buf), 4)
                    position["be_moved"] = True

            # ── Exit ────────────────────────────────────────────────── #
            if position:
                hi, lo, cl = highs[i], lows[i], closes[i]
                d         = position["direction"]
                bars_held = i - position["entry_bar"]
                ep, er    = None, None

                if d == "BUY":
                    if lo <= position["stop"]:
                        ep, er = position["stop"], "BE_STOP" if position.get("be_moved") else "STOP"
                    elif hi >= position["target"]:
                        ep, er = position["target"], "TARGET"
                    elif bars_held >= self.max_bars:
                        ep, er = cl, "TIME"
                else:
                    if hi >= position["stop"]:
                        ep, er = position["stop"], "BE_STOP" if position.get("be_moved") else "STOP"
                    elif lo <= position["target"]:
                        ep, er = position["target"], "TARGET"
                    elif bars_held >= self.max_bars:
                        ep, er = cl, "TIME"

                # EOD momentum: force-close at the end of the 3pm-4pm ET bar (hour 20 UTC)
                # This implements the Baltussen "final period" exit — never hold overnight
                if ep is None and position.get("strategy") == "eod_momentum":
                    if int(hour_utc[i]) == 20:
                        ep, er = cl, "EOD"

                if ep is not None:
                    entry = position["entry_price"]
                    risk  = position["risk"]
                    pnl   = (ep - entry) if d == "BUY" else (entry - ep)
                    trades.append(BacktestTrade(
                        strategy        = position["strategy"],
                        direction       = d,
                        entry_bar       = position["entry_bar"],
                        exit_bar        = i,
                        entry_price     = entry,
                        exit_price      = round(ep, 4),
                        stop_loss       = position["orig_stop"],
                        take_profit     = position["target"],
                        exit_reason     = er,
                        pnl_pts         = round(pnl, 4),
                        r_multiple      = round(pnl / risk, 3) if risk > 0 else 0.0,
                        bars_held       = bars_held,
                        composite_score = position["score"],
                        regime          = position["regime"],
                        be_moved        = position.get("be_moved", False),
                    ))
                    equity        += pnl
                    eq_curve.append(round(equity, 4))
                    position       = None
                    last_exit_bar  = i   # reset global cooldown
                continue

            # ── RTH session filter ───────────────────────────────────── #
            # Only enter trades during RTH for futures 1h to avoid overnight noise
            if self.rth_only:
                curr_hour = int(hour_utc[i])
                next_hour = int(hour_utc[i+1]) if i+1 < n else curr_hour
                if curr_hour not in self.RTH_HOURS or next_hour not in self.RTH_HOURS:
                    continue

            # ── Global post-trade cooldown ───────────────────────────── #
            # Prevents lower-quality strategies (ema21) from firing in the
            # tail of an ema55 trade where conditions are likely still poor
            if i - last_exit_bar < self.GLOBAL_COOLDOWN:
                continue

            # ── Regime detection ─────────────────────────────────────── #
            regime = self._regime(i, adx, dmp, dmn)

            # ── Entry scoring ───────────────────────────────────────── #
            best_buy  = (0.0, "none", "")
            best_sell = (0.0, "none", "")

            for strat in self.STRATEGIES:
                if strat == "eod_momentum":
                    continue  # handled separately below (needs session state)
                if i - last_signal.get(strat, -99) < cooldowns[strat]:
                    continue

                b, s = self._score(strat, i, regime,
                                   closes, opens, highs, lows, volumes,
                                   atrs, rsi, adx, dmp, dmn, ema8, ema21, ema55, ema200,
                                   vwap, vol_avg, bb_up, bb_lo,
                                   pdh, pdl, is_nr4)
                if b > best_buy[0]:  best_buy  = (b, strat, regime)
                if s > best_sell[0]: best_sell = (s, strat, regime)

            # ── EOD momentum (Baltussen et al. 2021) ────────────────── #
            # Fires on the bar BEFORE the final RTH hour so entry is at
            # the 3pm ET open. Session return from 9am open to 2pm close
            # positively predicts the final hour — structural LETF effect.
            if "eod_momentum" in self.STRATEGIES:
                if i - last_signal.get("eod_momentum", -99) >= cooldowns["eod_momentum"]:
                    eb, es = self._eod_momentum(
                        i, closes, opens, vwap, rsi, adx,
                        int(hour_utc[i]), session_open_price,
                    )
                    if eb > best_buy[0]:  best_buy  = (eb, "eod_momentum", regime)
                    if es > best_sell[0]: best_sell = (es, "eod_momentum", regime)

            if best_buy[0] >= self.min_score and best_buy[0] >= best_sell[0]:
                direction, score_used, strat_name = "BUY",  best_buy[0],  best_buy[1]
            elif best_sell[0] >= self.min_score and self.allow_short:
                direction, score_used, strat_name = "SELL", best_sell[0], best_sell[1]
            else:
                continue

            entry    = opens[i + 1]
            atr_v    = atrs[i] if not np.isnan(atrs[i]) else abs(closes[i] - closes[i-1])
            atr_v    = max(atr_v, closes[i] * 0.001)
            risk     = self.atr_stop * atr_v
            rr_used  = self.STRATEGY_RR.get(strat_name) or self.rr_ratio
            stop     = (entry - risk)              if direction == "BUY" else (entry + risk)
            target   = (entry + rr_used * risk)    if direction == "BUY" else (entry - rr_used * risk)

            last_signal[strat_name] = i
            position = dict(
                direction   = direction,
                strategy    = strat_name,
                entry_bar   = i + 1,
                entry_price = round(entry, 4),
                stop        = round(stop, 4),
                orig_stop   = round(stop, 4),
                target      = round(target, 4),
                risk        = risk,
                atr_entry   = atr_v,
                score       = round(score_used, 3),
                regime      = regime,
                be_moved    = False,
            )

        return BacktestResult(
            symbol=symbol, timeframe=timeframe, market=market,
            total_bars=n, trades=trades, equity_curve=eq_curve,
        )

    # ── Regime detection ───────────────────────────────────────────── #

    def _regime(self, i: int, adx: np.ndarray, dmp: np.ndarray, dmn: np.ndarray) -> str:
        if i < 5:
            return "neutral"
        av  = adx[i]   if not np.isnan(adx[i])   else 0.0
        av3 = adx[i-3] if not np.isnan(adx[i-3]) else av

        if av >= self.min_adx:
            return "trending" if av > av3 else "neutral"
        elif av < 18.0:
            return "ranging" if av <= av3 else "neutral"
        return "neutral"

    # ── Strategy dispatcher ────────────────────────────────────────── #

    def _score(
        self, strat: str, i: int, regime: str,
        closes, opens, highs, lows, volumes,
        atrs, rsi, adx, dmp, dmn, ema8, ema21, ema55, ema200,
        vwap, vol_avg, bb_up, bb_lo,
        pdh=None, pdl=None, is_nr4=None,
    ) -> Tuple[float, float]:
        """Route to the correct strategy scorer with regime gating."""

        if strat == "ema55_bounce":
            # Trending regime only
            if regime != "trending":
                return 0.0, 0.0
            return self._ema55_bounce(i, closes, highs, lows, ema21, ema55, ema200, adx, dmp, dmn)

        if strat == "ema21_bounce":
            # Trending regime only — shallow pullback complement to ema55
            if regime != "trending":
                return 0.0, 0.0
            return self._ema21_bounce(i, closes, highs, lows, ema21, ema55, ema200, adx, dmp, dmn)

        if strat == "pdhl_breakout":
            # Trending + neutral regimes — NR4 compression breakout
            # (ranging excluded: mean-reversion conditions, not breakout)
            if regime == "ranging":
                return 0.0, 0.0
            if pdh is None or pdl is None or is_nr4 is None:
                return 0.0, 0.0
            return self._pdhl_breakout(
                i, closes, opens, highs, lows, volumes, vol_avg,
                adx, dmp, dmn, ema21, ema55, pdh, pdl, is_nr4,
            )

        return 0.0, 0.0

    # ─────────────────────────────────────────────────────────────────── #
    # STRATEGY 1: EMA55 BOUNCE — DEEP PULLBACK IN TREND                   #
    # ─────────────────────────────────────────────────────────────────── #
    # Evidence: 3yr backtest ES 1h RTH → PF 1.838, WR 47.6%, 42 trades   #
    # Consistent across 2024 (PF 2.22), 2025 (PF 1.48), 2026 (PF 2.0)  #
    #                                                                      #
    # When EMA55 acts as support in a confirmed uptrend (EMA55 > EMA200), #
    # and price pulls back to touch EMA55 then bounces, institutional      #
    # cost-of-carry dynamics create genuine mean-reversion to trend.       #
    #                                                                      #
    # ALL conditions must be true (binary gate — no partial credit):       #
    # 1. Low went below EMA55 × 1.002 in last 7 bars (the pullback hit)   #
    # 2. Current close > EMA55 AND > EMA21 (bounce confirmed above both)  #
    # 3. Current bar bullish: close > prior close                          #
    # 4. EMA55 > EMA200 × 1.001 (medium trend above long trend)           #
    # 5. ADX ≥ min_adx AND ADX slope rising (trend strength growing)      #
    # 6. DI+ > DI- (directional movement confirms uptrend)                #
    # ─────────────────────────────────────────────────────────────────── #

    def _ema55_bounce(self, i, closes, highs, lows, ema21, ema55, ema200,
                     adx, dmp, dmn) -> Tuple[float, float]:
        if i < self.warmup:
            return 0.0, 0.0

        c  = closes[i]
        c1 = closes[i-1]

        e21  = ema21[i]  if not np.isnan(ema21[i])  else c
        e55  = ema55[i]  if not np.isnan(ema55[i])  else c
        e200 = ema200[i] if not np.isnan(ema200[i]) else c

        _adx = adx[i] if not np.isnan(adx[i]) else 0.0
        adx3 = adx[i-3] if i >= 3 and not np.isnan(adx[i-3]) else _adx
        _dmp = dmp[i] if not np.isnan(dmp[i]) else 0.0
        _dmn = dmn[i] if not np.isnan(dmn[i]) else 0.0

        # ═══ BULL SETUP ═══════════════════════════════════════════════ #

        # Gate 1: Recent low touched EMA55 zone (last 7 bars)
        lb7_bull = any(lows[i-k] < ema55[i-k] * 1.002
                       for k in range(1, 8)
                       if i-k >= 0 and not np.isnan(ema55[i-k]))

        # Gate 2: Current bar bounced back above BOTH EMA55 and EMA21
        above_both = c > e55 and c > e21

        # Gate 3: Bullish bar
        bull_bar = c > c1

        # Gate 4: EMA55 > EMA200 (medium-term trend in place)
        trend_intact = e55 > e200 * 1.001

        # Gate 5: ADX trending and strengthening
        adx_ok = _adx >= self.min_adx and _adx > adx3

        # Gate 6: DI+ confirms uptrend direction
        di_bull = _dmp > _dmn

        buy = 0.80 if (lb7_bull and above_both and bull_bar and trend_intact and adx_ok and di_bull) else 0.0

        # Short side disabled — backtesting shows 23% WR in bull-dominant periods.
        # EMA55 < EMA200 fires briefly during corrections but the broader trend
        # reverses, making these shorts consistently losing. Re-enable only after
        # validating against 2022 bear year data in isolation.
        return buy, 0.0

    # ─────────────────────────────────────────────────────────────────── #
    # STRATEGY 2: BB REVERSION — MEAN REVERSION IN RANGING MARKETS        #
    # ─────────────────────────────────────────────────────────────────── #
    # Regime: ranging (ADX < 18, flat/falling) only                       #
    # RR: 1.5:1 — closer target hits more often, typical for mean rev.    #
    # Research: Bookmap VWAP 2SD study, institutional algo execution data #
    # confirms mean-reversion edge at BB extreme in low-ADX environments  #
    #                                                                      #
    # ALL gates must be true:                                              #
    # LONG:                                                                #
    # 1. Close ≤ lower Bollinger Band (price at statistical extreme)       #
    # 2. Previous bar also at/below lower BB (sustained, not a 1-bar spike)#
    # 3. RSI < 38 (oversold confirmation)                                  #
    # 4. Bullish reversal candle: close > open (buyers stepping in)        #
    # 5. Close > prior close (momentum turning)                            #
    # SHORT: exact mirror — close ≥ upper BB, RSI > 62, bearish candle    #
    # ─────────────────────────────────────────────────────────────────── #

    def _bb_reversion(self, i, closes, opens, highs, lows,
                      rsi, bb_up, bb_lo, adx) -> Tuple[float, float]:
        if i < self.warmup:
            return 0.0, 0.0

        c   = closes[i];   o  = opens[i]
        c1  = closes[i-1]; o1 = opens[i-1]

        _bbl = bb_lo[i] if not np.isnan(bb_lo[i]) else np.nan
        _bbu = bb_up[i] if not np.isnan(bb_up[i]) else np.nan
        _bbl1 = bb_lo[i-1] if not np.isnan(bb_lo[i-1]) else np.nan
        _bbu1 = bb_up[i-1] if not np.isnan(bb_up[i-1]) else np.nan

        if np.isnan(_bbl) or np.isnan(_bbu):
            return 0.0, 0.0

        _rsi = rsi[i] if not np.isnan(rsi[i]) else 50.0

        buy = sell = 0.0

        # ═══ LONG SETUP (price at lower BB extreme) ═══════════════════ #
        at_lower_bb     = c <= _bbl * 1.001          # at or below lower band
        sustained_low   = (not np.isnan(_bbl1)) and c1 <= _bbl1 * 1.002  # prev bar also extreme
        rsi_oversold    = _rsi < 38
        bull_candle     = c > o                      # close above open (green candle)
        momentum_up     = c > c1                     # turning higher vs prior close

        if at_lower_bb and sustained_low and rsi_oversold and bull_candle and momentum_up:
            buy = 0.75
            if _rsi < 30:     buy = 0.80             # deeper oversold = stronger signal
            if c > o * 1.001: buy = min(buy + 0.05, 0.85)  # strong green body

        # ═══ SHORT SETUP (price at upper BB extreme) ══════════════════ #
        at_upper_bb     = c >= _bbu * 0.999
        sustained_high  = (not np.isnan(_bbu1)) and c1 >= _bbu1 * 0.998
        rsi_overbought  = _rsi > 62
        bear_candle     = c < o
        momentum_down   = c < c1

        if at_upper_bb and sustained_high and rsi_overbought and bear_candle and momentum_down:
            sell = 0.75
            if _rsi > 70:     sell = 0.80
            if c < o * 0.999: sell = min(sell + 0.05, 0.85)

        return buy, sell

    # ─────────────────────────────────────────────────────────────────── #
    # STRATEGY 3: PRIOR DAY HIGH/LOW BREAKOUT — COMPRESSION → EXPANSION   #
    # ─────────────────────────────────────────────────────────────────── #
    # Regime: neutral (ADX 15-25, transitioning from ranging to trending)  #
    # RR: 2:1 — breakouts that confirm run further than mean reversions    #
    # Research: Crabel (1990) — NR4/NR7 filter is the most important gate #
    # ORB Setups data: 55-62% WR on ES with NR day qualifier               #
    #                                                                      #
    # ALL gates must be true:                                              #
    # 1. Prior session was NR4 day (narrowest range of last 4 days)        #
    # 2. Close breaks ABOVE prior day high (long) / BELOW prior day low    #
    # 3. ADX ≥ 15 AND rising (momentum building from compression)          #
    # 4. DI+ > DI- for longs / DI- > DI+ for shorts                       #
    # 5. Close > EMA21 for longs (confirms upside momentum)                #
    # 6. Volume above average (breakouts require participation)            #
    # ─────────────────────────────────────────────────────────────────── #

    def _pdhl_breakout(self, i, closes, opens, highs, lows, volumes, vol_avg,
                       adx, dmp, dmn, ema21, ema55,
                       pdh, pdl, is_nr4) -> Tuple[float, float]:
        if i < self.warmup:
            return 0.0, 0.0

        c  = closes[i]
        c1 = closes[i-1]

        _pdh = pdh[i]; _pdl = pdl[i]
        if np.isnan(_pdh) or np.isnan(_pdl):
            return 0.0, 0.0

        # Gate 1: prior session was NR4 (compression prerequisite)
        if not is_nr4[i]:
            return 0.0, 0.0

        _adx  = adx[i]  if not np.isnan(adx[i])  else 0.0
        _adx3 = adx[i-3] if i >= 3 and not np.isnan(adx[i-3]) else _adx
        _dmp  = dmp[i]  if not np.isnan(dmp[i])  else 0.0
        _dmn  = dmn[i]  if not np.isnan(dmn[i])  else 0.0
        e21   = ema21[i] if not np.isnan(ema21[i]) else c
        e55   = ema55[i] if not np.isnan(ema55[i]) else c

        # Gate 3: ADX ≥ 14 and rising (expansion just starting from compression)
        adx_expanding = _adx >= 14.0 and _adx > _adx3

        # Gate 6: volume above average
        _vol    = volumes[i] if not np.isnan(volumes[i]) else 0.0
        _volavg = vol_avg[i] if not np.isnan(vol_avg[i]) else _vol
        vol_ok  = _vol >= _volavg * 0.9

        buy = sell = 0.0

        # ═══ LONG: close above prior day high ═════════════════════════ #
        broke_above = c > _pdh and c1 <= _pdh   # fresh breakout this bar
        di_bull     = _dmp > _dmn
        above_ema21 = c > e21

        if broke_above and adx_expanding and di_bull and above_ema21 and vol_ok:
            buy = 0.75
            if c > e55:                buy = min(buy + 0.05, 0.85)  # macro tailwind
            if _adx > _adx3 * 1.05:   buy = min(buy + 0.05, 0.85)  # ADX accelerating

        # ═══ SHORT: close below prior day low ═════════════════════════ #
        broke_below = c < _pdl and c1 >= _pdl
        di_bear     = _dmn > _dmp
        below_ema21 = c < e21

        if broke_below and adx_expanding and di_bear and below_ema21 and vol_ok:
            sell = 0.75
            if c < e55:                sell = min(sell + 0.05, 0.85)
            if _adx > _adx3 * 1.05:   sell = min(sell + 0.05, 0.85)

        return buy, sell

    # ─────────────────────────────────────────────────────────────────── #
    # LEGACY STRATEGY 2: EMA21 BOUNCE — SHALLOW PULLBACK IN ESTABLISHED TREND    #
    # ─────────────────────────────────────────────────────────────────── #
    # Evidence: Combined with EMA55, 3yr PF 1.729 | 77 trades | WR 48%   #
    # EMA21 bounce alone: PF 1.709, WR 50% (with spread ≥ 0.3% filter)  #
    #                                                                      #
    # KEY FILTER: EMA21/EMA55 spread must be ≥ 0.3%                       #
    # Rationale: If EMA21 is close to EMA55, the pullback risks going all  #
    # the way to EMA55 making this a weaker signal. Spread ≥ 0.3% means   #
    # EMA21 is a DISTINCT support level worth trading from.                #
    #                                                                      #
    # ALL conditions must be true (binary gate):                           #
    # 1. EMA21/EMA55 spread ≥ 0.3% (distinct levels, not clustered)      #
    # 2. Low went below EMA21 × 1.001 in last 5 bars                     #
    # 3. Current close > EMA21 (bounced back above)                       #
    # 4. EMA21 > EMA55 × 1.001 (trend alignment)                          #
    # 5. Current bar bullish: close > prior close                          #
    # 6. ADX ≥ min_adx AND rising                                          #
    # 7. DI+ > DI-                                                         #
    # 8. Close > EMA200 (macro uptrend)                                    #
    # ─────────────────────────────────────────────────────────────────── #

    def _ema21_bounce(self, i, closes, highs, lows, ema21, ema55, ema200,
                     adx, dmp, dmn) -> Tuple[float, float]:
        if i < self.warmup:
            return 0.0, 0.0

        c  = closes[i]
        c1 = closes[i-1]

        e21  = ema21[i]  if not np.isnan(ema21[i])  else c
        e55  = ema55[i]  if not np.isnan(ema55[i])  else c
        e200 = ema200[i] if not np.isnan(ema200[i]) else c

        _adx = adx[i] if not np.isnan(adx[i]) else 0.0
        adx3 = adx[i-3] if i >= 3 and not np.isnan(adx[i-3]) else _adx
        _dmp = dmp[i] if not np.isnan(dmp[i]) else 0.0
        _dmn = dmn[i] if not np.isnan(dmn[i]) else 0.0

        # ═══ BULL SETUP ═══════════════════════════════════════════════ #

        # Gate 1: EMA21/EMA55 spread ≥ 0.3% (distinct levels)
        ema_spread_bull = (e21 / e55 - 1.0) >= 0.003

        # Gate 2: Recent low touched EMA21 zone
        lb5_bull = any(lows[i-k] < ema21[i-k] * 1.001
                       for k in range(1, 6)
                       if i-k >= 0 and not np.isnan(ema21[i-k]))

        # Gate 3: Current bar bounced above EMA21
        above_ema21 = c > e21

        # Gate 4: EMA21 above EMA55 (trend aligned)
        trend_bull = e21 > e55 * 1.001

        # Gate 5: Bullish bar
        bull_bar = c > c1

        # Gate 6: ADX trending and strengthening
        adx_ok = _adx >= self.min_adx and _adx > adx3

        # Gate 7: DI+ confirms uptrend
        di_bull = _dmp > _dmn

        # Gate 8: Above macro trend (EMA200)
        macro_bull = c > e200

        buy = 0.75 if (ema_spread_bull and lb5_bull and above_ema21 and trend_bull
                       and bull_bar and adx_ok and di_bull and macro_bull) else 0.0

        # Short side disabled — not validated in bull-dominant conditions.
        return buy, 0.0

    # ─────────────────────────────────────────────────────────────────── #
    # STRATEGY 3: INSIDE BAR BREAKOUT                                     #
    # ─────────────────────────────────────────────────────────────────── #
    # Volatility compression (inside bar) precedes directional breakout.  #
    # EMA21 > EMA55 provides trend bias for direction.                    #
    # ─────────────────────────────────────────────────────────────────── #

    def _inside_bar(self, i, closes, highs, lows, volumes,
                   atrs, adx, dmp, dmn, ema21, ema55, vol_avg) -> Tuple[float, float]:
        if i < 25:
            return 0.0, 0.0

        h  = highs[i];  l  = lows[i]
        h1 = highs[i-1]; l1 = lows[i-1]
        c  = closes[i]

        if h >= h1 or l <= l1:
            return 0.0, 0.0

        prior_range  = h1 - l1
        inside_range = h - l

        if prior_range <= 0 or inside_range / prior_range > 0.65:
            return 0.0, 0.0

        e21  = ema21[i] if not np.isnan(ema21[i]) else c
        e55  = ema55[i] if not np.isnan(ema55[i]) else c
        _adx = adx[i]   if not np.isnan(adx[i])   else 0.0
        _dmp = dmp[i]   if not np.isnan(dmp[i])   else 0.0
        _dmn = dmn[i]   if not np.isnan(dmn[i])   else 0.0

        if _adx < 15:
            return 0.0, 0.0

        bar_pos = (c - l) / inside_range if inside_range > 0 else 0.5

        buy = sell = 0.0

        if e21 > e55 * 1.001:
            if bar_pos >= 0.55:
                buy = 0.65
                if inside_range / prior_range < 0.50: buy += 0.10
                if bar_pos >= 0.70:                   buy += 0.10
                if _dmp > _dmn:                       buy += 0.08
                if _adx >= 20:                        buy += 0.07

        if e21 < e55 * 0.999:
            if bar_pos <= 0.45:
                sell = 0.65
                if inside_range / prior_range < 0.50: sell += 0.10
                if bar_pos <= 0.30:                   sell += 0.10
                if _dmn > _dmp:                       sell += 0.08
                if _adx >= 20:                        sell += 0.07

        return min(buy, 1.0), min(sell, 1.0)

    # ─────────────────────────────────────────────────────────────────── #
    # STRATEGY 4: EOD MOMENTUM — FINAL-HOUR STRUCTURAL EDGE               #
    # ─────────────────────────────────────────────────────────────────── #
    # Source: Baltussen, Da, Lammers, Martens (2021) J. Financial         #
    # Economics — "Hedging Demand and Market Intraday Momentum"           #
    # 45-year study across 60+ futures markets confirms: the rest-of-day  #
    # return positively predicts the final period return.                  #
    # Mechanism: LETF rebalancers and gamma hedgers create directional     #
    # pressure in the final session hour that mirrors the day's direction. #
    #                                                                      #
    # Signal: fires on the 19 UTC bar (2pm-3pm ET) → enters at 3pm open  #
    # Exit: force-closed at close of 20 UTC bar (4pm ET bell) — never     #
    # held overnight. Stop = 1 ATR as normal.                              #
    #                                                                      #
    # Gates:                                                               #
    # 1. Must be the 19 UTC bar (one bar before the final RTH bar)        #
    # 2. Session return (9am open → 2pm close) > +0.15% for longs        #
    # 3. Price above session VWAP (confirms institutional bias)            #
    # 4. ADX > 15 (at least mild directional structure)                   #
    # ─────────────────────────────────────────────────────────────────── #

    def _eod_momentum(
        self,
        i: int,
        closes: np.ndarray,
        opens: np.ndarray,
        vwap: np.ndarray,
        rsi: np.ndarray,
        adx: np.ndarray,
        hour: int,
        session_open: float,
    ) -> Tuple[float, float]:
        # Only fires on the bar BEFORE the last RTH bar (19 UTC = 2pm-3pm ET)
        # so entry is at the open of hour 20 (3pm ET)
        if hour != 19:
            return 0.0, 0.0
        if np.isnan(session_open) or session_open <= 0:
            return 0.0, 0.0
        if i < 5:
            return 0.0, 0.0

        c  = closes[i]
        vw = vwap[i] if not np.isnan(vwap[i]) and vwap[i] > 0 else float("nan")
        _adx = adx[i] if not np.isnan(adx[i]) else 0.0

        session_return = (c - session_open) / session_open

        # Require at least mild directional structure
        if _adx < 15:
            return 0.0, 0.0

        buy = sell = 0.0

        # LONG: session trending up → final hour should continue upward
        if session_return > 0.0015:      # +0.15% minimum day return
            buy = 0.70
            if not np.isnan(vw) and c > vw:
                buy = 0.80               # above VWAP confirms institutional long bias
            if session_return > 0.005:   # strong day (>0.5%) gets max conviction
                buy = min(buy + 0.05, 0.90)

        # SHORT: session trending down → final hour should continue downward
        if session_return < -0.0015:
            sell = 0.70
            if not np.isnan(vw) and c < vw:
                sell = 0.80
            if session_return < -0.005:
                sell = min(sell + 0.05, 0.90)

        return buy, sell

    # ─────────────────────────────────────────────────────────────────── #
    # STRATEGY 4: VWAP MEAN REVERSION (ranging only)                      #
    # ─────────────────────────────────────────────────────────────────── #

    def _vwap_reversion(self, i, closes, opens, highs, lows, vwap, rsi, atrs, adx, volumes, vol_avg, bb_up, bb_lo) -> Tuple[float, float]:
        if i < 5:
            return 0.0, 0.0

        vw = vwap[i]
        if np.isnan(vw) or vw <= 0:
            return 0.0, 0.0

        c    = closes[i]; o = opens[i]
        _atr = atrs[i]  if not np.isnan(atrs[i])  else abs(c - closes[i-1])
        _atr = max(_atr, c * 0.001)
        _rsi = rsi[i]   if not np.isnan(rsi[i])   else 50.0
        _bbl = bb_lo[i] if not np.isnan(bb_lo[i]) else c * 0.98
        _bbu = bb_up[i] if not np.isnan(bb_up[i]) else c * 1.02

        dev_pts = c - vw
        dev_atr = abs(dev_pts) / _atr

        if dev_atr < 2.0:
            return 0.0, 0.0

        buy = sell = 0.0

        if dev_pts < 0:
            if dev_atr >= 3.0:    buy += 0.28
            elif dev_atr >= 2.5:  buy += 0.22
            else:                 buy += 0.15

            # Research (bookmap): fade at 2SD VWAP deviation in ranging market.
            # RSI gate relaxed from 25→40 range; still filters directionally neutral bars.
            if _rsi < 35:         buy += 0.30
            elif _rsi < 40:       buy += 0.22
            elif _rsi < 48:       buy += 0.12
            else: return 0.0, 0.0

            if c <= _bbl * 1.002: buy += 0.20
            if c > o:             buy += 0.12
            if c > closes[i-1]:   buy += 0.10

        if dev_pts > 0:
            if dev_atr >= 3.0:    sell += 0.28
            elif dev_atr >= 2.5:  sell += 0.22
            else:                 sell += 0.15

            if _rsi > 65:         sell += 0.30
            elif _rsi > 60:       sell += 0.22
            elif _rsi > 55:       sell += 0.12
            else: return 0.0, 0.0

            if c >= _bbu * 0.998: sell += 0.20
            if c < o:             sell += 0.12
            if c < closes[i-1]:   sell += 0.10

        return min(buy, 1.0), min(sell, 1.0)

    # ──────────────────────────────────────────────────────────────────────────── #
    # LIVE SIGNAL — evaluate the most recent complete bar for a live entry          #
    # ──────────────────────────────────────────────────────────────────────────── #

    def live_signal(
        self,
        df_raw: pd.DataFrame,
        current_price: float | None = None,
    ) -> "dict | None":
        """
        Evaluate the most recently completed 1h bar for a live trading signal.

        Returns None if no strategy fires.
        Returns a dict with all signal details (entry, stop, target, reasoning) if
        one of the three validated strategies fires on the last bar.

        Args:
            df_raw:         Historical OHLCV DataFrame (same format as run()).
                            Should cover at least warmup + 10 bars (≥ 210 bars
                            for the default warmup=200).
            current_price:  Optional live price to use as the approximate entry
                            (next bar open).  Falls back to last close if None.

        Signal bar  = df.iloc[-1]   (most recent COMPLETED bar).
        Entry       ≈ current_price (the next 1h bar hasn't formed yet).
        """
        df = add_all(df_raw.copy())
        n  = len(df)
        if n < self.warmup + 10:
            return None

        def _col(name: str) -> np.ndarray:
            return df[name].values if name in df.columns else np.full(n, np.nan)

        closes   = df["close"].values
        opens    = df["open"].values
        highs    = df["high"].values
        lows     = df["low"].values
        volumes  = df["volume"].values
        atrs     = _col("atr")
        rsi      = _col("rsi")
        adx      = _col("adx")
        dmp      = _col("dmp")
        dmn      = _col("dmn")
        ema8     = _col("ema_8")
        ema21    = _col("ema_21")
        ema55    = _col("ema_55")
        ema200   = _col("ema_200")
        vwap     = _col("vwap")
        vol_avg  = _col("vol_avg")
        bb_up    = _col("bb_upper")
        bb_lo    = _col("bb_lower")

        # RTH hour array
        if "timestamp" in df.columns:
            hour_utc = pd.to_datetime(df["timestamp"]).dt.hour.values
        else:
            hour_utc = np.full(n, 15)

        # ── Precompute PDH / PDL / NR4 (identical to run()) ─────────── #
        pdh    = np.full(n, np.nan)
        pdl    = np.full(n, np.nan)
        is_nr4 = np.zeros(n, dtype=bool)

        if "timestamp" in df.columns:
            dates        = pd.to_datetime(df["timestamp"]).dt.date.values
            unique_dates = sorted(set(dates))
            day_high: dict = {}
            day_low:  dict = {}
            for d in unique_dates:
                mask        = dates == d
                day_high[d] = highs[mask].max()
                day_low[d]  = lows[mask].min()
            day_range = {d: day_high[d] - day_low[d] for d in unique_dates}
            for idx in range(n):
                d     = dates[idx]
                d_pos = unique_dates.index(d)
                if d_pos >= 1:
                    prev_d   = unique_dates[d_pos - 1]
                    pdh[idx] = day_high[prev_d]
                    pdl[idx] = day_low[prev_d]
                    if d_pos >= 4:
                        prev_ranges = [day_range[unique_dates[d_pos - k]] for k in range(1, 5)]
                        is_nr4[idx] = prev_ranges[0] == min(prev_ranges)

        # ── Check signal on the last complete bar ────────────────────── #
        i = n - 1

        # RTH gate: current bar must be RTH, and next bar must also be RTH
        # (next bar ≈ current_hour + 1 for 1h data)
        if self.rth_only:
            curr_hour = int(hour_utc[i])
            next_hour = curr_hour + 1   # approximate; next 1h bar open
            if curr_hour not in self.RTH_HOURS or next_hour not in self.RTH_HOURS:
                return None

        regime = self._regime(i, adx, dmp, dmn)

        # Score all active strategies
        best_buy  = (0.0, "none")
        best_sell = (0.0, "none")

        for strat in self.STRATEGIES:
            b, s = self._score(
                strat, i, regime,
                closes, opens, highs, lows, volumes,
                atrs, rsi, adx, dmp, dmn, ema8, ema21, ema55, ema200,
                vwap, vol_avg, bb_up, bb_lo,
                pdh, pdl, is_nr4,
            )
            if b > best_buy[0]:  best_buy  = (b, strat)
            if s > best_sell[0]: best_sell = (s, strat)

        # Pick winning direction
        if best_buy[0] >= self.min_score and best_buy[0] >= best_sell[0]:
            direction, score, strat_name = "BUY",  best_buy[0],  best_buy[1]
        elif best_sell[0] >= self.min_score and self.allow_short:
            direction, score, strat_name = "SELL", best_sell[0], best_sell[1]
        else:
            return None

        # ── Compute levels ───────────────────────────────────────────── #
        entry   = float(current_price or closes[i])
        atr_v   = float(atrs[i]) if not np.isnan(atrs[i]) else abs(float(closes[i]) - float(closes[i - 1]))
        atr_v   = max(atr_v, entry * 0.001)
        risk    = self.atr_stop * atr_v
        rr_used = self.STRATEGY_RR.get(strat_name) or self.rr_ratio
        stop    = (entry - risk)           if direction == "BUY"  else (entry + risk)
        target  = (entry + rr_used * risk) if direction == "BUY"  else (entry - rr_used * risk)

        # ── Strategy readable labels ─────────────────────────────────── #
        _labels = {
            "ema55_bounce":  "EMA55 Bounce — deep pullback to key moving average in uptrend",
            "ema21_bounce":  "EMA21 Bounce — shallow pullback to fast moving average in uptrend",
            "pdhl_breakout": "PDHL Breakout — prior-day high/low compression breakout",
        }

        bar_time_str = ""
        if "timestamp" in df.columns:
            try:
                ts = pd.to_datetime(df["timestamp"].iloc[i])
                import pytz
                et = pytz.timezone("America/New_York")
                bar_time_str = ts.tz_convert(et).strftime("%I:%M %p ET") if ts.tzinfo else ts.strftime("%H:%M UTC")
            except Exception:
                bar_time_str = str(df["timestamp"].iloc[i])

        adx_val = float(adx[i]) if not np.isnan(adx[i]) else 0.0

        return {
            "direction":      direction,
            "strategy":       strat_name,
            "strategy_label": _labels.get(strat_name, strat_name),
            "regime":         regime,
            "entry":          round(entry, 2),
            "stop":           round(stop, 2),
            "target":         round(target, 2),
            "risk_pts":       round(risk, 2),
            "rr":             rr_used,
            "score":          round(score, 3),
            "adx":            round(adx_val, 1),
            "bar_time":       bar_time_str,
        }
