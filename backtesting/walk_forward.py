"""
Walk-Forward Backtest CLI
─────────────────────────
Runs all active engines over rolling train/validate windows and reports
per-window stats plus a summary of which engines hold up out-of-sample.

Usage:
    python3 -m backtesting.walk_forward --symbol ES=F --train 40 --validate 20 --step 5
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import pytz

# ── Engine imports ──────────────────────────────────────────────────────────── #
from backtesting.engine_aplus import BacktestEngineAPlus
from backtesting.engine_orb import ORBEngine
from backtesting.engine_fade_rip import FadeTheRipEngine
from backtesting.models import BacktestResult
from data.fetcher import fetch_historical

ET = pytz.timezone("America/New_York")

# ── Degradation thresholds ───────────────────────────────────────────────────── #
MIN_VALIDATE_EXPECTANCY = 0.10   # R
MIN_VALIDATE_PF         = 1.20

# ── Engine registry ──────────────────────────────────────────────────────────── #
ENGINE_MAP = {
    "aplus":    BacktestEngineAPlus,
    "orb":      ORBEngine,
    "fade_rip": FadeTheRipEngine,
}


# ─────────────────────────────────────────────────────────────────────────────── #
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────── #

def _to_et_date(ts: pd.Timestamp) -> date:
    """Convert a UTC-aware timestamp to an ET calendar date."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ET).date()


def slice_df(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Return rows whose ET date falls within [start, end] inclusive."""
    et_dates = df["timestamp"].apply(_to_et_date)
    mask = (et_dates >= start) & (et_dates <= end)
    return df.loc[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────── #
# Window generation
# ─────────────────────────────────────────────────────────────────────────────── #

@dataclass
class Window:
    index: int
    train_start: date
    train_end: date
    validate_start: date
    validate_end: date


def build_windows(
    data_start: date,
    train_days: int,
    validate_days: int,
    step_days: int,
    data_end: date,
) -> List[Window]:
    windows: List[Window] = []
    w = 0
    while True:
        train_start    = data_start + timedelta(days=w * step_days)
        train_end      = train_start + timedelta(days=train_days - 1)
        validate_start = train_end   + timedelta(days=1)
        validate_end   = validate_start + timedelta(days=validate_days - 1)
        if validate_end > data_end:
            break
        windows.append(Window(
            index          = w + 1,
            train_start    = train_start,
            train_end      = train_end,
            validate_start = validate_start,
            validate_end   = validate_end,
        ))
        w += 1
    return windows


# ─────────────────────────────────────────────────────────────────────────────── #
# Per-window result
# ─────────────────────────────────────────────────────────────────────────────── #

@dataclass
class WindowResult:
    window: Window
    engine_name: str
    train_result: Optional[BacktestResult]
    validate_result: Optional[BacktestResult]

    @property
    def train_trades(self) -> int:
        return self.train_result.total_trades if self.train_result else 0

    @property
    def validate_trades(self) -> int:
        return self.validate_result.total_trades if self.validate_result else 0

    @property
    def train_wr(self) -> float:
        return self.train_result.win_rate if self.train_result and self.train_result.total_trades else 0.0

    @property
    def validate_wr(self) -> float:
        return self.validate_result.win_rate if self.validate_result and self.validate_result.total_trades else 0.0

    @property
    def train_expectancy(self) -> float:
        return self.train_result.expectancy if self.train_result and self.train_result.total_trades else 0.0

    @property
    def validate_expectancy(self) -> float:
        return self.validate_result.expectancy if self.validate_result and self.validate_result.total_trades else 0.0

    @property
    def validate_pf(self) -> float:
        if not self.validate_result or not self.validate_result.total_trades:
            return 0.0
        pf = self.validate_result.profit_factor
        return pf if not math.isinf(pf) else 99.0

    @property
    def flag(self) -> str:
        """Return warning flag if validate stats are degraded."""
        if self.validate_trades == 0:
            return "? no trades"
        if self.validate_expectancy < 0:
            return "⚠ validate E<0"
        if self.validate_expectancy < MIN_VALIDATE_EXPECTANCY:
            return f"⚠ validate E<{MIN_VALIDATE_EXPECTANCY}R"
        if self.validate_pf < MIN_VALIDATE_PF:
            return f"⚠ validate PF<{MIN_VALIDATE_PF}"
        return "✓"

    @property
    def is_ok(self) -> bool:
        return self.flag == "✓"


# ─────────────────────────────────────────────────────────────────────────────── #
# Runner
# ─────────────────────────────────────────────────────────────────────────────── #

def run_engine_on_slice(
    engine_cls,
    df_slice: pd.DataFrame,
    symbol: str,
    timeframe: str = "5m",
) -> Optional[BacktestResult]:
    """Instantiate an engine and run it on a DataFrame slice. Returns None on error."""
    if df_slice.empty:
        return None
    try:
        engine = engine_cls()
        return engine.run(df_slice, symbol, timeframe)
    except Exception as exc:  # noqa: BLE001
        print(f"    [engine error: {exc}]", file=sys.stderr)
        return None


def run_walk_forward(
    symbol: str,
    train_days: int,
    validate_days: int,
    step_days: int,
    engine_names: List[str],
    min_trades: int,
    timeframe: str = "5m",
) -> List[WindowResult]:
    # Total calendar days needed (a generous buffer for market closed days)
    total_days_needed = train_days + validate_days + step_days * 20 + 10
    total_days_needed = max(total_days_needed, train_days + validate_days + 30)

    print(f"Fetching {total_days_needed} days of {timeframe} data for {symbol}…")
    df = fetch_historical(symbol, timeframe, days_back=total_days_needed)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Determine trading-day range from the data
    all_et_dates = sorted(set(df["timestamp"].apply(_to_et_date)))
    if len(all_et_dates) < train_days + validate_days:
        raise ValueError(
            f"Insufficient data: got {len(all_et_dates)} trading days, "
            f"need at least {train_days + validate_days}."
        )

    data_start = all_et_dates[0]
    data_end   = all_et_dates[-1]

    windows = build_windows(data_start, train_days, validate_days, step_days, data_end)
    if not windows:
        raise ValueError("No complete windows fit within the fetched data range.")

    print(f"Running {len(windows)} windows × {len(engine_names)} engines…\n")

    results: List[WindowResult] = []
    for w in windows:
        train_df    = slice_df(df, w.train_start,    w.train_end)
        validate_df = slice_df(df, w.validate_start, w.validate_end)

        for eng_name in engine_names:
            engine_cls = ENGINE_MAP[eng_name]

            train_res    = run_engine_on_slice(engine_cls, train_df,    symbol, timeframe)
            validate_res = run_engine_on_slice(engine_cls, validate_df, symbol, timeframe)

            # Skip windows that don't meet minimum trade threshold in validate
            if validate_res is not None and validate_res.total_trades < min_trades:
                validate_res = None  # mark as insufficient but keep the row

            results.append(WindowResult(
                window          = w,
                engine_name     = eng_name,
                train_result    = train_res,
                validate_result = validate_res,
            ))

    return results


# ─────────────────────────────────────────────────────────────────────────────── #
# Output
# ─────────────────────────────────────────────────────────────────────────────── #

def _fmt_e(val: float) -> str:
    return f"{val:+.1f}R"


def _fmt_pct(val: float) -> str:
    return f"{val:.0%}"


def _print_rich(
    results: List[WindowResult],
    symbol: str,
    train_days: int,
    validate_days: int,
    step_days: int,
) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        _RICH = True
    except ImportError:
        _RICH = False

    if not _RICH:
        _print_plain(results, symbol, train_days, validate_days, step_days)
        return

    console = Console()
    title = (
        f"Walk-Forward Report: [bold]{symbol}[/bold]  |  "
        f"Train=[cyan]{train_days}d[/cyan]  "
        f"Validate=[cyan]{validate_days}d[/cyan]  "
        f"Step=[cyan]{step_days}d[/cyan]"
    )
    console.print()
    console.rule(title)

    tbl = Table(box=box.SIMPLE_HEAD, show_footer=False)
    tbl.add_column("Window",  style="bold")
    tbl.add_column("Train Period")
    tbl.add_column("Validate Period")
    tbl.add_column("Engine",   style="cyan")
    tbl.add_column("Tr Trades", justify="right")
    tbl.add_column("Tr WR",     justify="right")
    tbl.add_column("Tr E",      justify="right")
    tbl.add_column("Va Trades", justify="right")
    tbl.add_column("Va WR",     justify="right")
    tbl.add_column("Va E",      justify="right")
    tbl.add_column("Flag")

    for r in results:
        w = r.window
        flag_style = "green" if r.is_ok else "yellow"
        tbl.add_row(
            f"W{w.index:02d}",
            f"{w.train_start} → {w.train_end}",
            f"{w.validate_start} → {w.validate_end}",
            r.engine_name,
            str(r.train_trades),
            _fmt_pct(r.train_wr),
            _fmt_e(r.train_expectancy),
            str(r.validate_trades) if r.validate_result else "—",
            _fmt_pct(r.validate_wr) if r.validate_result else "—",
            _fmt_e(r.validate_expectancy) if r.validate_result else "—",
            f"[{flag_style}]{r.flag}[/{flag_style}]",
        )

    console.print(tbl)
    _print_summary_rich(console, results)


def _print_summary_rich(console, results: List[WindowResult]) -> None:
    from rich.table import Table
    from rich import box

    console.rule("[bold]SUMMARY (validate windows)[/bold]")
    engine_names = sorted(set(r.engine_name for r in results))

    tbl = Table(box=box.SIMPLE_HEAD)
    tbl.add_column("Engine",    style="cyan bold")
    tbl.add_column("OK Windows")
    tbl.add_column("Avg Va E",  justify="right")
    tbl.add_column("Min Va E",  justify="right")
    tbl.add_column("Max Va E",  justify="right")
    tbl.add_column("Verdict")

    for eng in engine_names:
        eng_results = [r for r in results if r.engine_name == eng and r.validate_result]
        total = len(eng_results)
        if total == 0:
            tbl.add_row(eng, "0/0", "—", "—", "—", "[yellow]NO DATA[/yellow]")
            continue

        ok_count   = sum(1 for r in eng_results if r.is_ok)
        pass_rate  = ok_count / total
        expectancies = [r.validate_expectancy for r in eng_results]
        avg_e = sum(expectancies) / len(expectancies)
        min_e = min(expectancies)
        max_e = max(expectancies)

        verdict_style = "green" if pass_rate >= 0.70 else "red"
        verdict_text  = "OK" if pass_rate >= 0.70 else "→ CAUTION: <70% pass rate"

        tbl.add_row(
            eng,
            f"{ok_count}/{total}",
            _fmt_e(avg_e),
            _fmt_e(min_e),
            _fmt_e(max_e),
            f"[{verdict_style}]{verdict_text}[/{verdict_style}]",
        )

    console.print(tbl)
    console.print()


def _print_plain(
    results: List[WindowResult],
    symbol: str,
    train_days: int,
    validate_days: int,
    step_days: int,
) -> None:
    SEP = "=" * 100
    HDR = "-" * 100
    print()
    print(SEP)
    print(
        f"Walk-Forward Report: {symbol}  |  "
        f"Train={train_days}d  Validate={validate_days}d  Step={step_days}d"
    )
    print(SEP)
    hdr = (
        f"{'Window':<8} {'Train Period':<25} {'Engine':<12}"
        f"{'Tr Trades':>10} {'Tr WR':>7} {'Tr E':>8}"
        f"{'Va Trades':>10} {'Va WR':>7} {'Va E':>8}  Flag"
    )
    print(hdr)
    print(HDR)

    for r in results:
        w = r.window
        train_period    = f"{w.train_start} → {w.train_end}"
        validate_period = f"{w.validate_start} → {w.validate_end}"  # noqa: F841
        va_trades = str(r.validate_trades) if r.validate_result else "—"
        va_wr     = _fmt_pct(r.validate_wr) if r.validate_result else "—"
        va_e      = _fmt_e(r.validate_expectancy) if r.validate_result else "—"
        print(
            f"W{w.index:02d}     {train_period:<25} {r.engine_name:<12}"
            f"{r.train_trades:>10} {_fmt_pct(r.train_wr):>7} {_fmt_e(r.train_expectancy):>8}"
            f"{va_trades:>10} {va_wr:>7} {va_e:>8}  {r.flag}"
        )

    print(SEP)
    print("SUMMARY (validate windows):")
    engine_names = sorted(set(r.engine_name for r in results))
    for eng in engine_names:
        eng_results = [r for r in results if r.engine_name == eng and r.validate_result]
        total = len(eng_results)
        if total == 0:
            print(f"  {eng:<12}: no data")
            continue
        ok_count   = sum(1 for r in eng_results if r.is_ok)
        pass_rate  = ok_count / total
        expectancies = [r.validate_expectancy for r in eng_results]
        avg_e = sum(expectancies) / len(expectancies)
        min_e = min(expectancies)
        max_e = max(expectancies)
        caution = "  → CAUTION: <70% pass rate" if pass_rate < 0.70 else ""
        print(
            f"  {eng:<12}: {ok_count}/{total} windows OK  |  "
            f"avg validate E={_fmt_e(avg_e)}  |  "
            f"min={_fmt_e(min_e)}  max={_fmt_e(max_e)}{caution}"
        )
    print(SEP)
    print()


# ─────────────────────────────────────────────────────────────────────────────── #
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────── #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward backtest across all active engines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",     default="ES=F",              help="Ticker symbol")
    p.add_argument("--train",      type=int, default=40,         help="Training window in calendar days")
    p.add_argument("--validate",   type=int, default=20,         help="Validation window in calendar days")
    p.add_argument("--step",       type=int, default=5,          help="Days to slide each iteration")
    p.add_argument("--engines",    default="aplus,fade_rip,orb", help="Comma-separated engine list")
    p.add_argument("--min-trades", type=int, default=3,          help="Min validate trades to include window")
    p.add_argument("--timeframe",  default="5m",                 help="Bar timeframe (passed to fetcher + engines)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engine_names if e not in ENGINE_MAP]
    if unknown:
        print(f"Unknown engine(s): {unknown}. Valid: {sorted(ENGINE_MAP)}", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_walk_forward(
            symbol       = args.symbol,
            train_days   = args.train,
            validate_days= args.validate,
            step_days    = args.step,
            engine_names = engine_names,
            min_trades   = args.min_trades,
            timeframe    = args.timeframe,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Prefer rich output; fall back to plain text automatically
    _print_rich(results, args.symbol, args.train, args.validate, args.step)


if __name__ == "__main__":
    main()
