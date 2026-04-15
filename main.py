"""
Main trading bot orchestrator.
Fetches data → runs strategies → AI decision → paper execute.

Usage:
    python main.py                    # analyze ES with default settings
    python main.py --symbol NQ --tf 5m
    python main.py --symbol AAPL --market equities
"""
import argparse
import sys
from dataclasses import asdict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Initialize DB on first run
from data.store import init_db
init_db()

from data.fetcher import fetch_historical
from data.store import upsert_ohlcv
from strategies.aggregator import SignalAggregator
from risk.manager import RiskManager, TradeRequest
from execution.paper import PaperBroker
from ai.agent import TradingAgent
from config import settings

console = Console()


def run_analysis(symbol: str, timeframe: str, market: str, days_back: int = 30):
    console.rule(f"[bold cyan]{symbol} {timeframe} — {market.upper()}")

    # 1. Fetch data
    console.print(f"[dim]Fetching {days_back} days of {timeframe} data for {symbol}...[/dim]")
    try:
        df = fetch_historical(symbol, timeframe, days_back)
        upsert_ohlcv(df, symbol, timeframe)
        console.print(f"[green]✓ {len(df)} bars loaded[/green]")
    except Exception as e:
        console.print(f"[red]✗ Data fetch failed: {e}[/red]")
        return

    # 2. Run strategy engine
    aggregator = SignalAggregator(market=market)
    agg = aggregator.run(df, symbol=symbol, timeframe=timeframe)

    # Print signal table
    table = Table(title="Strategy Signals", show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Direction", style="bold")
    table.add_column("Confidence")
    table.add_column("Reasoning", max_width=60)

    for sig in agg.individual_signals:
        color = "green" if sig.direction == "BUY" else "red" if sig.direction == "SELL" else "dim"
        table.add_row(
            sig.strategy,
            f"[{color}]{sig.direction}[/{color}]",
            f"{sig.confidence:.0%}",
            sig.reasoning[:80] + "..." if len(sig.reasoning) > 80 else sig.reasoning
        )
    console.print(table)

    score_color = "green" if agg.composite_score > 0.3 else "red" if agg.composite_score < -0.3 else "yellow"
    console.print(Panel(
        f"Direction: [{score_color}]{agg.direction}[/{score_color}]  |  "
        f"Score: [{score_color}]{agg.composite_score:+.3f}[/{score_color}]  |  "
        f"Confidence: {agg.confidence:.0%}\n"
        f"Agreeing: {', '.join(agg.agreeing_strategies) or 'none'}  |  "
        f"Disagreeing: {', '.join(agg.disagreeing_strategies) or 'none'}",
        title="[bold]Aggregated Signal",
        border_style=score_color,
    ))

    # 3. Risk manager + paper broker
    risk_mgr = RiskManager()
    broker = PaperBroker(risk_manager=risk_mgr)
    portfolio = broker.portfolio_summary()

    console.print(f"\n[bold]Portfolio:[/bold] Balance=${portfolio['account_balance']:,.2f} | "
                  f"Open Positions={portfolio['open_positions']} | "
                  f"Daily P&L=${portfolio['daily_pnl']:+,.2f}")

    # 4. AI decision
    if agg.direction == "NEUTRAL" and abs(agg.composite_score) < 0.2:
        console.print("\n[dim]Signal too weak for AI analysis — waiting for a clear setup.[/dim]")
        return agg, None

    console.print(f"\n[bold cyan]Asking Claude to analyze this setup...[/bold cyan]")

    agent = TradingAgent()
    decision = agent.analyze(
        symbol=symbol,
        timeframe=timeframe,
        aggregated_signal=asdict(agg) if hasattr(agg, '__dataclass_fields__') else vars(agg),
        portfolio_summary=portfolio,
    )

    action_color = "green" if decision.action == "BUY" else "red" if decision.action == "SELL" else "yellow"
    console.print(Panel(
        f"[bold {action_color}]Action: {decision.action}[/bold {action_color}]  |  "
        f"Confidence: {decision.confidence:.0%}\n"
        f"Entry: {decision.entry}  |  Stop: {decision.stop_loss}  |  Target: {decision.target}  |  R:R: {decision.r_ratio}\n"
        f"Lead: {decision.lead_strategy}  |  Supporting: {', '.join(decision.supporting) or 'none'}\n\n"
        f"[italic]{decision.raw_response[:600]}[/italic]",
        title="[bold]Claude's Trade Decision",
        border_style=action_color,
    ))

    # 5. Paper execute if action is BUY or SELL
    if decision.action in ("BUY", "SELL") and decision.entry and decision.stop_loss and decision.target:
        console.print("\n[bold]Executing paper trade...[/bold]")
        success, msg, pos = broker.open_position(
            symbol=symbol,
            direction=decision.action,
            entry_price=decision.entry,
            stop_loss=decision.stop_loss,
            take_profit=decision.target,
            strategy_used=decision.lead_strategy,
            ai_reasoning=decision.reasoning,
        )
        color = "green" if success else "red"
        console.print(f"[{color}]{msg}[/{color}]")

    # 6. Interactive follow-up
    console.print("\n[dim]Ask follow-up questions (type 'quit' to exit):[/dim]")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = agent.ask(question)
        console.print(Panel(answer, border_style="cyan"))

    return agg, decision


def main():
    parser = argparse.ArgumentParser(description="AI Day Trading Assistant")
    parser.add_argument("--symbol", default="ES", help="Symbol to trade (default: ES)")
    parser.add_argument("--tf", default="5m", help="Timeframe (default: 5m)")
    parser.add_argument("--market", default="futures", choices=["futures", "equities"])
    parser.add_argument("--days", default=30, type=int, help="Days of history to load")
    args = parser.parse_args()

    run_analysis(args.symbol, args.tf, args.market, args.days)


if __name__ == "__main__":
    main()
