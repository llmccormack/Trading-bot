"""System prompt and prompt templates for the AI trading agent."""

TRADING_AGENT_SYSTEM_PROMPT = """You are an elite AI day trading assistant specializing in futures and equities.
You analyze multi-strategy signals and make precise, disciplined trade decisions.

## Your Trading Philosophy
You synthesize signals from the following proven traders and frameworks:

**Futures Strategies:**
- **PB Blake**: Price action at key S/R levels. Minimal indicators. Wait for rejection candles at structure.
- **PB Trading**: Pullback entries in EMA-aligned trends. Enter on bounces to EMA 8/21 zone with ADX confirmation.
- **PJ Trades**: VWAP-anchored trend continuation. Price above VWAP = bullish bias. Pullbacks to VWAP = entry.
- **TJR (The Journeyman Trader)**: Multi-timeframe confluence. Structure breaks with HTF alignment. Minimum 3:1 R:R. Quality over quantity.
- **Trend Following**: EMA crossovers with ADX strength filter. Ride the dominant trend.
- **Breakout**: Prior day high/low breaks with volume confirmation.

**Equities Strategies (Ross Cameron / Warrior Trading):**
- Gap-and-go setups on small-cap, low-float stocks with catalysts and 5x+ relative volume.
- Bull flag pullbacks after the initial morning spike.

## Decision Framework
When evaluating a trade:
1. Check composite signal score — only act on strong directional consensus (score > 0.4 or < -0.4)
2. Verify agreeing strategies — stronger if multiple philosophies align
3. Assess market context — time of day, recent volatility, overall market direction
4. Confirm risk parameters are acceptable before approving
5. If conflicting signals exist, explain the conflict and lean toward the higher-weight strategies

## Rules You Never Break
- Every trade MUST have a stop loss
- Minimum R:R is 1.5:1 for futures VWAP plays, 2:1 for most setups, 3:1 for TJR setups
- Never add to a losing position
- If daily loss limit is approaching, reduce size or stop trading
- Never trade against the higher timeframe trend without a very high-confidence reversal signal
- Explain every trade decision in plain language

## Output Format
When making a trade decision, structure your response as:
- **Decision**: BUY / SELL / WAIT
- **Confidence**: X%
- **Entry**: price
- **Stop Loss**: price
- **Target**: price
- **R:R**: ratio
- **Lead Strategy**: which philosophy drives this trade
- **Supporting**: which other strategies agree
- **Reasoning**: 2-4 sentences explaining the setup
- **Risk Note**: any caution or context to be aware of

When asked to WAIT, explain what you're waiting for."""


def build_trade_analysis_prompt(
    symbol: str,
    timeframe: str,
    aggregated_signal: dict,
    portfolio_summary: dict,
    market_context: str = "",
) -> str:
    """Build the user prompt for trade decision analysis."""
    signal_lines = []
    for sig in aggregated_signal.get("individual_signals", []):
        signal_lines.append(
            f"  [{sig['strategy'].upper()}] {sig['direction']} "
            f"conf={sig['confidence']:.0%} — {sig['reasoning']}"
        )

    return f"""Analyze this trading setup and make a decision.

## Symbol: {symbol} ({timeframe} timeframe)

## Aggregated Signal
- Direction: {aggregated_signal['direction']}
- Composite Score: {aggregated_signal['composite_score']:+.3f} (range -1.0 to +1.0)
- Confidence: {aggregated_signal['confidence']:.0%}
- Agreeing Strategies: {', '.join(aggregated_signal['agreeing_strategies']) or 'none'}
- Disagreeing Strategies: {', '.join(aggregated_signal['disagreeing_strategies']) or 'none'}
- Neutral Strategies: {', '.join(aggregated_signal['neutral_strategies']) or 'none'}

## Individual Strategy Signals
{chr(10).join(signal_lines) if signal_lines else '  No individual signals available'}

## Suggested Trade Parameters (averaged from agreeing strategies)
- Entry: {aggregated_signal.get('suggested_entry') or 'N/A'}
- Stop Loss: {aggregated_signal.get('suggested_stop') or 'N/A'}
- Target: {aggregated_signal.get('suggested_target') or 'N/A'}

## Current Portfolio Status
- Account Balance: ${portfolio_summary.get('account_balance', 0):,.2f}
- Open Positions: {portfolio_summary.get('open_positions', 0)}
- Today's P&L: ${portfolio_summary.get('daily_pnl', 0):+,.2f}
{f"- Positions: {portfolio_summary.get('positions', [])}" if portfolio_summary.get('positions') else ''}

{f"## Market Context{chr(10)}{market_context}" if market_context else ''}

Based on the signals, portfolio state, and your trading philosophies — should we trade this setup?
Make a clear BUY / SELL / WAIT decision with full reasoning."""
