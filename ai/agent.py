"""
AI trading decision agent — uses Claude to analyze multi-strategy signals
and decide whether to enter/exit trades.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any
import anthropic
from config import settings
from ai.prompts import TRADING_AGENT_SYSTEM_PROMPT, build_trade_analysis_prompt


@dataclass
class TradeDecision:
    action: str           # BUY | SELL | WAIT
    confidence: float     # 0.0 – 1.0
    entry: float | None
    stop_loss: float | None
    target: float | None
    r_ratio: float | None
    lead_strategy: str
    supporting: list[str]
    reasoning: str
    risk_note: str
    raw_response: str     # full Claude response text


class TradingAgent:
    """
    Wraps Claude API calls for trade analysis.
    Maintains conversation history for interactive Q&A.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.conversation_history: list[dict] = []
        self.model = "claude-opus-4-6"

    # ------------------------------------------------------------------ #
    # Core: analyze a setup and return a trade decision                   #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        aggregated_signal: dict,
        portfolio_summary: dict,
        market_context: str = "",
    ) -> TradeDecision:
        """Analyze a trading setup and return a structured trade decision."""

        user_prompt = build_trade_analysis_prompt(
            symbol=symbol,
            timeframe=timeframe,
            aggregated_signal=aggregated_signal,
            portfolio_summary=portfolio_summary,
            market_context=market_context,
        )

        # Use streaming for reliability on long outputs
        full_text = ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=TRADING_AGENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                full_text += text

        # Store in conversation history for follow-up Q&A
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": full_text})

        return self._parse_decision(full_text, aggregated_signal)

    # ------------------------------------------------------------------ #
    # Conversational Q&A — ask follow-up questions about the setup       #
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> str:
        """
        Ask a follow-up question about the current trade analysis.
        Maintains full conversation context.
        """
        if not self.conversation_history:
            return "No trade has been analyzed yet. Run analyze() first."

        self.conversation_history.append({"role": "user", "content": question})

        full_text = ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=TRADING_AGENT_SYSTEM_PROMPT,
            messages=self.conversation_history,
        ) as stream:
            for text in stream.text_stream:
                full_text += text

        self.conversation_history.append({"role": "assistant", "content": full_text})
        return full_text

    def clear_conversation(self) -> None:
        self.conversation_history = []

    # ------------------------------------------------------------------ #
    # Parse Claude's response into a structured TradeDecision             #
    # ------------------------------------------------------------------ #

    def _parse_decision(self, text: str, signal: dict) -> TradeDecision:
        """Extract structured fields from Claude's free-text response."""
        lines = text.lower()

        # Determine action
        if "**decision**: buy" in lines or "decision: buy" in lines:
            action = "BUY"
        elif "**decision**: sell" in lines or "decision: sell" in lines:
            action = "SELL"
        else:
            action = "WAIT"

        # Extract numeric fields with simple parsing
        entry = self._extract_price(text, ["entry:", "**entry**:"])
        stop_loss = self._extract_price(text, ["stop loss:", "**stop loss**:"])
        target = self._extract_price(text, ["target:", "**target**:"])
        confidence_pct = self._extract_pct(text, ["confidence:", "**confidence**:"])

        r_ratio = None
        if entry and stop_loss and target:
            risk = abs(entry - stop_loss)
            reward = abs(target - entry)
            r_ratio = round(reward / risk, 2) if risk > 0 else None

        # Extract lead strategy and supporting
        lead = signal.get("agreeing_strategies", ["unknown"])[0] if signal.get("agreeing_strategies") else "unknown"
        supporting = signal.get("agreeing_strategies", [])[1:]

        # Extract reasoning block
        reasoning = ""
        risk_note = ""
        if "**reasoning**:" in text.lower():
            parts = text.lower().split("**reasoning**:")
            if len(parts) > 1:
                reasoning_block = parts[1].split("**")[0].strip()
                reasoning = reasoning_block[:500]
        if "**risk note**:" in text.lower():
            parts = text.lower().split("**risk note**:")
            if len(parts) > 1:
                risk_note = parts[1].split("**")[0].strip()[:300]

        return TradeDecision(
            action=action,
            confidence=confidence_pct or 0.0,
            entry=entry or signal.get("suggested_entry"),
            stop_loss=stop_loss or signal.get("suggested_stop"),
            target=target or signal.get("suggested_target"),
            r_ratio=r_ratio,
            lead_strategy=lead,
            supporting=supporting,
            reasoning=reasoning or text[:300],
            risk_note=risk_note,
            raw_response=text,
        )

    def _extract_price(self, text: str, labels: list[str]) -> float | None:
        import re
        for label in labels:
            pattern = re.escape(label) + r"\s*\$?([\d,]+\.?\d*)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue
        return None

    def _extract_pct(self, text: str, labels: list[str]) -> float | None:
        import re
        for label in labels:
            pattern = re.escape(label) + r"\s*([\d]+)%"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) / 100
                except ValueError:
                    continue
        return None
