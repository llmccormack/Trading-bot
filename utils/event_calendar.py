"""
Economic event blackout calendar.

On FOMC decision days, CPI releases, and NFP mornings, price action is
driven by binary event risk that invalidates structural setups.  The ATR
spike filter catches the aftermath but not the event itself — this module
blocks *entries* on known scheduled-event days before the data drops.

Maintenance
-----------
Update the sets below each quarter.  Sources:
  FOMC  — https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  CPI   — https://www.bls.gov/schedule/news_release/cpi.htm
  NFP   — https://www.bls.gov/schedule/news_release/empsit.htm

Override via env:
  EVENT_BLACKOUT_EXTRA="2026-06-15,2026-07-04"   # comma-separated YYYY-MM-DD
  EVENT_BLACKOUT_OFF=true                          # disable entirely
"""
from __future__ import annotations

import os
from datetime import date


# ── FOMC rate-decision days ──────────────────────────────────────────── #
_FOMC: set[str] = {
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026 (from Fed calendar — confirm at federalreserve.gov)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
}

# ── CPI release days (8:30 AM ET — causes immediate gap risk) ───────── #
_CPI: set[str] = {
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-10", "2025-10-15", "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-09",
    "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
    "2026-09-09", "2026-10-14", "2026-11-11", "2026-12-09",
}

# ── NFP / Non-Farm Payrolls (first Friday of each month, 8:30 AM ET) ── #
_NFP: set[str] = {
    # 2025
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # 2026
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-08", "2026-06-05", "2026-07-10", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
}

_ALL_BLACKOUT: frozenset[str] = frozenset(_FOMC | _CPI | _NFP)


def is_news_blackout_today(today: date | None = None) -> tuple[bool, str]:
    """
    Return (True, reason) if today is a scheduled high-impact event day,
    (False, "") otherwise.

    Pass ``today`` explicitly in tests; defaults to UTC date in production
    (ET calendar date is used by the caller when applicable).
    """
    # Env kill-switch
    if os.getenv("EVENT_BLACKOUT_OFF", "").lower() in ("1", "true", "yes"):
        return False, ""

    d = (today or date.today()).isoformat()

    # Extra dates injected via env (comma-separated YYYY-MM-DD)
    extra_raw = os.getenv("EVENT_BLACKOUT_EXTRA", "")
    extra = {s.strip() for s in extra_raw.split(",") if s.strip()}

    if d in _ALL_BLACKOUT or d in extra:
        category = (
            "FOMC" if d in _FOMC else
            "CPI"  if d in _CPI  else
            "NFP"  if d in _NFP  else
            "custom"
        )
        return True, f"news blackout ({category}) — no new entries on {d}"

    return False, ""
