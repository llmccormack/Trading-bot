"""
Background Signal Monitor
─────────────────────────
Runs silently in the background. Checks ES=F for trading signals every hour
during market hours (9am–4pm ET, Mon–Fri). Sends a macOS notification the
moment a setup fires, with exact entry / stop / target pre-calculated.

Usage:
    python monitor.py            # run forever (Ctrl+C to stop)
    python monitor.py --check    # run one check right now and exit

Auto-start at login:
    python monitor.py --install  # installs a macOS LaunchAgent
    python monitor.py --uninstall
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytz

# ── Path setup ───────────────────────────────────────────────────── #
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from backtesting.engine import BacktestEngine
from backtesting.engine_5m import BacktestEngine5m
from backtesting.engine_aplus import BacktestEngineAPlus
from data.fetcher import fetch_historical, get_live_price

# ── Config ───────────────────────────────────────────────────────── #
SYMBOL       = "ES=F"
SYMBOL_SHORT = "ES"
ET           = pytz.timezone("America/New_York")
CHECK_INTERVAL_SECS = 60 * 5   # check every 5 minutes

# Mode-specific config (overridden at startup)
_MODE      = "1h"   # set by --mode flag; "1h" or "5m"
TIMEFRAME  = "1h"
DAYS_BACK  = 365

LOG_FILE = ROOT / "monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("monitor")

# Tracks signals we've already notified about (direction+strategy+bar_time)
_notified: set[str] = set()


# ── Market hours check ───────────────────────────────────────────── #

def is_market_hours() -> bool:
    """True if current ET time is Mon–Fri between 9:00 AM and 4:00 PM."""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    hour = now_et.hour
    return 9 <= hour < 16              # 9:00 AM – 3:59 PM ET


def next_market_open_secs() -> float:
    """Seconds until next market open (for sleeping over weekends/nights)."""
    now_et = datetime.now(ET)
    # Find next 9am ET weekday
    candidate = now_et.replace(hour=9, minute=0, second=0, microsecond=0)
    if candidate <= now_et:
        candidate += timedelta(days=1)
    # Skip weekends
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    delta = (candidate - now_et).total_seconds()
    return max(delta, 60.0)


# ── macOS notification ────────────────────────────────────────────── #

def _notify(title: str, message: str, subtitle: str = "") -> None:
    """Send a macOS notification via osascript."""
    sub_part = f'subtitle "{subtitle}" ' if subtitle else ""
    script = (
        f'display notification "{message}" '
        f'with title "{title}" '
        f'{sub_part}'
        f'sound name "Glass"'
    )
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception as e:
        log.warning(f"Notification failed: {e}")


# ── Signal check ─────────────────────────────────────────────────── #

def run_check() -> dict | None:
    """
    Fetch latest data, run live_signal(), return result dict or None.
    Uses global _MODE to select engine.
    """
    log.info(f"Checking {SYMBOL} ({_MODE}) for signals…")
    try:
        df      = fetch_historical(SYMBOL, TIMEFRAME, DAYS_BACK)
        snap    = get_live_price(SYMBOL_SHORT)
        cur_px  = snap.get("last_price") if snap else None

        if _MODE == "aplus":
            engine = BacktestEngineAPlus(
                min_adx=18.0, min_score=0.65,
                allow_short=False, require_macro_confirm=False,
            )
        elif _MODE == "5m":
            engine = BacktestEngine5m(
                warmup=50, max_bars=24, rr_ratio=2.0, atr_stop=1.5,
                min_score=0.50, allow_short=True, min_adx=18.0, rth_only=True,
            )
        else:
            engine = BacktestEngine(
                warmup=200, max_bars=24, rr_ratio=2.0, atr_stop=1.0,
                min_score=0.55, allow_short=True, min_adx=25.0, rth_only=True,
            )
        sig = engine.live_signal(df, current_price=cur_px)

        if sig:
            log.info(
                f"SIGNAL: {sig['direction']} {SYMBOL_SHORT} | "
                f"{sig['strategy']} | entry ~{sig['entry']:.2f} | "
                f"stop {sig['stop']:.2f} | target {sig['target']:.2f} | "
                f"score {sig['score']:.2f}"
            )
        else:
            log.info("No signal — all strategies NEUTRAL")

        return sig

    except Exception as e:
        log.error(f"Check failed: {e}")
        return None


def notify_signal(sig: dict) -> None:
    """Send a macOS notification for a firing signal."""
    direction  = sig["direction"]
    icon       = "🟢" if direction == "BUY" else "🔴"
    risk_usd   = int(sig["risk_pts"] * 50)
    reward_usd = int(sig["risk_pts"] * sig["rr"] * 50)

    title    = f"{icon} {direction} SIGNAL — ES Futures"
    subtitle = sig["strategy"].replace("_", " ").upper()
    message  = (
        f"Entry ~{sig['entry']:,.0f}  "
        f"Stop {sig['stop']:,.0f} (-${risk_usd:,})  "
        f"Target {sig['target']:,.0f} (+${reward_usd:,})"
    )

    _notify(title, message, subtitle)
    log.info(f"Notification sent: {title} | {message}")


# ── Monitor loop ─────────────────────────────────────────────────── #

def run_monitor() -> None:
    """Run the signal monitor loop indefinitely."""
    _mode_label = (
        "A+ (IB Breakout + VWAP-85 Retest)" if _MODE == "aplus"
        else "5m (ORB / VWAP / EMA Stack)" if _MODE == "5m"
        else "1h (EMA55/EMA21 Bounce)"
    )
    log.info("=" * 60)
    log.info(f"ES Futures Signal Monitor started  [mode: {_mode_label}]")
    log.info(f"Checking every {CHECK_INTERVAL_SECS // 60} minutes during market hours")
    log.info(f"Log file: {LOG_FILE}")
    log.info("=" * 60)

    _notify(
        "📈 Signal Monitor Started",
        f"Mode: {_mode_label}. Watching ES futures 9am–4pm ET.",
    )

    while True:
        now_et = datetime.now(ET)

        if not is_market_hours():
            sleep_secs = next_market_open_secs()
            wake_et    = datetime.now(ET) + timedelta(seconds=sleep_secs)
            log.info(
                f"Market closed — sleeping until "
                f"{wake_et.strftime('%a %b %d %I:%M %p ET')} "
                f"({sleep_secs / 3600:.1f}h)"
            )
            time.sleep(min(sleep_secs, 3600))  # wake up at most every hour to recheck
            continue

        sig = run_check()
        if sig:
            sig_key = f"{sig['direction']}_{sig['strategy']}_{sig['bar_time']}"
            if sig_key not in _notified:
                _notified.add(sig_key)
                notify_signal(sig)
            else:
                log.info("Signal already notified — skipping duplicate")

        time.sleep(CHECK_INTERVAL_SECS)


# ── LaunchAgent install / uninstall ──────────────────────────────── #

PLIST_NAME = "com.trading.signalmonitor"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{PLIST_NAME}.plist"

def install_launchagent() -> None:
    """Install a macOS LaunchAgent so the monitor auto-starts at login."""
    python_path = sys.executable
    script_path = str(Path(__file__).resolve())

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{LOG_FILE}</string>

    <key>StandardErrorPath</key>
    <string>{LOG_FILE}</string>

    <key>WorkingDirectory</key>
    <string>{ROOT}</string>
</dict>
</plist>
"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist_content)

    # Load the agent immediately
    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=False)

    print(f"✅ LaunchAgent installed at {PLIST_PATH}")
    print("   The monitor will now start automatically every time you log in.")
    print("   It's running right now in the background.")
    print(f"   Logs → {LOG_FILE}")
    print(f"\n   To stop:     python monitor.py --uninstall")


def uninstall_launchagent() -> None:
    """Remove the LaunchAgent."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
        print(f"✅ LaunchAgent removed — monitor will no longer auto-start.")
    else:
        print("No LaunchAgent found.")


# ── Entry point ──────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES Futures Signal Monitor")
    parser.add_argument("--check",     action="store_true", help="Run one check and exit")
    parser.add_argument("--install",   action="store_true", help="Install macOS LaunchAgent (auto-start at login)")
    parser.add_argument("--uninstall", action="store_true", help="Remove macOS LaunchAgent")
    parser.add_argument(
        "--mode", choices=["1h", "5m", "aplus"], default="1h",
        help="Engine mode: "
             "1h = EMA55/EMA21 Bounce (2-4 trades/month), "
             "5m = ORB+VWAP+EMA Stack (2-3 trades/day), "
             "aplus = A+ Institutional IB Breakout (1-2 setups/day). Default: 1h",
    )
    args = parser.parse_args()

    # Apply mode to globals
    _MODE = args.mode
    if _MODE in ("5m", "aplus"):
        TIMEFRAME = "5m"
        DAYS_BACK = 58   # yfinance 5m limit
    else:
        TIMEFRAME = "1h"
        DAYS_BACK = 365

    if args.install:
        install_launchagent()
    elif args.uninstall:
        uninstall_launchagent()
    elif args.check:
        sig = run_check()
        if sig:
            notify_signal(sig)
            print(f"\nSIGNAL FOUND:")
            print(f"  Direction : {sig['direction']}")
            print(f"  Strategy  : {sig['strategy_label']}")
            print(f"  Entry     : ~{sig['entry']:,.2f}")
            print(f"  Stop      : {sig['stop']:,.2f}")
            print(f"  Target    : {sig['target']:,.2f}")
            print(f"  Risk/Rwd  : {sig['rr']:.1f}:1")
        else:
            print("No signal right now.")
    else:
        run_monitor()
