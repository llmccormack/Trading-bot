"""Historical data fetcher — yfinance for free backtesting data, Polygon.io for production."""
import threading
import tempfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Literal
import httpx
from config import settings

# Point yfinance cache to /tmp so it works in read-only cloud environments (Railway)
try:
    yf.set_tz_cache_location(tempfile.gettempdir())
except Exception:
    pass

# yfinance is NOT thread-safe: parallel calls share internal state and produce
# duplicate/merged columns. This lock ensures only one download runs at a time.
_YF_LOCK = threading.Lock()

# yfinance timeframe map
YF_INTERVALS = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}

# Futures symbol map (yfinance uses different tickers for futures)
FUTURES_SYMBOLS = {
    "ES": "ES=F",    # S&P 500 E-mini
    "NQ": "NQ=F",    # Nasdaq E-mini
    "CL": "CL=F",    # Crude Oil
    "GC": "GC=F",    # Gold
    "RTY": "RTY=F",  # Russell 2000 E-mini
    "YM": "YM=F",    # Dow E-mini
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and ensure timestamp column exists."""
    df.columns = [c.lower() for c in df.columns]
    if df.index.name and "time" in df.index.name.lower():
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "timestamp"})
    elif "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()



# yfinance hard limits on intraday history
_YF_MAX_DAYS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "1d":  3650,
}

# Map days_back → yfinance period string (more reliable than start/end for intraday)
def _days_to_period(days: int, interval: str) -> str:
    max_days = _YF_MAX_DAYS.get(interval, 60)
    days = min(days, max_days - 1)   # stay 1 day under the hard limit
    if days <= 7:   return "5d"
    if days <= 30:  return "1mo"
    if days <= 60:  return "60d"
    if days <= 90:  return "3mo"
    if days <= 180: return "6mo"
    if days <= 365: return "1y"
    return "2y"


def fetch_historical_yf(
    symbol: str,
    timeframe: str = "5m",
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from yfinance (free, good for backtesting).
    For futures, use symbol shorthand like 'ES', 'NQ', 'CL'.
    For equities, use standard ticker like 'AAPL', 'TSLA'.
    """
    yf_symbol = FUTURES_SYMBOLS.get(symbol, symbol)
    interval  = YF_INTERVALS.get(timeframe, timeframe)
    period    = _days_to_period(days_back, interval)

    # Use period= instead of start/end — more reliable for intraday futures
    # Lock prevents yfinance thread-safety bug (parallel calls produce duplicate columns)
    with _YF_LOCK:
        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol} ({yf_symbol}) @ {timeframe}. "
            f"yfinance may not carry this symbol intraday — try 1h or 1d, "
            f"or use a different symbol (NQ, CL, GC)."
        )

    # yfinance returns MultiIndex columns — find which level holds the OHLCV field names
    if isinstance(df.columns, pd.MultiIndex):
        _price_fields = {"Close", "Open", "High", "Low", "Volume", "Adj Close"}
        level0_vals   = set(df.columns.get_level_values(0))
        if level0_vals & _price_fields:
            df.columns = df.columns.get_level_values(0)   # level 0 has field names
        else:
            df.columns = df.columns.get_level_values(1)   # level 1 has field names

    return _normalize_columns(df)


def fetch_historical_polygon(
    symbol: str,
    timeframe: str = "5",      # minutes
    multiplier: int = 1,
    days_back: int = 60,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Polygon.io (requires API key).
    Higher quality data, supports options and crypto too.
    """
    if not settings.polygon_api_key:
        raise ValueError("POLYGON_API_KEY not set in .env")

    end = datetime.now()
    start = end - timedelta(days=days_back)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range"
        f"/{multiplier}/{timeframe}"
        f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": settings.polygon_api_key,
    }

    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("resultsCount", 0) == 0:
        raise ValueError(f"No Polygon data for {symbol}")

    df = pd.DataFrame(data["results"])
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()


def fetch_historical(
    symbol: str,
    timeframe: str = "5m",
    days_back: int = 30,
    source: Literal["yfinance", "polygon"] = "yfinance",
) -> pd.DataFrame:
    """Unified fetch — auto-selects source. Use polygon in production."""
    if source == "polygon":
        tf_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "1d": "day"}
        return fetch_historical_polygon(symbol, tf_map.get(timeframe, timeframe), days_back=days_back)
    return fetch_historical_yf(symbol, timeframe, days_back)


def get_live_price(symbol: str) -> dict:
    """
    Fast lightweight price snapshot — single API call, no bar download.
    Returns last price, bid, ask, day high/low.
    Suitable for polling every 5-10 seconds.
    Note: yfinance intraday data is typically 1-2 minutes delayed for free tier.
    For true real-time (<1s), connect IBKR via ib_insync (see execution/live.py).
    """
    yf_symbol = FUTURES_SYMBOLS.get(symbol, symbol)
    try:
        fi = yf.Ticker(yf_symbol).fast_info
        return {
            "last_price": getattr(fi, "last_price", None),
            "bid":        getattr(fi, "bid",         None),
            "ask":        getattr(fi, "ask",         None),
            "day_high":   getattr(fi, "day_high",    None),
            "day_low":    getattr(fi, "day_low",     None),
            "open":       getattr(fi, "open",        None),
            "volume":     getattr(fi, "last_volume", None),
            "symbol":     symbol,
        }
    except Exception as e:
        return {"last_price": None, "symbol": symbol, "error": str(e)}
