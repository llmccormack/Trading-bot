"""Technical indicator helpers using the `ta` library (pure Python, no C deps)."""
import pandas as pd
import ta


def add_ema(df: pd.DataFrame, periods: list[int] = [8, 21, 55]) -> pd.DataFrame:
    for p in periods:
        df[f"ema_{p}"] = ta.trend.EMAIndicator(close=df["close"], window=p).ema_indicator()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-reset VWAP: resets at the start of each trading day.
    Groups by calendar date (ET) so overnight bars don't bleed into RTH VWAP.
    Falls back to whole-df VWAP if no timestamp column is present.
    """
    try:
        if "timestamp" in df.columns:
            import pytz
            _ET = pytz.timezone("America/New_York")
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            ts_et      = ts.dt.tz_convert(_ET)
            date_et    = ts_et.dt.date

            tmp = df[["high", "low", "close", "volume"]].copy()
            tmp["_date"] = date_et.values
            tmp["_tp"]   = (tmp["high"] + tmp["low"] + tmp["close"]) / 3
            tmp["_tpv"]  = tmp["_tp"] * tmp["volume"]
            cum_tpv = tmp.groupby("_date", sort=False)["_tpv"].cumsum()
            cum_vol = tmp.groupby("_date", sort=False)["volume"].cumsum()
            df["vwap"] = (cum_tpv / cum_vol.replace(0, float("nan"))).values
        else:
            # Fallback: whole-df cumulative VWAP (acceptable for single-session data)
            df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
            ).volume_weighted_average_price()
    except Exception:
        df["vwap"] = float("nan")
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=period).rsi()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=period
    )
    df["adx"] = adx.adx()
    df["dmp"] = adx.adx_pos()   # +DI
    df["dmn"] = adx.adx_neg()   # -DI
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(close=df["close"], window=period, window_dev=std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd = ta.trend.MACD(close=df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    return df


def add_volume_metrics(df: pd.DataFrame, avg_period: int = 20) -> pd.DataFrame:
    df["vol_avg"]   = df["volume"].rolling(avg_period).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"]
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add all standard indicators to a dataframe."""
    df = add_ema(df, [8, 21, 55, 200])
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_volume_metrics(df)
    return df


def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    highs = df["high"]
    return highs == highs.rolling(lookback * 2 + 1, center=True).max()


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    lows = df["low"]
    return lows == lows.rolling(lookback * 2 + 1, center=True).min()


def get_key_levels(df: pd.DataFrame, lookback: int = 5) -> dict:
    sh = df[find_swing_highs(df, lookback)]["high"].tail(5).tolist()
    sl = df[find_swing_lows(df, lookback)]["low"].tail(5).tolist()
    return {"resistance": sorted(sh, reverse=True), "support": sorted(sl, reverse=True)}


def prev_day_levels(df: pd.DataFrame) -> dict:
    df_copy = df.copy()
    if "timestamp" not in df_copy.columns:
        return {}
    df_copy = df_copy.set_index("timestamp")
    daily = df_copy[["open", "high", "low", "close"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if len(daily) < 2:
        return {}
    prev = daily.iloc[-2]
    return {
        "prev_high":  prev["high"],
        "prev_low":   prev["low"],
        "prev_close": prev["close"],
    }
