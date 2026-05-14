"""DuckDB storage layer for OHLCV data, signals, positions, and trade journal."""
import threading
import duckdb
import pandas as pd
from pathlib import Path
from config import settings

# One lock for the whole process — prevents concurrent write-write conflicts
# when Streamlit reruns the script on multiple threads simultaneously
_db_lock = threading.Lock()

# Use Railway persistent volume if available, otherwise fall back to project folder
import os
_DATA_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", str(Path(__file__).parent.parent))
DB_PATH = str(Path(_DATA_DIR) / "trading_bot.duckdb")
_DB_PATH = DB_PATH  # backwards-compat alias


def get_conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)


def init_db() -> None:
    """Create all tables if they don't exist."""
    with _db_lock:
        try:
            conn = get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol      VARCHAR NOT NULL,
                    timeframe   VARCHAR NOT NULL,
                    timestamp   TIMESTAMPTZ NOT NULL,
                    open        DOUBLE,
                    high        DOUBLE,
                    low         DOUBLE,
                    close       DOUBLE,
                    volume      DOUBLE,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id          VARCHAR DEFAULT gen_random_uuid(),
                    symbol      VARCHAR NOT NULL,
                    timestamp   TIMESTAMPTZ NOT NULL,
                    strategy    VARCHAR NOT NULL,
                    direction   VARCHAR NOT NULL,
                    confidence  DOUBLE,
                    reasoning   TEXT,
                    PRIMARY KEY (id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id                  VARCHAR DEFAULT gen_random_uuid(),
                    symbol              VARCHAR NOT NULL,
                    direction           VARCHAR NOT NULL,
                    qty                 DOUBLE NOT NULL,
                    entry_price         DOUBLE NOT NULL,
                    stop_loss           DOUBLE,
                    take_profit         DOUBLE,
                    target_1            DOUBLE DEFAULT 0.0,
                    be_moved            BOOLEAN DEFAULT FALSE,
                    original_stop       DOUBLE DEFAULT 0.0,
                    partial_pnl_booked  DOUBLE DEFAULT 0.0,
                    opened_at           TIMESTAMPTZ NOT NULL,
                    closed_at           TIMESTAMPTZ,
                    exit_price          DOUBLE,
                    realized_pnl        DOUBLE,
                    status              VARCHAR DEFAULT 'OPEN',
                    notes               TEXT,
                    PRIMARY KEY (id)
                )
            """)
            # Migrate older DBs that don't have the new columns yet
            for _col_sql in [
                "ALTER TABLE positions ADD COLUMN IF NOT EXISTS target_1 DOUBLE DEFAULT 0.0",
                "ALTER TABLE positions ADD COLUMN IF NOT EXISTS be_moved BOOLEAN DEFAULT FALSE",
                "ALTER TABLE positions ADD COLUMN IF NOT EXISTS original_stop DOUBLE DEFAULT 0.0",
                "ALTER TABLE positions ADD COLUMN IF NOT EXISTS partial_pnl_booked DOUBLE DEFAULT 0.0",
            ]:
                try:
                    conn.execute(_col_sql)
                except Exception:
                    pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id              VARCHAR DEFAULT gen_random_uuid(),
                    position_id     VARCHAR,
                    symbol          VARCHAR NOT NULL,
                    direction       VARCHAR NOT NULL,
                    entry_price     DOUBLE,
                    exit_price      DOUBLE,
                    qty             DOUBLE,
                    pnl             DOUBLE,
                    r_multiple      DOUBLE,
                    strategy_used   VARCHAR,
                    ai_reasoning    TEXT,
                    opened_at       TIMESTAMPTZ,
                    closed_at       TIMESTAMPTZ,
                    PRIMARY KEY (id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_signals (
                    id              VARCHAR PRIMARY KEY,
                    strategy        VARCHAR NOT NULL,
                    symbol          VARCHAR NOT NULL,
                    direction       VARCHAR NOT NULL,
                    entry           DOUBLE NOT NULL,
                    original_stop   DOUBLE NOT NULL,
                    current_stop    DOUBLE NOT NULL,
                    t1              DOUBLE NOT NULL,
                    t2              DOUBLE NOT NULL,
                    score           DOUBLE NOT NULL,
                    signal_time     TIMESTAMP NOT NULL,
                    status          VARCHAR DEFAULT 'open',
                    t1_exited       BOOLEAN DEFAULT FALSE,
                    exit_price      DOUBLE,
                    exit_reason     VARCHAR,
                    pnl_pts         DOUBLE,
                    r_multiple      DOUBLE,
                    closed_at       TIMESTAMP,
                    bars_managed    INT DEFAULT 0
                )
            """)
            conn.close()
        except Exception:
            # Tables already exist from a parallel Streamlit thread — that's fine
            pass


def upsert_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Insert or replace OHLCV candles."""
    df = df.copy()
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    with _db_lock:
        conn = get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume
            FROM df
        """)
        conn.close()


def get_open_positions() -> pd.DataFrame:
    """Return all currently-open positions from DuckDB."""
    conn = get_conn()
    df = conn.execute("""
        SELECT id, symbol, direction, qty, entry_price, stop_loss, take_profit,
               opened_at, notes
        FROM positions
        WHERE status = 'OPEN'
        ORDER BY opened_at DESC
    """).df()
    conn.close()
    return df


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, [symbol, timeframe, limit]).df()
    conn.close()
    return df.sort_values("timestamp").reset_index(drop=True)
