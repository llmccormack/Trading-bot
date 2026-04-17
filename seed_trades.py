"""
Seed historical paper trades so Railway retains them across redeploys.
Only inserts if the positions table is empty.
"""
import duckdb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trading_bot.duckdb"

POSITIONS = [
    # (id, symbol, direction, qty, entry, sl, tp, opened_at, closed_at, exit_price, realized_pnl, status, notes)
    ("001b10e6-fcee-4542-b33f-563cc428f8c3","ES","LONG",117.0,6637.317,6621.15,6672.56,
     "2026-04-01T11:04:48.407840-04:00","2026-04-01T14:28:57.574971-04:00",6617.8394,-2278.88,"CLOSED",
     "aplus: Autopilot: 5m IB Retest | regime=ib_retest_long | score=0.81"),
    ("56328953-9940-4416-8ab3-d3ba778fadb5","NQ","LONG",27.0,24261.3746,24194.45,24413.64,
     "2026-04-01T11:04:48.308386-04:00","2026-04-01T14:28:54.932438-04:00",24182.3528,-2133.59,"CLOSED",
     "aplus: Autopilot: 5m IB Retest | regime=ib_retest_long | score=0.79"),
    ("01df80a8-f479-43e2-bf5b-35be06f78119","ES","LONG",123.0,6636.5666,6621.06,6669.81,
     "2026-04-01T11:05:39.146791-04:00","2026-04-01T14:28:57.742544-04:00",6617.7495,-2314.5,"CLOSED",
     "aplus: Autopilot: 5m IB Retest | regime=ib_retest_long | score=0.83"),
    ("c9fafa50-f018-4d4e-b996-ec46ea50391b","ES","LONG",117.0,7033.515,7051.7875,7047.9,
     "2026-04-15T10:37:54.532433-04:00","2026-04-15T15:03:17.651422-04:00",7044.376,1270.74,"CLOSED",
     "aplus: Server autopilot | score=0.79 | regime=ib_retest_long | bar=2026-04-15 10:25 ET"),
]

JOURNAL = [
    # (id, position_id, symbol, direction, entry, exit, qty, pnl, r_multiple, strategy, opened_at, closed_at)
    ("5e73aa02-29b9-40b3-8f2e-46d8bc9ca477","001b10e6-fcee-4542-b33f-563cc428f8c3","ES","LONG",
     6637.317,6617.8394,117.0,-2278.88,-1.2047755822110715,"aplus",
     "2026-04-01T11:04:48.407840-04:00","2026-04-01T14:28:57.574971-04:00"),
    ("77762076-3a3d-47f2-8afc-21536fa7304a","56328953-9940-4416-8ab3-d3ba778fadb5","NQ","LONG",
     24261.3746,24182.3528,27.0,-2133.59,-1.1807594195834368,"aplus",
     "2026-04-01T11:04:48.308386-04:00","2026-04-01T14:28:54.932438-04:00"),
    ("46447290-ec56-4715-92bb-c7cd4ed6a998","01df80a8-f479-43e2-bf5b-35be06f78119","ES","LONG",
     6636.5666,6617.7495,123.0,-2314.5,-1.2134880096689191,"aplus",
     "2026-04-01T11:05:39.146791-04:00","2026-04-01T14:28:57.742544-04:00"),
    ("7eb1f313-2b32-42ca-8949-b6b7e529a0d3","c9fafa50-f018-4d4e-b996-ec46ea50391b","ES","LONG",
     7033.515,7041.94,117.0,985.72,0.8882445967315504,"aplus",
     "2026-04-15T10:37:54.532433-04:00","2026-04-15T14:22:59.694149-04:00"),
    ("8325f463-ca41-4181-a9c7-bbfdc70a542f","c9fafa50-f018-4d4e-b996-ec46ea50391b","ES","LONG",
     7033.515,7044.376,117.0,1270.74,0.5943918807511627,"aplus",
     "2026-04-15T10:37:54.532433-04:00","2026-04-15T15:03:17.651422-04:00"),
]


def seed():
    conn = duckdb.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    if count > 0:
        conn.close()
        return  # already seeded

    print("Seeding historical trades...")
    for p in POSITIONS:
        conn.execute("""
            INSERT OR IGNORE INTO positions
            (id,symbol,direction,qty,entry_price,stop_loss,take_profit,
             opened_at,closed_at,exit_price,realized_pnl,status,notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, list(p))

    for j in JOURNAL:
        conn.execute("""
            INSERT OR IGNORE INTO trade_journal
            (id,position_id,symbol,direction,entry_price,exit_price,
             qty,pnl,r_multiple,strategy_used,opened_at,closed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, list(j))

    conn.close()
    print(f"Seeded {len(POSITIONS)} positions and {len(JOURNAL)} journal entries.")


if __name__ == "__main__":
    seed()
