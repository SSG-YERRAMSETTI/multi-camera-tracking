"""
database.py  —  Async SQLite Database Layer
Phase 3 NEW component: DB Server with counts_table + config_table
Replaces: flat .txt / .csv log files from original system
"""

from __future__ import annotations

import asyncio
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

DB_PATH = "tracking.db"


# ─── Schema SQL ────────────────────────────────────────────────────────────────

CREATE_COUNTS_TABLE = """
CREATE TABLE IF NOT EXISTS counts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,           -- ISO-8601
    camera_id       INTEGER NOT NULL,
    mode            TEXT    NOT NULL,           -- 'object_tracking' | 'traffic_counting'
    object_class    TEXT    NOT NULL,           -- 'car', 'truck', 'bus', 'person'
    count           INTEGER NOT NULL DEFAULT 0,
    total_ids       INTEGER NOT NULL DEFAULT 0, -- unique track IDs seen this interval
    interval_start  TEXT    NOT NULL,
    interval_end    TEXT    NOT NULL
);
"""

CREATE_INTERSECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS intersections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    camera_id       INTEGER NOT NULL,
    track_id        INTEGER NOT NULL,
    object_class    TEXT    NOT NULL,
    x               REAL    NOT NULL,           -- intersection x coordinate
    y               REAL    NOT NULL,           -- intersection y coordinate
    angle           REAL    NOT NULL,           -- crossing angle in degrees
    direction       TEXT    NOT NULL            -- 'up' | 'down' | 'left' | 'right'
);
"""

CREATE_CONFIG_TABLE = """
CREATE TABLE IF NOT EXISTS config (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    updated TEXT NOT NULL
);
"""

CREATE_SYSTEM_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    level       TEXT NOT NULL,   -- 'INFO' | 'WARNING' | 'ERROR'
    camera_id   INTEGER,
    message     TEXT NOT NULL
);
"""

ALL_TABLES = [
    CREATE_COUNTS_TABLE,
    CREATE_INTERSECTIONS_TABLE,
    CREATE_CONFIG_TABLE,
    CREATE_SYSTEM_EVENTS_TABLE,
]


# ─── Database Manager ──────────────────────────────────────────────────────────

class Database:
    """
    Async SQLite wrapper.
    Single instance shared across the FastAPI app via lifespan context.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
        await self._conn.execute("PRAGMA foreign_keys=ON")
        for ddl in ALL_TABLES:
            await self._conn.execute(ddl)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    # ── Counts ─────────────────────────────────────────────────────────────────

    async def write_counts(
        self,
        camera_id: int,
        mode: str,
        counts_by_class: dict,        # {"car": 5, "truck": 2, ...}
        total_ids: int,
        interval_start: datetime,
        interval_end: datetime,
    ) -> None:
        now = datetime.utcnow().isoformat()
        rows = [
            (
                now, camera_id, mode, cls, count, total_ids,
                interval_start.isoformat(), interval_end.isoformat(),
            )
            for cls, count in counts_by_class.items()
        ]
        await self._conn.executemany(
            """INSERT INTO counts
               (timestamp, camera_id, mode, object_class, count, total_ids,
                interval_start, interval_end)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        await self._conn.commit()

    async def get_counts(
        self,
        camera_id: Optional[int] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[dict]:
        sql = "SELECT * FROM counts WHERE 1=1"
        params: list = []
        if camera_id is not None:
            sql += " AND camera_id=?"
            params.append(camera_id)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += f" ORDER BY timestamp DESC LIMIT {limit}"
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def export_counts_csv(self, camera_id: Optional[int] = None) -> str:
        rows = await self.get_counts(camera_id=camera_id, limit=10000)
        if not rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    # ── Intersections (traffic counting) ──────────────────────────────────────

    async def write_intersection(
        self,
        camera_id: int,
        track_id: int,
        object_class: str,
        x: float,
        y: float,
        angle: float,
        direction: str,
    ) -> None:
        await self._conn.execute(
            """INSERT INTO intersections
               (timestamp, camera_id, track_id, object_class, x, y, angle, direction)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.utcnow().isoformat(), camera_id, track_id,
             object_class, x, y, angle, direction),
        )
        await self._conn.commit()

    async def export_intersections_csv(self, camera_id: Optional[int] = None) -> str:
        sql = "SELECT * FROM intersections"
        params: list = []
        if camera_id is not None:
            sql += " WHERE camera_id=?"
            params.append(camera_id)
        sql += " ORDER BY timestamp DESC LIMIT 10000"
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        if not rows:
            return ""
        rows = [dict(r) for r in rows]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    # ── Config table ──────────────────────────────────────────────────────────

    async def set_config(self, key: str, value: str) -> None:
        await self._conn.execute(
            """INSERT INTO config (key, value, updated)
               VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated""",
            (key, value, datetime.utcnow().isoformat()),
        )
        await self._conn.commit()

    async def get_config(self, key: str) -> Optional[str]:
        async with self._conn.execute(
            "SELECT value FROM config WHERE key=?", (key,)
        ) as cur:
            row = await cur.fetchone()
        return row["value"] if row else None

    async def get_all_config(self) -> dict:
        async with self._conn.execute("SELECT key, value FROM config") as cur:
            rows = await cur.fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ── System events ─────────────────────────────────────────────────────────

    async def log_event(
        self,
        level: str,
        message: str,
        camera_id: Optional[int] = None,
    ) -> None:
        await self._conn.execute(
            """INSERT INTO system_events (timestamp, level, camera_id, message)
               VALUES (?,?,?,?)""",
            (datetime.utcnow().isoformat(), level, camera_id, message),
        )
        await self._conn.commit()

    async def get_recent_events(self, limit: int = 50) -> List[dict]:
        async with self._conn.execute(
            "SELECT * FROM system_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Analytics summary ─────────────────────────────────────────────────────

    async def get_analytics_summary(self) -> dict:
        """Returns aggregated totals for the analytics dashboard."""
        async with self._conn.execute(
            """SELECT object_class, SUM(count) as total, SUM(total_ids) as ids
               FROM counts GROUP BY object_class"""
        ) as cur:
            rows = await cur.fetchall()
        class_totals = {r["object_class"]: {"count": r["total"], "ids": r["ids"]} for r in rows}

        async with self._conn.execute("SELECT COUNT(*) as n FROM intersections") as cur:
            row = await cur.fetchone()
        total_intersections = row["n"] if row else 0

        return {
            "by_class": class_totals,
            "total_objects": sum(v["count"] for v in class_totals.values()),
            "total_intersections": total_intersections,
        }


# ─── Module-level singleton ────────────────────────────────────────────────────
db = Database()
