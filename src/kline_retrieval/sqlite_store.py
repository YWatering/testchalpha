"""Transactional SQLite snapshots for live-refreshable market data."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    amount REAL,
    return_rate REAL,
    pre_close REAL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol_date
    ON daily_bars (symbol, date);
CREATE TABLE IF NOT EXISTS dataset_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS instruments (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    board TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync_checkpoint (
    run_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    completed_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_key, symbol)
);
CREATE TABLE IF NOT EXISTS sync_staging_bars (
    run_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    PRIMARY KEY (run_key, symbol, date)
);
CREATE TABLE IF NOT EXISTS universe_symbols (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    board TEXT NOT NULL,
    status TEXT NOT NULL,
    data_start_date TEXT,
    data_end_date TEXT,
    bar_count INTEGER NOT NULL DEFAULT 0,
    valid_sample_count INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_universe_symbols_status
    ON universe_symbols (status, symbol);
CREATE TABLE IF NOT EXISTS universe_batches (
    batch_id TEXT PRIMARY KEY,
    batch_number INTEGER NOT NULL,
    symbols_json TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_universe_batches_status
    ON universe_batches (status, batch_number);
"""


def write_snapshot(
    database_path: str | Path,
    rows: Iterable[Mapping[str, object]],
    *,
    source_hash: str,
    manifest: Mapping[str, object],
    instruments: Iterable[Mapping[str, object]] = (),
) -> None:
    """Replace the database in one transaction so readers see whole snapshots."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        (
            str(row["symbol"]),
            str(row["date"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]) if row.get("volume") not in (None, "") else None,
        )
        for row in rows
    ]
    metadata = {
        "source_hash": source_hash,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": json.dumps(manifest, ensure_ascii=False, sort_keys=True),
    }
    instrument_rows = [
        (
            str(item["symbol"]),
            str(item.get("name", item["symbol"])),
            str(item.get("exchange", "")),
            str(item.get("board", "")),
            str(item.get("status", "listed")),
        )
        for item in instruments
    ]
    try:
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(SCHEMA)
            _ensure_optional_columns(connection)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM daily_bars")
            connection.executemany(
                """
                INSERT INTO daily_bars
                    (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                normalized,
            )
            if instrument_rows:
                connection.execute("DELETE FROM instruments")
                connection.executemany(
                    """
                    INSERT INTO instruments (symbol, name, exchange, board, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    instrument_rows,
                )
            connection.executemany(
                """
                INSERT INTO dataset_meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                metadata.items(),
            )
            connection.commit()
    except sqlite3.Error as exc:
        raise OSError(f"写入行情数据库失败: {path}") from exc


def read_metadata(database_path: str | Path) -> dict[str, str]:
    path = Path(database_path)
    try:
        with sqlite3.connect(path, timeout=5) as connection:
            connection.executescript(SCHEMA)
            _ensure_optional_columns(connection)
            return dict(connection.execute("SELECT key, value FROM dataset_meta"))
    except sqlite3.Error as exc:
        raise OSError(f"读取行情数据库元数据失败: {path}") from exc


def read_instruments(database_path: str | Path) -> list[dict[str, str]]:
    path = Path(database_path)
    try:
        with sqlite3.connect(path, timeout=5) as connection:
            connection.executescript(SCHEMA)
            _ensure_optional_columns(connection)
            rows = connection.execute(
                """
                SELECT symbol, name, exchange, board, status
                FROM instruments
                ORDER BY symbol
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise OSError(f"读取股票元数据失败: {path}") from exc
    return [
        {
            "symbol": row[0],
            "name": row[1],
            "exchange": row[2],
            "board": row[3],
            "status": row[4],
        }
        for row in rows
    ]


def load_checkpoint(
    database_path: str | Path,
    run_key: str,
) -> tuple[set[str], list[dict[str, object]]]:
    path = Path(database_path)
    if not path.exists():
        return set(), []
    try:
        with sqlite3.connect(path, timeout=10) as connection:
            connection.executescript(SCHEMA)
            _ensure_optional_columns(connection)
            connection.execute("DELETE FROM sync_checkpoint WHERE run_key <> ?", (run_key,))
            connection.execute("DELETE FROM sync_staging_bars WHERE run_key <> ?", (run_key,))
            connection.commit()
            completed = {
                row[0]
                for row in connection.execute(
                    "SELECT symbol FROM sync_checkpoint WHERE run_key = ?",
                    (run_key,),
                )
            }
            rows = [
                {
                    "symbol": row[0],
                    "date": row[1],
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "volume": row[6],
                }
                for row in connection.execute(
                    """
                    SELECT symbol, date, open, high, low, close, volume
                    FROM sync_staging_bars
                    WHERE run_key = ?
                    ORDER BY symbol, date
                    """,
                    (run_key,),
                )
            ]
            return completed, rows
    except sqlite3.Error as exc:
        raise OSError(f"读取同步断点失败: {path}") from exc


def stage_symbol(
    database_path: str | Path,
    run_key: str,
    symbol: str,
    rows: Iterable[Mapping[str, object]],
) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        (
            run_key,
            symbol,
            str(row["date"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]) if row.get("volume") not in (None, "") else None,
        )
        for row in rows
    ]
    try:
        with sqlite3.connect(path, timeout=30) as connection:
            connection.executescript(SCHEMA)
            _ensure_optional_columns(connection)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM sync_staging_bars WHERE run_key = ? AND symbol = ?",
                (run_key, symbol),
            )
            connection.executemany(
                """
                INSERT INTO sync_staging_bars
                    (run_key, symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                normalized,
            )
            connection.execute(
                """
                INSERT INTO sync_checkpoint (run_key, symbol, completed_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(run_key, symbol) DO UPDATE SET
                    completed_at_utc=excluded.completed_at_utc
                """,
                (run_key, symbol, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
    except sqlite3.Error as exc:
        raise OSError(f"保存同步断点失败: {path}") from exc


def clear_checkpoint(database_path: str | Path, run_key: str) -> None:
    path = Path(database_path)
    try:
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute("DELETE FROM sync_checkpoint WHERE run_key = ?", (run_key,))
            connection.execute("DELETE FROM sync_staging_bars WHERE run_key = ?", (run_key,))
            connection.commit()
    except sqlite3.Error as exc:
        raise OSError(f"清理同步断点失败: {path}") from exc


def _ensure_optional_columns(connection: sqlite3.Connection) -> None:
    """Migrate databases created before the CSMAR fields were added."""

    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(daily_bars)")
    }
    for name in ("amount", "return_rate", "pre_close"):
        if name not in columns:
            connection.execute(f"ALTER TABLE daily_bars ADD COLUMN {name} REAL")
