"""Prepare and inspect staged full-market backfill status."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG, ProjectConfig
from .sqlite_store import SCHEMA


def prepare_universe_batches(
    database_path: str | Path,
    *,
    config: ProjectConfig = DEFAULT_CONFIG,
    batch_size: int = 200,
) -> dict[str, object]:
    """Create idempotent symbol status and backfill batch records."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    path = Path(database_path)
    now = datetime.now(timezone.utc).isoformat()
    min_history = config.monthly_period * config.daily_window
    max_future = max(config.future_horizons)
    with sqlite3.connect(path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
        instruments = connection.execute(
            "SELECT symbol, name, exchange, board, status FROM instruments ORDER BY symbol"
        ).fetchall()
        if not instruments:
            raise ValueError("数据库 instruments 表中没有股票")

        status_rows: list[tuple[object, ...]] = []
        for symbol, name, exchange, board, status in instruments:
            dates = [
                row[0]
                for row in connection.execute(
                    "SELECT date FROM daily_bars WHERE symbol = ? ORDER BY date", (symbol,)
                )
            ]
            valid_count = max(0, len(dates) - min_history - max_future + 1)
            status_rows.append(
                (
                    symbol,
                    name,
                    exchange,
                    board,
                    "ready" if valid_count else "pending",
                    dates[0] if dates else None,
                    dates[-1] if dates else None,
                    len(dates),
                    valid_count,
                    now,
                )
            )
        connection.executemany(
            """
            INSERT INTO universe_symbols
                (symbol, name, exchange, board, status, data_start_date, data_end_date,
                 bar_count, valid_sample_count, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                name=excluded.name, exchange=excluded.exchange, board=excluded.board,
                data_start_date=excluded.data_start_date, data_end_date=excluded.data_end_date,
                bar_count=excluded.bar_count, valid_sample_count=excluded.valid_sample_count,
                status=CASE WHEN universe_symbols.status = 'failed'
                            THEN universe_symbols.status ELSE excluded.status END,
                updated_at_utc=excluded.updated_at_utc
            """,
            status_rows,
        )

        existing_batches = {
            row[0]: row[1:]
            for row in connection.execute(
                "SELECT batch_id, status, attempts, last_error FROM universe_batches"
            )
        }
        batch_rows: list[tuple[object, ...]] = []
        symbols = [row[0] for row in instruments]
        for offset in range(0, len(symbols), batch_size):
            number = offset // batch_size + 1
            batch_id = f"a-share-{number:04d}"
            old = existing_batches.get(batch_id, ("pending", 0, None))
            batch_rows.append(
                (
                    batch_id,
                    number,
                    json.dumps(symbols[offset : offset + batch_size], ensure_ascii=False),
                    old[0],
                    old[1],
                    old[2],
                    now,
                )
            )
        connection.executemany(
            """
            INSERT INTO universe_batches
                (batch_id, batch_number, symbols_json, status, attempts, last_error, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_id) DO UPDATE SET
                batch_number=excluded.batch_number, symbols_json=excluded.symbols_json,
                updated_at_utc=excluded.updated_at_utc
            """,
            batch_rows,
        )
        connection.execute(
            "INSERT INTO dataset_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("universe_updated_at_utc", now),
        )
        connection.commit()

    return coverage_report(path)


def coverage_report(database_path: str | Path) -> dict[str, object]:
    """Return coverage, quality and batch state for the configured snapshot."""

    path = Path(database_path)
    with sqlite3.connect(path, timeout=10) as connection:
        connection.executescript(SCHEMA)
        symbols = connection.execute(
            "SELECT COUNT(*), SUM(status = 'ready'), SUM(status = 'failed'), SUM(valid_sample_count) FROM universe_symbols"
        ).fetchone()
        latest = connection.execute("SELECT MAX(date) FROM daily_bars").fetchone()[0]
        batches = connection.execute(
            "SELECT status, COUNT(*) FROM universe_batches GROUP BY status ORDER BY status"
        ).fetchall()
    total, ready, failed, valid_samples = symbols
    total = int(total or 0)
    return {
        "total_symbols": total,
        "ready_symbols": int(ready or 0),
        "failed_symbols": int(failed or 0),
        "coverage_ratio": round((int(ready or 0) / total), 6) if total else 0.0,
        "valid_sample_count": int(valid_samples or 0),
        "latest_trading_date": latest,
        "batches": {status: count for status, count in batches},
    }


def read_universe_symbols(
    database_path: str | Path,
    *,
    loaded_symbols: Sequence[str] = (),
    query: str = "",
    exchange: str = "",
    board: str = "",
    watchlist: Sequence[str] = (),
    offset: int = 0,
    limit: int = 10_000,
) -> tuple[list[dict[str, object]], int]:
    """Read filtered universe rows for the stock selector API."""

    if offset < 0 or limit <= 0:
        raise ValueError("offset 不能为负数且 limit 必须大于 0")
    clauses = ["1 = 1"]
    params: list[object] = []
    if query.strip():
        clauses.append("(symbol LIKE ? OR name LIKE ?)")
        value = f"%{query.strip()}%"
        params.extend((value, value))
    if exchange:
        clauses.append("exchange = ?")
        params.append(exchange)
    if board:
        clauses.append("board = ?")
        params.append(board)
    selected = tuple(dict.fromkeys(item.strip() for item in watchlist if item.strip()))
    if selected:
        placeholders = ",".join("?" for _ in selected)
        clauses.append(f"symbol IN ({placeholders})")
        params.extend(selected)
    loaded = tuple(dict.fromkeys(item.strip() for item in loaded_symbols if item.strip()))
    if loaded:
        placeholders = ",".join("?" for _ in loaded)
        clauses.append(f"symbol IN ({placeholders})")
        params.extend(loaded)
    where = " AND ".join(clauses)
    path = Path(database_path)
    with sqlite3.connect(path, timeout=10) as connection:
        connection.executescript(SCHEMA)
        total = connection.execute(f"SELECT COUNT(*) FROM universe_symbols WHERE {where}", params).fetchone()[0]
        rows = connection.execute(
            f"SELECT symbol, name, exchange, board, status, valid_sample_count FROM universe_symbols WHERE {where} ORDER BY symbol LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    return (
        [
            {
                "symbol": row[0],
                "name": row[1],
                "exchange": row[2],
                "board": row[3],
                "status": row[4],
                "valid_sample_count": row[5],
            }
            for row in rows
        ],
        int(total),
    )
