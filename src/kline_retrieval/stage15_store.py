"""Resumable SQLite storage for stage-fifteen weekly predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE15_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    index_version TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    status TEXT NOT NULL,
    target_count INTEGER NOT NULL,
    completed_count INTEGER NOT NULL DEFAULT 0,
    updated_at_utc TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    run_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (run_key, symbol, signal_date),
    FOREIGN KEY (run_key) REFERENCES prediction_runs(run_key)
);
CREATE INDEX IF NOT EXISTS idx_stage15_predictions_scope
    ON predictions(run_key, signal_date, symbol);
CREATE TABLE IF NOT EXISTS prediction_matches (
    run_key TEXT NOT NULL,
    query_symbol TEXT NOT NULL,
    query_date TEXT NOT NULL,
    rank INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_key, query_symbol, query_date, rank),
    FOREIGN KEY (run_key) REFERENCES prediction_runs(run_key)
);
CREATE INDEX IF NOT EXISTS idx_stage15_matches_scope
    ON prediction_matches(run_key, query_date, query_symbol);
"""


class Stage15PredictionStore:
    """Persist each query atomically and resume only missing query keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(STAGE15_STORE_SCHEMA)

    @staticmethod
    def run_key(
        *,
        model: str,
        index_version: str,
        source_hash: str,
        parameters: dict[str, Any],
        scope: dict[str, Any],
    ) -> tuple[str, str, str]:
        parameters_json = _normalized_json(parameters)
        scope_json = _normalized_json(scope)
        material = _normalized_json(
            [model, index_version, source_hash, parameters_json, scope_json]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest(), parameters_json, scope_json

    def begin_run(
        self,
        *,
        model: str,
        index_version: str,
        source_hash: str,
        parameters: dict[str, Any],
        scope: dict[str, Any],
        target_count: int,
    ) -> str:
        if target_count < 0:
            raise ValueError("target_count cannot be negative")
        run_key, parameters_json, scope_json = self.run_key(
            model=model,
            index_version=index_version,
            source_hash=source_hash,
            parameters=parameters,
            scope=scope,
        )
        now = _utc_now()
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT INTO prediction_runs
                    (run_key, model, index_version, source_hash, parameters_json,
                     scope_json, status, target_count, completed_count, updated_at_utc, error)
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?, 0, ?, NULL)
                ON CONFLICT(run_key) DO UPDATE SET
                    status='running',
                    target_count=excluded.target_count,
                    completed_count=(
                        SELECT COUNT(*) FROM predictions WHERE predictions.run_key=excluded.run_key
                    ),
                    updated_at_utc=excluded.updated_at_utc,
                    error=NULL
                """,
                (
                    run_key,
                    model,
                    index_version,
                    source_hash,
                    parameters_json,
                    scope_json,
                    target_count,
                    now,
                ),
            )
        return run_key

    def completed_keys(self, run_key: str) -> set[tuple[str, date]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT symbol, signal_date FROM predictions WHERE run_key=?",
                (run_key,),
            )
        return {(str(symbol), date.fromisoformat(str(signal_date))) for symbol, signal_date in rows}

    def put_query(
        self,
        run_key: str,
        prediction: dict[str, object],
        matches: Iterable[dict[str, object]],
    ) -> None:
        symbol = str(prediction["symbol"])
        signal_date = date.fromisoformat(str(prediction["signal_date"]))
        match_rows = list(matches)
        now = _utc_now()
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions
                    (run_key, symbol, signal_date, payload_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_key, symbol, signal_date.isoformat(), _normalized_json(prediction), now),
            )
            connection.execute(
                "DELETE FROM prediction_matches WHERE run_key=? AND query_symbol=? AND query_date=?",
                (run_key, symbol, signal_date.isoformat()),
            )
            connection.executemany(
                """
                INSERT INTO prediction_matches
                    (run_key, query_symbol, query_date, rank, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_key,
                        symbol,
                        signal_date.isoformat(),
                        int(item["rank"]),
                        _normalized_json(item),
                    )
                    for item in match_rows
                ],
            )
            completed = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE run_key=?", (run_key,)
            ).fetchone()[0]
            connection.execute(
                """
                UPDATE prediction_runs
                SET completed_count=?, updated_at_utc=?
                WHERE run_key=?
                """,
                (completed, now, run_key),
            )

    def finish_run(self, run_key: str, *, status: str, error: str | None = None) -> None:
        if status not in {"partial", "complete", "failed"}:
            raise ValueError("invalid run status")
        with sqlite3.connect(self.path) as connection:
            completed = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE run_key=?", (run_key,)
            ).fetchone()[0]
            connection.execute(
                """
                UPDATE prediction_runs
                SET status=?, completed_count=?, updated_at_utc=?, error=?
                WHERE run_key=?
                """,
                (status, completed, _utc_now(), error, run_key),
            )

    def run_status(self, run_key: str) -> dict[str, object]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT model, index_version, source_hash, parameters_json, scope_json,
                       status, target_count, completed_count, updated_at_utc, error
                FROM prediction_runs WHERE run_key=?
                """,
                (run_key,),
            ).fetchone()
        if row is None:
            raise KeyError(run_key)
        return {
            "run_key": run_key,
            "model": row[0],
            "index_version": row[1],
            "source_hash": row[2],
            "parameters": json.loads(row[3]),
            "scope": json.loads(row[4]),
            "status": row[5],
            "target_count": int(row[6]),
            "completed_count": int(row[7]),
            "updated_at_utc": row[8],
            "error": row[9],
        }

    def load_predictions(self, run_key: str) -> list[dict[str, object]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM predictions
                WHERE run_key=? ORDER BY signal_date, symbol
                """,
                (run_key,),
            )
        return [json.loads(row[0]) for row in rows]

    def load_matches(self, run_key: str) -> list[dict[str, object]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM prediction_matches
                WHERE run_key=? ORDER BY query_date, query_symbol, rank
                """,
                (run_key,),
            )
        return [json.loads(row[0]) for row in rows]

    def export_csv(
        self,
        run_key: str,
        *,
        predictions_path: str | Path,
        matches_path: str | Path,
    ) -> None:
        _write_dict_rows(predictions_path, self.load_predictions(run_key))
        _write_dict_rows(matches_path, self.load_matches(run_key))


def _write_dict_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _normalized_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
