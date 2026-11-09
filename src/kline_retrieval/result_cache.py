"""Versioned compact cache for materialized retrieval rankings."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .retrieval import RetrievalSample, SampleEmbeddings, SimilarityMatch


RESULT_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_results (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    index_version TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    symbol TEXT NOT NULL,
    query_date TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    matches_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_results_version
    ON retrieval_results(model, index_version, source_hash);
CREATE INDEX IF NOT EXISTS idx_retrieval_results_scope
    ON retrieval_results(model, index_version, source_hash, parameters_json, symbol, query_date);
CREATE TABLE IF NOT EXISTS retrieval_jobs (
    job_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    index_version TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    target_count INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    scope_json TEXT NOT NULL DEFAULT '{}',
    updated_at_utc TEXT NOT NULL,
    error TEXT
);
"""


class RetrievalResultCache:
    """SQLite-backed result cache storing candidate IDs and distances, never images."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(RESULT_CACHE_SCHEMA)
            job_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(retrieval_jobs)")
            }
            if "scope_json" not in job_columns:
                connection.execute(
                    "ALTER TABLE retrieval_jobs ADD COLUMN scope_json TEXT NOT NULL DEFAULT '{}'"
                )

    @staticmethod
    def key(
        *,
        model: str,
        index_version: str,
        source_hash: str,
        symbol: str,
        query_date: date,
        parameters: dict[str, Any],
    ) -> tuple[str, str]:
        normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        material = json.dumps(
            [model, index_version, source_hash, symbol, query_date.isoformat(), normalized],
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest(), normalized

    def get(
        self,
        *,
        model: str,
        index_version: str,
        source_hash: str,
        symbol: str,
        query_date: date,
        parameters: dict[str, Any],
    ) -> list[SimilarityMatch] | None:
        cache_key, _ = self.key(
            model=model,
            index_version=index_version,
            source_hash=source_hash,
            symbol=symbol,
            query_date=query_date,
            parameters=parameters,
        )
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT matches_json FROM retrieval_results WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return _decode_matches(row[0])

    def put(
        self,
        matches: list[SimilarityMatch],
        *,
        model: str,
        index_version: str,
        source_hash: str,
        symbol: str,
        query_date: date,
        parameters: dict[str, Any],
    ) -> None:
        cache_key, normalized = self.key(
            model=model,
            index_version=index_version,
            source_hash=source_hash,
            symbol=symbol,
            query_date=query_date,
            parameters=parameters,
        )
        payload = [
            {
                "rank": match.rank,
                "symbol": match.sample.symbol,
                "date": match.sample.date.isoformat(),
                "distance": match.distance,
                "monthly_distance": match.monthly_distance,
                "weekly_distance": match.weekly_distance,
                "daily_distance": match.daily_distance,
            }
            for match in matches
        ]
        encoded_payload = _encode_matches(payload)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT OR REPLACE INTO retrieval_results
                   (cache_key, model, index_version, source_hash, symbol, query_date,
                    parameters_json, matches_json, created_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cache_key,
                    model,
                    index_version,
                    source_hash,
                    symbol,
                    query_date.isoformat(),
                    normalized,
                    sqlite3.Binary(encoded_payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def begin_job(
        self,
        *,
        model: str,
        index_version: str,
        source_hash: str,
        parameters: dict[str, Any],
        target_count: int,
        scope: dict[str, Any] | None = None,
    ) -> str:
        normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        normalized_scope = json.dumps(scope or {}, sort_keys=True, separators=(",", ":"))
        job_key = hashlib.sha256(
            f"{model}:{index_version}:{source_hash}:{normalized}:{normalized_scope}".encode("utf-8")
        ).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """INSERT INTO retrieval_jobs
                   (job_key, model, index_version, source_hash, parameters_json, status,
                    target_count, completed_count, scope_json, updated_at_utc, error)
                   VALUES (?, ?, ?, ?, ?, 'running', ?, 0, ?, ?, NULL)
                   ON CONFLICT(job_key) DO UPDATE SET status='running', target_count=excluded.target_count,
                   scope_json=excluded.scope_json, updated_at_utc=excluded.updated_at_utc, error=NULL""",
                (
                    job_key,
                    model,
                    index_version,
                    source_hash,
                    normalized,
                    target_count,
                    normalized_scope,
                    now,
                ),
            )
        return job_key

    def update_job(self, job_key: str, completed_count: int, *, status: str, error: str | None = None) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """UPDATE retrieval_jobs SET completed_count=?, status=?, error=?, updated_at_utc=?
                   WHERE job_key=?""",
                (completed_count, status, error, datetime.now(timezone.utc).isoformat(), job_key),
            )

    def count(self, *, model: str, index_version: str, source_hash: str, parameters: dict[str, Any]) -> int:
        normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """SELECT COUNT(*) FROM retrieval_results
                   WHERE model=? AND index_version=? AND source_hash=? AND parameters_json=?""",
                (model, index_version, source_hash, normalized),
            ).fetchone()
        return int(row[0])

    def count_scope(
        self,
        *,
        model: str,
        index_version: str,
        source_hash: str,
        parameters: dict[str, Any],
        date_ranges: dict[str, tuple[date, date]],
    ) -> int:
        normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self.path) as connection:
            total = 0
            for symbol, (first_date, last_date) in date_ranges.items():
                total += connection.execute(
                    """SELECT COUNT(*) FROM retrieval_results
                       WHERE model=? AND index_version=? AND source_hash=? AND parameters_json=?
                       AND symbol=? AND query_date BETWEEN ? AND ?""",
                    (
                        model,
                        index_version,
                        source_hash,
                        normalized,
                        symbol,
                        first_date.isoformat(),
                        last_date.isoformat(),
                    ),
                ).fetchone()[0]
        return int(total)

    def status(self, *, model: str | None = None, index_version: str | None = None) -> dict[str, object]:
        clauses: list[str] = []
        values: list[str] = []
        if model is not None:
            clauses.append("model=?")
            values.append(model)
        if index_version is not None:
            clauses.append("index_version=?")
            values.append(index_version)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with sqlite3.connect(self.path) as connection:
            cached = connection.execute(
                f"SELECT COUNT(*) FROM retrieval_results{where}", values
            ).fetchone()[0]
            jobs = connection.execute(
                "SELECT model, index_version, source_hash, status, target_count, completed_count, scope_json, updated_at_utc, error "
                "FROM retrieval_jobs ORDER BY updated_at_utc DESC LIMIT 20"
            ).fetchall()
        return {
            "cached_query_count": int(cached),
            "jobs": [
                {
                    "model": row[0],
                    "index_version": row[1],
                    "source_hash": row[2],
                    "status": row[3],
                    "target_count": row[4],
                    "completed_count": row[5],
                    "scope": json.loads(row[6]),
                    "updated_at_utc": row[7],
                    "error": row[8],
                }
                for row in jobs
            ],
        }


def _encode_matches(values: list[dict[str, Any]]) -> bytes:
    chunks = [b"RSC1", struct.pack("!H", len(values))]
    for item in values:
        symbol = str(item["symbol"]).encode("utf-8")
        if len(symbol) > 255:
            raise ValueError("缓存中的股票代码长度超过255字节")
        chunks.append(
            struct.pack(
                "!BHI4d",
                len(symbol),
                int(item["rank"]),
                date.fromisoformat(item["date"]).toordinal(),
                float(item["distance"]),
                float(item["monthly_distance"]),
                float(item["weekly_distance"]),
                float(item["daily_distance"]),
            )
        )
        chunks.append(symbol)
    return b"".join(chunks)


def _decode_matches(payload: str | bytes | memoryview) -> list[SimilarityMatch]:
    if isinstance(payload, str):
        values = json.loads(payload)
    else:
        raw = bytes(payload)
        if not raw.startswith(b"RSC1"):
            raise ValueError("检索结果缓存的二进制版本标记无效")
        count = struct.unpack_from("!H", raw, 4)[0]
        values = []
        offset = 6
        record_header = struct.Struct("!BHI4d")
        for _ in range(count):
            symbol_length, rank, ordinal, distance, monthly, weekly, daily = (
                record_header.unpack_from(raw, offset)
            )
            offset += record_header.size
            symbol = raw[offset : offset + symbol_length].decode("utf-8")
            offset += symbol_length
            values.append(
                {
                    "rank": rank,
                    "symbol": symbol,
                    "date": date.fromordinal(ordinal).isoformat(),
                    "distance": distance,
                    "monthly_distance": monthly,
                    "weekly_distance": weekly,
                    "daily_distance": daily,
                }
            )
    return [
        SimilarityMatch(
            rank=int(item["rank"]),
            sample=RetrievalSample(
                symbol=item["symbol"],
                date=date.fromisoformat(item["date"]),
                embeddings=SampleEmbeddings((), (), ()),
            ),
            distance=float(item["distance"]),
            monthly_distance=float(item["monthly_distance"]),
            weekly_distance=float(item["weekly_distance"]),
            daily_distance=float(item["daily_distance"]),
        )
        for item in values
    ]
