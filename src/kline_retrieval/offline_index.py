"""Persistent ColProj feature storage and exact multi-scale retrieval."""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from .analytics import calculate_forward_returns
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData
from .retrieval import (
    RetrievalSample,
    SampleEmbeddings,
    SimilarityMatch,
    build_sample_embedding,
    weighted_distance,
)


INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_versions (
    version_id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    symbols_json TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    status TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    version_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    monthly BLOB NOT NULL,
    weekly BLOB NOT NULL,
    daily BLOB NOT NULL,
    PRIMARY KEY (version_id, symbol, date),
    FOREIGN KEY (version_id) REFERENCES feature_versions(version_id)
);
CREATE INDEX IF NOT EXISTS idx_samples_version_date
    ON samples (version_id, date);
CREATE INDEX IF NOT EXISTS idx_samples_version_symbol_date
    ON samples (version_id, symbol, date);
CREATE TABLE IF NOT EXISTS return_labels (
    version_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    base_close REAL NOT NULL,
    return_1 REAL NOT NULL,
    return_5 REAL NOT NULL,
    return_10 REAL NOT NULL,
    return_20 REAL NOT NULL,
    PRIMARY KEY (version_id, symbol, date),
    FOREIGN KEY (version_id) REFERENCES feature_versions(version_id)
);
"""

INDEX_SCAN_CHUNK_SIZE = 65536


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    index_path: Path
    version_id: str
    symbol_count: int
    sample_count: int
    inserted_count: int
    skipped_count: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index_path": str(self.index_path),
            "version_id": self.version_id,
            "symbol_count": self.symbol_count,
            "sample_count": self.sample_count,
            "inserted_count": self.inserted_count,
            "skipped_count": self.skipped_count,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def build_colproj_index(
    database_path: str | Path,
    index_path: str | Path,
    *,
    symbols: Sequence[str] | None = None,
    config: ProjectConfig = DEFAULT_CONFIG,
    replace: bool = False,
    batch_size: int = 500,
) -> IndexBuildResult:
    """Build or incrementally extend a versioned exact ColProj index."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if tuple(config.future_horizons) != (1, 5, 10, 20):
        raise ValueError("阶段九索引要求 future_horizons 固定为 (1, 5, 10, 20)")
    started = perf_counter()
    source_data = MarketData.from_sqlite(database_path, symbols=symbols)
    selected_symbols = tuple(symbols) if symbols else source_data.symbols
    if tuple(selected_symbols) != source_data.symbols:
        selected_symbols = tuple(symbol for symbol in selected_symbols if symbol in source_data.symbols)
    if not selected_symbols:
        raise ValueError("没有可用于构建索引的股票")

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    version_id = _version_id(config, selected_symbols)
    config_json = json.dumps(_index_config(config), ensure_ascii=False, sort_keys=True)
    symbols_json = json.dumps(list(selected_symbols), ensure_ascii=False)
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(path, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(INDEX_SCHEMA)
        existing = connection.execute(
            "SELECT status, config_json FROM feature_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        if existing and json.loads(existing[1]) != json.loads(config_json):
            raise ValueError("索引版本配置与当前配置不一致，请更换 index_path 或使用 --replace")
        if replace:
            connection.execute("DELETE FROM return_labels WHERE version_id = ?", (version_id,))
            connection.execute("DELETE FROM samples WHERE version_id = ?", (version_id,))
        connection.execute(
            """
            INSERT INTO feature_versions
                (version_id, source_hash, config_json, symbols_json, model, dimension,
                 status, sample_count, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, 'colproj', ?, 'building', 0, ?, ?)
            ON CONFLICT(version_id) DO UPDATE SET
                source_hash=excluded.source_hash,
                symbols_json=excluded.symbols_json,
                status='building',
                updated_at_utc=excluded.updated_at_utc
            """,
            (version_id, source_data.source_hash or "", config_json, symbols_json,
             config.chart_width, now, now),
        )
        existing_keys = set()
        if not replace:
            existing_keys = {
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT symbol, date FROM samples WHERE version_id = ?", (version_id,)
                )
            }

        sample_rows: list[tuple[object, ...]] = []
        label_rows: list[tuple[object, ...]] = []
        inserted_count = 0
        skipped_count = 0
        for symbol in selected_symbols:
            dates = source_data.dates(symbol)
            first_valid = config.monthly_period * config.daily_window - 1
            last_valid = len(dates) - max(config.future_horizons) - 1
            if last_valid < first_valid:
                continue
            for index in range(first_valid, last_valid + 1):
                trading_date = dates[index]
                key = (symbol, trading_date.isoformat())
                if key in existing_keys:
                    skipped_count += 1
                    continue
                sample = build_sample_embedding(
                    source_data, symbol, trading_date, config=config
                )
                returns = calculate_forward_returns(
                    source_data, symbol, trading_date, config.future_horizons
                )
                values = dict(returns.values)
                sample_rows.append(
                    (
                        version_id,
                        symbol,
                        trading_date.isoformat(),
                        _pack_vector(sample.embeddings.monthly),
                        _pack_vector(sample.embeddings.weekly),
                        _pack_vector(sample.embeddings.daily),
                    )
                )
                label_rows.append(
                    (
                        version_id,
                        symbol,
                        trading_date.isoformat(),
                        returns.base_close,
                        values[1],
                        values[5],
                        values[10],
                        values[20],
                    )
                )
                if len(sample_rows) >= batch_size:
                    inserted_count += _insert_batches(connection, sample_rows, label_rows)
                    sample_rows.clear()
                    label_rows.clear()
        if sample_rows:
            inserted_count += _insert_batches(connection, sample_rows, label_rows)

        sample_count = connection.execute(
            "SELECT COUNT(*) FROM samples WHERE version_id = ?", (version_id,)
        ).fetchone()[0]
        connection.execute(
            """
            UPDATE feature_versions
            SET source_hash = ?, status = 'ready', sample_count = ?, updated_at_utc = ?
            WHERE version_id = ?
            """,
            (source_data.source_hash or "", sample_count, datetime.now(timezone.utc).isoformat(), version_id),
        )
        connection.commit()

    return IndexBuildResult(
        index_path=path,
        version_id=version_id,
        symbol_count=len(selected_symbols),
        sample_count=sample_count,
        inserted_count=inserted_count,
        skipped_count=skipped_count,
        elapsed_seconds=perf_counter() - started,
    )


class OfflineFeatureIndex:
    """Read a ready index and return the same ``SimilarityMatch`` contract as scanning."""

    def __init__(self, index_path: str | Path, version_id: str | None = None) -> None:
        self.path = Path(index_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"离线特征索引不存在: {self.path}")
        with sqlite3.connect(self.path) as connection:
            connection.executescript(INDEX_SCHEMA)
            rows = connection.execute(
                """
                SELECT version_id, source_hash, config_json, symbols_json, model,
                       dimension, status, sample_count
                FROM feature_versions
                WHERE status = 'ready'
                ORDER BY updated_at_utc DESC
                """
            ).fetchall()
        if not rows:
            raise ValueError(f"离线特征索引没有 ready 版本: {self.path}")
        selected = next((row for row in rows if row[0] == version_id), rows[0])
        self.version_id = selected[0]
        self.source_hash = selected[1]
        self.config = json.loads(selected[2])
        self.symbols = tuple(json.loads(selected[3]))
        self.model = selected[4]
        self.dimension = int(selected[5])
        self.sample_count = int(selected[7])

    def is_compatible(self, source_hash: str | None, config: ProjectConfig) -> bool:
        return self.source_hash == (source_hash or "") and self.config == _index_config(config)

    def search(
        self,
        query: RetrievalSample,
        market_data: MarketData,
        *,
        month_weight: float = 1.0,
        week_weight: float = 1.0,
        day_weight: float = 1.0,
        top_k: int = 20,
        embargo_trading_days: int = 0,
    ) -> tuple[list[SimilarityMatch], dict[str, int | float]]:
        started = perf_counter()
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query_dates = market_data.dates(query.symbol)
        query_index = query_dates.index(query.date)
        ranked_rows: list[tuple[float, tuple[str, str, bytes, bytes, bytes]]] = []
        candidate_count = 0
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                SELECT symbol, date, monthly, weekly, daily
                FROM samples
                WHERE version_id = ? AND date < ?
                ORDER BY date, symbol
                """,
                (self.version_id, query.date.isoformat()),
            )
            while True:
                rows = cursor.fetchmany(INDEX_SCAN_CHUNK_SIZE)
                if not rows:
                    break
                candidate_rows: list[tuple[str, str, bytes, bytes, bytes]] = []
                for row in rows:
                    symbol, date_text = row[0], row[1]
                    candidate_date = date.fromisoformat(date_text)
                    if symbol == query.symbol:
                        try:
                            distance_in_trading_days = query_index - market_data.dates(symbol).index(candidate_date)
                        except ValueError:
                            continue
                        if distance_in_trading_days <= embargo_trading_days:
                            continue
                    candidate_rows.append((symbol, date_text, row[2], row[3], row[4]))
                if not candidate_rows:
                    continue

                distances = _distance_rows(
                    query.embeddings,
                    candidate_rows,
                    month_weight=month_weight,
                    week_weight=week_weight,
                    day_weight=day_weight,
                )
                candidate_count += len(candidate_rows)
                ranked_rows.extend(zip(distances, candidate_rows))
                ranked_rows.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
                del ranked_rows[top_k:]

        if not ranked_rows:
            return [], {"index_candidate_count": 0, "index_ms": round((perf_counter() - started) * 1000, 3)}

        ranked = ranked_rows
        matches: list[SimilarityMatch] = []
        for rank, (distance, row) in enumerate(ranked, 1):
            symbol, date_text, monthly, weekly, daily = row
            sample = RetrievalSample(
                symbol=symbol,
                date=date.fromisoformat(date_text),
                embeddings=SampleEmbeddings(
                    monthly=_unpack_vector(monthly),
                    weekly=_unpack_vector(weekly),
                    daily=_unpack_vector(daily),
                ),
            )
            total, month_distance, week_distance, day_distance = weighted_distance(
                query.embeddings,
                sample.embeddings,
                month_weight=month_weight,
                week_weight=week_weight,
                day_weight=day_weight,
            )
            matches.append(
                SimilarityMatch(
                    rank=rank,
                    sample=sample,
                    distance=total,
                    monthly_distance=month_distance,
                    weekly_distance=week_distance,
                    daily_distance=day_distance,
                )
            )
        return matches, {
            "index_candidate_count": candidate_count,
            "index_ms": round((perf_counter() - started) * 1000, 3),
        }


class CompositeOfflineFeatureIndex:
    """Query multiple ready index shards as one exact logical index."""

    def __init__(self, index_paths: Sequence[str | Path]) -> None:
        paths = tuple(Path(path) for path in index_paths)
        if not paths:
            raise ValueError("至少需要一个索引分片")
        self.indexes = tuple(OfflineFeatureIndex(path) for path in paths)
        first = self.indexes[0]
        self.paths = paths
        self.config = first.config
        self.model = first.model
        self.dimension = first.dimension
        self.version_id = "composite-" + sha256(
            "|".join(index.version_id for index in self.indexes).encode("utf-8")
        ).hexdigest()[:16]
        self.source_hash = first.source_hash
        self.symbols = tuple(sorted({symbol for index in self.indexes for symbol in index.symbols}))
        self.sample_count = sum(index.sample_count for index in self.indexes)

    def is_compatible(self, source_hash: str | None, config: ProjectConfig) -> bool:
        return all(index.is_compatible(source_hash, config) for index in self.indexes)

    def search(
        self,
        query: RetrievalSample,
        market_data: MarketData,
        *,
        month_weight: float = 1.0,
        week_weight: float = 1.0,
        day_weight: float = 1.0,
        top_k: int = 20,
        embargo_trading_days: int = 0,
    ) -> tuple[list[SimilarityMatch], dict[str, int | float]]:
        started = perf_counter()
        matches: list[SimilarityMatch] = []
        candidate_count = 0
        shard_times = 0.0
        for index in self.indexes:
            shard_matches, diagnostics = index.search(
                query,
                market_data,
                month_weight=month_weight,
                week_weight=week_weight,
                day_weight=day_weight,
                top_k=top_k,
                embargo_trading_days=embargo_trading_days,
            )
            matches.extend(shard_matches)
            candidate_count += int(diagnostics.get("index_candidate_count", 0))
            shard_times += float(diagnostics.get("index_ms", 0.0))
        ranked = sorted(
            matches,
            key=lambda item: (item.distance, item.sample.symbol, item.sample.date),
        )[:top_k]
        ranked = [
            SimilarityMatch(
                rank=rank,
                sample=item.sample,
                distance=item.distance,
                monthly_distance=item.monthly_distance,
                weekly_distance=item.weekly_distance,
                daily_distance=item.daily_distance,
            )
            for rank, item in enumerate(ranked, 1)
        ]
        return ranked, {
            "index_candidate_count": candidate_count,
            "index_shard_count": len(self.indexes),
            "index_shard_ms": round(shard_times, 3),
            "index_ms": round((perf_counter() - started) * 1000, 3),
        }


def _distance_rows(
    query: SampleEmbeddings,
    rows: Sequence[tuple[str, str, bytes, bytes, bytes]],
    *,
    month_weight: float,
    week_weight: float,
    day_weight: float,
) -> list[float]:
    try:
        import numpy as np
    except ImportError:
        return [
            weighted_distance(
                query,
                SampleEmbeddings(
                    monthly=_unpack_vector(row[2]),
                    weekly=_unpack_vector(row[3]),
                    daily=_unpack_vector(row[4]),
                ),
                month_weight=month_weight,
                week_weight=week_weight,
                day_weight=day_weight,
            )[0]
            for row in rows
        ]

    monthly = np.vstack([np.frombuffer(row[2], dtype="<f4") for row in rows])
    weekly = np.vstack([np.frombuffer(row[3], dtype="<f4") for row in rows])
    daily = np.vstack([np.frombuffer(row[4], dtype="<f4") for row in rows])
    query_monthly = np.asarray(query.monthly, dtype=np.float32)
    query_weekly = np.asarray(query.weekly, dtype=np.float32)
    query_daily = np.asarray(query.daily, dtype=np.float32)
    monthly_distance = 1.0 - monthly @ query_monthly
    weekly_distance = 1.0 - weekly @ query_weekly
    daily_distance = 1.0 - daily @ query_daily
    total_weight = month_weight + week_weight + day_weight
    return (
        (month_weight * monthly_distance + week_weight * weekly_distance + day_weight * daily_distance)
        / total_weight
    ).tolist()


def _insert_batches(
    connection: sqlite3.Connection,
    sample_rows: Sequence[tuple[object, ...]],
    label_rows: Sequence[tuple[object, ...]],
) -> int:
    connection.executemany(
        """
        INSERT OR REPLACE INTO samples
            (version_id, symbol, date, monthly, weekly, daily)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        sample_rows,
    )
    connection.executemany(
        """
        INSERT OR REPLACE INTO return_labels
            (version_id, symbol, date, base_close, return_1, return_5, return_10, return_20)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        label_rows,
    )
    return len(sample_rows)


def _pack_vector(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(value: bytes) -> tuple[float, ...]:
    if len(value) % 4:
        raise ValueError("离线向量 BLOB 长度不是 float32 的整数倍")
    return struct.unpack(f"<{len(value) // 4}f", value)


def _index_config(config: ProjectConfig) -> dict[str, object]:
    return {
        "monthly_period": config.monthly_period,
        "weekly_period": config.weekly_period,
        "daily_window": config.daily_window,
        "future_chart_window": config.future_chart_window,
        "chart_size": list(config.chart_size),
        "future_horizons": list(config.future_horizons),
        "model": "colproj",
        "dimension": config.chart_width,
    }


def _version_id(config: ProjectConfig, symbols: Sequence[str]) -> str:
    payload = json.dumps(
        {"config": _index_config(config), "symbols": list(symbols)},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return f"colproj-{sha256(payload).hexdigest()[:16]}"
