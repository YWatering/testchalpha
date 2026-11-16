"""Experimental FAISS IVF-PQ candidate indexes for one feature-index shard."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from urllib.parse import quote
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .offline_index import OfflineFeatureIndex, _unpack_vector
from .retrieval import (
    RetrievalSample,
    SampleEmbeddings,
    SimilarityMatch,
    cosine_distance,
    weighted_distance,
)


SCALES = ("monthly", "weekly", "daily")


class MultiScaleANNIndex:
    """Optional multi-shard ANN candidate layer with exact weighted reranking."""

    def __init__(self, manifest_path: str | Path, *, candidate_k: int = 10_000, nprobe: int = 64) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("安装 ANN 依赖: pip install -e '.[ann]'") from exc
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "kline-ann-manifest-v1":
            raise ValueError("ANN 清单格式不受支持")
        shards = manifest.get("shards", [])
        if not shards:
            raise ValueError("ANN 清单没有分片")
        self.model = str(manifest["model"])
        self.source_hash = str(manifest["source_hash"])
        self.dimension = int(manifest["dimension"])
        self.model_version = str(manifest.get("model_version", self.model))
        self.version_id = str(manifest["version_id"])
        self.checkpoint_sha256 = manifest.get("checkpoint_sha256")
        self.candidate_k = int(candidate_k)
        self.nprobe = int(nprobe)
        self._faiss = faiss
        self._eligible_row_id_cache: tuple[frozenset[date], tuple[np.ndarray, ...]] | None = None
        self._eligible_search_params_cache: tuple[frozenset[date], tuple[object, ...]] | None = None
        self._shards: list[dict[str, object]] = []
        symbols: set[str] = set()
        sample_count = 0
        for shard in shards:
            source = _resolve_manifest_path(self.manifest_path, shard["source"])
            ann_dir = _resolve_manifest_path(self.manifest_path, shard["ann"])
            shard_manifest = json.loads((ann_dir / "manifest.json").read_text(encoding="utf-8"))
            if shard_manifest.get("source_hash") != self.source_hash:
                raise ValueError("ANN 分片行情快照不一致")
            indexes = {scale: faiss.read_index(str(ann_dir / f"{scale}.faiss")) for scale in SCALES}
            combined_path = ann_dir / "combined.faiss"
            if combined_path.is_file():
                indexes["combined"] = faiss.read_index(str(combined_path))
            for index in indexes.values():
                index.nprobe = min(self.nprobe, index.nlist)
            connection = _open_ann_source(source)
            feature = connection.execute(
                "SELECT symbols_json, sample_count FROM feature_versions WHERE version_id=?",
                (shard_manifest["version_id"],),
            ).fetchone()
            if feature is None:
                connection.close()
                raise ValueError(f"ANN 源分片没有对应 ready 版本: {source}")
            symbols.update(json.loads(feature[0]))
            sample_count += int(feature[1])
            self._shards.append(
                {
                    "source": source,
                    "version_id": shard_manifest["version_id"],
                    "indexes": indexes,
                    "connection": connection,
                }
            )
        self.symbols = tuple(sorted(symbols))
        self.sample_count = sample_count

    def close(self) -> None:
        """Close read-only SQLite connections held by all ANN shards."""

        for shard in self._shards:
            connection = shard.get("connection")
            if connection is not None:
                connection.close()
                shard["connection"] = None

    def is_compatible(self, source_hash: str | None, config: object, *, price_basis: str | None = None) -> bool:
        return self.source_hash == (source_hash or "")

    def search(
        self,
        query: RetrievalSample,
        market_data: object,
        *,
        month_weight: float = 1.0,
        week_weight: float = 1.0,
        day_weight: float = 1.0,
        top_k: int = 20,
        embargo_trading_days: int = 0,
        eligible_candidate_dates: set[date] | None = None,
    ) -> tuple[list[SimilarityMatch], dict[str, int | float]]:
        started = perf_counter()
        query_vectors = (query.embeddings.monthly, query.embeddings.weekly, query.embeddings.daily)
        query_arrays = tuple(np.asarray(vector, dtype=np.float32) for vector in query_vectors)
        query_norms = tuple(float(np.linalg.norm(vector)) for vector in query_arrays)
        query_dates = market_data.dates(query.symbol)
        query_positions = {trading_date: index for index, trading_date in enumerate(query_dates)}
        query_index = query_positions.get(query.date)
        if query_index is None:
            raise ValueError(f"查询日期不在行情数据中: {query.symbol} {query.date}")
        candidates: list[tuple[float, str, str, float, float, float, SampleEmbeddings]] = []
        candidate_count = 0
        ann_search_ms = 0.0
        candidate_load_ms = 0.0
        metadata_load_ms = 0.0
        rerank_ms = 0.0
        prefilter_count = 0
        combined_candidate_ids: dict[int, set[int]] | None = None
        use_combined = (
            month_weight == 1.0
            and week_weight == 1.0
            and all("combined" in shard["indexes"] for shard in self._shards)
        )
        eligible_row_ids = (
            self._eligible_row_ids(eligible_candidate_dates)
            if eligible_candidate_dates is not None
            else None
        )
        eligible_search_params = (
            self._eligible_search_params(eligible_row_ids)
            if eligible_row_ids is not None
            else None
        )
        if use_combined:
            combined_started = perf_counter()
            combined_query = np.concatenate(query_arrays) / np.sqrt(3.0)
            approximate: list[tuple[float, int, int]] = []
            for shard_index, shard in enumerate(self._shards):
                scores, ids = _index_search(
                    shard["indexes"]["combined"],
                    combined_query.reshape(1, -1),
                    self.candidate_k,
                    eligible_search_params[shard_index] if eligible_search_params is not None else None,
                )
                approximate.extend(
                    (float(score), shard_index, int(rowid))
                    for score, rowid in zip(scores[0], ids[0])
                    if rowid >= 0
                )
            approximate.sort(key=lambda item: (-item[0], item[1], item[2]))
            global_budget_multiplier = 6 if eligible_candidate_dates is not None else len(self._shards)
            global_budget = min(
                len(approximate),
                max(self.candidate_k * 3, top_k * global_budget_multiplier),
            )
            if eligible_candidate_dates is not None and embargo_trading_days == 0:
                combined_candidate_ids = _group_top_candidate_ids(approximate, global_budget)
                prefilter_count = len(approximate)
            else:
                candidate_metadata: dict[tuple[int, int], tuple[str, str]] = {}
                ids_by_shard: dict[int, list[int]] = {}
                for _, shard_index, rowid in approximate:
                    ids_by_shard.setdefault(shard_index, []).append(rowid)
                for shard_index, rowids in ids_by_shard.items():
                    shard = self._shards[shard_index]
                    connection = shard["connection"]
                    version_id = shard["version_id"]
                    for start in range(0, len(rowids), 900):
                        batch = rowids[start : start + 900]
                        load_started = perf_counter()
                        placeholders = ",".join("?" for _ in batch)
                        rows = connection.execute(
                            f"SELECT rowid, symbol, date FROM samples "
                            f"WHERE version_id=? AND rowid IN ({placeholders})",
                            (version_id, *batch),
                        ).fetchall()
                        metadata_load_ms += (perf_counter() - load_started) * 1000
                        prefilter_count += len(rows)
                        candidate_metadata.update(
                            ((shard_index, int(rowid)), (symbol, date_text))
                            for rowid, symbol, date_text in rows
                        )
                combined_candidate_ids = _select_temporally_eligible_candidates(
                    approximate,
                    candidate_metadata,
                    query_symbol=query.symbol,
                    query_date=query.date,
                    query_positions=query_positions,
                    query_index=query_index,
                    embargo_trading_days=embargo_trading_days,
                    eligible_candidate_dates=eligible_candidate_dates,
                    budget=global_budget,
                )
            ann_search_ms += (perf_counter() - combined_started) * 1000

        for shard_index, shard in enumerate(self._shards):
            shard_search_started = perf_counter()
            indexes = shard["indexes"]
            if combined_candidate_ids is not None:
                candidate_ids = combined_candidate_ids.get(shard_index, set())
            else:
                candidate_ids: set[int] = set()
                for scale, vector in zip(SCALES, query_vectors):
                    _, ids = _index_search(
                        indexes[scale],
                        np.asarray(vector, dtype=np.float32).reshape(1, -1),
                        self.candidate_k,
                        eligible_search_params[shard_index] if eligible_search_params is not None else None,
                    )
                    candidate_ids.update(int(item) for item in ids[0] if item >= 0)
            if combined_candidate_ids is None:
                ann_search_ms += (perf_counter() - shard_search_started) * 1000
            if not candidate_ids:
                continue
            version_id = shard["version_id"]
            ordered_ids = sorted(candidate_ids)
            shard_candidates: list[tuple[float, str, str, float, float, float, SampleEmbeddings]] = []
            connection = shard["connection"]
            for start in range(0, len(ordered_ids), 900):
                batch = ordered_ids[start : start + 900]
                load_started = perf_counter()
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT rowid, symbol, date, monthly, weekly, daily FROM samples "
                    f"WHERE version_id=? AND rowid IN ({placeholders})",
                    (version_id, *batch),
                ).fetchall()
                candidate_load_ms += (perf_counter() - load_started) * 1000
                candidate_count += len(rows)
                eligible: list[tuple[str, str, bytes, bytes, bytes]] = []
                for _, symbol, date_text, monthly, weekly, daily in rows:
                    candidate_date = date.fromisoformat(date_text)
                    if candidate_date >= query.date:
                        continue
                    if (
                        eligible_candidate_dates is not None
                        and candidate_date not in eligible_candidate_dates
                    ):
                        continue
                    if symbol == query.symbol:
                        candidate_index = query_positions.get(candidate_date)
                        if candidate_index is None or query_index - candidate_index <= embargo_trading_days:
                            continue
                    eligible.append((symbol, date_text, monthly, weekly, daily))
                if not eligible:
                    continue

                rerank_started = perf_counter()
                monthly_matrix = _decode_vector_batch(
                    [item[2] for item in eligible], len(query_arrays[0])
                )
                weekly_matrix = _decode_vector_batch(
                    [item[3] for item in eligible], len(query_arrays[1])
                )
                daily_matrix = _decode_vector_batch(
                    [item[4] for item in eligible], len(query_arrays[2])
                )
                monthly_distances = _batch_cosine_distances(
                    query_arrays[0], query_norms[0], monthly_matrix
                )
                weekly_distances = _batch_cosine_distances(
                    query_arrays[1], query_norms[1], weekly_matrix
                )
                daily_distances = _batch_cosine_distances(
                    query_arrays[2], query_norms[2], daily_matrix
                )
                denominator = month_weight + week_weight + day_weight
                total_distances = (
                    month_weight * monthly_distances
                    + week_weight * weekly_distances
                    + day_weight * daily_distances
                ) / denominator
                batch_order = np.argsort(total_distances, kind="stable")[:top_k]
                for row_index in batch_order.tolist():
                    symbol, date_text, *_ = eligible[row_index]
                    shard_candidates.append(
                        (
                            float(total_distances[row_index]),
                            symbol,
                            date_text,
                            float(monthly_distances[row_index]),
                            float(weekly_distances[row_index]),
                            float(daily_distances[row_index]),
                            SampleEmbeddings(
                                tuple(float(value) for value in monthly_matrix[row_index]),
                                tuple(float(value) for value in weekly_matrix[row_index]),
                                tuple(float(value) for value in daily_matrix[row_index]),
                            ),
                        )
                    )
                rerank_ms += (perf_counter() - rerank_started) * 1000

            shard_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            candidates.extend(shard_candidates[:top_k])
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        matches = [
            SimilarityMatch(
                rank=rank,
                sample=RetrievalSample(symbol, date.fromisoformat(date_text), embeddings),
                distance=distance,
                monthly_distance=monthly_distance,
                weekly_distance=weekly_distance,
                daily_distance=daily_distance,
            )
            for rank, (
                distance,
                symbol,
                date_text,
                monthly_distance,
                weekly_distance,
                daily_distance,
                embeddings,
            ) in enumerate(candidates[:top_k], 1)
        ]
        return matches, {
            "ann_candidate_count": candidate_count,
            "ann_shard_count": len(self._shards),
            "ann_ms": round((perf_counter() - started) * 1000, 3),
            "ann_search_ms": round(ann_search_ms, 3),
            "ann_candidate_load_ms": round(candidate_load_ms, 3),
            "ann_metadata_load_ms": round(metadata_load_ms, 3),
            "ann_prefilter_count": prefilter_count,
            "ann_rerank_ms": round(rerank_ms, 3),
        }

    def _eligible_row_ids(
        self, eligible_candidate_dates: set[date]
    ) -> tuple[np.ndarray, ...]:
        key = frozenset(eligible_candidate_dates)
        if self._eligible_row_id_cache is not None and self._eligible_row_id_cache[0] == key:
            return self._eligible_row_id_cache[1]
        if not key:
            row_ids = tuple(np.empty(0, dtype=np.int64) for _ in self._shards)
        else:
            dates = tuple(sorted(key))
            placeholders = ",".join("?" for _ in dates)
            row_ids = tuple(
                np.fromiter(
                    (
                        int(row[0])
                        for row in shard["connection"].execute(
                            f"SELECT rowid FROM samples WHERE version_id=? AND date IN ({placeholders})",
                            (shard["version_id"], *(item.isoformat() for item in dates)),
                        )
                    ),
                    dtype=np.int64,
                )
                for shard in self._shards
            )
        self._eligible_row_id_cache = (key, row_ids)
        return row_ids

    def _eligible_search_params(self, row_ids: tuple[np.ndarray, ...]) -> tuple[object, ...]:
        key = self._eligible_row_id_cache[0] if self._eligible_row_id_cache else frozenset()
        if (
            self._eligible_search_params_cache is not None
            and self._eligible_search_params_cache[0] == key
        ):
            return self._eligible_search_params_cache[1]
        params = []
        for index_number, shard in enumerate(self._shards):
            search_params = self._faiss.SearchParametersIVF()
            search_params.nprobe = self.nprobe
            search_params.sel = self._faiss.IDSelectorBatch(row_ids[index_number])
            params.append(search_params)
        result = tuple(params)
        self._eligible_search_params_cache = (key, result)
        return result


def _index_search(index: object, query: np.ndarray, candidate_k: int, params: object | None):
    if params is None:
        return index.search(query, candidate_k)
    return index.search(query, candidate_k, params=params)


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _open_ann_source(source: Path) -> sqlite3.Connection:
    """Open immutable feature storage with a bounded per-shard read cache."""
    uri = f"file:{quote(str(source.resolve()))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA mmap_size=268435456")
    return connection


def _decode_vector_batch(values: list[bytes], dimension: int) -> np.ndarray:
    """Decode one SQLite blob column without constructing per-row tuples."""
    if not values:
        return np.empty((0, dimension), dtype=np.float32)
    expected_bytes = dimension * 4
    if any(len(value) != expected_bytes for value in values):
        raise ValueError("ANN 候选向量维度不一致")
    return np.frombuffer(b"".join(values), dtype="<f4").reshape(len(values), dimension)


def _batch_cosine_distances(
    query: np.ndarray,
    query_norm: float,
    candidates: np.ndarray,
) -> np.ndarray:
    if not len(candidates):
        return np.empty((0,), dtype=np.float32)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    similarities = np.zeros(len(candidates), dtype=np.float32)
    valid = (query_norm > 0.0) & (candidate_norms > 0.0)
    if np.any(valid):
        similarities[valid] = (candidates[valid] @ query) / (candidate_norms[valid] * query_norm)
    np.clip(similarities, -1.0, 1.0, out=similarities)
    distances = 1.0 - similarities
    both_zero = (query_norm == 0.0) & (candidate_norms == 0.0)
    distances[both_zero] = 0.0
    return distances


def _select_temporally_eligible_candidates(
    approximate: list[tuple[float, int, int]],
    metadata: dict[tuple[int, int], tuple[str, str]],
    *,
    query_symbol: str,
    query_date: date,
    query_positions: dict[date, int],
    query_index: int,
    embargo_trading_days: int,
    budget: int,
    eligible_candidate_dates: set[date] | None = None,
) -> dict[int, set[int]]:
    selected: dict[int, set[int]] = {}
    selected_count = 0
    for _, shard_index, rowid in approximate:
        candidate = metadata.get((shard_index, rowid))
        if candidate is None:
            continue
        symbol, date_text = candidate
        candidate_date = date.fromisoformat(date_text)
        if candidate_date >= query_date:
            continue
        if eligible_candidate_dates is not None and candidate_date not in eligible_candidate_dates:
            continue
        if symbol == query_symbol:
            candidate_index = query_positions.get(candidate_date)
            if candidate_index is None or query_index - candidate_index <= embargo_trading_days:
                continue
        selected.setdefault(shard_index, set()).add(rowid)
        selected_count += 1
        if selected_count >= budget:
            break
    return selected


def _group_top_candidate_ids(
    approximate: Sequence[tuple[float, int, int]],
    budget: int,
) -> dict[int, set[int]]:
    selected: dict[int, set[int]] = {}
    for _, shard_index, rowid in approximate[:budget]:
        selected.setdefault(shard_index, set()).add(rowid)
    return selected


def build_faiss_shard(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    training_rows: int = 100_000,
    nlist: int | None = None,
    batch_size: int = 16_384,
    training_end_date: str | None = None,
) -> dict[str, object]:
    """Build three scale indexes plus a default-weight combined IVF-PQ index."""

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("安装 ANN 依赖: pip install -e '.[ann]'") from exc
    source = Path(source_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as connection:
        version = connection.execute(
            "SELECT version_id, source_hash, model, dimension, sample_count FROM feature_versions "
            "WHERE status='ready' ORDER BY updated_at_utc DESC LIMIT 1"
        ).fetchone()
        if version is None:
            raise ValueError(f"索引分片没有 ready 版本: {source}")
        version_id, source_hash, model, dimension, row_count = version
        training_filter = " AND date<=?" if training_end_date else ""
        training_parameters = (version_id, training_end_date) if training_end_date else (version_id,)
        training_population = int(connection.execute(
            f"SELECT count(*) FROM samples WHERE version_id=?{training_filter}",
            training_parameters,
        ).fetchone()[0])
        if training_population < 1:
            raise ValueError("训练截止日期之前没有可用于训练 IVF 索引的样本")
        selected_nlist = nlist or min(1024, max(64, int(math.sqrt(training_population))))
        selected_nlist = min(selected_nlist, max(1, training_population // 39))
        if selected_nlist < 1:
            raise ValueError("样本量不足以训练 IVF 索引")
        configuration = {
            "format": "kline-faiss-multiscale-ivfpq-v1",
            "source_path": str(source.resolve()),
            "version_id": version_id,
            "source_hash": source_hash,
            "model": model,
            "dimension": int(dimension),
            "sample_count": int(row_count),
            "training_end_date": training_end_date,
            "training_sample_count": training_population,
            "nlist": selected_nlist,
            "pq_m": 16 if int(dimension) % 16 == 0 else 8,
            "pq_bits": 8,
        }
        (destination / "manifest.json").write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        result: dict[str, object] = {**configuration, "indexes": {}}
        for scale in SCALES:
            train_count = min(training_population, max(training_rows, selected_nlist * 39))
            cursor = connection.execute(
                f"SELECT rowid, {scale} FROM samples WHERE version_id=?{training_filter} ORDER BY rowid",
                training_parameters,
            )
            train_values = np.empty((train_count, int(dimension)), dtype=np.float32)
            seen = 0
            train_stride = max(1, training_population // train_count)
            for row_index, (_, blob) in enumerate(cursor):
                if row_index % train_stride == 0 and seen < train_count:
                    train_values[seen] = np.frombuffer(blob, dtype="<f4")
                    seen += 1
            train_values = train_values[:seen]
            if not len(train_values):
                raise ValueError(f"{scale} 没有训练向量")

            quantizer = faiss.IndexFlatIP(int(dimension))
            index = faiss.IndexIVFPQ(
                quantizer,
                int(dimension),
                selected_nlist,
                int(configuration["pq_m"]),
                int(configuration["pq_bits"]),
                faiss.METRIC_INNER_PRODUCT,
            )
            index.train(train_values)
            cursor = connection.execute(
                f"SELECT rowid, {scale} FROM samples AS s WHERE version_id=? ORDER BY rowid",
                (version_id,),
            )
            batch_ids: list[int] = []
            batch_vectors: list[np.ndarray] = []
            added = 0
            for rowid, blob in cursor:
                batch_ids.append(int(rowid))
                batch_vectors.append(np.frombuffer(blob, dtype="<f4"))
                if len(batch_ids) >= batch_size:
                    index.add_with_ids(
                        np.asarray(batch_vectors, dtype=np.float32),
                        np.asarray(batch_ids, dtype=np.int64),
                    )
                    added += len(batch_ids)
                    batch_ids.clear()
                    batch_vectors.clear()
            if batch_ids:
                index.add_with_ids(
                    np.asarray(batch_vectors, dtype=np.float32),
                    np.asarray(batch_ids, dtype=np.int64),
                )
                added += len(batch_ids)
            faiss.write_index(index, str(destination / f"{scale}.faiss"))
            result["indexes"][scale] = {"vectors": added, "training_vectors": len(train_values)}

        result["indexes"]["combined"] = _build_combined_faiss_index(
            connection,
            version_id=version_id,
            destination=destination,
            dimension=int(dimension),
            row_count=int(row_count),
            selected_nlist=selected_nlist,
            pq_bits=int(configuration["pq_bits"]),
            training_rows=training_rows,
            batch_size=batch_size,
            training_end_date=training_end_date,
            training_population=training_population,
        )
        configuration["combined_dimension"] = int(dimension) * len(SCALES)
        configuration["combined_weights"] = [1.0, 1.0, 1.0]
        (destination / "manifest.json").write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def build_combined_faiss_shard(
    source_path: str | Path,
    ann_dir: str | Path,
    *,
    training_rows: int = 100_000,
    batch_size: int = 16_384,
) -> dict[str, object]:
    """Add the default-weight combined index without rebuilding scale indexes."""

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("安装 ANN 依赖: pip install -e '.[ann]'") from exc
    del faiss  # Import here validates the optional dependency before opening files.
    source = Path(source_path)
    destination = Path(ann_dir)
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"ANN 分片清单不存在: {manifest_path}")
    configuration = json.loads(manifest_path.read_text(encoding="utf-8"))
    if configuration.get("format") != "kline-faiss-multiscale-ivfpq-v1":
        raise ValueError("ANN 分片清单格式不受支持")
    if Path(configuration["source_path"]).resolve() != source.resolve():
        raise ValueError("ANN 分片清单与源索引路径不匹配")
    with sqlite3.connect(source) as connection:
        version = connection.execute(
            "SELECT version_id, source_hash, model, dimension, sample_count FROM feature_versions "
            "WHERE status='ready' ORDER BY updated_at_utc DESC LIMIT 1"
        ).fetchone()
        if version is None:
            raise ValueError(f"索引分片没有 ready 版本: {source}")
        version_id, source_hash, model, dimension, row_count = version
        if version_id != configuration["version_id"] or source_hash != configuration["source_hash"]:
            raise ValueError("ANN 分片源版本已变化，请重新构建完整 ANN 分片")
        selected_nlist = int(configuration["nlist"])
        result = _build_combined_faiss_index(
            connection,
            version_id=version_id,
            destination=destination,
            dimension=int(dimension),
            row_count=int(row_count),
            selected_nlist=selected_nlist,
            pq_bits=int(configuration.get("pq_bits", 8)),
            training_rows=training_rows,
            batch_size=batch_size,
            training_end_date=configuration.get("training_end_date"),
            training_population=int(configuration.get("training_sample_count", row_count)),
        )
    configuration["combined_dimension"] = int(dimension) * len(SCALES)
    configuration["combined_weights"] = [1.0, 1.0, 1.0]
    manifest_path.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "source_path": str(source),
        "version_id": version_id,
        "model": model,
        "combined": result,
    }


def _build_combined_faiss_index(
    connection: sqlite3.Connection,
    *,
    version_id: str,
    destination: Path,
    dimension: int,
    row_count: int,
    selected_nlist: int,
    pq_bits: int,
    training_rows: int,
    batch_size: int,
    training_end_date: str | None = None,
    training_population: int | None = None,
) -> dict[str, object]:
    import faiss

    combined_dimension = dimension * len(SCALES)
    combined_m = 16 if combined_dimension % 16 == 0 else 8
    training_filter = " AND date<=?" if training_end_date else ""
    training_parameters = (version_id, training_end_date) if training_end_date else (version_id,)
    training_population = row_count if training_population is None else training_population
    combined_train_count = min(training_population, max(training_rows, selected_nlist * 39))
    combined_train_values = np.empty((combined_train_count, combined_dimension), dtype=np.float32)
    combined_seen = 0
    combined_stride = max(1, training_population // combined_train_count)
    cursor = connection.execute(
        f"SELECT rowid, monthly, weekly, daily FROM samples WHERE version_id=?{training_filter} ORDER BY rowid",
        training_parameters,
    )
    for row_index, (_, monthly, weekly, daily) in enumerate(cursor):
        if row_index % combined_stride == 0 and combined_seen < combined_train_count:
            combined_train_values[combined_seen] = _concat_scales(monthly, weekly, daily) / np.sqrt(3.0)
            combined_seen += 1
    combined_train_values = combined_train_values[:combined_seen]
    if not len(combined_train_values):
        raise ValueError("综合 ANN 没有训练向量")
    combined_index = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(combined_dimension),
        combined_dimension,
        selected_nlist,
        combined_m,
        pq_bits,
        faiss.METRIC_INNER_PRODUCT,
    )
    combined_index.train(combined_train_values)
    cursor = connection.execute(
        "SELECT rowid, monthly, weekly, daily FROM samples WHERE version_id=? ORDER BY rowid",
        (version_id,),
    )
    batch_ids: list[int] = []
    batch_vectors: list[np.ndarray] = []
    added = 0
    for rowid, monthly, weekly, daily in cursor:
        batch_ids.append(int(rowid))
        batch_vectors.append(_concat_scales(monthly, weekly, daily) / np.sqrt(3.0))
        if len(batch_ids) >= batch_size:
            combined_index.add_with_ids(
                np.asarray(batch_vectors, dtype=np.float32),
                np.asarray(batch_ids, dtype=np.int64),
            )
            added += len(batch_ids)
            batch_ids.clear()
            batch_vectors.clear()
    if batch_ids:
        combined_index.add_with_ids(
            np.asarray(batch_vectors, dtype=np.float32),
            np.asarray(batch_ids, dtype=np.int64),
        )
        added += len(batch_ids)
    destination.mkdir(parents=True, exist_ok=True)
    faiss.write_index(combined_index, str(destination / "combined.faiss"))
    return {"vectors": added, "training_vectors": len(combined_train_values), "weights": [1.0, 1.0, 1.0]}


def build_temporal_composite_index(
    source_manifest_path: str | Path,
    output_manifest_path: str | Path,
    *,
    training_end_date: str,
    training_rows: int = 100_000,
) -> dict[str, object]:
    """Build or resume a composite index whose IVF-PQ codebooks use only earlier samples."""

    source_manifest_path = Path(source_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("format") != "kline-ann-manifest-v1":
        raise ValueError("ANN 清单格式不受支持")
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ann_root = output_manifest_path.parent / "stage15_oos_ann" / str(source_manifest["model"])
    output_shards: list[dict[str, str]] = []
    for shard_number, shard in enumerate(source_manifest["shards"], start=1):
        source = _resolve_manifest_path(source_manifest_path, shard["source"])
        ann_dir = ann_root / Path(shard["source"]).stem
        shard_manifest_path = ann_dir / "manifest.json"
        complete = all((ann_dir / f"{scale}.faiss").is_file() for scale in (*SCALES, "combined"))
        if complete and shard_manifest_path.is_file():
            current = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            complete = (
                current.get("source_hash") == source_manifest["source_hash"]
                and current.get("training_end_date") == training_end_date
            )
        if not complete:
            print(f"building temporal ANN shard {shard_number}/{len(source_manifest['shards'])}: {source.name}", flush=True)
            build_faiss_shard(
                source,
                ann_dir,
                training_rows=training_rows,
                training_end_date=training_end_date,
            )
        output_shards.append({
            "source": os.path.relpath(source, output_manifest_path.parent),
            "ann": os.path.relpath(ann_dir, output_manifest_path.parent),
        })
    manifest = {
        **{key: value for key, value in source_manifest.items() if key not in {"created_at_utc", "version_id", "shards"}},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version_id": f"{source_manifest['version_id']}-train-through-{training_end_date}",
        "training_end_date": training_end_date,
        "training_policy": "ivf_pq_codebooks_fit_only_on_samples_on_or_before_training_end_date; all_vectors_indexed",
        "shards": output_shards,
    }
    temporary_path = output_manifest_path.with_suffix(output_manifest_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_manifest_path)
    return {"manifest": str(output_manifest_path), "shard_count": len(output_shards), "training_end_date": training_end_date}


def _concat_scales(monthly: bytes, weekly: bytes, daily: bytes) -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(monthly, dtype="<f4"),
            np.frombuffer(weekly, dtype="<f4"),
            np.frombuffer(daily, dtype="<f4"),
        ]
    )


def evaluate_faiss_recall(
    source_path: str | Path,
    ann_dir: str | Path,
    database_path: str | Path,
    *,
    queries: int = 100,
    candidate_k: int = 1000,
    top_k: int = 20,
    nprobe: int = 16,
    month_weight: float = 1.0,
    week_weight: float = 1.0,
    embargo_trading_days: int = 0,
) -> dict[str, object]:
    """Compare ANN candidate reranking against exact rankings on a date-stratified sample."""

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("安装 ANN 依赖: pip install -e '.[ann]'") from exc
    source = Path(source_path)
    root = Path(ann_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_path") != str(source.resolve()):
        raise ValueError("ANN 清单与源分片路径不匹配")
    indexes = {scale: faiss.read_index(str(root / f"{scale}.faiss")) for scale in SCALES}
    for index in indexes.values():
        index.nprobe = min(nprobe, index.nlist)
    base = OfflineFeatureIndex(source, version_id=manifest["version_id"])
    with sqlite3.connect(source) as connection:
        sample_ids = [
            row[0]
            for row in connection.execute(
                "SELECT rowid FROM samples WHERE version_id=? ORDER BY date, symbol",
                (manifest["version_id"],),
            )
        ]
    if not sample_ids:
        raise ValueError("源分片没有样本")
    selected_indices = np.linspace(0, len(sample_ids) - 1, min(queries, len(sample_ids)), dtype=int)
    recalls: list[float] = []
    elapsed_ann: list[float] = []
    elapsed_exact: list[float] = []
    skipped_no_history = 0
    for sample_index in selected_indices:
        with sqlite3.connect(source) as connection:
            rowid, symbol, date_text, monthly, weekly, daily = connection.execute(
                "SELECT rowid, symbol, date, monthly, weekly, daily FROM samples WHERE version_id=? AND rowid=?",
                (manifest["version_id"], sample_ids[int(sample_index)]),
            ).fetchone()
        query = SampleEmbeddings(
            _unpack_vector(monthly), _unpack_vector(weekly), _unpack_vector(daily)
        )
        query_sample = _as_sample(symbol, date_text, query)
        query_market = _load_one_symbol_market(database_path, symbol)
        exact_started = perf_counter()
        exact, _ = base.search(
            query_sample,
            query_market,
            month_weight=month_weight,
            week_weight=week_weight,
            top_k=top_k,
            embargo_trading_days=embargo_trading_days,
        )
        elapsed_exact.append((perf_counter() - exact_started) * 1000)
        if not exact:
            skipped_no_history += 1
            continue
        candidate_ids: set[int] = set()
        ann_started = perf_counter()
        for scale, vector in zip(SCALES, (query.monthly, query.weekly, query.daily)):
            _, ids = indexes[scale].search(
                np.asarray(vector, dtype=np.float32).reshape(1, -1), candidate_k
            )
            candidate_ids.update(int(item) for item in ids[0] if item >= 0)
        elapsed_ann.append((perf_counter() - ann_started) * 1000)
        if not candidate_ids:
            recalls.append(0.0)
            continue
        ordered_ids = sorted(candidate_ids)
        rows = []
        with sqlite3.connect(source) as connection:
            for start in range(0, len(ordered_ids), 500):
                batch = ordered_ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows.extend(connection.execute(
                    f"SELECT rowid, symbol, date, monthly, weekly, daily FROM samples "
                    f"WHERE version_id=? AND rowid IN ({placeholders})",
                    (manifest["version_id"], *batch),
                ).fetchall())
        candidate_rows = []
        for candidate_rowid, candidate_symbol, candidate_date, cm, cw, cd in rows:
            if candidate_date >= date_text:
                continue
            candidate_rows.append(
                (
                    candidate_rowid,
                    candidate_symbol,
                    candidate_date,
                    SampleEmbeddings(_unpack_vector(cm), _unpack_vector(cw), _unpack_vector(cd)),
                )
            )
        if not candidate_rows:
            recalls.append(0.0)
            continue
        query_positions = {day: position for position, day in enumerate(query_market.dates(symbol))}
        query_position = query_positions.get(query_sample.date)
        scored = []
        for _, candidate_symbol, candidate_date, embeddings in candidate_rows:
            if candidate_symbol == symbol:
                candidate_position = query_positions.get(date.fromisoformat(candidate_date))
                if candidate_position is None or query_position - candidate_position <= embargo_trading_days:
                    continue
            distance = weighted_distance(
                query,
                embeddings,
                month_weight=month_weight,
                week_weight=week_weight,
            )[0]
            scored.append((distance, candidate_symbol, candidate_date))
        approximate = sorted(scored)[:top_k]
        exact_keys = {(item.sample.symbol, item.sample.date.isoformat()) for item in exact}
        approximate_keys = {(item[1], item[2]) for item in approximate}
        recalls.append(len(exact_keys & approximate_keys) / max(1, len(exact_keys)))
    return {
        "queries": len(recalls),
        "skipped_queries_without_eligible_history": skipped_no_history,
        "evaluation_scope": "single_shard_exact_ground_truth",
        "candidate_k_per_scale": candidate_k,
        "top_k": top_k,
        "nprobe": nprobe,
        "mean_recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "minimum_recall_at_k": float(np.min(recalls)) if recalls else 0.0,
        "recall_at_k": recalls,
        "mean_ann_candidate_ms": float(np.mean(elapsed_ann)) if elapsed_ann else None,
        "mean_exact_search_ms": float(np.mean(elapsed_exact)) if elapsed_exact else None,
    }


def _load_one_symbol_market(source: str | Path, symbol: str):
    from .market_data import MarketData

    return MarketData.from_sqlite(source, symbols=(symbol,), price_basis="active")


def _as_sample(symbol: str, date_text: str, embeddings: SampleEmbeddings):
    from .retrieval import RetrievalSample

    return RetrievalSample(symbol, date.fromisoformat(date_text), embeddings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--training-rows", type=int, default=100_000)
    build.add_argument("--nlist", type=int)
    build.add_argument("--training-end-date")
    composite = subparsers.add_parser("build-temporal-composite")
    composite.add_argument("--source-manifest", type=Path, required=True)
    composite.add_argument("--output-manifest", type=Path, required=True)
    composite.add_argument("--training-end-date", required=True)
    composite.add_argument("--training-rows", type=int, default=100_000)
    combined = subparsers.add_parser("build-combined")
    combined.add_argument("--source", type=Path, required=True)
    combined.add_argument("--ann", type=Path, required=True)
    combined.add_argument("--training-rows", type=int, default=100_000)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--ann", type=Path, required=True)
    evaluate.add_argument("--database", type=Path, required=True)
    evaluate.add_argument("--queries", type=int, default=100)
    evaluate.add_argument("--candidate-k", type=int, default=1000)
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.add_argument("--nprobe", type=int, default=16)
    evaluate.add_argument("--month-weight", type=float, default=1.0)
    evaluate.add_argument("--week-weight", type=float, default=1.0)
    evaluate.add_argument("--embargo-trading-days", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        print(json.dumps(build_faiss_shard(
            args.source,
            args.output,
            training_rows=args.training_rows,
            nlist=args.nlist,
            training_end_date=args.training_end_date,
        ), indent=2))
    elif args.command == "build-temporal-composite":
        print(json.dumps(build_temporal_composite_index(
            args.source_manifest,
            args.output_manifest,
            training_end_date=args.training_end_date,
            training_rows=args.training_rows,
        ), indent=2))
    elif args.command == "build-combined":
        print(json.dumps(
            build_combined_faiss_shard(
                args.source,
                args.ann,
                training_rows=args.training_rows,
            ),
            indent=2,
        ))
    else:
        print(json.dumps(evaluate_faiss_recall(
            args.source, args.ann, args.database, queries=args.queries, candidate_k=args.candidate_k,
            top_k=args.top_k, nprobe=args.nprobe, month_weight=args.month_weight,
            week_weight=args.week_weight, embargo_trading_days=args.embargo_trading_days,
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
