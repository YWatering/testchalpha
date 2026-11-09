"""Experimental FAISS IVF-PQ candidate indexes for one feature-index shard."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import date
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
            for index in indexes.values():
                index.nprobe = min(self.nprobe, index.nlist)
            with sqlite3.connect(source) as connection:
                feature = connection.execute(
                    "SELECT symbols_json, sample_count FROM feature_versions WHERE version_id=?",
                    (shard_manifest["version_id"],),
                ).fetchone()
            if feature is None:
                raise ValueError(f"ANN 源分片没有对应 ready 版本: {source}")
            symbols.update(json.loads(feature[0]))
            sample_count += int(feature[1])
            self._shards.append(
                {"source": source, "version_id": shard_manifest["version_id"], "indexes": indexes}
            )
        self.symbols = tuple(sorted(symbols))
        self.sample_count = sample_count

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
    ) -> tuple[list[SimilarityMatch], dict[str, int | float]]:
        started = perf_counter()
        query_vectors = (query.embeddings.monthly, query.embeddings.weekly, query.embeddings.daily)
        query_dates = market_data.dates(query.symbol)
        query_positions = {trading_date: index for index, trading_date in enumerate(query_dates)}
        query_index = query_positions.get(query.date)
        if query_index is None:
            raise ValueError(f"查询日期不在行情数据中: {query.symbol} {query.date}")
        candidates: list[tuple[float, str, str, SampleEmbeddings]] = []
        candidate_count = 0
        for shard in self._shards:
            indexes = shard["indexes"]
            candidate_ids: set[int] = set()
            for scale, vector in zip(SCALES, query_vectors):
                _, ids = indexes[scale].search(
                    np.asarray(vector, dtype=np.float32).reshape(1, -1), self.candidate_k
                )
                candidate_ids.update(int(item) for item in ids[0] if item >= 0)
            if not candidate_ids:
                continue
            source = shard["source"]
            version_id = shard["version_id"]
            ordered_ids = sorted(candidate_ids)
            with sqlite3.connect(source) as connection:
                for start in range(0, len(ordered_ids), 500):
                    batch = ordered_ids[start : start + 500]
                    placeholders = ",".join("?" for _ in batch)
                    rows = connection.execute(
                        f"SELECT rowid, symbol, date, monthly, weekly, daily FROM samples "
                        f"WHERE version_id=? AND rowid IN ({placeholders})",
                        (version_id, *batch),
                    ).fetchall()
                    candidate_count += len(rows)
                    for _, symbol, date_text, monthly, weekly, daily in rows:
                        candidate_date = date.fromisoformat(date_text)
                        if candidate_date >= query.date:
                            continue
                        if symbol == query.symbol:
                            candidate_index = query_positions.get(candidate_date)
                            if candidate_index is None or query_index - candidate_index <= embargo_trading_days:
                                continue
                        embeddings = SampleEmbeddings(
                            _unpack_vector(monthly), _unpack_vector(weekly), _unpack_vector(daily)
                        )
                        distances = weighted_distance(
                            query.embeddings,
                            embeddings,
                            month_weight=month_weight,
                            week_weight=week_weight,
                            day_weight=day_weight,
                        )
                        candidates.append((distances[0], symbol, date_text, embeddings))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        matches = [
            SimilarityMatch(
                rank=rank,
                sample=RetrievalSample(symbol, date.fromisoformat(date_text), embeddings),
                distance=distance,
                monthly_distance=cosine_distance(query.embeddings.monthly, embeddings.monthly),
                weekly_distance=cosine_distance(query.embeddings.weekly, embeddings.weekly),
                daily_distance=cosine_distance(query.embeddings.daily, embeddings.daily),
            )
            for rank, (distance, symbol, date_text, embeddings) in enumerate(candidates[:top_k], 1)
        ]
        return matches, {
            "ann_candidate_count": candidate_count,
            "ann_shard_count": len(self._shards),
            "ann_ms": round((perf_counter() - started) * 1000, 3),
        }


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def build_faiss_shard(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    training_rows: int = 100_000,
    nlist: int | None = None,
    batch_size: int = 16_384,
) -> dict[str, object]:
    """Build three compressed cosine IVF-PQ indexes keyed by SQLite sample rowid."""

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
        selected_nlist = nlist or min(1024, max(64, int(math.sqrt(row_count))))
        selected_nlist = min(selected_nlist, max(1, row_count // 39))
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
            "nlist": selected_nlist,
            "pq_m": 16 if int(dimension) % 16 == 0 else 8,
            "pq_bits": 8,
        }
        (destination / "manifest.json").write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        result: dict[str, object] = {**configuration, "indexes": {}}
        for scale in SCALES:
            field = f"s.{scale}"
            cursor = connection.execute(
                f"SELECT rowid, {scale} FROM samples AS s WHERE version_id=? ORDER BY rowid",
                (version_id,),
            )
            train_count = min(int(row_count), max(training_rows, selected_nlist * 39))
            train_values = np.empty((train_count, int(dimension)), dtype=np.float32)
            seen = 0
            train_stride = max(1, int(row_count) // train_count)
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
    return result


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
        print(json.dumps(build_faiss_shard(args.source, args.output, training_rows=args.training_rows, nlist=args.nlist), indent=2))
    else:
        print(json.dumps(evaluate_faiss_recall(
            args.source, args.ann, args.database, queries=args.queries, candidate_k=args.candidate_k,
            top_k=args.top_k, nprobe=args.nprobe, month_weight=args.month_weight,
            week_weight=args.week_weight, embargo_trading_days=args.embargo_trading_days,
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
