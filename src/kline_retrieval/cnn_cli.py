"""Command-line entry points for the stage-11 CNN prototype."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .cnn import (
    CNNEmbedder,
    CNNTrainingConfig,
    build_cnn_sample_embedding,
    build_split_manifest,
    load_split_manifest,
    train_cnn,
    write_split_manifest,
)
from .config import ProjectConfig
from .analytics import calculate_forward_returns
from .market_data import MarketData
from .retrieval import build_sample_embedding, search_similar_samples


DEFAULT_SYMBOLS = ("000002", "000333", "300750", "600519", "688981", "920885")


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练阶段十一 CNN 原型")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output-dir", default="data/processed/cnn_stage11")
    parser.add_argument("--symbols", default=','.join(DEFAULT_SYMBOLS))
    parser.add_argument("--max-per-split", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args(argv)

    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    config = ProjectConfig(
        data_path=Path(args.database),
        active_symbols=symbols,
    )
    market_data = MarketData.from_sqlite(args.database, symbols=symbols)
    manifest = build_split_manifest(
        market_data,
        symbols,
        config=config,
        max_per_symbol_split=args.max_per_split,
    )
    output_dir = Path(args.output_dir)
    manifest_path = write_split_manifest(manifest, output_dir / "splits.json")
    result = train_cnn(
        market_data,
        manifest,
        output_dir / "cnn_stage11.pt",
        config=CNNTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )
    result["manifest"] = str(manifest_path)
    (output_dir / "training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估 CNN 与 ColProj 的固定查询集差异")
    parser.add_argument("--database", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="data/processed/cnn_stage11/evaluation.json")
    parser.add_argument("--max-queries", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    manifest = load_split_manifest(args.manifest)
    test_items = _balanced_items(manifest["splits"]["test"], args.max_queries)
    candidate_items = _balanced_items(manifest["splits"]["test"], args.max_candidates)
    symbols = sorted({item["symbol"] for item in test_items + candidate_items})
    config = ProjectConfig(data_path=Path(args.database), active_symbols=tuple(symbols))
    market_data = MarketData.from_sqlite(args.database, symbols=symbols)
    embedder = CNNEmbedder(args.checkpoint)

    overlaps: list[float] = []
    rows: list[dict[str, object]] = []
    return_values = {"colproj": {1: [], 5: [], 10: [], 20: []}, "cnn": {1: [], 5: [], 10: [], 20: []}}
    started = perf_counter()
    for item in test_items:
        query_date = date.fromisoformat(item["date"])
        colproj_query = build_sample_embedding(market_data, item["symbol"], query_date, config=config)
        cnn_query = build_cnn_sample_embedding(market_data, item["symbol"], query_date, embedder, config=config)
        colproj_candidates = [
            build_sample_embedding(market_data, candidate["symbol"], date.fromisoformat(candidate["date"]), config=config)
            for candidate in candidate_items
            if candidate["symbol"] != item["symbol"] or candidate["date"] != item["date"]
        ]
        cnn_candidates = [
            build_cnn_sample_embedding(market_data, candidate["symbol"], date.fromisoformat(candidate["date"]), embedder, config=config)
            for candidate in candidate_items
            if candidate["symbol"] != item["symbol"] or candidate["date"] != item["date"]
        ]
        colproj_matches = search_similar_samples(colproj_query, colproj_candidates, top_k=args.top_k)
        cnn_matches = search_similar_samples(cnn_query, cnn_candidates, top_k=args.top_k)
        colproj_keys = {(match.sample.symbol, match.sample.date.isoformat()) for match in colproj_matches}
        cnn_keys = {(match.sample.symbol, match.sample.date.isoformat()) for match in cnn_matches}
        overlap = len(colproj_keys & cnn_keys) / args.top_k
        overlaps.append(overlap)
        for name, matches in (("colproj", colproj_matches), ("cnn", cnn_matches)):
            for match in matches:
                values = calculate_forward_returns(
                    market_data,
                    match.sample.symbol,
                    match.sample.date,
                    (1, 5, 10, 20),
                ).values
                for horizon in return_values[name]:
                    return_values[name][horizon].append(
                        next(value for current, value in values if current == horizon)
                    )
        rows.append({
            "query": item,
            "top_k_overlap": overlap,
            "colproj": sorted(f"{symbol}:{trading_date}" for symbol, trading_date in colproj_keys),
            "cnn": sorted(f"{symbol}:{trading_date}" for symbol, trading_date in cnn_keys),
        })

    report = {
        "model_version": embedder.model_version,
        "source_hash": manifest.get("source_hash"),
        "query_count": len(test_items),
        "candidate_count": len(candidate_items),
        "top_k": args.top_k,
        "mean_top_k_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "mean_top_k_return": {
            model: {
                str(horizon): sum(values) / len(values) if values else 0.0
                for horizon, values in horizons.items()
            }
            for model, horizons in return_values.items()
        },
        "evaluation_ms": round((perf_counter() - started) * 1000, 3),
        "queries": rows,
        "decision": "keep_colproj_production_until_cnn_passes_quality_gate",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _balanced_items(items: Sequence[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    groups: dict[str, list[dict[str, str]]] = {}
    for item in items:
        groups.setdefault(item["symbol"], []).append(item)
    ordered = [groups[key] for key in sorted(groups)]
    result: list[dict[str, str]] = []
    position = 0
    while len(result) < limit and ordered:
        progressed = False
        for group in ordered:
            if position < len(group) and len(result) < limit:
                result.append(group[position])
                progressed = True
        if not progressed:
            break
        position += 1
    return result
