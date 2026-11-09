"""FastAPI application for the browser retrieval page."""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import date
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analytics import ReturnAnalysisError, analyze_matches
from .charts import build_query_charts
from .cnn import CNNEmbedder, build_cnn_sample_embedding
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData, MarketDataError
from .offline_index import CompositeOfflineFeatureIndex, OfflineFeatureIndex
from .retrieval import (
    RetrievalError,
    SampleEmbeddingCache,
    build_sample_embedding,
    search_market_data,
)
from .result_cache import RetrievalResultCache
from .sqlite_store import read_instruments, read_metadata
from .universe import coverage_report, read_universe_symbols


class SearchRequest(BaseModel):
    symbol: str = Field(min_length=1)
    date: str
    model: Literal["colproj", "cnn"] = "colproj"
    month_weight: float = Field(default=1.0, ge=0.0)
    week_weight: float = Field(default=1.0, ge=0.0)
    day_weight: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=20, ge=1, le=100)
    embargo_trading_days: int = Field(default=0, ge=0, le=100)


def create_app(
    data_path: str | Path | None = None,
    config: ProjectConfig | None = None,
) -> FastAPI:
    config = config or ProjectConfig.from_env()
    resolved_data_path = Path(data_path or config.data_path)
    market_data = _load_market_data(resolved_data_path, config)
    offline_index = _load_offline_index(config.index_path, market_data, config)
    web_root = Path(__file__).resolve().parents[2] / "web"

    app = FastAPI(title="K-Line Pattern Retrieval", version="0.1.0")
    app.state.config = config
    app.state.market_data = market_data
    app.state.data_path = resolved_data_path
    app.state.market_data_token = _data_token(resolved_data_path)
    app.state.embedding_cache = SampleEmbeddingCache()
    app.state.offline_index = offline_index
    app.state.cnn_index = _load_offline_index(config.cnn_index_path, market_data, config)
    app.state.cnn_embedder = _load_cnn_embedder(config.cnn_checkpoint_path)
    cache_path = config.retrieval_cache_path or resolved_data_path.with_name(
        f"{resolved_data_path.stem}.retrieval-cache.sqlite3"
    )
    app.state.retrieval_result_cache = RetrievalResultCache(cache_path)
    app.state.data_reload_lock = RLock()
    app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_root / "index.html")

    def current_market_data() -> MarketData:
        token = _data_token(resolved_data_path)
        if token != app.state.market_data_token:
            with app.state.data_reload_lock:
                token = _data_token(resolved_data_path)
                if token != app.state.market_data_token:
                    app.state.market_data = _load_market_data(resolved_data_path, config)
                    app.state.offline_index = _load_offline_index(
                        config.index_path, app.state.market_data, config
                    )
                    app.state.cnn_index = _load_offline_index(
                        config.cnn_index_path, app.state.market_data, config
                    )
                    app.state.market_data_token = token
                    app.state.embedding_cache = SampleEmbeddingCache()
        return app.state.market_data

    @app.get("/api/health")
    def health() -> dict[str, object]:
        market_data = current_market_data()
        manifest_path = resolved_data_path.with_name(
            f"{resolved_data_path.stem}.manifest.json"
        )
        dataset: dict[str, object] = {
            "provider": "synthetic"
            if resolved_data_path.name == "demo_market_data.csv"
            else "local_file"
        }
        if resolved_data_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            try:
                stored_manifest = read_metadata(resolved_data_path).get("manifest")
                if stored_manifest:
                    manifest = json.loads(stored_manifest)
                    if manifest.get("sha256") == market_data.source_hash:
                        dataset = manifest
            except (OSError, json.JSONDecodeError):
                pass
        elif manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("sha256") == market_data.source_hash:
                    dataset = manifest
            except (OSError, json.JSONDecodeError):
                pass
        try:
            coverage = coverage_report(resolved_data_path) if resolved_data_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"} else None
        except (OSError, sqlite3.Error):
            coverage = None
        return {
            "status": "ok",
            "symbols": int((coverage or {}).get("total_symbols", len(market_data.symbols))),
            "loaded_symbols": len(market_data.symbols),
            "data_path": str(resolved_data_path),
            "source_hash": market_data.source_hash,
            "price_basis": market_data.price_basis,
            "storage": "sqlite"
            if resolved_data_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}
            else "file",
            "dataset": dataset,
            "index": {
                "mode": _index_mode(app.state.offline_index),
                "version_id": app.state.offline_index.version_id
                if app.state.offline_index is not None
                else None,
                "sample_count": app.state.offline_index.sample_count
                if app.state.offline_index is not None
                else 0,
                "indexed_symbols": len(app.state.offline_index.symbols)
                if app.state.offline_index is not None
                else 0,
            },
            "models": {
                "colproj": _model_health(app.state.offline_index),
                "cnn": _model_health(app.state.cnn_index),
            },
            "coverage": coverage,
        }

    @app.get("/api/symbols")
    def symbols(
        q: str = Query(default=""),
        exchange: str = Query(default=""),
        board: str = Query(default=""),
        watchlist: str = Query(default=""),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10_000, ge=1, le=20_000),
    ) -> dict[str, object]:
        loaded_symbols = list(current_market_data().symbols)
        instruments: list[dict[str, object]] = []
        if resolved_data_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            try:
                instruments, total = read_universe_symbols(
                    resolved_data_path,
                    query=q,
                    exchange=exchange,
                    board=board,
                    watchlist=watchlist.split(","),
                    offset=offset,
                    limit=limit,
                )
            except (OSError, ValueError, sqlite3.Error):
                instruments = []
                total = 0
            if not instruments and not q and not exchange and not board and not watchlist:
                all_instruments = read_instruments(resolved_data_path)
                total = len(all_instruments)
                instruments = all_instruments[offset : offset + limit]
        else:
            instruments = [{"symbol": symbol, "name": symbol} for symbol in loaded_symbols[offset : offset + limit]]
            total = len(loaded_symbols)
        indexed_symbols = (
            set(app.state.offline_index.symbols)
            if app.state.offline_index is not None
            else set()
        )
        cnn_symbols = (
            set(app.state.cnn_index.symbols)
            if app.state.cnn_index is not None and app.state.cnn_embedder is not None
            else set()
        )
        for item in instruments:
            if item.get("status") == "ready" and indexed_symbols and item["symbol"] not in indexed_symbols:
                item["status"] = "index_pending"
            item["colproj_indexed"] = item["symbol"] in indexed_symbols
            item["cnn_indexed"] = item["symbol"] in cnn_symbols
        return {
            "symbols": [item["symbol"] for item in instruments],
            "instruments": instruments,
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/dates")
    def dates(symbol: str = Query(min_length=1)) -> dict[str, object]:
        try:
            available_dates = _dates_for_symbol(current_market_data(), resolved_data_path, symbol)
        except MarketDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"symbol": symbol, "dates": [item.isoformat() for item in available_dates]}

    @app.get("/api/valid-dates")
    def valid_dates(symbol: str = Query(min_length=1)) -> dict[str, object]:
        try:
            available_dates = _dates_for_symbol(current_market_data(), resolved_data_path, symbol)
        except MarketDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        history_count = config.monthly_period * config.daily_window
        future_count = max(config.future_horizons)
        first = history_count - 1
        last = len(available_dates) - future_count - 1
        dates = available_dates[first : last + 1] if last >= first else ()
        return {
            "symbol": symbol,
            "dates": [item.isoformat() for item in dates],
            "required_history": history_count,
            "required_future": future_count,
        }

    @app.get("/api/retrieval/config")
    def retrieval_config() -> dict[str, object]:
        active_market_data = current_market_data()
        return {
            "model": "colproj",
            "models": [
                model
                for model, index in (
                    ("colproj", app.state.offline_index),
                    ("cnn", app.state.cnn_index),
                )
                if index is not None and (model != "cnn" or app.state.cnn_embedder is not None)
            ] or ["colproj"],
            "chart_size": list(config.chart_size),
            "price_basis": active_market_data.price_basis,
            "future_horizons": list(config.future_horizons),
            "default_weights": {
                "month": config.month_weight,
                "week": config.week_weight,
                "day": config.day_weight,
            },
            "default_top_k": config.default_top_k,
            "similarity_temperature": config.similarity_temperature,
            "index_mode": _index_mode(app.state.offline_index),
            "index_path": str(config.index_path) if config.index_path else None,
            "index_paths": [str(path) for path in _index_paths(config.index_path)],
            "models_config": {
                "colproj": _model_health(app.state.offline_index),
                "cnn": _model_health(app.state.cnn_index),
            },
        }

    @app.get("/api/retrieval/cache")
    def retrieval_cache_status() -> dict[str, object]:
        return {
            "path": str(cache_path),
            "cache": app.state.retrieval_result_cache.status(),
            "models": {
                "colproj": _cache_model_status(
                    app.state.retrieval_result_cache, app.state.offline_index
                ),
                "cnn": _cache_model_status(
                    app.state.retrieval_result_cache, app.state.cnn_index
                ),
            },
        }

    @app.post("/api/retrieval/search")
    def search(request: SearchRequest) -> dict[str, object]:
        request_started = perf_counter()
        market_data = current_market_data()
        if request.day_weight != 1.0:
            raise HTTPException(status_code=400, detail="日线权重必须固定为 1.0")
        try:
            market_data = _market_data_for_symbol(market_data, resolved_data_path, request.symbol)
            query_date = date.fromisoformat(request.date)
            diagnostics: dict[str, int | float] = {}
            selected_index = (
                app.state.cnn_index if request.model == "cnn" else app.state.offline_index
            )
            if request.model == "cnn" and selected_index is None:
                raise HTTPException(
                    status_code=503,
                    detail="CNN 全市场索引尚未完成或当前未配置，请先完成 CNN 分片构建。",
                )
            if selected_index is not None:
                if request.symbol not in selected_index.symbols:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"股票 {request.symbol} 尚未进入所选 {request.model} 索引覆盖范围；"
                            "请等待对应模型的全市场分片发布后再检索。"
                        ),
                    )
                cache_parameters = {
                    "month_weight": request.month_weight,
                    "week_weight": request.week_weight,
                    "day_weight": request.day_weight,
                    "top_k": request.top_k,
                    "embargo_trading_days": request.embargo_trading_days,
                }
                cached_matches = None
                cache_mode: str | None = None
                cache_identity = None
                if request.model == "colproj" and config.ann_index_path is not None:
                    cache_identity = _ann_cache_identity(
                        config.ann_index_path, selected_index.source_hash
                    )
                    if cache_identity is not None:
                        ann_version, ann_source_hash = cache_identity
                        ann_parameters = {
                            **cache_parameters,
                            "retrieval_engine": "faiss-ann-exact-rerank-v1",
                            "ann_candidate_k": config.ann_candidate_k,
                            "ann_nprobe": config.ann_nprobe,
                        }
                        cached_matches = app.state.retrieval_result_cache.get(
                            model="colproj",
                            index_version=ann_version,
                            source_hash=ann_source_hash,
                            symbol=request.symbol,
                            query_date=query_date,
                            parameters=ann_parameters,
                        )
                        if cached_matches is not None:
                            cache_mode = "materialized_ann_colproj"
                if cached_matches is None:
                    cached_matches = app.state.retrieval_result_cache.get(
                        model=request.model,
                        index_version=selected_index.version_id,
                        source_hash=selected_index.source_hash,
                        symbol=request.symbol,
                        query_date=query_date,
                        parameters=cache_parameters,
                    )
                    if cached_matches is not None:
                        cache_mode = f"materialized_{request.model}"
                if cached_matches is not None:
                    matches = cached_matches
                    diagnostics["result_cache_hit"] = 1
                    diagnostics["search_mode"] = cache_mode or f"materialized_{request.model}"
                    diagnostics["index_candidate_count"] = 0
                    diagnostics["index_ms"] = 0.0
                else:
                    if request.model == "cnn":
                        if app.state.cnn_embedder is None:
                            raise HTTPException(
                                status_code=503,
                                detail="CNN 检查点未配置，请设置 KLINE_CNN_CHECKPOINT_PATH。",
                            )
                        query = build_cnn_sample_embedding(
                            market_data,
                            request.symbol,
                            query_date,
                            app.state.cnn_embedder,
                            config=config,
                        )
                    else:
                        query = build_sample_embedding(
                            market_data, request.symbol, query_date, config=config
                        )
                    matches, index_diagnostics = selected_index.search(
                        query,
                        market_data,
                        month_weight=request.month_weight,
                        week_weight=request.week_weight,
                        top_k=request.top_k,
                        embargo_trading_days=request.embargo_trading_days,
                    )
                    app.state.retrieval_result_cache.put(
                        matches,
                        model=request.model,
                        index_version=selected_index.version_id,
                        source_hash=selected_index.source_hash,
                        symbol=request.symbol,
                        query_date=query_date,
                        parameters=cache_parameters,
                    )
                    diagnostics.update(index_diagnostics)
                    diagnostics["result_cache_hit"] = 0
                    diagnostics["search_mode"] = f"offline_{request.model}"
            else:
                matches = search_market_data(
                    market_data,
                    request.symbol,
                    query_date,
                    config=config,
                    month_weight=request.month_weight,
                    week_weight=request.week_weight,
                    top_k=request.top_k,
                    embargo_trading_days=request.embargo_trading_days,
                    embedding_cache=app.state.embedding_cache,
                    diagnostics=diagnostics,
                )
                diagnostics["search_mode"] = "request_scan"
            market_data = _market_data_for_symbols(
                market_data,
                resolved_data_path,
                {match.sample.symbol for match in matches},
            )
            analysis = analyze_matches(market_data, matches, config=config)
            query_charts = build_query_charts(
                market_data,
                request.symbol,
                query_date,
                config,
            )
        except (ValueError, MarketDataError, RetrievalError, ReturnAnalysisError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        match_payload: list[dict[str, object]] = []
        for outcome in analysis.outcomes:
            sample = outcome.match.sample
            historical_charts = build_query_charts(
                market_data,
                sample.symbol,
                sample.date,
                config,
            )
            match_payload.append(
                {
                    **outcome.as_dict(),
                    "charts": _encode_charts(historical_charts.as_dict()),
                    "future_chart": _encode_image(outcome.future_chart),
                }
            )

        response = {
            "price_basis": market_data.price_basis,
            "query": {
                "symbol": request.symbol,
                "date": query_date.isoformat(),
                "charts": _encode_charts(query_charts.as_dict()),
            },
            "parameters": request.model_dump(),
            "model": {
                "name": request.model,
                "model_version": selected_index.model_version if selected_index is not None else None,
                "index_version": selected_index.version_id if selected_index is not None else None,
                "checkpoint_sha256": (
                    selected_index.checkpoint_sha256 if selected_index is not None else None
                ),
            },
            "matches": match_payload,
            "performance": diagnostics,
            "statistics": {
                str(statistic.horizon): statistic.as_dict()
                for statistic in analysis.statistics
            },
        }
        diagnostics["request_ms"] = round((perf_counter() - request_started) * 1000, 3)
        return response

    return app


def _encode_charts(charts: dict[str, object]) -> dict[str, str]:
    return {name: _encode_image(image) for name, image in charts.items()}


def _encode_image(image: object) -> str:
    encoded = base64.b64encode(image.to_png_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class ConfiguredDataApp:
    """Load the configured local dataset lazily and report setup errors via HTTP."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope: dict[str, object], receive: object, send: object) -> None:
        if self._app is None:
            try:
                self._app = create_app()
            except (MarketDataError, OSError) as exc:
                response = JSONResponse(
                    status_code=503,
                    content={"detail": f"无法加载配置的行情数据: {exc}"},
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


app = ConfiguredDataApp()


def _data_token(path: Path) -> tuple[object, ...]:
    """Return a cheap snapshot token so SQLite updates are visible without restart."""

    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        metadata = read_metadata(path)
        return metadata.get("source_hash", ""), metadata.get("updated_at_utc", "")
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _load_market_data(path: Path, config: ProjectConfig) -> MarketData:
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        return MarketData.from_sqlite(
            path,
            symbols=config.active_symbols or None,
            price_basis=config.price_basis,
        )
    return MarketData.from_file(path)


def _market_data_for_symbol(base: MarketData, path: Path, symbol: str) -> MarketData:
    """Add a lazily loaded symbol to the active in-memory market snapshot."""

    return _market_data_for_symbols(base, path, {symbol})


def _market_data_for_symbols(
    base: MarketData,
    path: Path,
    symbols: set[str],
) -> MarketData:
    """Add missing symbols needed by a query and its Top-K matches."""

    missing = tuple(sorted(symbol for symbol in symbols if symbol not in base.symbols))
    if not missing or path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        return base

    selected = MarketData.from_sqlite(
        path, symbols=missing, price_basis=base.price_basis
    )
    rows = []
    for loaded_symbol in base.symbols:
        rows.extend(
            {
                "symbol": bar.symbol,
                "date": bar.date.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "return_rate": bar.return_rate,
                "pre_close": bar.pre_close,
                "raw_open": bar.raw_open,
                "raw_high": bar.raw_high,
                "raw_low": bar.raw_low,
                "raw_close": bar.raw_close,
                "price_basis": bar.price_basis,
                "reconstruction_status": bar.reconstruction_status,
                "reconstruction_scale": bar.reconstruction_scale,
                "reconstruction_anchor_date": bar.reconstruction_anchor_date,
                "reconstruction_anchor_close": bar.reconstruction_anchor_close,
            }
            for bar in base.bars(loaded_symbol)
        )
    for selected_symbol in selected.symbols:
        rows.extend(
            {
                "symbol": bar.symbol,
                "date": bar.date.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "return_rate": bar.return_rate,
                "pre_close": bar.pre_close,
                "raw_open": bar.raw_open,
                "raw_high": bar.raw_high,
                "raw_low": bar.raw_low,
                "raw_close": bar.raw_close,
                "price_basis": bar.price_basis,
                "reconstruction_status": bar.reconstruction_status,
                "reconstruction_scale": bar.reconstruction_scale,
                "reconstruction_anchor_date": bar.reconstruction_anchor_date,
                "reconstruction_anchor_close": bar.reconstruction_anchor_close,
            }
            for bar in selected.bars(selected_symbol)
        )
    return MarketData.from_rows(
        rows,
        source_path=base.source_path,
        source_hash=base.source_hash,
        price_basis=base.price_basis,
    )


def _dates_for_symbol(base: MarketData, path: Path, symbol: str) -> tuple[date, ...]:
    """Read dates for a dropdown-selected symbol without rebuilding the active pool."""

    if symbol in base.symbols:
        return base.dates(symbol)
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        return MarketData.from_sqlite(
            path, symbols=(symbol,), price_basis=base.price_basis
        ).dates(symbol)
    return base.dates(symbol)


def _load_offline_index(
    index_path: Path | None,
    market_data: MarketData,
    config: ProjectConfig,
) -> OfflineFeatureIndex | None:
    paths = _index_paths(index_path)
    if not paths:
        return None
    try:
        index = OfflineFeatureIndex(paths[0]) if len(paths) == 1 else CompositeOfflineFeatureIndex(paths)
    except (OSError, ValueError, sqlite3.Error):
        return None
    return (
        index
        if index.is_compatible(
            market_data.source_hash,
            config,
            price_basis=market_data.price_basis,
        )
        else None
    )


def _load_cnn_embedder(checkpoint_path: Path | None) -> CNNEmbedder | None:
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    try:
        return CNNEmbedder(checkpoint_path)
    except (OSError, RuntimeError, ValueError):
        return None


def _model_health(index: OfflineFeatureIndex | CompositeOfflineFeatureIndex | None) -> dict[str, object]:
    return {
        "available": index is not None,
        "mode": _index_mode(index),
        "version_id": index.version_id if index is not None else None,
        "sample_count": index.sample_count if index is not None else 0,
        "indexed_symbols": len(index.symbols) if index is not None else 0,
        "model": getattr(index, "model", None),
        "dimension": getattr(index, "dimension", None),
        "model_version": getattr(index, "model_version", None),
    }


def _ann_cache_identity(path: Path, expected_source_hash: str) -> tuple[str, str] | None:
    """Read cache identity only; API intentionally never imports FAISS."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("format") != "kline-ann-manifest-v1"
        or manifest.get("model") != "colproj"
        or manifest.get("source_hash") != expected_source_hash
        or not manifest.get("version_id")
    ):
        return None
    return str(manifest["version_id"]), str(manifest["source_hash"])


def _cache_model_status(
    cache: RetrievalResultCache,
    index: OfflineFeatureIndex | CompositeOfflineFeatureIndex | None,
) -> dict[str, object]:
    if index is None:
        return {"available": False, "cached_query_count": 0, "index_version": None}
    return {
        "available": True,
        "cached_query_count": cache.status(
            model=index.model, index_version=index.version_id
        )["cached_query_count"],
        "target_sample_count": index.sample_count,
        "index_version": index.version_id,
        "source_hash": index.source_hash,
    }


def _index_paths(index_path: Path | None) -> tuple[Path, ...]:
    if index_path is None:
        return ()
    if index_path.suffix.lower() == ".json":
        try:
            manifest = json.loads(index_path.read_text(encoding="utf-8"))
            paths = manifest.get("indexes", [])
            return tuple(
                (index_path.parent / Path(item)).resolve()
                if not Path(item).is_absolute()
                else Path(item)
                for item in paths
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return ()
    return tuple(Path(item.strip()) for item in str(index_path).split(",") if item.strip())


def _index_mode(index: object | None) -> str:
    if index is None:
        return "request_scan"
    model = getattr(index, "model", "colproj")
    sharded = isinstance(index, CompositeOfflineFeatureIndex)
    return f"offline_{model}{'_sharded' if sharded else ''}"
