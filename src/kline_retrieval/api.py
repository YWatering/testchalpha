"""FastAPI application for the browser retrieval page."""

from __future__ import annotations

import base64
import json
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
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData, MarketDataError
from .retrieval import RetrievalError, SampleEmbeddingCache, search_market_data
from .sqlite_store import read_instruments, read_metadata


class SearchRequest(BaseModel):
    symbol: str = Field(min_length=1)
    date: str
    model: Literal["colproj"] = "colproj"
    month_weight: float = Field(default=1.0, ge=0.0)
    week_weight: float = Field(default=1.0, ge=0.0)
    day_weight: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=20, ge=1, le=100)
    embargo_trading_days: int = Field(default=0, ge=0, le=100)


def create_app(data_path: str | Path | None = None) -> FastAPI:
    config = ProjectConfig.from_env()
    resolved_data_path = Path(data_path or config.data_path)
    market_data = _load_market_data(resolved_data_path, config)
    web_root = Path(__file__).resolve().parents[2] / "web"

    app = FastAPI(title="K-Line Pattern Retrieval", version="0.1.0")
    app.state.config = config
    app.state.market_data = market_data
    app.state.data_path = resolved_data_path
    app.state.market_data_token = _data_token(resolved_data_path)
    app.state.embedding_cache = SampleEmbeddingCache()
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
        return {
            "status": "ok",
            "symbols": len(market_data.symbols),
            "data_path": str(resolved_data_path),
            "source_hash": market_data.source_hash,
            "storage": "sqlite"
            if resolved_data_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}
            else "file",
            "dataset": dataset,
        }

    @app.get("/api/symbols")
    def symbols() -> dict[str, object]:
        loaded_symbols = list(current_market_data().symbols)
        instruments: list[dict[str, str]] = []
        if resolved_data_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            instruments = [
                item
                for item in read_instruments(resolved_data_path)
                if item["symbol"] in loaded_symbols
            ]
        return {"symbols": loaded_symbols, "instruments": instruments}

    @app.get("/api/dates")
    def dates(symbol: str = Query(min_length=1)) -> dict[str, object]:
        try:
            available_dates = current_market_data().dates(symbol)
        except MarketDataError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"symbol": symbol, "dates": [item.isoformat() for item in available_dates]}

    @app.get("/api/retrieval/config")
    def retrieval_config() -> dict[str, object]:
        return {
            "model": "colproj",
            "models": ["colproj"],
            "chart_size": list(config.chart_size),
            "future_horizons": list(config.future_horizons),
            "default_weights": {
                "month": config.month_weight,
                "week": config.week_weight,
                "day": config.day_weight,
            },
            "default_top_k": config.default_top_k,
        }

    @app.post("/api/retrieval/search")
    def search(request: SearchRequest) -> dict[str, object]:
        request_started = perf_counter()
        market_data = current_market_data()
        if request.day_weight != 1.0:
            raise HTTPException(status_code=400, detail="日线权重必须固定为 1.0")
        try:
            query_date = date.fromisoformat(request.date)
            diagnostics: dict[str, int | float] = {}
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
            "query": {
                "symbol": request.symbol,
                "date": query_date.isoformat(),
                "charts": _encode_charts(query_charts.as_dict()),
            },
            "parameters": request.model_dump(),
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
        return MarketData.from_sqlite(path, symbols=config.active_symbols or None)
    return MarketData.from_file(path)
