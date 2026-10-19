"""FastAPI application for the browser retrieval page."""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analytics import ReturnAnalysisError, analyze_matches
from .charts import build_query_charts
from .config import DEFAULT_CONFIG, ProjectConfig
from .market_data import MarketData, MarketDataError
from .retrieval import RetrievalError, search_market_data


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
    market_data = MarketData.from_file(resolved_data_path)
    web_root = Path(__file__).resolve().parents[2] / "web"

    app = FastAPI(title="K-Line Pattern Retrieval", version="0.1.0")
    app.state.config = config
    app.state.market_data = market_data
    app.state.data_path = resolved_data_path
    app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "symbols": len(market_data.symbols),
            "data_path": str(resolved_data_path),
            "source_hash": market_data.source_hash,
        }

    @app.get("/api/symbols")
    def symbols() -> dict[str, list[str]]:
        return {"symbols": list(market_data.symbols)}

    @app.get("/api/dates")
    def dates(symbol: str = Query(min_length=1)) -> dict[str, object]:
        try:
            available_dates = market_data.dates(symbol)
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
        if request.day_weight != 1.0:
            raise HTTPException(status_code=400, detail="日线权重必须固定为 1.0")
        try:
            query_date = date.fromisoformat(request.date)
            matches = search_market_data(
                market_data,
                request.symbol,
                query_date,
                config=config,
                month_weight=request.month_weight,
                week_weight=request.week_weight,
                top_k=request.top_k,
                embargo_trading_days=request.embargo_trading_days,
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

        return {
            "query": {
                "symbol": request.symbol,
                "date": query_date.isoformat(),
                "charts": _encode_charts(query_charts.as_dict()),
            },
            "parameters": request.model_dump(),
            "matches": match_payload,
            "statistics": {
                str(statistic.horizon): statistic.as_dict()
                for statistic in analysis.statistics
            },
        }

    return app


def _encode_charts(charts: dict[str, object]) -> dict[str, str]:
    return {name: _encode_image(image) for name, image in charts.items()}


def _encode_image(image: object) -> str:
    encoded = base64.b64encode(image.to_png_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


app = create_app()
