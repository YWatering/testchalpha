"""Reconstruct a comparable OHLC price path from daily returns.

The source OHLC values remain untouched.  A reconstructed close is anchored
to one source close and then propagated with the supplied daily return.  The
same day's raw OHLC shape is scaled by the reconstructed/raw close ratio.
"""

from __future__ import annotations

import math
import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Sequence

from .market_data import MarketBar
from .sqlite_store import SCHEMA, _ensure_optional_columns

SCALE_REVIEW_MIN = 0.1
SCALE_REVIEW_MAX = 10.0


class PriceReconstructionError(ValueError):
    """Raised when a symbol cannot be reconstructed safely."""


@dataclass(frozen=True, slots=True)
class ReconstructionQuality:
    symbol: str
    status: str
    row_count: int
    reconstructed_count: int
    missing_return_count: int
    invalid_return_count: int
    invalid_ohlc_count: int
    scale_min: float | None
    scale_max: float | None
    extreme_scale_count: int
    anchor_date: str | None
    anchor_close: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "row_count": self.row_count,
            "reconstructed_count": self.reconstructed_count,
            "missing_return_count": self.missing_return_count,
            "invalid_return_count": self.invalid_return_count,
            "invalid_ohlc_count": self.invalid_ohlc_count,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "extreme_scale_count": self.extreme_scale_count,
            "anchor_date": self.anchor_date,
            "anchor_close": self.anchor_close,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    bars: tuple[MarketBar, ...]
    quality: ReconstructionQuality


def reconstruct_symbol_bars(
    bars: Sequence[MarketBar],
    *,
    anchor_date: date | str | None = None,
    anchor_close: float | None = None,
) -> ReconstructionResult:
    """Return reconstructed bars while preserving each bar's raw OHLC fields.

    The default anchor is the latest available source close for the symbol.
    A symbol with incomplete or invalid return data is returned unchanged and
    marked with a non-ready quality status; it is never silently interpolated.
    """

    ordered = tuple(sorted(bars, key=lambda item: item.date))
    if not ordered:
        raise PriceReconstructionError("不能重建空的股票行情序列")
    symbol = ordered[0].symbol
    if any(bar.symbol != symbol for bar in ordered):
        raise PriceReconstructionError("一次只能重建一只股票")

    if anchor_date is None:
        anchor_index = len(ordered) - 1
    else:
        parsed_anchor = _as_date(anchor_date)
        try:
            anchor_index = next(index for index, bar in enumerate(ordered) if bar.date == parsed_anchor)
        except StopIteration as exc:
            raise PriceReconstructionError(
                f"锚点日期不属于 {symbol}: {parsed_anchor.isoformat()}"
            ) from exc

    # For a backward step, the next day's return is used; for a forward step,
    # the current day's return is used. The first row's return is never needed.
    used_returns = [bar.return_rate for bar in ordered[1:]]
    missing_return_count = sum(value is None for value in used_returns)
    invalid_return_count = sum(
        value is not None and (not math.isfinite(value) or 1.0 + value <= 0.0)
        for value in used_returns
    )
    if missing_return_count or invalid_return_count:
        quality = ReconstructionQuality(
            symbol=symbol,
            status="raw_fallback_missing_or_invalid_return",
            row_count=len(ordered),
            reconstructed_count=0,
            missing_return_count=missing_return_count,
            invalid_return_count=invalid_return_count,
            invalid_ohlc_count=0,
            scale_min=None,
            scale_max=None,
            extreme_scale_count=0,
            anchor_date=ordered[anchor_index].date.isoformat(),
            anchor_close=ordered[anchor_index].raw_close,
        )
        return ReconstructionResult(ordered, quality)

    base_close = ordered[anchor_index].raw_close
    if anchor_close is None:
        anchor_value = base_close
    else:
        anchor_value = float(anchor_close)
    if not math.isfinite(anchor_value) or anchor_value <= 0:
        raise PriceReconstructionError("锚点收盘价必须是正数")

    reconstructed_closes = [0.0] * len(ordered)
    reconstructed_closes[anchor_index] = anchor_value
    for index in range(anchor_index + 1, len(ordered)):
        rate = float(ordered[index].return_rate)  # validated above
        reconstructed_closes[index] = reconstructed_closes[index - 1] * (1.0 + rate)
    for index in range(anchor_index - 1, -1, -1):
        rate = float(ordered[index + 1].return_rate)
        reconstructed_closes[index] = reconstructed_closes[index + 1] / (1.0 + rate)

    reconstructed: list[MarketBar] = []
    invalid_ohlc_count = 0
    scales: list[float] = []
    for bar, reconstructed_close in zip(ordered, reconstructed_closes):
        scale = reconstructed_close / bar.raw_close
        scales.append(scale)
        values = {
            "open": bar.raw_open * scale,
            "high": bar.raw_high * scale,
            "low": bar.raw_low * scale,
            "close": reconstructed_close,
        }
        tolerance = 1e-12 * max(1.0, *values.values())
        if values["high"] + tolerance >= max(values["open"], values["close"]):
            values["high"] = max(values["high"], values["open"], values["close"])
        if values["low"] - tolerance <= min(values["open"], values["close"]):
            values["low"] = min(values["low"], values["open"], values["close"])
        valid = (
            all(math.isfinite(value) and value > 0 for value in values.values())
            and values["high"] >= max(values["open"], values["close"], values["low"])
            and values["low"] <= min(values["open"], values["close"], values["high"])
        )
        if not valid:
            invalid_ohlc_count += 1
        reconstructed.append(
            replace(
                bar,
                **values,
                price_basis="reconstructed",
                reconstruction_status="reconstructed" if valid else "invalid_ohlc",
                reconstruction_scale=scale,
                reconstruction_anchor_date=ordered[anchor_index].date,
                reconstruction_anchor_close=anchor_value,
            )
        )

    quality = ReconstructionQuality(
        symbol=symbol,
        status="reconstructed" if invalid_ohlc_count == 0 else "invalid_ohlc",
        row_count=len(ordered),
        reconstructed_count=len(reconstructed),
        missing_return_count=0,
        invalid_return_count=0,
        invalid_ohlc_count=invalid_ohlc_count,
        scale_min=min(scales),
        scale_max=max(scales),
        extreme_scale_count=sum(
            scale < SCALE_REVIEW_MIN or scale > SCALE_REVIEW_MAX for scale in scales
        ),
        anchor_date=ordered[anchor_index].date.isoformat(),
        anchor_close=anchor_value,
    )
    return ReconstructionResult(tuple(reconstructed), quality)


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def materialize_reconstructed_prices(
    database_path: str | Path,
    *,
    quality_report_path: str | Path | None = None,
    anchor_date: date | str | None = None,
    anchor_close: float | None = None,
    batch_size: int = 10_000,
) -> dict[str, object]:
    """Write reconstructed OHLC columns and quality metadata into SQLite."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    database = Path(database_path)
    if not database.is_file():
        raise FileNotFoundError(f"行情数据库不存在: {database}")

    with sqlite3.connect(database, timeout=120) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(SCHEMA)
        _ensure_optional_columns(connection)
        metadata = dict(connection.execute("SELECT key, value FROM dataset_meta").fetchall())
        source_hash = metadata.get("source_hash", "")
        symbols = [row[0] for row in connection.execute("SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol")]
        if not symbols:
            raise ValueError("行情数据库没有 daily_bars 数据")

        quality_rows: list[dict[str, object]] = []
        status_counts: dict[str, int] = {}
        total_rows = 0
        reconstructed_rows = 0
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM dataset_meta WHERE key IN ('price_basis', 'reconstructed_source_hash')")
            for symbol in symbols:
                raw_rows = connection.execute(
                    """SELECT symbol, date, open, high, low, close, volume,
                              return_rate, pre_close
                       FROM daily_bars WHERE symbol = ? ORDER BY date""",
                    (symbol,),
                ).fetchall()
                bars = tuple(
                    MarketBar(
                        symbol=row[0], date=date.fromisoformat(row[1]),
                        open=float(row[2]), high=float(row[3]), low=float(row[4]),
                        close=float(row[5]), volume=row[6],
                        return_rate=row[7], pre_close=row[8],
                    )
                    for row in raw_rows
                )
                result = reconstruct_symbol_bars(
                    bars, anchor_date=anchor_date, anchor_close=anchor_close
                )
                quality_rows.append(result.quality.as_dict())
                status_counts[result.quality.status] = status_counts.get(result.quality.status, 0) + 1
                reconstructed_rows += result.quality.reconstructed_count
                total_rows += len(result.bars)
                updates = []
                for bar in result.bars:
                    updates.append(
                        (
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close,
                            bar.reconstruction_scale or 1.0,
                            bar.reconstruction_status,
                            bar.reconstruction_anchor_date.isoformat()
                            if bar.reconstruction_anchor_date
                            else None,
                            bar.reconstruction_anchor_close,
                            bar.symbol,
                            bar.date.isoformat(),
                        )
                    )
                    if len(updates) >= batch_size:
                        connection.executemany(_RECONSTRUCTION_UPDATE, updates)
                        updates.clear()
                if updates:
                    connection.executemany(_RECONSTRUCTION_UPDATE, updates)

            reconstruction_manifest = {
                "format": "return-reconstructed-prices-v1",
                "price_basis": "return_reconstructed_v1",
                "source_hash": source_hash,
                "anchor_date": str(anchor_date) if anchor_date is not None else "latest_per_symbol",
                "anchor_close": anchor_close,
                "symbol_count": len(symbols),
                "row_count": total_rows,
                "reconstructed_row_count": reconstructed_rows,
                "status_counts": status_counts,
                "quality": quality_rows,
            }
            reconstructed_hash = sha256(
                json.dumps(reconstruction_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            reconstruction_manifest["reconstructed_hash"] = reconstructed_hash
            if status_counts.get("reconstructed") == len(symbols):
                connection.execute(
                    """INSERT INTO dataset_meta (key, value) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                    ("price_basis", "return_reconstructed_v1"),
                )
                connection.execute(
                    """INSERT INTO dataset_meta (key, value) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                    ("reconstructed_source_hash", sha256(f"{source_hash}:{reconstructed_hash}".encode()).hexdigest()),
                )
            connection.execute(
                """INSERT INTO dataset_meta (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                ("reconstruction_manifest", json.dumps(reconstruction_manifest, ensure_ascii=False, sort_keys=True)),
            )
            connection.execute(
                """INSERT INTO dataset_meta (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                ("updated_at_utc", datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    report = {
        **reconstruction_manifest,
        "database": str(database),
        "quality_report_path": str(quality_report_path) if quality_report_path else None,
    }
    if quality_report_path:
        report_path = Path(quality_report_path)
    else:
        report_path = database.with_suffix(".reconstruction.quality.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


_RECONSTRUCTION_UPDATE = """UPDATE daily_bars
    SET reconstructed_open=?, reconstructed_high=?, reconstructed_low=?, reconstructed_close=?,
        reconstruction_scale=?, reconstruction_status=?, reconstruction_anchor_date=?,
        reconstruction_anchor_close=?
    WHERE symbol=? AND date=?"""
