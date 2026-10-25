"""Stream CSMAR daily A-share CSV exports into the project SQLite database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from .data_sync import build_instrument_metadata
from .sqlite_store import SCHEMA


CSMAR_FIELDS = (
    "Stkcd",
    "Trddt",
    "Opnprc",
    "Hiprc",
    "Loprc",
    "Clsprc",
    "Dnshrtrd",
    "Dnvaltrd",
    "Dretnd",
    "PreClosePrice",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="包含 CSMAR CSV 分片的目录；可重复传入多个目录",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/csmar_a_share_daily.sqlite3"),
    )
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = import_csmar_csvs(
            args.input_dir,
            args.database,
            quality_report_path=args.quality_report,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"CSMAR 数据导入失败: {exc}")
        return 2

    print(f"database: {args.database}")
    print(f"rows: {report['row_count']}")
    print(f"symbols: {report['symbol_count']}")
    print(f"date_range: {report['data_start_date']} -> {report['data_end_date']}")
    print(f"duplicate_rows: {report['duplicate_rows']}")
    print(f"invalid_rows: {report['invalid_rows']}")
    return 0


def import_csmar_csvs(
    input_dirs: Iterable[str | Path],
    database_path: str | Path,
    *,
    quality_report_path: str | Path | None = None,
    batch_size: int = 10_000,
) -> dict[str, object]:
    """Import all CSV files from the supplied directories without loading them all into RAM."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    directories = tuple(Path(item) for item in input_dirs)
    files = _discover_csv_files(directories)
    if not files:
        raise ValueError("输入目录中没有找到 CSV 文件")

    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "provider": "csmar",
        "source_type": "local_csv_export",
        "source_paths": [str(path) for path in files],
        "source_files": [],
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "exclusion_claim": "user_supplied_dataset_claims_st_and_finance_excluded",
        "exclusion_verified_from_daily_table": False,
        "row_count": 0,
        "symbol_count": 0,
        "duplicate_rows": 0,
        "invalid_rows": 0,
        "data_start_date": None,
        "data_end_date": None,
    }
    symbols: set[str] = set()
    file_summaries: list[dict[str, object]] = []

    with sqlite3.connect(database, timeout=60) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(SCHEMA)
        _ensure_optional_columns(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM daily_bars")
        connection.execute("DELETE FROM instruments")
        connection.execute("DELETE FROM dataset_meta")
        batch: list[tuple[object, ...]] = []
        try:
            for path in files:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                file_rows = 0
                file_invalid = 0
                file_duplicate = 0
                for normalized in _iter_normalized_rows(path):
                    if normalized is None:
                        file_invalid += 1
                        continue
                    symbol, trading_date, open_price, high, low, close, volume, amount, return_rate, pre_close = normalized
                    try:
                        batch.append(
                            (
                                symbol,
                                trading_date,
                                open_price,
                                high,
                                low,
                                close,
                                volume,
                                amount,
                                return_rate,
                                pre_close,
                            )
                        )
                        if len(batch) >= batch_size:
                            inserted, duplicates = _insert_batch(connection, batch)
                            file_duplicate += duplicates
                            file_rows += inserted
                            batch.clear()
                    except sqlite3.Error:
                        connection.rollback()
                        raise
                    symbols.add(symbol)
                    report["data_start_date"] = _min_date(report["data_start_date"], trading_date)
                    report["data_end_date"] = _max_date(report["data_end_date"], trading_date)
                if batch:
                    inserted, duplicates = _insert_batch(connection, batch)
                    file_duplicate += duplicates
                    file_rows += inserted
                    batch.clear()
                file_summaries.append(
                    {
                        "path": str(path),
                        "sha256": file_hash,
                        "row_count": file_rows,
                        "invalid_rows": file_invalid,
                        "duplicate_rows": file_duplicate,
                    }
                )

            instrument_rows = build_instrument_metadata(sorted(symbols))
            connection.executemany(
                """
                INSERT INTO instruments (symbol, name, exchange, board, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["symbol"],
                        item["name"],
                        item["exchange"],
                        item["board"],
                        item["status"],
                    )
                    for item in instrument_rows
                ],
            )
            report["row_count"] = sum(int(item["row_count"]) for item in file_summaries)
            report["duplicate_rows"] = sum(int(item["duplicate_rows"]) for item in file_summaries)
            report["invalid_rows"] = sum(int(item["invalid_rows"]) for item in file_summaries)
            report["symbol_count"] = len(symbols)
            report["source_files"] = file_summaries
            report["source_hash"] = _combined_hash(file_summaries)
            report["sha256"] = report["source_hash"]
            connection.execute(
                "INSERT INTO dataset_meta (key, value) VALUES (?, ?)",
                ("source_hash", str(report["source_hash"])),
            )
            connection.execute(
                "INSERT INTO dataset_meta (key, value) VALUES (?, ?)",
                ("updated_at_utc", str(report["fetched_at_utc"])),
            )
            connection.execute(
                "INSERT INTO dataset_meta (key, value) VALUES (?, ?)",
                ("manifest", json.dumps(report, ensure_ascii=False, sort_keys=True)),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    report_path = Path(quality_report_path) if quality_report_path else database.with_suffix(".quality.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _discover_csv_files(directories: Sequence[Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"输入目录不存在: {directory}")
        files.extend(sorted(path for path in directory.glob("*.csv") if path.is_file()))
    return tuple(dict.fromkeys(files))


def _iter_normalized_rows(path: Path) -> Iterator[tuple[object, ...] | None]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in CSMAR_FIELDS):
            raise ValueError(f"CSMAR 文件字段不完整: {path}")
        for row in reader:
            try:
                symbol = str(row["Stkcd"]).strip()
                trading_date = str(row["Trddt"]).strip()[:10]
                date.fromisoformat(trading_date)
                values = [float(row[field]) for field in CSMAR_FIELDS[2:]]
                open_price, high, low, close, volume, amount, return_rate, pre_close = values
                if not symbol or len(symbol) != 6 or not symbol.isdigit():
                    raise ValueError("invalid symbol")
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("non-finite value")
                if min(open_price, high, low, close) <= 0 or high < max(open_price, close, low):
                    raise ValueError("invalid OHLC")
                if volume < 0 or amount < 0 or pre_close <= 0:
                    raise ValueError("invalid volume or pre-close")
                return_rate = return_rate if math.isfinite(return_rate) else None
                yield (
                    symbol,
                    trading_date,
                    open_price,
                    high,
                    low,
                    close,
                    volume,
                    amount,
                    return_rate,
                    pre_close,
                )
            except (TypeError, ValueError, OverflowError):
                yield None


def _insert_batch(connection: sqlite3.Connection, rows: Sequence[tuple[object, ...]]) -> tuple[int, int]:
    inserted = 0
    duplicates = 0
    for row in rows:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO daily_bars
                (symbol, date, open, high, low, close, volume, amount, return_rate, pre_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        if cursor.rowcount == 1:
            inserted += 1
        else:
            duplicates += 1
    return inserted, duplicates


def _combined_hash(file_summaries: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in file_summaries:
        digest.update(str(item["path"]).encode("utf-8"))
        digest.update(str(item["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _min_date(current: object, candidate: str) -> str:
    return candidate if current is None or candidate < str(current) else str(current)


def _max_date(current: object, candidate: str) -> str:
    return candidate if current is None or candidate > str(current) else str(current)


def _ensure_optional_columns(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(daily_bars)")}
    for name in ("amount", "return_rate", "pre_close"):
        if name not in columns:
            connection.execute(f"ALTER TABLE daily_bars ADD COLUMN {name} REAL")


if __name__ == "__main__":
    raise SystemExit(main())
