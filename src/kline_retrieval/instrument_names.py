"""Synchronize A-share symbol/name metadata into the local SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .data_sync import DEFAULT_A_SHARE_NAMES
from .sqlite_store import SCHEMA


CNINFO_ENDPOINT = "https://www.cninfo.com.cn/new/data/szse_stock.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/csmar_a_share_daily.sqlite3"),
        help="需要更新 instruments 表的 SQLite 数据库",
    )
    parser.add_argument("--page-size", type=int, default=0, help="兼容旧命令参数；中国信息披露网接口无需分页")
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = sync_instrument_names(
            args.database,
            page_size=args.page_size,
            timeout=args.timeout,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"股票名称同步失败: {exc}")
        return 2

    print(f"database: {args.database}")
    print(f"database_symbols: {report['database_symbols']}")
    print(f"remote_symbols: {report['remote_symbols']}")
    print(f"updated_symbols: {report['updated_symbols']}")
    print(f"fallback_symbols: {report['fallback_symbols']}")
    return 0


def sync_instrument_names(
    database_path: str | Path,
    *,
    page_size: int = 0,
    timeout: float = 20.0,
    fetcher=None,
) -> dict[str, object]:
    """Update names for symbols already present in the local database."""

    if page_size < 0:
        raise ValueError("page_size 不能小于 0")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")

    path = Path(database_path)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.executescript(SCHEMA)
        rows = connection.execute("SELECT symbol, name FROM instruments ORDER BY symbol").fetchall()
        database_symbols = {str(symbol) for symbol, _ in rows}
        names = dict(DEFAULT_A_SHARE_NAMES)
        names.update((fetcher or fetch_cninfo_names)(page_size=page_size, timeout=timeout))
        matched = {
            symbol: names[symbol]
            for symbol in database_symbols
            if symbol in names and names[symbol].strip()
        }
        connection.executemany(
            "UPDATE instruments SET name = ? WHERE symbol = ?",
            [(name.strip(), symbol) for symbol, name in matched.items()],
        )
        fetched_at = datetime.now(timezone.utc).isoformat()
        connection.executemany(
            """
            INSERT INTO dataset_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            [
                ("instrument_names_source", CNINFO_ENDPOINT),
                ("instrument_names_updated_at_utc", fetched_at),
                ("instrument_names_remote_count", str(len(names))),
            ],
        )
        connection.commit()

    return {
        "database_symbols": len(database_symbols),
        "remote_symbols": len(names),
        "updated_symbols": len(matched),
        "fallback_symbols": len(database_symbols - matched.keys()),
        "updated_at_utc": fetched_at,
    }


def fetch_cninfo_names(*, page_size: int = 0, timeout: float = 20.0) -> dict[str, str]:
    """Fetch current A-share code/name pairs from CNINFO's public list endpoint."""

    del page_size  # The CNINFO endpoint returns the complete list in one response.
    request = Request(
        CNINFO_ENDPOINT,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.cninfo.com.cn/",
            "User-Agent": "Mozilla/5.0 kline-retrieval/0.1",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise OSError(f"无法获取中国信息披露网股票名称列表: {exc}") from exc

    stock_list = payload.get("stockList") or []
    if not isinstance(stock_list, list):
        raise OSError("中国信息披露网股票名称列表格式异常")
    names: dict[str, str] = {}
    for item in stock_list:
        symbol = str(item.get("code", "")).strip()
        name = str(item.get("zwjc", "")).strip()
        category = str(item.get("category", "")).strip()
        if len(symbol) == 6 and symbol.isdigit() and name and category == "A股":
            names[symbol] = name
    return names


if __name__ == "__main__":
    raise SystemExit(main())
