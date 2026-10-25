"""Build a persistent exact ColProj index from a SQLite market snapshot."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig
from .offline_index import build_colproj_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/colproj_stage9.sqlite3"),
    )
    parser.add_argument("--symbols", help="逗号分隔的股票代码")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--limit", type=int, help="未指定代码时按代码排序取前 N 只股票")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        symbols = _resolve_symbols(args.symbols, args.symbols_file)
        if symbols is None and args.limit is not None and args.limit <= 0:
            raise ValueError("limit 必须大于 0")
        if symbols is None and args.limit:
            with sqlite3.connect(args.database) as connection:
                symbols = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT symbol FROM instruments ORDER BY symbol LIMIT ?",
                        (args.limit,),
                    )
                )
            if not symbols:
                raise ValueError("数据库 instruments 表中没有股票")
        result = build_colproj_index(
            args.database,
            args.index,
            symbols=symbols[: args.limit] if symbols and args.limit else symbols,
            config=ProjectConfig.from_env(),
            replace=args.replace,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as exc:
        print(f"ColProj 索引构建失败: {exc}")
        return 2
    print(f"index: {result.index_path}")
    print(f"version: {result.version_id}")
    print(f"symbols: {result.symbol_count}")
    print(f"samples: {result.sample_count}")
    print(f"inserted: {result.inserted_count}")
    print(f"skipped: {result.skipped_count}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.3f}")
    return 0


def _resolve_symbols(value: str | None, path: Path | None) -> tuple[str, ...] | None:
    if value and path:
        raise ValueError("--symbols 和 --symbols-file 只能选择一个")
    if value:
        symbols = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
        return symbols or None
    if path:
        if not path.is_file():
            raise ValueError(f"股票代码文件不存在: {path}")
        symbols = tuple(
            dict.fromkeys(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        )
        return symbols or None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
