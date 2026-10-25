"""Atomically publish a set of compatible ColProj index shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", action="append", type=Path, required=True)
    return parser


def publish_index_manifest(output: str | Path, index_paths: Sequence[str | Path]) -> dict[str, object]:
    destination = Path(output)
    paths = tuple(Path(path) for path in index_paths)
    if not paths:
        raise ValueError("至少需要一个索引分片")
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(f"索引分片不存在: {missing}")
    manifest = {
        "format": "kline-index-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "indexes": [os.path.relpath(path, destination.parent) for path in paths],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = publish_index_manifest(args.output, args.index)
    except (OSError, ValueError) as exc:
        print(f"索引清单发布失败: {exc}")
        return 2
    print(f"manifest: {args.output}")
    print(f"indexes: {len(manifest['indexes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

