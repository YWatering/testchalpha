"""Generate deterministic multi-symbol data for local UI and API smoke tests."""

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path


def generate_rows(symbol: str, count: int, base: float, phase: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = date(2023, 1, 3)
    index = 0
    while len(rows) < count:
        if current.weekday() < 5:
            wave = math.sin(index / 8.0 + phase) * 2.4
            cycle = math.sin(index / 23.0 + phase * 0.7) * 1.2
            trend = index * 0.035
            close = base + trend + wave + cycle
            open_price = close - math.sin(index / 4.0 + phase) * 0.7
            high = max(open_price, close) + 1.2 + abs(math.sin(index / 5.0)) * 0.4
            low = min(open_price, close) - 1.0 - abs(math.cos(index / 6.0)) * 0.35
            rows.append(
                {
                    "symbol": symbol,
                    "date": current.isoformat(),
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": 1_000_000 + index * 1_250,
                }
            )
            index += 1
        current += timedelta(days=1)
    return rows


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data" / "demo_market_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        generate_rows("DEMO", 360, 100.0, 0.0)
        + generate_rows("ALPHA", 360, 62.0, 1.4)
        + generate_rows("BETA", 360, 148.0, 2.8)
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("symbol", "date", "open", "high", "low", "close", "volume"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
