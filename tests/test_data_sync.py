import csv
import json
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.data_provider import (  # noqa: E402
    AKShareProvider,
    DataProviderError,
)
from kline_retrieval.data_sync import (  # noqa: E402
    DataSyncError,
    normalize_provider_rows,
    sync_a_share_daily,
)
from kline_retrieval.market_data import MarketData  # noqa: E402


def ak_row(day: str, close: float = 10.0) -> dict[str, object]:
    return {
        "日期": day,
        "开盘": close - 0.2,
        "最高": close + 0.3,
        "最低": close - 0.4,
        "收盘": close,
        "成交量": 1000,
    }


class FakeProvider:
    name = "akshare"
    version = "test"
    endpoint = "eastmoney"

    def __init__(self, rows: dict[str, list[dict[str, object]]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, date, date, str]] = []

    def fetch_daily(self, symbol, start_date, end_date, adjust):
        self.calls.append((symbol, start_date, end_date, adjust))
        return self.rows[symbol]


class DataSyncTests(unittest.TestCase):
    def test_akshare_adapter_calls_daily_endpoint_and_maps_records(self) -> None:
        calls = []

        class Frame:
            empty = False

            def to_dict(self, orient):
                self.assertion = orient
                return [ak_row("2024-01-02")]

        def fetcher(**kwargs):
            calls.append(kwargs)
            return Frame()

        with patch.object(sys, "version_info", (3, 12, 0)):
            provider = AKShareProvider(fetcher=fetcher)
        rows = provider.fetch_daily(
            "000001", date(2024, 1, 1), date(2024, 1, 31), "qfq"
        )

        self.assertEqual(calls[0]["period"], "daily")
        self.assertEqual(calls[0]["start_date"], "20240101")
        self.assertEqual(calls[0]["adjust"], "qfq")
        self.assertEqual(rows[0]["日期"], "2024-01-02")

    def test_akshare_sina_endpoint_uses_prefixed_symbol_and_normalizes_english_fields(self) -> None:
        calls = []

        class Frame:
            empty = False

            def to_dict(self, orient):
                return [
                    {
                        "date": date(2024, 1, 2),
                        "open": 9.8,
                        "high": 10.3,
                        "low": 9.6,
                        "close": 10.0,
                        "volume": 1000,
                    }
                ]

        def fetcher(**kwargs):
            calls.append(kwargs)
            return Frame()

        with patch.object(sys, "version_info", (3, 12, 0)):
            provider = AKShareProvider(fetcher=fetcher, endpoint="sina")
        records = provider.fetch_daily(
            "000001", date(2024, 1, 1), date(2024, 1, 31), ""
        )

        self.assertEqual(calls[0]["symbol"], "sz000001")
        normalized = normalize_provider_rows("000001", records)
        self.assertEqual(normalized[0]["date"], "2024-01-02")

    def test_akshare_adapter_reports_runtime_requirement(self) -> None:
        with patch.object(sys, "version_info", (3, 10, 4)):
            with self.assertRaisesRegex(DataProviderError, "Python 3.11"):
                AKShareProvider(fetcher=lambda **_: [])

    def test_sync_writes_validated_csv_and_manifest(self) -> None:
        provider = FakeProvider(
            {"000001": [ak_row("2024-01-02"), ak_row("2024-01-03", 11.0)]}
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "a_share.csv"
            result = sync_a_share_daily(
                provider,
                ["000001"],
                date(2024, 1, 1),
                date(2024, 1, 31),
                output,
                sleep=lambda _: None,
            )

            loaded = MarketData.from_csv(output)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded.symbols, ("000001",))
            self.assertEqual(len(loaded.dates("000001")), 2)
            self.assertEqual(manifest["adjust"], "qfq")
            self.assertEqual(manifest["sha256"], result.source_hash)
            self.assertEqual(manifest["row_count"], 2)

    def test_incremental_sync_refreshes_overlap_and_deduplicates(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "a_share.csv"
            first = FakeProvider(
                {"000001": [ak_row("2024-01-02"), ak_row("2024-01-03", 11.0)]}
            )
            sync_a_share_daily(
                first,
                ["000001"],
                date(2024, 1, 1),
                date(2024, 1, 3),
                output,
                sleep=lambda _: None,
            )

            second = FakeProvider(
                {"000001": [ak_row("2024-01-03", 12.0), ak_row("2024-01-04", 13.0)]}
            )
            result = sync_a_share_daily(
                second,
                ["000001"],
                date(2024, 1, 2),
                date(2024, 1, 4),
                output,
                sleep=lambda _: None,
            )

            loaded = MarketData.from_csv(output)
            self.assertEqual(result.row_count, 3)
            self.assertEqual(second.calls[0][1], date(2024, 1, 2))
            self.assertEqual(loaded.get_bar("000001", "2024-01-03").close, 12.0)
            with output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_invalid_provider_data_does_not_replace_previous_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "a_share.csv"
            initial = FakeProvider({"000001": [ak_row("2024-01-02")]})
            sync_a_share_daily(
                initial,
                ["000001"],
                date(2024, 1, 1),
                date(2024, 1, 2),
                output,
                sleep=lambda _: None,
            )
            previous_bytes = output.read_bytes()
            invalid = ak_row("2024-01-03", -2.0)
            provider = FakeProvider({"000001": [invalid]})

            with self.assertRaises(DataSyncError):
                sync_a_share_daily(
                    provider,
                    ["000001"],
                    date(2024, 1, 1),
                    date(2024, 1, 3),
                    output,
                    incremental=False,
                    sleep=lambda _: None,
                )

            self.assertEqual(output.read_bytes(), previous_bytes)

    def test_provider_retries_transient_failures_then_succeeds(self) -> None:
        class FlakyProvider(FakeProvider):
            def __init__(self):
                super().__init__({"000001": [ak_row("2024-01-02")]})
                self.attempts = 0

            def fetch_daily(self, symbol, start_date, end_date, adjust):
                self.attempts += 1
                if self.attempts < 3:
                    raise DataProviderError("temporary source failure")
                return super().fetch_daily(symbol, start_date, end_date, adjust)

        provider = FlakyProvider()
        with TemporaryDirectory() as directory:
            result = sync_a_share_daily(
                provider,
                ["000001"],
                date(2024, 1, 1),
                date(2024, 1, 31),
                Path(directory) / "a_share.csv",
                sleep=lambda _: None,
            )

        self.assertEqual(provider.attempts, 3)
        self.assertEqual(result.row_count, 1)

    def test_empty_provider_result_is_explained(self) -> None:
        provider = FakeProvider({"000001": []})
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataSyncError, "未返回"):
                sync_a_share_daily(
                    provider,
                    ["000001"],
                    date(2024, 1, 1),
                    date(2024, 1, 31),
                    Path(directory) / "empty.csv",
                    sleep=lambda _: None,
                )


if __name__ == "__main__":
    unittest.main()
