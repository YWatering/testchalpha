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
    DEFAULT_A_SHARE_SYMBOLS,
    DataSyncError,
    normalize_provider_rows,
    sync_a_share_daily,
)
from kline_retrieval.market_data import MarketData  # noqa: E402
from kline_retrieval.sqlite_store import (  # noqa: E402
    clear_checkpoint,
    load_checkpoint,
    read_metadata,
    stage_symbol,
    write_snapshot,
)


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
    def test_default_a_share_pool_covers_nine_symbols(self) -> None:
        self.assertEqual(len(DEFAULT_A_SHARE_SYMBOLS), 9)
        self.assertEqual(len(set(DEFAULT_A_SHARE_SYMBOLS)), 9)

    def test_sqlite_snapshot_is_readable_after_transactional_write(self) -> None:
        rows = [
            {
                "symbol": "000001",
                "date": "2024-01-02",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1000,
            }
        ]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            write_snapshot(
                database,
                rows,
                source_hash="snapshot-1",
                manifest={"provider": "test"},
            )
            loaded = MarketData.from_sqlite(database)
            metadata = read_metadata(database)

        self.assertEqual(loaded.symbols, ("000001",))
        self.assertEqual(loaded.source_hash, "snapshot-1")
        self.assertEqual(metadata["manifest"], '{"provider": "test"}')

    def test_sqlite_failed_snapshot_keeps_previous_committed_data(self) -> None:
        first = {
            "symbol": "000001",
            "date": "2024-01-02",
            "open": 10.0,
            "high": 10.5,
            "low": 9.8,
            "close": 10.2,
            "volume": 1000,
        }
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            write_snapshot(database, [first], source_hash="snapshot-1", manifest={})
            with self.assertRaises(OSError):
                write_snapshot(
                    database,
                    [first, first],
                    source_hash="snapshot-2",
                    manifest={},
                )
            loaded = MarketData.from_sqlite(database)

        self.assertEqual(loaded.source_hash, "snapshot-1")
        self.assertEqual(loaded.get_bar("000001", "2024-01-02").close, 10.2)

    def test_sqlite_sync_checkpoint_can_resume_and_clear(self) -> None:
        rows = normalize_provider_rows("000001", [ak_row("2024-01-02")])
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.sqlite3"
            stage_symbol(database, "run-1", "000001", rows)
            completed, staged_rows = load_checkpoint(database, "run-1")
            clear_checkpoint(database, "run-1")
            cleared, _ = load_checkpoint(database, "run-1")

        self.assertEqual(completed, {"000001"})
        self.assertEqual(staged_rows[0]["date"], "2024-01-02")
        self.assertEqual(cleared, set())

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

    def test_akshare_sina_endpoint_routes_beijing_symbol(self) -> None:
        calls = []

        class Frame:
            empty = True

        def fetcher(**kwargs):
            calls.append(kwargs)
            return Frame()

        with patch.object(sys, "version_info", (3, 12, 0)):
            provider = AKShareProvider(fetcher=fetcher, endpoint="sina")
        provider.fetch_daily("920001", date(2024, 1, 1), date(2024, 1, 31), "")

        self.assertEqual(calls[0]["symbol"], "bj920001")

    def test_akshare_adapter_reports_runtime_requirement(self) -> None:
        with patch.object(sys, "version_info", (3, 10, 4)):
            with self.assertRaisesRegex(DataProviderError, "Python 3.11"):
                AKShareProvider(fetcher=lambda **_: [])

    def test_akshare_universe_normalizes_listed_market_metadata(self) -> None:
        class Frame:
            def to_dict(self, orient):
                return [
                    {"code": "688001", "name": "科创样例"},
                    {"code": 920001, "name": "北交样例"},
                    {"code": "000001", "name": "深市样例"},
                    {"name": "缺少代码"},
                    {"code": "000002", "name": ""},
                ]

        with patch.object(sys, "version_info", (3, 12, 0)):
            provider = AKShareProvider(
                fetcher=lambda **_: [],
                universe_fetcher=lambda: Frame(),
            )
        universe = provider.list_a_share_universe()

        self.assertEqual([item["symbol"] for item in universe], ["000001", "688001", "920001"])
        self.assertEqual(universe[1]["board"], "STAR")
        self.assertEqual(universe[2]["exchange"], "BSE")

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

    def test_full_market_sync_isolates_a_failed_symbol(self) -> None:
        class PartialProvider(FakeProvider):
            def fetch_daily(self, symbol, start_date, end_date, adjust):
                if symbol == "000002":
                    raise DataProviderError("temporary source failure")
                return super().fetch_daily(symbol, start_date, end_date, adjust)

        provider = PartialProvider(
            {
                "000001": [ak_row("2024-01-02")],
                "000002": [],
            }
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "partial.csv"
            database = Path(directory) / "partial.sqlite3"
            result = sync_a_share_daily(
                provider,
                ["000001", "000002"],
                date(2024, 1, 1),
                date(2024, 1, 31),
                output,
                database_path=database,
                continue_on_error=True,
                retries=1,
                sleep=lambda _: None,
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            loaded = MarketData.from_sqlite(database)

        self.assertEqual(result.failed_symbols, ("000002",))
        self.assertEqual(manifest["failed_symbols"], ["000002"])
        self.assertEqual(loaded.symbols, ("000001",))


if __name__ == "__main__":
    unittest.main()
