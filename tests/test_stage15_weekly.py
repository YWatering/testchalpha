import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.retrieval import (  # noqa: E402
    RetrievalSample,
    SampleEmbeddings,
    SimilarityMatch,
)
from kline_retrieval.stage15_weekly import (  # noqa: E402
    TradableLabel,
    build_weekly_schedule,
    calculate_tradable_label,
    filter_labeled_matches,
)


class Stage15WeeklyTests(unittest.TestCase):
    def test_schedule_uses_last_observed_session_and_global_horizon(self) -> None:
        market_dates = tuple(
            date.fromisoformat(value)
            for value in (
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
                "2024-02-08",
                "2024-02-19",
                "2024-02-20",
                "2024-02-21",
                "2024-02-22",
                "2024-02-23",
                "2024-02-26",
            )
        )

        schedule = build_weekly_schedule(
            market_dates,
            start_date="2024-02-01",
            end_date="2024-02-29",
            horizon_sessions=5,
        )

        self.assertEqual(schedule[0].signal_date, date(2024, 2, 8))
        self.assertEqual(schedule[0].entry_date, date(2024, 2, 19))
        self.assertEqual(schedule[0].exit_date, date(2024, 2, 23))
        self.assertEqual(schedule[1].signal_date, date(2024, 2, 23))

    def test_label_uses_market_entry_and_exit_dates_not_symbol_next_rows(self) -> None:
        schedule = build_weekly_schedule(
            tuple(
                date.fromisoformat(value)
                for value in (
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                    "2024-01-12",
                )
            ),
            start_date="2024-01-05",
            end_date="2024-01-05",
        )[0]
        bars = {
            ("600001", date(2024, 1, 5)): (9.5, 10.0, 100.0),
            ("600001", date(2024, 1, 8)): (10.0, 10.2, 100.0),
            ("600001", date(2024, 1, 12)): (10.8, 11.0, 100.0),
        }

        label = calculate_tradable_label("600001", schedule, bars)

        self.assertEqual(label.status, "tradable")
        self.assertAlmostEqual(label.gross_return, 0.1)

        missing_entry = calculate_tradable_label(
            "600001",
            schedule,
            {key: value for key, value in bars.items() if key[1] != date(2024, 1, 8)},
        )
        self.assertEqual(missing_entry.status, "missing_entry_bar")

    def test_filter_rejects_candidate_until_label_exit_is_observable(self) -> None:
        market_dates = tuple(
            date.fromisoformat(value)
            for value in (
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-15",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
            )
        )
        schedules = build_weekly_schedule(
            market_dates,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        first = schedules[0]
        embeddings = SampleEmbeddings((1.0,), (1.0,), (1.0,))
        match = SimilarityMatch(
            rank=1,
            sample=RetrievalSample("600001", first.signal_date, embeddings),
            distance=0.1,
            monthly_distance=0.1,
            weekly_distance=0.1,
            daily_distance=0.1,
        )
        label = TradableLabel(
            symbol="600001",
            signal_date=first.signal_date,
            entry_date=first.entry_date,
            exit_date=first.exit_date,
            status="tradable",
            entry_open=10.0,
            exit_close=11.0,
            entry_amount=100.0,
            exit_amount=100.0,
            gross_return=0.1,
        )
        schedule_by_date = {item.signal_date: item for item in schedules}
        labels = {("600001", first.signal_date): label}

        hidden = filter_labeled_matches(
            [match],
            query_date=date(2024, 1, 11),
            schedule_by_date=schedule_by_date,
            labels=labels,
            top_k=20,
        )
        visible = filter_labeled_matches(
            [match],
            query_date=date(2024, 1, 12),
            schedule_by_date=schedule_by_date,
            labels=labels,
            top_k=20,
        )

        self.assertEqual(hidden, ())
        self.assertEqual(len(visible), 1)


if __name__ == "__main__":
    unittest.main()
