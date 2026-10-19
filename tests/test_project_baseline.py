import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.cli import main  # noqa: E402
from kline_retrieval.config import DEFAULT_CONFIG, ProjectConfig  # noqa: E402


class ProjectBaselineTests(unittest.TestCase):
    def test_default_config_matches_project_contract(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.monthly_period, 20)
        self.assertEqual(DEFAULT_CONFIG.weekly_period, 5)
        self.assertEqual(DEFAULT_CONFIG.daily_window, 12)
        self.assertEqual(DEFAULT_CONFIG.chart_size, (36, 36))
        self.assertEqual(DEFAULT_CONFIG.future_horizons, (1, 5, 10, 12, 21))
        self.assertEqual(DEFAULT_CONFIG.day_weight, 1.0)

    def test_config_can_be_overridden_by_environment_values(self) -> None:
        config = ProjectConfig.from_env(
            {
                "KLINE_DATA_PATH": "data/custom.csv",
                "KLINE_MONTHLY_PERIOD": "10",
                "KLINE_WEEK_WEIGHT": "2.5",
            }
        )

        self.assertEqual(config.data_path, Path("data/custom.csv"))
        self.assertEqual(config.monthly_period, 10)
        self.assertEqual(config.week_weight, 2.5)

    def test_cli_loads_and_validates_sample_data(self) -> None:
        result = main(
            [
                "--data",
                str(ROOT / "data" / "sample_market_data.csv"),
                "--symbol",
                "DEMO",
                "--date",
                "2024-01-10",
                "--history-count",
                "5",
                "--future-count",
                "2",
            ]
        )

        self.assertEqual(result, 0)

    def test_cli_rejects_only_one_sample_selector(self) -> None:
        result = main(["--symbol", "DEMO"])
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
