import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kline_retrieval.stage15_portfolio import (  # noqa: E402
    CostConfig,
    RawBar,
    _is_limit_down,
    _is_limit_up,
    _floor_lot,
    _mark_price_at_open,
    build_equal_weight_benchmark,
    simulate_portfolio,
)


class Stage15PortfolioTests(unittest.TestCase):
    def test_equal_weight_benchmark_uses_calendar_and_full_pool(self) -> None:
        signal = date(2024, 1, 5)
        entry = date(2024, 1, 8)
        exit_date = date(2024, 1, 12)
        benchmark = build_equal_weight_benchmark(
            {
                signal: [
                    {
                        "symbol": "000001",
                        "entry_date": entry,
                        "exit_date": exit_date,
                        "predicted_return": -0.02,
                        "signal_strength": None,
                    }
                ]
            },
            ("000001", "000002"),
        )

        self.assertEqual(
            [row["symbol"] for row in benchmark[signal]],
            ["000001", "000002"],
        )
        self.assertTrue(
            all(row["predicted_return"] == 1.0 for row in benchmark[signal])
        )

    def test_weekly_simulation_applies_lot_size_costs_and_cash_weight_cap(self) -> None:
        signal = date(2024, 1, 5)
        entry = date(2024, 1, 8)
        exit_date = date(2024, 1, 12)
        bars = {
            ("000001", entry): RawBar("000001", entry, 10, 10.2, 9.9, 10.1, 10),
            ("000001", exit_date): RawBar("000001", exit_date, 10.8, 11, 10.7, 11, 10.1),
        }
        result = simulate_portfolio(
            {
                signal: [
                    {
                        "symbol": "000001",
                        "signal_date": signal,
                        "entry_date": entry,
                        "exit_date": exit_date,
                        "predicted_return": 0.03,
                        "signal_strength": 0.5,
                    }
                ]
            },
            bars,
            {"000001": "MAIN"},
            (signal, entry, date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11), exit_date),
            threshold=0.01,
            min_signal_strength=None,
            initial_capital=100_000,
            costs=CostConfig(commission_rate=0.001, stamp_duty_rate=0.001, slippage_rate=0.001, minimum_commission=0, lot_size=100),
        )

        trade = result["trades"][0]
        self.assertEqual(trade.action, "buy")
        self.assertEqual(trade.quantity, 900)
        self.assertAlmostEqual(trade.price, 10.01)
        self.assertAlmostEqual(result["metrics"]["final_value"], 100_000 + 900 * 11 - trade.notional - trade.commission)
        self.assertAlmostEqual(result["weekly"][0]["transaction_cost"], trade.commission + trade.slippage)

    def test_single_asset_can_use_full_position_weight(self) -> None:
        signal = date(2024, 1, 5)
        entry = date(2024, 1, 8)
        exit_date = date(2024, 1, 12)
        bar = RawBar("510300", entry, 10, 10, 10, 10, 10)
        result = simulate_portfolio(
            {
                signal: [
                    {
                        "symbol": "510300",
                        "entry_date": entry,
                        "exit_date": exit_date,
                        "predicted_return": 1.0,
                        "signal_strength": None,
                    }
                ]
            },
            {
                ("510300", entry): bar,
                ("510300", exit_date): RawBar(
                    "510300", exit_date, 10, 10, 10, 10, 10
                ),
            },
            {"510300": "ETF"},
            (signal, entry, exit_date),
            threshold=0,
            min_signal_strength=None,
            initial_capital=100_000,
            costs=CostConfig(
                commission_rate=0,
                stamp_duty_rate=0,
                slippage_rate=0,
                minimum_commission=0,
                lot_size=100,
            ),
            rebalance=False,
            max_position_weight=1.0,
        )

        self.assertEqual(result["trades"][0].quantity, 10_000)
        self.assertEqual(result["weekly"][0]["cash_end"], 0)

    def test_buy_and_hold_does_not_rebalance_after_initial_entry(self) -> None:
        first_signal = date(2024, 1, 5)
        first_entry = date(2024, 1, 8)
        first_exit = date(2024, 1, 12)
        second_signal = first_exit
        second_entry = date(2024, 1, 15)
        second_exit = date(2024, 1, 19)
        rows = build_equal_weight_benchmark(
            {
                first_signal: [
                    {
                        "entry_date": first_entry,
                        "exit_date": first_exit,
                    }
                ],
                second_signal: [
                    {
                        "entry_date": second_entry,
                        "exit_date": second_exit,
                    }
                ],
            },
            ("000001", "000002"),
        )
        bars = {
            ("000001", first_entry): RawBar("000001", first_entry, 10, 10, 10, 10, 10),
            ("000002", first_entry): RawBar("000002", first_entry, 20, 20, 20, 20, 20),
            ("000001", first_exit): RawBar("000001", first_exit, 20, 20, 20, 20, 10),
            ("000002", first_exit): RawBar("000002", first_exit, 20, 20, 20, 20, 20),
            ("000001", second_entry): RawBar("000001", second_entry, 20, 20, 20, 20, 20),
            ("000002", second_entry): RawBar("000002", second_entry, 20, 20, 20, 20, 20),
            ("000001", second_exit): RawBar("000001", second_exit, 22, 22, 22, 22, 20),
            ("000002", second_exit): RawBar("000002", second_exit, 18, 18, 18, 18, 20),
        }
        result = simulate_portfolio(
            rows,
            bars,
            {"000001": "MAIN", "000002": "MAIN"},
            (first_signal, first_entry, first_exit, second_entry, second_exit),
            threshold=0,
            min_signal_strength=None,
            initial_capital=100_000,
            costs=CostConfig(
                commission_rate=0,
                stamp_duty_rate=0,
                slippage_rate=0,
                minimum_commission=0,
                lot_size=100,
            ),
            rebalance=False,
        )

        self.assertEqual(len(result["trades"]), 2)
        self.assertEqual(result["weekly"][1]["executed_trade_count"], 0)
        self.assertEqual(result["weekly"][1]["turnover"], 0)
        self.assertEqual(result["weekly"][1]["holding_count"], 2)

    def test_missing_initial_bar_is_recorded_as_blocked(self) -> None:
        signal = date(2024, 1, 5)
        entry = date(2024, 1, 8)
        exit_date = date(2024, 1, 12)
        rows = build_equal_weight_benchmark(
            {
                signal: [
                    {
                        "entry_date": entry,
                        "exit_date": exit_date,
                    }
                ]
            },
            ("000001",),
        )
        result = simulate_portfolio(
            rows,
            {},
            {"000001": "MAIN"},
            (signal, entry, exit_date),
            threshold=0,
            min_signal_strength=None,
            initial_capital=100_000,
            costs=CostConfig(),
            rebalance=False,
        )

        self.assertEqual(result["weekly"][0]["blocked_trade_count"], 1)
        self.assertEqual(result["trades"][0].reason, "missing_entry_bar")
        self.assertEqual(result["trades"][0].quantity, 0)

    def test_target_below_one_lot_is_recorded_as_blocked(self) -> None:
        signal = date(2024, 1, 5)
        entry = date(2024, 1, 8)
        exit_date = date(2024, 1, 12)
        rows = build_equal_weight_benchmark(
            {
                signal: [
                    {
                        "entry_date": entry,
                        "exit_date": exit_date,
                    }
                ]
            },
            ("000001", "000002", "000003", "000004", "000005", "000006"),
        )
        bars = {
            (symbol, entry): RawBar(symbol, entry, 2_000, 2_000, 2_000, 2_000, 2_000)
            for symbol in ("000001", "000002", "000003", "000004", "000005", "000006")
        }
        result = simulate_portfolio(
            rows,
            bars,
            {symbol: "MAIN" for symbol, _ in bars},
            (signal, entry, exit_date),
            threshold=0,
            min_signal_strength=None,
            initial_capital=100_000,
            costs=CostConfig(),
            rebalance=False,
        )

        self.assertEqual(result["metrics"]["blocked_trade_count"], 6)
        self.assertTrue(
            all(trade.reason == "target_below_one_lot" for trade in result["trades"])
        )

    def test_selected_set_exit_sells_at_next_open_and_charges_sell_costs(self) -> None:
        first_signal = date(2024, 1, 5)
        first_entry = date(2024, 1, 8)
        first_exit = date(2024, 1, 12)
        second_signal = first_exit
        second_entry = date(2024, 1, 15)
        second_exit = date(2024, 1, 19)
        bars = {
            ("000001", first_entry): RawBar("000001", first_entry, 10, 10, 10, 10, 10),
            ("000001", first_exit): RawBar("000001", first_exit, 11, 11, 11, 11, 10),
            ("000001", second_entry): RawBar("000001", second_entry, 10.5, 10.6, 10.4, 10.5, 11),
            ("000001", second_exit): RawBar("000001", second_exit, 10.5, 10.6, 10.4, 10.5, 10.5),
        }
        yes = {"symbol": "000001", "entry_date": first_entry, "exit_date": first_exit, "predicted_return": .02, "signal_strength": .5}
        no = {"symbol": "000001", "entry_date": second_entry, "exit_date": second_exit, "predicted_return": -.01, "signal_strength": .5}
        result = simulate_portfolio(
            {first_signal: [yes], second_signal: [no]},
            bars,
            {"000001": "MAIN"},
            (first_signal, first_entry, date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11), first_exit, second_entry, date(2024, 1, 16), date(2024, 1, 17), date(2024, 1, 18), second_exit),
            threshold=0,
            min_signal_strength=None,
            initial_capital=100_000,
            costs=CostConfig(commission_rate=0, stamp_duty_rate=.001, slippage_rate=0, minimum_commission=0, lot_size=100),
        )

        self.assertEqual([item.action for item in result["trades"]], ["buy", "sell"])
        self.assertGreater(result["trades"][1].stamp_duty, 0)
        self.assertEqual(result["weekly"][1]["holding_count"], 0)

    def test_one_price_limit_moves_block_at_open(self) -> None:
        limit_up = RawBar("000001", date(2024, 1, 8), 11, 11, 11, 11, 10)
        limit_down = RawBar("000001", date(2024, 1, 8), 9, 9, 9, 9, 10)
        self.assertTrue(_is_limit_up(limit_up, "MAIN"))
        self.assertTrue(_is_limit_down(limit_down, "MAIN"))
        self.assertFalse(_is_limit_up(limit_up, "STAR"))

    def test_lot_size_floors_down(self) -> None:
        self.assertEqual(_floor_lot(299.9, 100), 200)
        self.assertEqual(_floor_lot(99.9, 100), 0)

    def test_rebalance_equity_uses_entry_open_not_same_day_close(self) -> None:
        day = date(2024, 1, 8)
        bar = RawBar("000001", day, 10, 15, 9, 14, 9.5)
        self.assertEqual(_mark_price_at_open("000001", day, {("000001", day): bar}), 10)


if __name__ == "__main__":
    unittest.main()
