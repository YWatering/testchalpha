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
    simulate_portfolio,
)


class Stage15PortfolioTests(unittest.TestCase):
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
