"""Fee-aware weekly portfolio simulation for stage fifteen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RawBar:
    symbol: str
    session: date
    open: float
    high: float
    low: float
    close: float
    pre_close: float | None


@dataclass(frozen=True, slots=True)
class CostConfig:
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.0005
    minimum_commission: float = 5.0
    lot_size: int = 100


@dataclass(frozen=True, slots=True)
class Trade:
    signal_date: date
    entry_date: date
    symbol: str
    action: str
    quantity: int
    price: float | None
    notional: float
    commission: float
    stamp_duty: float
    slippage: float
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=Path("data/processed/csmar_a_share_daily.sqlite3")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("research/stage15/colproj_validation_predictions.csv"),
    )
    parser.add_argument(
        "--stock-pool", type=Path, default=Path("research/stage15/stock_pool.csv")
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--predicted-return-threshold", type=float, default=0.01)
    parser.add_argument("--threshold-grid", default="0,0.005,0.01,0.015,0.02,0.05")
    parser.add_argument("--min-signal-strength", type=float)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--minimum-commission", type=float, default=5.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument(
        "--weekly-output",
        type=Path,
        default=Path("research/stage15/colproj_portfolio_weekly.csv"),
    )
    parser.add_argument(
        "--trades-output",
        type=Path,
        default=Path("research/stage15/colproj_portfolio_trades.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("research/stage15/colproj_portfolio_manifest.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("research/stage15/COLPROJ_PORTFOLIO_RESULTS.md"),
    )
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=Path("research/stage15/colproj_portfolio_thresholds.csv"),
    )
    parser.add_argument(
        "--evaluation-period",
        choices=("validation", "out-of-sample"),
        default="validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.initial_capital <= 0:
        raise ValueError("initial-capital 必须大于 0")
    costs = CostConfig(
        commission_rate=args.commission_rate,
        stamp_duty_rate=args.stamp_duty_rate,
        slippage_rate=args.slippage_rate,
        minimum_commission=args.minimum_commission,
        lot_size=args.lot_size,
    )
    predictions = load_predictions(args.predictions, args.start_date, args.end_date)
    symbols = load_stock_pool(args.stock_pool)
    bars, boards, market_dates = load_database_data(
        args.database,
        symbols,
        args.start_date,
        (date.fromisoformat(args.end_date) + timedelta(days=14)).isoformat(),
    )
    result = simulate_portfolio(
        predictions,
        bars,
        boards,
        market_dates,
        threshold=args.predicted_return_threshold,
        min_signal_strength=args.min_signal_strength,
        initial_capital=args.initial_capital,
        costs=costs,
    )
    grid = []
    for threshold in _parse_threshold_grid(args.threshold_grid):
        candidate = simulate_portfolio(
            predictions,
            bars,
            boards,
            market_dates,
            threshold=threshold,
            min_signal_strength=args.min_signal_strength,
            initial_capital=args.initial_capital,
            costs=costs,
        )
        grid.append(
            {
                "predicted_return_threshold": threshold,
                "selected_signal_count": sum(
                    int(row["selected_count"]) for row in candidate["weekly"]
                ),
                **candidate["metrics"],
            }
        )
    write_result(
        result,
        args.weekly_output,
        args.trades_output,
        args.manifest_output,
        args.report_output,
        args.thresholds_output,
        threshold_grid=grid,
        evaluation_period=args.evaluation_period,
    )
    print(f"weekly_periods: {len(result['weekly'])}")
    print(f"trade_count: {len(result['trades'])}")
    print(f"final_value: {result['metrics']['final_value']:.2f}")
    print(f"cumulative_return: {result['metrics']['cumulative_return']:.4f}")
    print(f"report: {args.report_output}")
    return 0


def load_predictions(
    path: str | Path, start_date: str, end_date: str
) -> dict[date, list[dict[str, object]]]:
    output: dict[date, list[dict[str, object]]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            if row.get("actual_status") not in (None, "", "tradable"):
                continue
            signal_date = date.fromisoformat(str(row["signal_date"]))
            if not date.fromisoformat(start_date) <= signal_date <= date.fromisoformat(end_date):
                continue
            predicted = row.get("predicted_return")
            if predicted in (None, "") or not row.get("entry_date") or not row.get("exit_date"):
                continue
            output.setdefault(signal_date, []).append(
                {
                    "symbol": str(row["symbol"]),
                    "signal_date": signal_date,
                    "entry_date": date.fromisoformat(str(row["entry_date"])),
                    "exit_date": date.fromisoformat(str(row["exit_date"])),
                    "predicted_return": float(predicted),
                    "signal_strength": (
                        float(row["signal_strength"])
                        if row.get("signal_strength") not in (None, "")
                        else None
                    ),
                }
            )
    if not output:
        raise ValueError("预测文件中没有可模拟的股票—周记录")
    return output


def load_stock_pool(path: str | Path) -> tuple[str, ...]:
    symbols: list[str] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            symbol = str(row.get("symbol", "")).strip()
            if symbol:
                symbols.append(symbol)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("股票池为空或包含重复代码")
    return tuple(symbols)


def load_database_data(
    database: str | Path,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
) -> tuple[dict[tuple[str, date], RawBar], dict[str, str], tuple[date, ...]]:
    placeholders = ",".join("?" for _ in symbols)
    with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            f"""
            SELECT symbol, date, open, high, low, close, pre_close
            FROM daily_bars
            WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?
            """,
            (*symbols, start_date, end_date),
        )
        bars = {
            (str(symbol), date.fromisoformat(str(session))): RawBar(
                symbol=str(symbol),
                session=date.fromisoformat(str(session)),
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=float(close),
                pre_close=float(pre_close) if pre_close is not None else None,
            )
            for symbol, session, open_price, high, low, close, pre_close in rows
        }
        boards = {
            str(symbol): str(board)
            for symbol, board in connection.execute(
                f"SELECT symbol, board FROM instruments WHERE symbol IN ({placeholders})", symbols
            )
        }
        market_dates = tuple(
            date.fromisoformat(str(session))
            for (session,) in connection.execute(
                "SELECT DISTINCT date FROM daily_bars WHERE date BETWEEN ? AND ? ORDER BY date",
                (start_date, end_date),
            )
        )
    return bars, boards, market_dates


def simulate_portfolio(
    predictions: dict[date, list[dict[str, object]]],
    bars: dict[tuple[str, date], RawBar],
    boards: dict[str, str],
    market_dates: Sequence[date],
    *,
    threshold: float,
    min_signal_strength: float | None,
    initial_capital: float,
    costs: CostConfig,
) -> dict[str, object]:
    if costs.lot_size <= 0 or any(
        value < 0
        for value in (
            costs.commission_rate,
            costs.stamp_duty_rate,
            costs.slippage_rate,
            costs.minimum_commission,
        )
    ):
        raise ValueError("交易成本参数不能为负，lot_size 必须为正")
    sessions = set(market_dates)
    holdings: dict[str, int] = {}
    cash = float(initial_capital)
    weekly: list[dict[str, object]] = []
    trades: list[Trade] = []
    previous_close_value = float(initial_capital)
    for signal_date in sorted(predictions):
        rows = predictions[signal_date]
        entry_date = _single_date(rows, "entry_date")
        exit_date = _single_date(rows, "exit_date")
        if entry_date not in sessions or exit_date not in sessions:
            continue
        entry_bars = {symbol: bars.get((symbol, entry_date)) for symbol in set(holdings) | {str(row["symbol"]) for row in rows}}
        selected = [
            row
            for row in rows
            if float(row["predicted_return"]) >= threshold
            and (
                min_signal_strength is None
                or row["signal_strength"] is not None
                and float(row["signal_strength"]) >= min_signal_strength
            )
        ]
        selected_symbols = tuple(sorted({str(row["symbol"]) for row in selected}))
        pre_trade_equity = cash + sum(
            quantity * _mark_price_at_open(symbol, entry_date, bars)
            for symbol, quantity in holdings.items()
            if _mark_bar(symbol, entry_date, bars) is not None
        )
        target_weight = min(1.0 / len(selected_symbols), 0.10) if selected_symbols else 0.0
        desired = {
            symbol: _floor_lot(
                pre_trade_equity * target_weight / (entry_bars[symbol].open * (1 + costs.slippage_rate)),
                costs.lot_size,
            )
            for symbol in selected_symbols
            if entry_bars.get(symbol) is not None and entry_bars[symbol].open > 0
        }
        period_trades: list[Trade] = []
        blocked = 0
        for symbol in sorted(set(holdings) | set(desired)):
            current = holdings.get(symbol, 0)
            target = desired.get(symbol, 0)
            if current <= target:
                continue
            bar = entry_bars.get(symbol)
            quantity = current - target
            if bar is None:
                blocked += 1
                period_trades.append(_blocked_trade(signal_date, entry_date, symbol, "sell", quantity, "missing_entry_bar"))
                continue
            if _is_limit_down(bar, boards.get(symbol, "")):
                blocked += 1
                period_trades.append(_blocked_trade(signal_date, entry_date, symbol, "sell", quantity, "opening_limit_down"))
                continue
            trade, cash = _execute_sell(signal_date, entry_date, symbol, quantity, bar, cash, costs)
            holdings[symbol] = current - quantity
            period_trades.append(trade)
        for symbol in sorted(desired):
            current = holdings.get(symbol, 0)
            quantity = desired[symbol] - current
            bar = entry_bars.get(symbol)
            if quantity <= 0:
                continue
            if bar is None:
                blocked += 1
                period_trades.append(_blocked_trade(signal_date, entry_date, symbol, "buy", quantity, "missing_entry_bar"))
                continue
            if _is_limit_up(bar, boards.get(symbol, "")):
                blocked += 1
                period_trades.append(_blocked_trade(signal_date, entry_date, symbol, "buy", quantity, "opening_limit_up"))
                continue
            quantity = _affordable_buy_quantity(quantity, bar.open * (1 + costs.slippage_rate), cash, costs)
            if quantity <= 0:
                blocked += 1
                period_trades.append(_blocked_trade(signal_date, entry_date, symbol, "buy", desired[symbol] - current, "insufficient_cash_after_costs"))
                continue
            trade, cash = _execute_buy(signal_date, entry_date, symbol, quantity, bar, cash, costs)
            holdings[symbol] = current + quantity
            period_trades.append(trade)
        trades.extend(period_trades)
        exit_bars = {symbol: _mark_bar(symbol, exit_date, bars) for symbol in holdings}
        end_value = cash + sum(
            quantity * exit_bars[symbol].close
            for symbol, quantity in holdings.items()
            if exit_bars.get(symbol) is not None
        )
        if end_value <= 0:
            raise ValueError("组合期末权益必须大于0")
        weekly.append(
            {
                "signal_date": signal_date.isoformat(),
                "entry_date": entry_date.isoformat(),
                "exit_date": exit_date.isoformat(),
                "candidate_count": len(rows),
                "selected_count": len(selected_symbols),
                "executed_trade_count": sum(item.quantity > 0 and item.price is not None for item in period_trades),
                "blocked_trade_count": blocked,
                "start_value": previous_close_value,
                "end_value": end_value,
                "net_return": end_value / previous_close_value - 1,
                "cash_end": cash,
                "holding_count": sum(quantity > 0 for quantity in holdings.values()),
                "turnover": sum(item.notional for item in period_trades if item.price is not None) / pre_trade_equity,
                "transaction_cost": sum(item.commission + item.stamp_duty + item.slippage for item in period_trades),
            }
        )
        previous_close_value = end_value
    metrics = summarize_portfolio(weekly, initial_capital)
    return {"weekly": weekly, "trades": trades, "metrics": metrics, "parameters": {"threshold": threshold, "min_signal_strength": min_signal_strength, "initial_capital": initial_capital, "costs": asdict(costs)}}


def summarize_portfolio(weekly: Sequence[dict[str, object]], initial_capital: float) -> dict[str, float]:
    values = [initial_capital] + [float(row["end_value"]) for row in weekly]
    returns = [float(row["net_return"]) for row in weekly]
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    volatility = math.sqrt(sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)) if len(returns) > 1 else 0.0
    final_value = values[-1]
    return {
        "period_count": float(len(weekly)),
        "initial_capital": initial_capital,
        "final_value": final_value,
        "cumulative_return": final_value / initial_capital - 1,
        "annualized_return": (final_value / initial_capital) ** (52 / len(weekly)) - 1 if weekly and final_value > 0 else 0.0,
        "annualized_volatility": volatility * math.sqrt(52),
        "max_drawdown": max_drawdown,
        "weekly_win_rate": sum(value > 0 for value in returns) / len(returns) if returns else 0.0,
        "sharpe_zero_rf": mean_return / volatility * math.sqrt(52) if volatility > 0 else 0.0,
        "average_holding_count": sum(float(row["holding_count"]) for row in weekly) / len(weekly) if weekly else 0.0,
        "average_cash_ratio": sum(float(row["cash_end"]) / float(row["end_value"]) for row in weekly) / len(weekly) if weekly else 0.0,
        "total_turnover": sum(float(row["turnover"]) for row in weekly),
        "total_transaction_cost": sum(float(row["transaction_cost"]) for row in weekly),
    }


def _parse_threshold_grid(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("threshold-grid 必须是逗号分隔的数字") from exc
    if not thresholds or any(not math.isfinite(item) for item in thresholds):
        raise ValueError("threshold-grid 至少需要一个有限阈值")
    return tuple(dict.fromkeys(thresholds))


def write_result(
    result: dict[str, object],
    weekly_path: str | Path,
    trades_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
    thresholds_path: str | Path,
    *,
    threshold_grid: Sequence[dict[str, object]],
    evaluation_period: str = "validation",
) -> None:
    _write_rows(weekly_path, result["weekly"])
    _write_rows(trades_path, [asdict(trade) for trade in result["trades"]])
    _write_rows(thresholds_path, threshold_grid)
    manifest = {"schema_version": "stage15-portfolio-v1", "parameters": result["parameters"], "metrics": result["metrics"], "weekly_count": len(result["weekly"]), "trade_count": len(result["trades"]), "threshold_grid": list(threshold_grid), "execution_price_basis": "raw_open_and_raw_close", "limitations": ["收益为价格收益，未包含现金分红总回报。", "涨跌停状态根据日线OHLC、昨收盘和板块比例保守推断，数据库没有交易所原始涨跌停状态字段。", "期末持仓按最后可见收盘价估值，未额外模拟期末强制平仓成本。", "结果是研究模拟，不是实盘建议。"]}
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = result["metrics"]
    period_name = "样本外测试" if evaluation_period == "out-of-sample" else "验证期"
    weekly_name = Path(weekly_path).name
    trades_name = Path(trades_path).name
    thresholds_name = Path(thresholds_path).name
    text = f"""# 阶段十五：ColProj周频组合模拟

## 模拟口径

- 预测输入：网站同口径生成的 ColProj 周频预测结果。
- 买入条件：预测收益不低于 `{result['parameters']['threshold']:.2%}`""" + (f"，稳定性分数不低于 `{result['parameters']['min_signal_strength']:.2f}`。" if result["parameters"]["min_signal_strength"] is not None else "。") + f"""
- 交易：下一市场交易日开盘调仓，第5个市场交易日收盘估值。
- 仓位：满足条件的股票等权，单只股票目标权重不超过10%，剩余资金保留现金。
- 成本：佣金、卖出印花税、双边滑点和100股整手约束均纳入。
- 价格：交易使用原始开盘价和原始收盘价，收益率重建价格只用于形态检索。

## 结果

| 指标 | 数值 |
| --- | ---: |
| 周期数 | {int(metrics['period_count'])} |
| 初始资金 | {metrics['initial_capital']:.2f} |
| 期末权益 | {metrics['final_value']:.2f} |
| 累计收益 | {metrics['cumulative_return']:.2%} |
| 年化收益 | {metrics['annualized_return']:.2%} |
| 年化波动率 | {metrics['annualized_volatility']:.2%} |
| 最大回撤 | {metrics['max_drawdown']:.2%} |
| 周胜率 | {metrics['weekly_win_rate']:.2%} |
| Sharpe（无风险利率0） | {metrics['sharpe_zero_rf']:.3f} |
| 平均持仓数 | {metrics['average_holding_count']:.2f} |
| 平均现金比例 | {metrics['average_cash_ratio']:.2%} |
| 累计换手率 | {metrics['total_turnover']:.2%} |
| 交易成本 | {metrics['total_transaction_cost']:.2f} |

逐周结果见 `{weekly_name}`，逐笔交易见 `{trades_name}`。涨跌停阻断和缺失行情不会被静默当作已成交，而会在逐笔交易文件中记录原因。

## {period_name}阈值敏感性

| 预测收益阈值 | 入选信号数 | 累计收益 | 最大回撤 | Sharpe | 交易成本 |
| ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        "| {threshold:.2%} | {count} | {return_:.2%} | {drawdown:.2%} | {sharpe:.3f} | {cost:.2f} |".format(
            threshold=float(row["predicted_return_threshold"]),
            count=int(row["selected_signal_count"]),
            return_=float(row["cumulative_return"]),
            drawdown=float(row["max_drawdown"]),
            sharpe=float(row["sharpe_zero_rf"]),
            cost=float(row["total_transaction_cost"]),
        )
        for row in threshold_grid
    ) + f"""

阈值网格仅作为同一{period_name}上的敏感性描述；不能根据表中最高收益挑选并宣称该阈值有效。阈值结果CSV见 `{thresholds_name}`。

## 限制

- 默认费率是假设值：佣金万分之3且最低5元、卖出印花税万分之5、买卖滑点各万分之5；请按实际账户费率覆盖参数。
- 这是{period_name}的研究模拟；除事先指定的1%参考规则外，其他阈值不得用于回看挑选策略。
- 收益为原始价格收益，未包含现金分红总回报。
- 涨跌停根据日线OHLC、昨收盘和板块涨跌幅近似推断一字板；没有交易所权威涨跌停状态、集合竞价队列或逐笔成交，仍可能高估成交能力。
- 期末持仓按最后可见收盘价估值，未额外模拟期末强制平仓成本。
- 当前 ColProj 预测统计尚未确认稳定买入优势，报告不是实盘收益预测。
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text, encoding="utf-8")


def _single_date(rows: Sequence[dict[str, object]], field: str) -> date:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ValueError(f"同一信号日的 {field} 不一致")
    return next(iter(values))


def _mark_bar(
    symbol: str, session: date, bars: dict[tuple[str, date], RawBar]
) -> RawBar | None:
    candidates = [
        bar for (bar_symbol, bar_date), bar in bars.items()
        if bar_symbol == symbol and bar_date <= session
    ]
    return max(candidates, key=lambda bar: bar.session) if candidates else None


def _mark_price_at_open(
    symbol: str, session: date, bars: dict[tuple[str, date], RawBar]
) -> float:
    current = bars.get((symbol, session))
    if current is not None:
        return current.open
    previous = _mark_bar(symbol, session, bars)
    if previous is None:
        return 0.0
    return previous.close


def _floor_lot(quantity: float, lot_size: int) -> int:
    return max(0, int(quantity // lot_size) * lot_size)


def _commission(notional: float, costs: CostConfig) -> float:
    return max(notional * costs.commission_rate, costs.minimum_commission)


def _execute_buy(signal_date: date, entry_date: date, symbol: str, quantity: int, bar: RawBar, cash: float, costs: CostConfig) -> tuple[Trade, float]:
    price = bar.open * (1 + costs.slippage_rate)
    notional = quantity * price
    commission = _commission(notional, costs)
    slippage = quantity * bar.open * costs.slippage_rate
    return Trade(signal_date, entry_date, symbol, "buy", quantity, price, notional, commission, 0.0, slippage, "executed"), cash - notional - commission


def _execute_sell(signal_date: date, entry_date: date, symbol: str, quantity: int, bar: RawBar, cash: float, costs: CostConfig) -> tuple[Trade, float]:
    price = bar.open * (1 - costs.slippage_rate)
    notional = quantity * price
    commission = _commission(notional, costs)
    stamp_duty = notional * costs.stamp_duty_rate
    slippage = quantity * bar.open * costs.slippage_rate
    return Trade(signal_date, entry_date, symbol, "sell", quantity, price, notional, commission, stamp_duty, slippage, "executed"), cash + notional - commission - stamp_duty


def _affordable_buy_quantity(target: int, price: float, cash: float, costs: CostConfig) -> int:
    quantity = target
    while quantity > 0:
        notional = quantity * price
        if notional + _commission(notional, costs) <= cash + 1e-9:
            return quantity
        quantity -= costs.lot_size
    return 0


def _blocked_trade(signal_date: date, entry_date: date, symbol: str, action: str, quantity: int, reason: str) -> Trade:
    return Trade(signal_date, entry_date, symbol, action, quantity, None, 0.0, 0.0, 0.0, 0.0, reason)


def _limit_price(pre_close: float, ratio: float, direction: int) -> float:
    value = Decimal(str(pre_close)) * (Decimal("1") + Decimal(str(ratio)) * direction)
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _is_limit_up(bar: RawBar, board: str) -> bool:
    if bar.pre_close is None or not board:
        return False
    limit = _limit_price(bar.pre_close, _limit_ratio(board), 1)
    return abs(bar.open - limit) < 0.005 and abs(bar.high - limit) < 0.005 and abs(bar.low - limit) < 0.005 and abs(bar.close - limit) < 0.005


def _is_limit_down(bar: RawBar, board: str) -> bool:
    if bar.pre_close is None or not board:
        return False
    limit = _limit_price(bar.pre_close, _limit_ratio(board), -1)
    return abs(bar.open - limit) < 0.005 and abs(bar.high - limit) < 0.005 and abs(bar.low - limit) < 0.005 and abs(bar.close - limit) < 0.005


def _limit_ratio(board: str) -> float:
    if board in {"STAR", "CHINEXT"}:
        return 0.20
    if board == "BSE":
        return 0.30
    return 0.10


def _write_rows(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
