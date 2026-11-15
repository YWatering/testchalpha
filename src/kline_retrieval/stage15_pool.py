"""Build the fixed, point-in-time stock pool used by stage fifteen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class StockPoolRules:
    build_date: str
    pool_size: int = 50
    lookback_sessions: int = 60
    min_history_sessions: int = 400
    min_history_calendar_days: int = 730
    min_recent_sessions: int = 55
    min_liquid_sessions: int = 55
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str | None = None

    def __post_init__(self) -> None:
        date.fromisoformat(self.build_date)
        for field_name in (
            "pool_size",
            "lookback_sessions",
            "min_history_sessions",
            "min_recent_sessions",
            "min_liquid_sessions",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.min_history_calendar_days < 0:
            raise ValueError("min_history_calendar_days cannot be negative")
        if self.min_recent_sessions > self.lookback_sessions:
            raise ValueError("min_recent_sessions cannot exceed lookback_sessions")
        if self.min_liquid_sessions > self.lookback_sessions:
            raise ValueError("min_liquid_sessions cannot exceed lookback_sessions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/csmar_a_share_daily.sqlite3"),
    )
    parser.add_argument("--build-date", default="2023-12-29")
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--lookback-sessions", type=int, default=60)
    parser.add_argument("--min-history-sessions", type=int, default=400)
    parser.add_argument("--min-history-calendar-days", type=int, default=730)
    parser.add_argument("--min-recent-sessions", type=int, default=55)
    parser.add_argument("--min-liquid-sessions", type=int, default=55)
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--validation-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end")
    parser.add_argument(
        "--colproj-index-manifest",
        type=Path,
        default=Path("data/processed/colproj_stage10.json"),
    )
    parser.add_argument(
        "--cnn-index-manifest",
        type=Path,
        default=Path("data/processed/cnn_stage12.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("research/stage15/stock_pool.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("research/stage15/stock_pool_manifest.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("research/stage15/README.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rules = StockPoolRules(
        build_date=args.build_date,
        pool_size=args.pool_size,
        lookback_sessions=args.lookback_sessions,
        min_history_sessions=args.min_history_sessions,
        min_history_calendar_days=args.min_history_calendar_days,
        min_recent_sessions=args.min_recent_sessions,
        min_liquid_sessions=args.min_liquid_sessions,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    try:
        result = select_stock_pool(
            args.database,
            rules,
            index_manifests={
                "colproj": args.colproj_index_manifest,
                "cnn": args.cnn_index_manifest,
            },
        )
        write_stock_pool_outputs(
            result,
            csv_path=args.output_csv,
            manifest_path=args.output_manifest,
            report_path=args.output_report,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"阶段十五股票池生成失败: {exc}")
        return 2

    print(f"build_date: {rules.build_date}")
    print(f"eligible_symbols: {result['manifest']['eligible_symbol_count']}")
    print(f"selected_symbols: {len(result['rows'])}")
    print(f"csv: {args.output_csv}")
    print(f"manifest: {args.output_manifest}")
    print(f"report: {args.output_report}")
    return 0


def select_stock_pool(
    database_path: str | Path,
    rules: StockPoolRules,
    *,
    index_manifests: dict[str, str | Path] | None = None,
) -> dict[str, object]:
    """Select liquid stocks using only records available on the build date."""

    path = Path(database_path)
    if not path.is_file():
        raise OSError(f"行情数据库不存在: {path}")

    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        market_dates = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT date
                FROM daily_bars
                WHERE date <= ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (rules.build_date, rules.lookback_sessions),
            )
        ]
        if len(market_dates) != rules.lookback_sessions:
            raise ValueError(
                f"建池日前只有 {len(market_dates)} 个市场交易日，"
                f"不足 {rules.lookback_sessions} 个"
            )
        if market_dates[0] != rules.build_date:
            raise ValueError(f"建池日期不是数据库交易日: {rules.build_date}")

        recent_start = market_dates[-1]
        candidates = _load_candidates(connection, rules, recent_start)
        if len(candidates) < rules.pool_size:
            raise ValueError(
                f"符合条件的股票只有 {len(candidates)} 只，少于目标 {rules.pool_size} 只"
            )

        metadata = dict(connection.execute("SELECT key, value FROM dataset_meta"))
        database_end_date = str(
            connection.execute("SELECT MAX(date) FROM daily_bars").fetchone()[0]
        )

    strata: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        stratum = str(candidate["stratum"])
        strata.setdefault(stratum, []).append(candidate)
    for rows in strata.values():
        rows.sort(
            key=lambda item: (
                -float(item["avg_amount_60d"]),
                -int(item["history_sessions"]),
                str(item["symbol"]),
            )
        )

    quotas = allocate_proportional_quotas(
        {stratum: len(rows) for stratum, rows in strata.items()},
        rules.pool_size,
    )
    selected: list[dict[str, object]] = []
    for stratum in sorted(strata):
        quota = quotas[stratum]
        for stratum_rank, item in enumerate(strata[stratum][:quota], start=1):
            selected.append(
                {
                    **item,
                    "stratum_rank": stratum_rank,
                    "stratum_eligible_count": len(strata[stratum]),
                    "stratum_quota": quota,
                }
            )

    selected.sort(
        key=lambda item: (
            -float(item["avg_amount_60d"]),
            str(item["symbol"]),
        )
    )
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        item["avg_amount_60d_cny_100m"] = round(
            float(item["avg_amount_60d"]) / 100_000_000,
            4,
        )
        item["selection_reason"] = (
            f"截至{rules.build_date}可观测历史跨度{item['history_calendar_days']}个自然日、"
            f"有{item['history_sessions']}个交易日历史；"
            f"近{rules.lookback_sessions}个市场交易日有{item['liquid_sessions']}日有效成交；"
            f"在{item['stratum']}层内按60日平均成交金额排名第{item['stratum_rank']}"
        )

    index_coverage = {}
    for model, manifest_path in (index_manifests or {}).items():
        coverage = inspect_index_coverage(manifest_path, [str(item["symbol"]) for item in selected])
        if coverage["missing_symbols"]:
            raise ValueError(
                f"{model} 索引缺少股票: {', '.join(coverage['missing_symbols'])}"
            )
        index_coverage[model] = coverage

    selection_identity = {
        "build_date": rules.build_date,
        "lookback_sessions": rules.lookback_sessions,
        "min_history_sessions": rules.min_history_sessions,
        "min_history_calendar_days": rules.min_history_calendar_days,
        "min_recent_sessions": rules.min_recent_sessions,
        "min_liquid_sessions": rules.min_liquid_sessions,
        "source_hash": metadata.get("source_hash"),
        "selected_symbols": [str(item["symbol"]) for item in selected],
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(selection_identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "stage15-stock-pool-v1",
        "database": str(path),
        "source_hash": metadata.get("source_hash"),
        "price_basis": metadata.get("price_basis"),
        "database_end_date": database_end_date,
        "build_date": rules.build_date,
        "selection_data_end_date": rules.build_date,
        "lookback_start_date": recent_start,
        "lookback_end_date": rules.build_date,
        "lookback_sessions": rules.lookback_sessions,
        "pool_size": rules.pool_size,
        "eligible_symbol_count": len(candidates),
        "eligibility_rules": {
            "min_history_sessions": rules.min_history_sessions,
            "min_history_calendar_days": rules.min_history_calendar_days,
            "min_recent_sessions": rules.min_recent_sessions,
            "min_liquid_sessions": rules.min_liquid_sessions,
            "last_observation_must_equal_build_date": True,
            "amount_must_be_positive_for_liquid_session": True,
        },
        "ranking_rule": "descending_60_session_average_trading_amount_within_stratum",
        "stratification_rule": "proportional allocation by exchange and board using largest remainder",
        "stratum_eligible_counts": {
            key: len(strata[key]) for key in sorted(strata)
        },
        "stratum_quotas": {key: quotas[key] for key in sorted(quotas)},
        "validation_period": {
            "start": rules.validation_start,
            "end": rules.validation_end,
        },
        "test_period": {
            "start": rules.test_start,
            "end": rules.test_end or database_end_date,
        },
        "name_metadata": {
            "source": metadata.get("instrument_names_source"),
            "updated_at_utc": metadata.get("instrument_names_updated_at_utc"),
            "used_for_selection": False,
        },
        "index_coverage": index_coverage,
        "dataset_scope_claim": "user_supplied_dataset_claims_st_and_finance_excluded",
        "selection_sha256": selection_sha256,
        "limitations": [
            "CSMAR daily table has no point-in-time industry field, so the first pool uses exchange-board stratification instead of industry caps.",
            "The non-ST and non-finance scope is inherited from the supplied dataset claim and cannot be independently verified from the daily table.",
            "Instrument names are current display metadata and are not used by the historical selection rule.",
            "This is a research sample pool, not a list of investment recommendations.",
        ],
        "selected_symbols": [str(item["symbol"]) for item in selected],
    }
    return {"rows": selected, "manifest": manifest}


def inspect_index_coverage(
    manifest_path: str | Path,
    symbols: Sequence[str],
) -> dict[str, object]:
    """Verify that every selected stock is available in a published index."""

    path = Path(manifest_path)
    if not path.is_file():
        raise OSError(f"索引清单不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    index_paths = payload.get("indexes")
    if not isinstance(index_paths, list) or not index_paths:
        raise ValueError(f"索引清单没有有效分片: {path}")

    counts: dict[str, int] = {}
    placeholders = ",".join("?" for _ in symbols)
    for item in index_paths:
        shard = path.parent / str(item)
        if not shard.is_file():
            raise OSError(f"索引分片不存在: {shard}")
        uri = f"file:{shard.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=30) as connection:
            version = connection.execute(
                """
                SELECT version_id
                FROM feature_versions
                WHERE status = 'ready'
                ORDER BY updated_at_utc DESC
                LIMIT 1
                """
            ).fetchone()
            if version is None:
                continue
            rows = connection.execute(
                f"""
                SELECT symbol, COUNT(*)
                FROM samples
                WHERE version_id = ? AND symbol IN ({placeholders})
                GROUP BY symbol
                """,
                (version[0], *symbols),
            ).fetchall()
            for symbol, sample_count in rows:
                counts[str(symbol)] = counts.get(str(symbol), 0) + int(sample_count)

    missing = sorted(set(symbols) - counts.keys())
    return {
        "manifest": str(path),
        "covered_symbol_count": len(counts),
        "missing_symbols": missing,
        "minimum_sample_count": min(counts.values()) if counts else 0,
        "maximum_sample_count": max(counts.values()) if counts else 0,
    }


def _load_candidates(
    connection: sqlite3.Connection,
    rules: StockPoolRules,
    recent_start: str,
) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        WITH history AS (
            SELECT
                symbol,
                COUNT(*) AS history_sessions,
                MIN(date) AS first_date,
                MAX(date) AS last_date
            FROM daily_bars
            WHERE date <= ?
            GROUP BY symbol
        ),
        recent AS (
            SELECT
                symbol,
                COUNT(*) AS recent_sessions,
                SUM(CASE WHEN amount > 0 THEN 1 ELSE 0 END) AS liquid_sessions,
                AVG(CASE WHEN amount > 0 THEN amount END) AS avg_amount_60d
            FROM daily_bars
            WHERE date BETWEEN ? AND ?
            GROUP BY symbol
        )
        SELECT
            h.symbol,
            i.name,
            i.exchange,
            i.board,
            h.first_date,
            h.last_date,
            h.history_sessions,
            CAST(julianday(?) - julianday(h.first_date) AS INTEGER) AS history_calendar_days,
            r.recent_sessions,
            r.liquid_sessions,
            r.avg_amount_60d
        FROM history h
        JOIN recent r ON r.symbol = h.symbol
        JOIN instruments i ON i.symbol = h.symbol
        WHERE h.history_sessions >= ?
          AND julianday(?) - julianday(h.first_date) >= ?
          AND h.last_date = ?
          AND r.recent_sessions >= ?
          AND r.liquid_sessions >= ?
          AND r.avg_amount_60d > 0
        """,
        (
            rules.build_date,
            recent_start,
            rules.build_date,
            rules.build_date,
            rules.min_history_sessions,
            rules.build_date,
            rules.min_history_calendar_days,
            rules.build_date,
            rules.min_recent_sessions,
            rules.min_liquid_sessions,
        ),
    ).fetchall()
    return [
        {
            "symbol": str(row[0]),
            "name": str(row[1]),
            "exchange": str(row[2]),
            "board": str(row[3]),
            "stratum": f"{row[2]}_{row[3]}",
            "first_date": str(row[4]),
            "last_date": str(row[5]),
            "history_sessions": int(row[6]),
            "history_calendar_days": int(row[7]),
            "recent_sessions": int(row[8]),
            "liquid_sessions": int(row[9]),
            "avg_amount_60d": float(row[10]),
        }
        for row in rows
    ]


def allocate_proportional_quotas(
    eligible_counts: dict[str, int],
    pool_size: int,
) -> dict[str, int]:
    """Allocate integer quotas using proportional largest remainders."""

    positive = {key: count for key, count in eligible_counts.items() if count > 0}
    total = sum(positive.values())
    if pool_size <= 0 or pool_size > total:
        raise ValueError("pool_size must be positive and no larger than eligible symbols")
    if pool_size < len(positive):
        raise ValueError("pool_size is too small to represent every non-empty stratum")

    exact = {key: pool_size * count / total for key, count in positive.items()}
    quotas = {key: max(1, math.floor(value)) for key, value in exact.items()}

    while sum(quotas.values()) < pool_size:
        candidates = [key for key in quotas if quotas[key] < positive[key]]
        key = max(
            candidates,
            key=lambda item: (exact[item] - quotas[item], positive[item], item),
        )
        quotas[key] += 1

    while sum(quotas.values()) > pool_size:
        candidates = [key for key in quotas if quotas[key] > 1]
        key = max(
            candidates,
            key=lambda item: (quotas[item] - exact[item], -positive[item], item),
        )
        quotas[key] -= 1

    return quotas


def write_stock_pool_outputs(
    result: dict[str, object],
    *,
    csv_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
) -> None:
    rows = list(result["rows"])
    manifest = dict(result["manifest"])
    csv_output = Path(csv_path)
    manifest_output = Path(manifest_path)
    report_output = Path(report_path)
    for path in (csv_output, manifest_output, report_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    fields = (
        "rank",
        "symbol",
        "name",
        "exchange",
        "board",
        "stratum",
        "first_date",
        "last_date",
        "history_sessions",
        "history_calendar_days",
        "recent_sessions",
        "liquid_sessions",
        "avg_amount_60d",
        "avg_amount_60d_cny_100m",
        "stratum_rank",
        "stratum_eligible_count",
        "stratum_quota",
        "selection_reason",
    )
    with csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(_render_report(rows, manifest), encoding="utf-8")


def _render_report(rows: list[dict[str, object]], manifest: dict[str, object]) -> str:
    quotas = manifest["stratum_quotas"]
    lines = [
        "# 阶段十五：50只股票研究池",
        "",
        "## 建池结论",
        "",
        f"- 建池日期：`{manifest['build_date']}`。",
        f"- 数据选择截止日：`{manifest['selection_data_end_date']}`。",
        f"- 符合基础条件：{manifest['eligible_symbol_count']}只；最终选择：{manifest['pool_size']}只。",
        f"- 流动性观察窗口：`{manifest['lookback_start_date']}` 至 `{manifest['lookback_end_date']}`，共{manifest['lookback_sessions']}个市场交易日。",
        "- 排名方法：在交易所—板块分层内，按近60个交易日平均成交金额由高到低选择。",
        "- 时间划分：2024年用于阈值验证；2025-01-01至数据库截止日用于冻结参数后的测试。",
        "- 双模型覆盖：所选50只股票均已通过现有ColProj和CNN全市场索引覆盖检查。",
        f"- 股票池版本哈希：`{manifest['selection_sha256']}`。",
        "- 这是一组策略研究样本，不是投资推荐名单。",
        "",
        "## 选择依据",
        "",
        "1. 数据库首条行情距建池日至少730个自然日，并至少有400个交易日历史，保证至少两年可观测行情。",
        "2. 最近60个市场交易日至少有55日行情且成交金额为正，降低停牌和缺失数据影响。",
        "3. 建池日当天必须有行情，避免把已经停止交易的证券加入股票池。",
        "4. 按沪市主板、深市主板、创业板、科创板和北交所的合格股票数量比例分配名额。",
        "5. 每个分层内部按60日平均成交金额排序，优先选择更容易按真实价格成交的股票。",
        "6. 股票名称只用于显示，不参与历史筛选，避免使用当前名称反向影响历史选择。",
        "",
        "## 分层名额",
        "",
        "| 分层 | 合格股票数 | 选择名额 |",
        "| --- | ---: | ---: |",
    ]
    eligible_counts = manifest["stratum_eligible_counts"]
    for stratum in sorted(quotas):
        lines.append(f"| {stratum} | {eligible_counts[stratum]} | {quotas[stratum]} |")

    lines.extend(
        [
            "",
            "## 股票清单",
            "",
            "| 排名 | 股票 | 交易所/板块 | 可观测历史跨度 | 历史交易日 | 近60日有效成交 | 60日平均成交金额（亿元） | 层内排名 |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in rows:
        lines.append(
            "| {rank} | {symbol}·{name} | {stratum} | {history_calendar_days} | {history_sessions} | "
            "{liquid_sessions} | {avg_amount_60d_cny_100m:.2f} | "
            "{stratum_rank}/{stratum_eligible_count} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## 当前限制",
            "",
            "- CSMAR日行情表没有历史行业字段，因此首版使用交易所—板块分层，尚未实现每个行业最多5只的约束。",
            "- 日行情表没有正式上市日期；730日条件按数据库内首条行情计算，表示至少两年可观测历史，不等同于证券主表上市日期。",
            "- 非ST、非金融行业范围继承自用户提供数据集的说明，无法仅凭当前日行情表独立复核。",
            "- 股票名称来自后续同步的当前名称，只用于显示；正式论文或报告若要求历史行业均衡，需要补充建池日可获得的行业分类文件后重新生成。",
            "- 后续预测和回测必须继续使用本清单，不得根据2024年以后收益表现替换股票。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
