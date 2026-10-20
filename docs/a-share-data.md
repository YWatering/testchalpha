# A 股本地行情数据

## 来源与模式

`kline-download-a-share` 默认使用 AKShare `stock_zh_a_hist`（Eastmoney）获取 A 股历史日线；当该上游不可用时，可显式选择 AKShare `stock_zh_a_daily`（Sina）。两种端点都记录在 manifest 中，不会在失败时静默切换数据源。数据下载由可替换的 `MarketDataProvider` 协议隔离，AKShare 只在创建 provider 时导入，不影响 CSV 检索、API 和基础测试。

支持 `qfq`、`hfq` 和 `raw`（传给 AKShare 的空调整参数）。一份输出快照只允许使用一种口径。复权历史的 OHLC 需通过当前正价与 OHLC 关系校验；若源数据不满足约束，下载会失败并保留原有文件，不会对负价格做偏移或静默丢弃。

## 下载

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[web,a-share]'
.venv/bin/kline-download-a-share \
  --symbols 000001,600519,300750,000333,002594,300059,600276,600900,601318 \
  --start-date 2018-01-01 \
  --adjust qfq \
  --endpoint sina
```

当前环境使用 macOS 系统代理时，Sina 直连命令需显式将 Sina 域名加入代理绕过名单：

```bash
NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn \
  .venv/bin/kline-download-a-share --endpoint sina
```

参数：

- `--symbols`：逗号分隔的六位证券代码；默认 9 只股票：平安银行、贵州茅台、宁德时代、美的集团、比亚迪、东方财富、恒瑞医药、长江电力、中国平安，覆盖多个行业。
- `--start-date` / `--end-date`：行情范围，日期为 `YYYY-MM-DD`。
- `--adjust`：`qfq`、`hfq` 或 `raw`。
- `--endpoint`：`eastmoney`（默认）或 `sina`；失败不会自动换源，避免数据来源不透明。
- `--output`：标准化 CSV 路径，默认 `data/processed/a_share_daily.csv`。
- `--database`：SQLite 数据库路径，默认 `data/processed/a_share_daily.sqlite3`。
- `--all-listed`：自动发现当前上市的沪深京 A 股。
- `--offset` / `--limit`：分批同步全市场，降低上游限流和长任务中断风险。
- `--full-refresh`：丢弃旧快照并从指定起始日重建。
- `--request-delay` / `--retries`：控制请求间隔与有限重试。

首次下载建议少量股票、逐个请求。Sina 源逐只返回个股全历史后在本地裁剪请求区间，耗时和流量会高于 Eastmoney。AKShare 上游接口受网络和上游服务变化影响，失败时保留原数据；请勿使用高并发绕过访问限制。

## 增量和可追溯性

增量更新读取相邻 `*.manifest.json`，要求数据源、复权方式和 SHA-256 与已有快照匹配。对每只股票从最新交易日之前的重叠区间重新抓取，更新相同 `(symbol, date)` 记录，再追加新日期。扩大历史区间会自动回补更早日期。来源请求或数据校验失败不会替换原始快照。

manifest 记录 provider、provider 版本、复权方式、股票池、请求与数据日期范围、行数、失败股票、抓取时间和数据 SHA-256。下载数据存放于 `data/processed/`，该目录在 `.gitignore` 中，不提交行情副本。

## SQLite 和自动刷新

将已有快照导入 SQLite：

```bash
.venv/bin/kline-build-database \
  --input data/processed/a_share_daily.csv \
  --database data/processed/a_share_daily.sqlite3
```

刷新一次或持续刷新：

```bash
NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn \
  .venv/bin/kline-refresh-a-share --endpoint sina

NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn \
  .venv/bin/kline-refresh-a-share --endpoint sina \
  --watch --interval-seconds 3600
```

SQLite 使用 WAL 和事务提交。同步中的暂存数据与已完成股票断点保存在库内；进程中断后，相同参数再次运行会继续未完成股票。单股失败在 `--all-listed` 模式下被隔离并记录，上一份完整快照在新事务提交前持续可读。

当前上市股票池由 AKShare 沪深京列表生成。2026-10-20 本机验证发现 5,575 只当前上市股票。全量历史下载耗时长且可能触发上游限流，建议用 `--offset` 和 `--limit` 分批执行。数据库可以保存全市场，但在阶段 10 的向量索引完成前，网页建议通过 `KLINE_ACTIVE_SYMBOLS=000001,600519,...` 只加载研究股票池。

## 启动

```bash
KLINE_DATA_PATH=data/processed/a_share_daily.sqlite3 \
  KLINE_ACTIVE_SYMBOLS=000001,600519,300750 \
  .venv/bin/uvicorn kline_retrieval.api:app --host 127.0.0.1 --port 8001
```

API 会检测 SQLite 快照版本，事务提交后自动重载数据并清空旧特征缓存，无需重启 Web 服务。页面顶部展示来源、复权方式和 SQLite 自动刷新状态，浏览器每分钟检查一次版本。当前只实现日线自动更新，不提供盘中逐笔行情或 WebSocket 推送。
