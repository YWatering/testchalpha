# A 股本地行情数据

## 来源与模式

`kline-download-a-share` 默认使用 AKShare `stock_zh_a_hist`（Eastmoney）获取 A 股历史日线；当该上游不可用时，可显式选择 AKShare `stock_zh_a_daily`（Sina）。两种端点都记录在 manifest 中，不会在失败时静默切换数据源。数据下载由可替换的 `MarketDataProvider` 协议隔离，AKShare 只在创建 provider 时导入，不影响 CSV 检索、API 和基础测试。

支持 `qfq`、`hfq` 和 `raw`（传给 AKShare 的空调整参数）。一份输出快照只允许使用一种口径。复权历史的 OHLC 需通过当前正价与 OHLC 关系校验；若源数据不满足约束，下载会失败并保留原有文件，不会对负价格做偏移或静默丢弃。

## 下载

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[web,a-share]'
.venv/bin/kline-download-a-share \
  --symbols 000001,600519,300750 \
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

- `--symbols`：逗号分隔的六位证券代码；默认 `000001,600519,300750`。
- `--start-date` / `--end-date`：行情范围，日期为 `YYYY-MM-DD`。
- `--adjust`：`qfq`、`hfq` 或 `raw`。
- `--endpoint`：`eastmoney`（默认）或 `sina`；失败不会自动换源，避免数据来源不透明。
- `--output`：标准化 CSV 路径，默认 `data/processed/a_share_daily.csv`。
- `--full-refresh`：丢弃旧快照并从指定起始日重建。
- `--request-delay` / `--retries`：控制请求间隔与有限重试。

首次下载建议少量股票、逐个请求。Sina 源逐只返回个股全历史后在本地裁剪请求区间，耗时和流量会高于 Eastmoney。AKShare 上游接口受网络和上游服务变化影响，失败时保留原数据；请勿使用高并发绕过访问限制。

## 增量和可追溯性

增量更新读取相邻 `*.manifest.json`，要求数据源、复权方式和 SHA-256 与已有快照匹配。对每只股票从最新交易日之前的重叠区间重新抓取，更新相同 `(symbol, date)` 记录，再追加新日期。扩大历史区间会自动回补更早日期。来源请求或数据校验失败不会替换原始快照。

manifest 记录 provider、provider 版本、复权方式、股票池、请求与数据日期范围、行数、抓取时间和数据 SHA-256。下载数据存放于 `data/processed/`，该目录在 `.gitignore` 中，不提交行情副本。

## 启动

```bash
KLINE_DATA_PATH=data/processed/a_share_daily.csv \
  .venv/bin/python -m kline_retrieval.server
```

数据文件在服务启动时加载；下载更新后需要重启服务。页面顶部的数据标记展示来源与复权方式，提示框展示行情覆盖范围和哈希。阶段六已用 Sina 端点的真实 A 股快照完成浏览器检索演示；阶段七仍需完成完整真实行情回归和桌面/窄屏验收。当前快照详情及测量结果见 `validation-stage-7.md`。
