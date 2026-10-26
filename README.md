# testchalpha

## 当前阶段

阶段 0 至阶段 9 已完成。阶段 10-A/B/C 已完成：P1 股票下拉框现在直接加载本地数据库中的全部 5,428 只股票代码和名称，页面只通过下拉窗口选择股票，并显示数据状态；搜索股票、交易所、板块和自选股控件已移除；28 个 ColProj 分片已生成，覆盖 5,164 只具备有效样本的股票和 6,862,102 个样本。阶段 10-D 的全量检索性能与随机抽样验收仍在进行，CNN 原型继续安排在阶段 11。

阶段 0 交付内容：

- 默认配置模块和环境变量配置。
- 统一日志配置。
- Python 包安装配置和命令行启动入口。
- 前端、后端和研究核心的架构边界文档。

阶段 1 交付内容：

数据层位于 `src/kline_retrieval/market_data.py`，提供：

- CSV 加载和数据版本哈希。
- 可选 Parquet 加载适配。
- 必要字段、OHLC 关系、重复记录和数值合法性校验。
- 按股票和交易日查询历史窗口与未来窗口。
- 样本历史数据和未来数据完整性判断。

阶段 2 交付内容：

- 月线、周线和日线 OHLC 聚合。
- 未来20个交易日对照图生成。
- 黑白 `64×64` 图像渲染。
- PNG/PGM 序列化和样本图批量保存。

阶段 3 交付内容：

- 64 维 ColProj 列投影特征。
- 月线、周线和日线加权余弦距离。
- 历史候选样本过滤和 Top-K 排名。
- 单尺度距离和综合距离结果。
- 查询样本不依赖未来图，候选未来数据不进入特征计算。

阶段 4 交付内容：

- 未来 1/5/10/20 个交易日收益计算。
- 相似样本未来20日图关联。
- 按相似度距离加权的平均收益、加权标准差、ESS、标准误和稳定性区间。
- 平均值、中位数、胜率和分位数统计。
- 收益直方图和 Gaussian KDE 数据。
- 可直接转换为 JSON 的分析结果结构。

阶段 5 交付内容：

- FastAPI 健康检查、股票、日期和检索接口。
- 静态浏览器检索页面。
- 股票、交易日、Top-K 和尺度权重控制。
- 查询三尺度图、历史样本图和未来20日图展示。
- 收益周期切换、统计卡片、直方图和 KDE 曲线。
- 加载、错误、无结果和数据状态提示。

阶段 8 验证基础与当前真实行情结果：

- API 进程内使用容量受限的 LRU 缓存复用样本向量，缓存与数据集实例隔离。
- 每次检索响应提供候选样本数、向量缓存命中/未命中、特征构建耗时、排序耗时和总耗时。
- 页面展示本次检索候选规模、总耗时和缓存命中数。
- 增加未来行情变更不影响查询向量的防泄漏测试，以及缓存前后排序一致性测试。
- 增加冷启动/热缓存性能基线命令：`python3 scripts/benchmark_retrieval.py`。
- 已导入本地 CSMAR 日线快照：5,428 只股票、9,074,433 条日线，日期范围为 2018-01-02 至 2026-10-23；质量检查发现 0 条无效记录、0 条重复主键记录。详细结果见 [阶段八验证报告](docs/validation-stage-8.md)。
- 前200只股票已生成 ColProj 离线索引，共321,571个有效样本；离线检索验证见 [阶段九验证报告](docs/validation-stage-9.md)。

## 下载真实 A 股数据

AKShare 下载器是可选功能，需要 Python 3.11 或更高版本。核心检索仍可在 Python 3.10 运行。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[web,a-share]'
NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn .venv/bin/kline-download-a-share \
  --endpoint sina \
  --symbols 000001,600519,300750,000333,002594,300059,600276,600900,601318 \
  --start-date 2018-01-01 \
  --adjust qfq
```

若本机代理拦截 Sina 请求，可在 macOS/Linux 对该命令显式直连 Sina 域名：`NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn`。Eastmoney 是默认端点；如果本地代理允许访问，也可省略 `--endpoint sina` 使用默认 Eastmoney。

默认股票池为 9 只多行业 A 股。下载命令同时生成标准 CSV、manifest 和 `data/processed/a_share_daily.sqlite3`。SQLite 使用 WAL 和事务快照；默认执行增量更新并重抓末尾若干天，完整重建加 `--full-refresh`。真实行情目录已加入 `.gitignore`，不会随项目推送。

持续自动刷新日线数据库：

```bash
NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn .venv/bin/kline-refresh-a-share \
  --endpoint sina --watch --interval-seconds 3600
```

自动发现当前上市沪深京 A 股并分批建立全市场数据库：

```bash
NO_PROXY=.sina.com.cn,.sse.com.cn,.szse.cn,.bse.cn \
no_proxy=.sina.com.cn,.sse.com.cn,.szse.cn,.bse.cn \
  .venv/bin/kline-refresh-a-share --endpoint sina --all-listed \
  --offset 0 --limit 200
```

使用 `--offset 200 --limit 200` 继续下一批。成功股票会保存断点，单股失败会记录到 manifest，不会让整个批次失败。全市场数据库可通过 `KLINE_ACTIVE_SYMBOLS` 限制网页当前加载的检索股票池，避免阶段 9 建立向量索引前进行超大规模内存扫描。

## 构建和使用 ColProj 离线索引

阶段九使用 SQLite 保存 `feature_versions`、`samples` 和 `return_labels`。以下命令为前200只股票生成索引，重复执行时会跳过已有 `(股票,日期)` 样本；数据快照更新后继续执行即可补充新增有效样本。

```bash
.venv/bin/kline-build-colproj-index \
  --database data/processed/csmar_a_share_daily.sqlite3 \
  --index data/processed/colproj_stage9_200.sqlite3 \
  --limit 200 \
  --replace
```

使用索引启动网站：

```bash
KLINE_DATA_PATH=data/processed/csmar_a_share_daily.sqlite3 \
KLINE_INDEX_PATH=data/processed/colproj_stage9_200.sqlite3 \
KLINE_ACTIVE_SYMBOLS=000002,000004 \
  .venv/bin/uvicorn kline_retrieval.api:app --host 127.0.0.1 --port 8001
```

启用索引后，`/api/retrieval/config` 会返回 `index_mode=offline_colproj`，检索响应的 `performance.search_mode` 也会标记为 `offline_colproj`。索引检索仍使用精确三尺度距离和可调月线、周线权重。

## 阶段十调整：P1 直接加载全量股票

当前阶段十试做版中的 P2 搜索框、交易所和板块控件不属于最终页面。最终页面启动后，P1 的“股票”下拉框必须直接加载完整研究股票池，并显示类似 `000333·美的集团` 的代码—名称组合；用户只通过这个下拉窗口选择股票。

当前 CSMAR 研究数据库包含5,428只股票。生产模式不能通过 `KLINE_ACTIVE_SYMBOLS` 限制为前200只；该变量只能用于开发调试。阶段十只有在 P1 加载数量与数据库证券主表一致，并且全部具备有效样本的股票完成索引后，才可以重新标记为完成。

当前实现进度（2026-10-26）：

- `/api/symbols?limit=20000` 已返回 5,428/5,428 条股票元数据，前端 P1 原生下拉框直接渲染代码—名称组合。
- 选择不在当前进程内存中的股票时，日期接口会从 SQLite 按需读取该股票行情；当前数据库最新交易日为 2026-10-23。
- 5,164 只股票具备完整历史和未来窗口，264 只仍显示在下拉框中并标记为“数据不足”。
- 当前已发布的阶段十分片覆盖全部 28 个批次、5,164 只具备有效样本的股票和 6,862,102 个样本；另外 264 只股票因历史窗口或未来窗口不足而没有可建立的样本。
- 阶段十-C 的全市场 ColProj 分片构建已完成；阶段十-D 仍需完成随机抽样检索和全量候选扫描性能验收，因此本阶段暂不标记为最终完成。

## 阶段十试做版：全市场批次和分片索引

现有批次和分片命令保留，作为阶段十重构后的底层实现。为避免一次性构建数 GB 级全市场索引，先准备批次状态：

```bash
.venv/bin/kline-plan-a-share-batches \
  --database data/processed/csmar_a_share_daily.sqlite3 \
  --batch-size 200
```

每次执行下面的命令会自动选择下一个待处理批次，构建独立 ColProj 分片，并将已经完成的分片写入原子索引清单：

```bash
.venv/bin/kline-build-next-index-batch \
  --database data/processed/csmar_a_share_daily.sqlite3 \
  --index-dir data/processed/colproj_stage10 \
  --manifest data/processed/colproj_stage10.json \
  --batch-size 200
```

使用已发布的分片清单启动网站：

```bash
KLINE_DATA_PATH=data/processed/csmar_a_share_daily.sqlite3 \
KLINE_INDEX_PATH=data/processed/colproj_stage10.json \
  .venv/bin/uvicorn kline_retrieval.api:app --host 127.0.0.1 --port 8001
```

阶段十试做版新增接口（股票搜索和筛选接口不作为最终页面入口）：

- `GET /api/health`：返回股票覆盖率、有效样本数、最新交易日、失败数、批次状态和当前索引版本。
- `GET /api/symbols?q=美的&exchange=SZSE&board=MAIN`：按代码/名称、交易所、板块和分页查询股票。
- `GET /api/valid-dates?symbol=000333`：只返回满足400日历史和20日未来窗口的可检索日期。
- `/api/symbols?watchlist=000333,600519`：只返回指定自选股票池。

当前真实 CSMAR 快照状态为5,428只股票、5,164只具备完整可检索窗口、6,862,102个有效样本、最新交易日2026-10-23。数据库和索引分片仍属于本地运行数据，不推送到 GitHub。

用下载的数据启动网页：

```bash
KLINE_DATA_PATH=data/processed/a_share_daily.sqlite3 \
  .venv/bin/uvicorn kline_retrieval.api:app --host 127.0.0.1 --port 8001
```

访问 `http://127.0.0.1:8001`。数据库提交新快照后 API 自动重载并清空旧向量缓存，浏览器每分钟检查数据版本。这里的“自动刷新”是交易日行情更新，不是逐笔或毫秒级盘中实时行情。默认 Eastmoney 端点仍受当前机器代理网络影响，详见 [数据接入说明](docs/a-share-data.md)。

## 数据格式

行情文件至少需要以下字段：

```text
symbol,date,open,high,low,close,volume
```

其中 `volume` 可选，日期建议使用 `YYYY-MM-DD` 格式。示例文件位于 `data/sample_market_data.csv`。

## 导入本地 CSMAR 数据

当前仓库支持将 CSMAR `TRD_Dalyr` CSV 分片流式导入 SQLite。原始数据目录和生成的数据库均在 `.gitignore` 中，不会推送到 GitHub。

```bash
.venv/bin/kline-import-csmar \
  --input-dir '日个股回报率文件（1）' \
  --input-dir '日个股回报率文件（2）' \
  --database data/processed/csmar_a_share_daily.sqlite3 \
  --quality-report data/processed/csmar_a_share_daily.quality.json
```

导入器会校验证券代码、日期、OHLC、成交量、成交金额、昨收盘和日收益率，并在数据库中保留 `volume`、`amount`、`return_rate`、`pre_close` 字段。由于当前 CSMAR 日行情表没有证券名称和行业字段，系统只能记录“由用户提供的数据集声明已剔除 ST 和金融业”，不能仅凭这张日行情表独立复核该筛选条件。

### 同步股票代码和名称

CSMAR 日行情表不包含证券简称，导入后可执行下面的命令，从中国信息披露网证券列表补齐当前数据库中的代码—名称映射：

```bash
.venv/bin/kline-sync-instrument-names \
  --database data/processed/csmar_a_share_daily.sqlite3
```

同步结果写入 SQLite 的 `instruments` 表和 `dataset_meta` 元数据。网页的股票下拉框会显示为 `000333·美的集团` 这样的格式；名称列表未返回的代码会保留代码本身作为兜底，不会影响行情检索。

## 使用示例

```python
from kline_retrieval import MarketData

market_data = MarketData.from_csv("data/sample_market_data.csv")
print(market_data.symbols)
print(market_data.dates("DEMO"))

validation = market_data.validate_sample(
    "DEMO",
    "2024-01-10",
    history_count=5,
    future_count=2,
)
print(validation.valid, validation.reasons)
```

## 运行项目测试

安装当前本地包：

```bash
python3 -m pip install -e .
```

安装 Web 依赖：

```bash
python3 -m pip install -e '.[web]'
```

当前测试使用 Python 标准库，不需要安装 pytest：

```bash
python3 -m unittest discover -s tests -v
```

## 生成多尺度 K 线图

图像模块提供以下接口：

```python
from kline_retrieval import MarketData, build_sample_charts, save_sample_charts

market_data = MarketData.from_csv("data/market_data.csv")
charts = build_sample_charts(market_data, "AAPL", "2024-01-31")
paths = save_sample_charts(charts, "data/cache", prefix="AAPL_2024-01-31")
```

查询使用的月线、周线和日线图会参与后续特征提取；`future` 只用于事后对照，不会进入检索特征。默认图像为64×64，月线、周线、日线均生成20根柱。

## 启动数据检查命令

安装本地包后，可以直接加载并检查样例数据：

```bash
python3 -m kline_retrieval --data data/sample_market_data.csv
python3 -m kline_retrieval \
  --data data/sample_market_data.csv \
  --symbol DEMO \
  --date 2024-01-10 \
  --history-count 5 \
  --future-count 2
```

完整的技术边界和错误处理约定见 `docs/architecture.md`。

阶段三的 ColProj 和检索规则见 `docs/retrieval.md`。

阶段四的收益公式和统计输出见 `docs/analytics.md`。

运行性能基线脚本会在 JSON 输出中记录数据哈希、候选数量、缓存统计和分段耗时；耗时会随设备和数据规模变化，不作为固定阈值。

## 启动检索网站

```bash
python3 -m kline_retrieval.server
```

启动后访问 `http://127.0.0.1:8000`。默认使用 `data/demo_market_data.csv`；可通过 `KLINE_DATA_PATH` 指向自己的 CSV 或 Parquet 行情文件。
