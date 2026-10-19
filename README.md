# testchalpha

## 当前阶段

阶段 0 至阶段 6 已完成：项目基线、行情数据、多尺度图像、ColProj 检索、收益分析、检索网站，以及 AKShare A 股历史数据接入。阶段 7 正在进行真实行情回归与性能复核；阶段 8 起再接入 CNN、规模化索引和回测。

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
- 未来 12 个交易日对照图生成。
- 黑白 `36×36` 图像渲染。
- PNG/PGM 序列化和样本图批量保存。

阶段 3 交付内容：

- 36 维 ColProj 列投影特征。
- 月线、周线和日线加权余弦距离。
- 历史候选样本过滤和 Top-K 排名。
- 单尺度距离和综合距离结果。
- 查询样本不依赖未来图，候选未来数据不进入特征计算。

阶段 4 交付内容：

- 未来 1/5/10/12/21 个交易日收益计算。
- 相似样本未来 12 日图关联。
- 平均值、中位数、胜率和分位数统计。
- 收益直方图和 Gaussian KDE 数据。
- 可直接转换为 JSON 的分析结果结构。

阶段 5 交付内容：

- FastAPI 健康检查、股票、日期和检索接口。
- 静态浏览器检索页面。
- 股票、交易日、Top-K 和尺度权重控制。
- 查询三尺度图、历史样本图和未来 12 日图展示。
- 收益周期切换、统计卡片、直方图和 KDE 曲线。
- 加载、错误、无结果和数据状态提示。

阶段 7 验证基础与当前真实行情结果：

- API 进程内使用容量受限的 LRU 缓存复用样本向量，缓存与数据集实例隔离。
- 每次检索响应提供候选样本数、向量缓存命中/未命中、特征构建耗时、排序耗时和总耗时。
- 页面展示本次检索候选规模、总耗时和缓存命中数。
- 增加未来行情变更不影响查询向量的防泄漏测试，以及缓存前后排序一致性测试。
- 增加冷启动/热缓存性能基线命令：`python3 scripts/benchmark_retrieval.py`。
- 新浪 AKShare 端点真实数据快照：3 只股票、6,288 条日线、5,508 个有效检索样本；真实数据冷/热检索基准见 [阶段 7 验证报告](docs/validation-stage-7.md)。

## 下载真实 A 股数据

AKShare 下载器是可选功能，需要 Python 3.11 或更高版本。核心检索仍可在 Python 3.10 运行。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[web,a-share]'
NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn .venv/bin/kline-download-a-share \
  --endpoint sina \
  --symbols 000001,600519,300750 \
  --start-date 2018-01-01 \
  --adjust qfq
```

若本机代理拦截 Sina 请求，可在 macOS/Linux 对该命令显式直连 Sina 域名：`NO_PROXY=.sina.com.cn no_proxy=.sina.com.cn`。Eastmoney 是默认端点；如果本地代理允许访问，也可省略 `--endpoint sina` 使用默认 Eastmoney。

默认写入 `data/processed/a_share_daily.csv`，旁边会生成包含来源、AKShare 版本、复权方式、日期范围和 SHA-256 的 manifest。默认执行增量更新并重抓末尾若干天以吸收近期修订；完整重建加 `--full-refresh`。真实行情目录已加入 `.gitignore`，不会随项目推送。

用下载的数据启动网页：

```bash
KLINE_DATA_PATH=data/processed/a_share_daily.csv \
  .venv/bin/uvicorn kline_retrieval.api:app --host 127.0.0.1 --port 8001
```

访问 `http://127.0.0.1:8001`。页面数据标记显示行情来源和复权方式，悬停可查看覆盖日期与数据哈希。该下载器获取的是历史日线数据，不提供实时行情。阶段 6 本地已用 AKShare Sina 端点完成真实网页检索；默认 Eastmoney 端点仍受当前机器代理网络影响，详见 [数据接入说明](docs/a-share-data.md)。

## 数据格式

行情文件至少需要以下字段：

```text
symbol,date,open,high,low,close,volume
```

其中 `volume` 可选，日期建议使用 `YYYY-MM-DD` 格式。示例文件位于 `data/sample_market_data.csv`。

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

查询使用的月线、周线和日线图会参与后续特征提取；`future_12d` 只用于事后对照，不会进入检索特征。

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
