# testchalpha

## 当前阶段

项目当前已完成阶段 0 至阶段 6：项目基线、行情数据、多尺度图像、ColProj 检索、收益分析、检索网站，以及第一版验证和性能优化。

下一阶段已调整为阶段 7：接入真实 A 股历史行情并构建本地演示数据集。当前网站尚未内置联网行情下载器，默认仍使用合成演示行情；真实 A 股数据接入完成前，可通过 `KLINE_DATA_PATH` 加载已有的标准 CSV/Parquet 文件。阶段 8 起再接入 CNN、规模化索引和回测。

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

阶段 6 交付内容：

- API 进程内使用容量受限的 LRU 缓存复用样本向量，缓存与数据集实例隔离。
- 每次检索响应提供候选样本数、向量缓存命中/未命中、特征构建耗时、排序耗时和总耗时。
- 页面展示本次检索候选规模、总耗时和缓存命中数。
- 增加未来行情变更不影响查询向量的防泄漏测试，以及缓存前后排序一致性测试。
- 增加冷启动/热缓存性能基线命令：`python3 scripts/benchmark_retrieval.py`。

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
