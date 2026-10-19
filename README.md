# testchalpha

## 当前阶段

项目当前已完成阶段 0：项目基线和技术准备、阶段 1：行情数据和样本基础、阶段 2：多尺度 K 线图生成，以及阶段 3：ColProj 特征和相似度检索。

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
