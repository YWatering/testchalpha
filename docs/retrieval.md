# 阶段三检索实现说明

## ColProj

当前默认特征模型是 `ColumnProjectionEmbedder`，对64×64图像逐列投影，输出64维向量：

1. 对图像的每一列读取黑色像素强度。
2. 使用黑色像素作为权重，计算该列的垂直质心。
3. 将质心转换为向上为正的价格轨迹值。
4. 对没有像素的列进行线性插值。
5. 对 64 个轨迹值去均值并做 L2 归一化。

因此输出向量表达的是图像的列方向价格形状，而不是股票的绝对价格水平。

## 多尺度距离

月线、周线和日线分别计算余弦距离，再按权重组合：

```text
distance = (
    month_weight * monthly_distance +
    week_weight  * weekly_distance +
    day_weight   * daily_distance
) / (month_weight + week_weight + day_weight)
```

日线权重固定为 `1.0`，月线和周线权重可以调整。

## 检索流程

```python
from kline_retrieval import MarketData, search_market_data

market_data = MarketData.from_csv("data/market_data.csv")
matches = search_market_data(
    market_data,
    "AAPL",
    "2024-01-31",
    top_k=20,
    month_weight=1.0,
    week_weight=1.0,
    embargo_trading_days=3,
)
```

每个 `SimilarityMatch` 包含：

- `rank`：排名。
- `sample.symbol` 和 `sample.date`：历史样本标识。
- `distance`：综合距离。
- `monthly_distance`、`weekly_distance`、`daily_distance`：各尺度距离。

## 候选过滤

候选样本必须满足：

- 候选日期早于查询日期。
- 历史窗口至少包含 400 个交易日（20 根月线，每根20日）。
- 未来窗口至少包含 20 个交易日。
- 查询样本自身不会进入结果。
- 同一股票可以通过 `embargo_trading_days` 排除相邻样本。

未来 20 日数据只用于确认候选样本可以作为历史收益标签，不进入图像、向量或距离计算。查询样本只需要历史窗口，因此即使查询日期接近数据末尾，也可以执行检索。
