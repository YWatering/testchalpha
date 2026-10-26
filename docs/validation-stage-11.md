# 阶段十一 CNN 原型验证报告

## 1. 结论

阶段十一已完成 CNN 原型训练、检查点保存、三尺度推理和 ColProj 对照评估。该小规模对照用于说明两种模型的结果差异，不用于判定是否向用户开放 CNN。根据产品要求，阶段十二将为全市场建立独立 CNN 索引，并允许用户主动选择 ColProj 或 CNN；生产默认值和模型说明将在阶段十二明确。

## 2. 模型和数据口径

- 输入：月线、周线、日线三张 `64×64` 黑白图像。
- 网络：4 个卷积模块，随后 2 层全连接，输出 128 维归一化向量。
- 训练方式：同一样本的两种增强视图进行对比学习，不使用未来图像或未来收益标签作为输入。
- 训练设备：Apple MPS。
- 代表股票：`000002`、`000333`、`300750`、`600519`、`688981`、`920885`，覆盖主板、创业板、科创板和北交所。
- 数据切分：按交易日期切分，训练集、验证集、测试集各 96 个样本；训练截止日为 `2023-11-20`，验证截止日为 `2025-04-22`。
- 数据源哈希：`173b465f311686a7cfba51e6057364c9ead10b2ca08dd9be8476476bc938d291`。

## 3. 训练结果

模型版本为 `cnn-stage11-v1`，训练 3 轮，对比损失从 `2.2954` 降至 `1.3379`。本地生成的检查点、切分清单和训练记录位于 `data/processed/cnn_stage11/`，这些运行数据不推送到 GitHub。

## 4. 对照结果

在固定测试查询集上使用 Top-5 结果比较：

- 查询数：12。
- 候选数：72。
- CNN 与 ColProj 的平均 Top-5 重合度：`21.67%`。
- 平均未来收益仅作为观察指标，不作为训练标签或模型选择的唯一依据。
- CNN 未来 1/5/10/20 日 Top-5 平均收益：`0.52% / 0.98% / 2.49% / 4.81%`。
- ColProj 未来 1/5/10/20 日 Top-5 平均收益：`-0.27% / 0.54% / 2.92% / 4.06%`。

当前样本规模仍属于原型验证，不能据此宣称 CNN 优于或劣于 ColProj。阶段十二需要扩大分层测试集，并增加跨年份、跨行业和滚动时间外推验证；这些评估用于向用户透明展示模型差异和稳定性，而不是取消用户选择其中一种模型的功能。

## 5. 可复现实验

安装可选依赖：

```bash
.venv/bin/pip install -e '.[cnn]'
```

训练原型：

```bash
.venv/bin/kline-train-cnn \
  --database data/processed/csmar_a_share_daily.sqlite3 \
  --output-dir data/processed/cnn_stage11 \
  --symbols 000002,000333,300750,600519,688981,920885 \
  --max-per-split 16 \
  --epochs 3 \
  --batch-size 8 \
  --device auto
```

运行对照评估：

```bash
.venv/bin/kline-evaluate-cnn \
  --database data/processed/csmar_a_share_daily.sqlite3 \
  --manifest data/processed/cnn_stage11/splits.json \
  --checkpoint data/processed/cnn_stage11/cnn_stage11.pt \
  --output data/processed/cnn_stage11/evaluation.json \
  --max-queries 12 \
  --max-candidates 72 \
  --top-k 5
```
