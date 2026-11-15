# 阶段十五：ColProj验证期批量预测结果

## 当前进度

- 验证期信号范围：`2024-01-01` 至 `2024-12-31`。
- 全球周历筛选后共有51个周度信号。
- 固定股票池：50只。
- 目标查询数：2,550条股票—周预测。
- 已完成：2,550/2,550条，覆盖率100%。
- 相似样本明细：51,000条，每个查询均保存完整Top-20。
- 预测状态：2,550条均为`ok`，实际标签均为`tradable`。
- 时间防泄漏检查：0条候选标签退出日晚于查询日。
- 全量计算耗时约63分钟；SQLite断点库仅保存在本地，不提交到GitHub。
- 任务版本：`fd94dc496f9fee87fc11751b64c5761651c517bac5cecfc1ee08f1c6924e147c`。

完整预测和匹配明细见`colproj_validation_predictions.csv`与`colproj_validation_matches.csv`；统计结论见[验证期结果报告](COLPROJ_VALIDATION_RESULTS.md)。本验证期方向准确率约48.75%，预测与实际收益的Pearson相关约-0.036，当前结果未能确认可用的买入阈值。

## 断点续跑

预测摘要和相似样本明细先在本地SQLite中按查询事务写入，再导出CSV。相同的行情哈希、ANN版本、股票池、时间范围和参数会恢复同一任务，已经完成的股票—周查询不会重复计算。

```bash
.venv/bin/kline-stage15-weekly predict-colproj-batch \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --max-new-queries 100
```

处理全部剩余查询时使用：

```bash
.venv/bin/kline-stage15-weekly predict-colproj-batch \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --all
```

## 实测性能

- ANN冷启动和首次磁盘读取耗时明显高于后续查询。
- 热查询通常约1.1至1.5秒。
- 完整2,550条预测耗时约3,775.5秒（约63分钟），包括ANN载入、查询、逐条事务写入和CSV导出。
- 平均吞吐约0.68条/秒；任务支持分批运行和断点恢复。

## 版本约束

- 行情重建版本：`4cdf937ddeb4c34fefd5c1ad558a9cd3ded8086bc72d4f1c5a135b59a2128f35`。
- ANN版本：`ann-composite-c33c9e6e03bf9f40`。
- Top-K：20；过采样：500；ANN候选基数：1,000；`nprobe=64`。
- 月线、周线、日线距离权重均为1；温度参数为0.25。
- 历史候选必须是周末信号样本，且其未来5日标签退出日不得晚于查询日。
