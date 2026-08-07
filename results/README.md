# 结果文件说明

主要可引用结果：

- `direct_compare_d24_seed{31,47,59,71,89}.json`：未参与调参的五种子直接确认集。
- `direct_compare_summary.md` / `.json`：固定步数、近似反传预算、墙钟预算和配对区间。
- `direct_compare_d48_seed{31,47,59}.json`：48 层、100 步的三种子描述性深度审计。
- `direct_compare_d48_summary.md`：48 层固定步数与近似反传预算汇总。
- `gradient_smoothing_calibration_alpha*.json`：种子7上的 `alpha` 开发集校准；因其解释
  正式GS超参数选择而保留。
- `ablation_seed7.json`、`ablation_seed11.json`、`ablation_seed23.json`：24 层三随机种子消融。
- `ablation_summary.md` / `.json`：在共同诊断步 35 聚合的可读与机器格式结果。
- `stress_depth48_scale3.json`：48 层、残差比例 3 的基线与三项正则压力实验。
- `stress_depth48_scale3_warmup.json`：相同正则的 10 步 warmup 变体。
- `ode_consistent_depth24.json`：单位区间与 `1/24` 残差步长的连续深度一致实验。

## 不进入发布候选集的开发数据

冒烟测试、24层pilot与压力pilot、benchmark、被替代的单种子消融、门控v2/v4/v5/v6
开发轨迹、废弃的10000步试跑和协议失效预运行均已移入被忽略的本地研究归档。主候选集
只保留解释正式超参数选择所必需的校准结果，以及具有独立研究结论的主动停止证据。

本地归档不随仓库分发；运行标识、排除理由和必要摘要保留在对应方法报告与停止审计中。

所有 JSON 都记录配置、PyTorch 版本、逐步指标和时间。当前结果来自 CPU 小模型，只能用于
机制验证，不可外推为大模型吞吐或语言建模性能结论。
