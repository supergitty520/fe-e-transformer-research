# 实验登记表

**中文** | [English](en/experiment_registry.md)

该表用于区分正式证据、探索结果、主动停止和失效运行。详细配置以对应manifest和报告为准。

| 阶段 | 规模与协议 | 主要问题 | 状态 | 主要入口 |
|---|---|---|---|---|
| PyTorch消融 | 24层，3种子，40步 | 刚度/质量/熵是否互补 | 机制实验 | `docs/research_report.md` |
| GS正面对照 | 24层，5种子，200步 | 固定步数与反传预算 | 小规模正式对照 | `results/direct_compare_summary.md` |
| 深度审计 | 48层，3种子 | 早期结论是否随深度保持 | 描述性 | `results/direct_compare_d48_summary.md` |
| MLX 24层 | 干净/学习率冲击，3种子 | 六方案工程比较 | 正式；周期脉冲被否定 | `docs/mlx_d24_experiment_report.md` |
| MLX 96层 | 干净/学习率冲击，3种子 | 深度扩展与等时预算 | 正式 | `docs/mlx_d96_experiment_report.md` |
| MLX 1024层 | 单种子冻结外推 | 超深可行性 | 主动停止，不作性能结论 | `docs/mlx_d1024_exploratory_stop_note.md` |
| MLX 192层 | 250步，3种子，5-batch验证 | 短程深层稳定性 | 正式描述性 | `docs/mlx_d192_s3_step250_eval5_report.md` |
| MLX 128层长程 | 1000步，单种子，8次验证 | 长程AUC与瞬态冲击 | 探索性 | `docs/mlx_d128_s1_step1000_eval8_report.md` |
| 5000步三方案 | 单种子串行运行 | 损失阈值与墙钟效率 | 主动停止；墙钟作废 | `docs/mlx_d128_s1_step5000_threecase_stop_note.md` |
| 持续门控v3 | 128层，99%连续终点 | 低频门控是否缩短相变 | 探索性正信号 | `docs/mlx_d128_acc99_persistent_gate_v3_report.md` |
| 三种子GSF/GS/AdamW | seed 31/47/59，99%终点 | 困难种子可靠性与总体性能 | 提前停止后的正式三种子汇总 | `docs/mlx_d128_acc99_s3_fee_gs_adamw_report.md` |
| AdamW同构sham | seed 31 | 分离观察器与FE-E作用 | 单种子同构机制对照 | `docs/mlx_d128_s31_adamw_fee_vs_sham_report.md` |
| 192层同构冲击 | seed 47，持续学习率冲击 | 深层GSF是否优于GS | 可信负机制案例 | `docs/mlx_d192_s47_acc99_gs_sham_vs_gsf_lrshock_report.md` |
| 剂量×结构噪声 | 128层，4环境×4剂量，seed 47 | FE-E有效前置条件 | 最新单种子条件性实验 | `docs/mlx_d128_s47_fee_dose_noise_acc99_report.md` |

## 无效运行处理

开发期、smoke、遗漏冲击或未完成运行保留在 `results/`，但不得因目录存在就进入正式汇总。
正式分析脚本使用明确的纳入列表，并在报告中说明排除原因。
