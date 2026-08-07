# FE-E：Transformer 深度伴随梯度的有限元—熵约束

**中文** | [English](README_EN.md)

这是一个工程研究原型，用一维线性有限元的刚度、质量装配约束
Transformer 残差流上的任务梯度，并用相对熵覆盖度描述梯度能量沿深度的集中。

方法正式命名为 **FE-E（Finite-Element and Entropy）**。这是研究档案，而不是已经成熟的
训练配方。当前证据不支持把 FE-E 作为通用优化器或默认正则项：在最新的128层单种子
剂量—传播噪声实验中，12个FE-E配对只有3个更早确认99% token accuracy，9个更晚；
正向结果集中在持续的层间能量集中和低剂量高频扰动。更稳妥的研究方向，是把刚度、质量
与熵先作为传播观测器，再检验它们能否预测未来失稳并触发少量、可回滚的干预。

> **Private research preview · 2026-08-07**  
> 仓库保留正结果、负结果、主动停止、失效试跑和方法变更。任何单种子结果都不得外推为
> 7B级语言模型的训练收益。

## 仓库导航

- [中英文发布对应表](docs/bilingual_publication_map.md)：双语入口、共同证据源和同步规则。
- [研究过程](docs/research_process.md)：问题如何从“常开正则”演变为“条件观测与干预”。
- [研究方法](docs/methodology.md)：假设、对照、终点、日志、门控和证据等级。
- [阶段性结论](docs/research_status.md)：目前能说什么、不能说什么、下一步验证什么。
- [实验登记表](docs/experiment_registry.md)：正式、探索、停止和失效运行的索引。
- [复现指南](REPRODUCIBILITY.md)：PyTorch、MLX、结果重建与平台限制。
- [数据与制品](DATA_AND_ARTIFACTS.md)：原始日志、汇总、图表、论文和附件的边界。
- [许可证适用范围](LICENSE_SCOPE.md)：代码Apache-2.0，研究文档、图表与数据CC BY 4.0。
- [授权策略](LICENSE_POLICY.md)：私有可见性、再使用权、专利和第三方内容边界。
- [私有发布检查清单](docs/private_github_release_checklist.md)：创建远程仓库前后的操作顺序。
- [私有发布报告](PRIVATE_REPO_PREP_REPORT.md)：候选集、排除项、测试与发布状态。
- [English research entry](README_EN.md)：英文读者的等价研究入口。
- [英文预印本](paper/fe_e_preprint.md)与
  [中文研究文章](output/markdown/刚度约束和信息熵对深度学习的反作用及引申.md)。

完整数学审计见[研究报告](docs/research_report.md)，五随机种子早期正面对照见
[直接对照汇总](results/direct_compare_summary.md)，最新条件性实验见
[剂量—传播噪声报告](docs/mlx_d128_s47_fee_dose_noise_acc99_report.md)。

新增的 Apple MLX 24 层工程实验比较了 AdamW、Gradient Smoothing、常开 FE-E、固定
周期 FE-E、观测器门控 FE-E 和 GS + 门控 FE-E，并包含干净训练与学习率冲击两个
三种子协议。完整结论、配对区间和日志审计见
[MLX 24 层实验报告](docs/mlx_d24_experiment_report.md)。
固定周期 FE-E 已由正式结果否定，后续主实验默认只运行其余五种方案；该变体仍可通过
显式指定 `--variants fe_e_periodic` 复现。

冻结 24 层参数后的 Apple MLX 96 层正式实验也已完成。干净训练中，常开 FE-E 的三种子
平均损失为 3.3263，低于 AdamW 的 3.3867 和 Gradient Smoothing 的 3.3423，但每步耗时
约为 AdamW 的 1.98 倍；学习率冲击实验中 Gradient Smoothing 以 3.3686 略优于常开
FE-E 的 3.3864。完整的等步数/等时间对照、配对区间、观测器审计和原始证据位置见
[MLX 96 层实验报告](docs/mlx_d96_experiment_report.md)。
96层可复现实验附件可由打包脚本从仓库内的源码、30个运行日志和两份环境清单重新生成。
ZIP不进入主Git历史；获得明确发布授权后再上传到对应Release。
论文级曲线图位于 `output/figures/fee_d96_validation_loss_curves.{png,svg}`；绘图脚本为
`scripts/plot_mlx_d96_curves.py`。

192 层、250 步、三种子、每检查点 5 个验证 batch 的正式对比已完成。主终点上 AdamW、
常开 FE-E 与 GS + 门控 FE-E 均约为 3.406；常开 FE-E 的跨种子 SD 最低，并在最后三个
检查点平均口径下达到 3.4017。详见
[192 层三种子报告](docs/mlx_d192_s3_step250_eval5_report.md)；曲线位于
`output/figures/fee_d192_s3_step250_eval5_curves.{png,svg}`。

128 层、1000 步、单种子长程试跑也已完成，共 8 个验证检查点、每点 8 个验证 batch。
GS + 门控 FE-E 的终点损失为 3.1585，低于常开 FE-E 的 3.2159 与 AdamW 的 3.2627；
常开 FE-E 在第 750 步出现可恢复的瞬态冲击。按 AdamW 1000 步训练时间预算进行离散
匹配时，常开 FE-E 与混合方案也略优于 AdamW。该结果只构成探索性单种子证据，详见
[128 层长程试跑报告](docs/mlx_d128_s1_step1000_eval8_report.md)；曲线位于
`output/figures/fee_d128_s1_step1000_eval8_curves.{png,svg}`。

后续 128 层三方案 5000 步实验因串行运行期间系统负载与热态改变而主动停止，墙钟效率
比较作废。仍然有效的固定更新数结果显示：纯 Gradient Smoothing 在第 3250 步首次达到
0.001；GS + 门控 FE-E 在主动停止前的第 2750 步为 0.002316；常开 FE-E 跑满 5000
步仍为 3.0239，并在第 3125 步出现不利状态跃迁。详见
[5000 步主动停止审计](docs/mlx_d128_s1_step5000_threecase_stop_note.md)。

最新实验把终点改为更有工程意义的稳定任务掌握：token accuracy ≥99%，连续 3 个验证
检查点确认。持续伤害门控要求当下至少两个传播指标异常、最近 4 次中至少 3 次异常、固定
哨兵任务连续受损 2 次，并加入快速学习相变保护；每次只介入 1 步，冷却 48 步，FE-E
梯度比率按 5%→10%→20% 递增。128 层单种子中，混合方案首次/确认达标为
1375/1625 步，纯 GS 为 2250/2500 步，混合方案仅 0.74% 更新使用 FE-E。详见
[99% 终点持续门控报告](docs/mlx_d128_acc99_persistent_gate_v3_report.md)，曲线位于
`output/figures/fee_d128_acc99_persistent_gate_v3.{png,svg}`。这是探索性正向信号；首个
介入前的 Metal 轨迹已经出现微小数值分离，因此仍需同构观察器控制和多种子复核，不能把
875 步差异直接视为已证明的因果收益。

在同一 99% 终点协议下，随后完成 seed 31、47、59 的三方案配对，并按用户要求停止原定
五种子实验。混合方案 3/3 达标，纯 GS 2/3，AdamW 3/3；以 5000 步为删失上限的受限
平均确认步数分别为 2625、3083、2083。23 次 FE-E 介入全部集中在纯 GS 未达标的
seed 31；seed 47、59 中门控 0 次介入，混合方案与纯 GS 严格平局。当前证据因此支持
“FE-E 可能提高 GS 的困难种子可靠性”，但不支持其总体优于 AdamW。详见
[三种子提前停止报告](docs/mlx_d128_acc99_s3_fee_gs_adamw_report.md)，结构化审计位于
`results/mlx_d128_acc99_s3_fee_gs_adamw_analysis.json`，曲线位于
`output/figures/fee_d128_acc99_s3_fee_gs_adamw.{png,svg}`。

为分离 FE-E 本身与观察器计算的影响，seed 31 又完成了 AdamW + 门控 FE-E 对 AdamW +
同构 sham observer 的配对实验。两边执行相同观察器和哨兵探测，首次门控决定一致；FE-E
用 10 次单步介入（0.5%）把首次/确认终点从 1875/2125 提前到 1750/2000。首次介入前
49 步最大任务损失差仅 1.43e-6，因果匹配明显优于此前串行异构对照。不过历史纯 AdamW
也在 2000 步确认，因此该结果只证明 FE-E 优于 sham，不证明其获得相对精简 AdamW 的
生产净收益。详见
[AdamW 同构观察器报告](docs/mlx_d128_s31_adamw_fee_vs_sham_report.md)，曲线位于
`output/figures/fee_d128_s31_adamw_fee_vs_sham.{png,svg}`。

英文预印本由 **XUEZHENG WANG** 署名，位于 `output/pdf/FE-E_preprint_v0.1.pdf`；
PDF 内嵌完整代码与结果ZIP；独立ZIP属于可重新生成的Release制品，不进入主Git历史。
可编辑正文在 `paper/fe_e_preprint.md`。

## 授权

软件、测试、实验和分析代码采用Apache License 2.0；论文、研究文档、图表和程序生成
实验数据采用CC BY 4.0。完整路径规则见[许可证适用范围](LICENSE_SCOPE.md)。GitHub
Private只控制访问，不撤销开放许可证已经授予合法接收者的权利。

## 目录

- `src/fe_entropy/regularizer.py`：一致质量矩阵、刚度矩阵、熵带和二阶反传入口。
- `src/fe_entropy/gradient_smoothing.py`：按论文 Algorithm 1 与 AdamW 附录实现的窗口平滑。
- `src/fe_entropy/spectral.py`：小矩阵 Jacobian 的谱熵与幅值诊断。
- `src/fe_entropy/model.py`：支持高阶自动微分的微型 Pre-LN Transformer。
- `src/fe_entropy/experiment.py`：离线反转序列实验和消融。
- `src/fe_entropy/mlx_experiment.py`：Apple MLX 原生模型、逐层 VJP 伴随场、门控状态机、
  FE 梯度信赖域和逐步 JSONL 日志。
- `tests/test_regularizer.py`：有限元解析解、方向翻转、熵、谱尺度和双反传测试。
- `results/`：原始 JSON、汇总数据和压力实验。
- `scripts/summarize_direct_compare.py`：五种子配对区间与计算预算对照。
- `scripts/summarize_mlx_d24.py`：MLX 干净/冲击实验的配对区间与工程审计报告。
- `scripts/summarize_mlx_d96.py`：五方案 96 层正式实验、计算匹配和观测器审计报告。
- `scripts/summarize_mlx_d192_s3.py`：192 层三种子主终点、尾段敏感性、配对区间、
  计算匹配和观测器审计。
- `scripts/plot_mlx_d192_s3.py`：生成 192 层三种子均值 ± 标准差曲线。
- `scripts/summarize_mlx_d128_s1_step1000_eval8.py`：128 层长程试跑的 AUC、等时预算、
  瞬态冲击、门控和完整性审计。
- `scripts/plot_mlx_d128_s1_step1000_eval8.py`：生成 128 层长程试跑的论文曲线并标注
  第 750 步瞬态冲击。
- `scripts/analyze_mlx_d128_acc99_gate_v3.py`：汇总稳定 99% 主终点、门控行为、日志完整性
  与系统效率字段，并生成最新三联曲线及中文报告。
- `scripts/analyze_mlx_d128_acc99_s3_fee_gs_adamw.py`：聚合三个完整种子的 99% 确认终点、
  5000 步删失、配对胜负、门控介入和完整性证据，生成 PNG/SVG 图、JSON 与中文报告。
- `scripts/analyze_mlx_d128_s31_adamw_fee_vs_sham.py`：分析 AdamW + 门控 FE-E 与同构
  sham observer 的首次介入前匹配、主终点、门控事件、内存和局部步耗时。
- `scripts/build_preprint.py`：离线生成预印本 PDF。
- `scripts/package_preprint.py`：生成代码附件并嵌入 PDF。

## 快速验证

需要 Python 3.10+ 与 PyTorch 2.2+：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

本机已有的 PyTorch Docker 镜像可这样运行：

```bash
docker run --rm \
  -v "$PWD:/work" -w /work -e PYTHONPATH=/work/src \
  wzsp-demucs-e830-ready:latest \
  python3 -m unittest discover -s tests -v
```

复现实验：

```bash
PYTHONPATH=src python3 -m fe_entropy.experiment \
  --steps 40 --layers 24 --width 32 \
  --lambda-stiffness 2 --lambda-energy 0.02 --lambda-entropy 2 \
  --entropy-lower 0.90 --entropy-upper 0.98 \
  --variants baseline,stiffness,energy,entropy,fe_e \
  --output results/example.json
```

普通 Transformer 应使用默认的 `--depth-domain layer_index`。只有残差步长按
`O(1/L)` 缩放、确实解释为固定时间区间上的网格加密时，才使用
`--depth-domain unit_interval`。

Apple Silicon 上可直接运行 MLX 正式协议（需要 `mlx>=0.31`）：

```bash
PYTHONPATH=src python3 -m fe_entropy.mlx_experiment \
  --layers 24 --steps 200 --seeds 31,47,59 \
  --lambda-stiffness 0.1 --lambda-energy 0.02 --lambda-entropy 2 \
  --entropy-lower 0.5 --entropy-upper 0.98 \
  --gated-fee-gradient-ratio 0.5
```

96 层复现只需将 `--layers` 改为 `96`。正式主对比不包含固定周期脉冲；可使用
`--variants baseline,gradient_smoothing,fe_e_always,fe_e_gated,gs_fe_e_gated` 明确冻结方法集。

## 最小集成方式

模型前向需返回残差流节点 `states = [h_0, ..., h_L]`：

```python
from fe_entropy import FEERegularizer

regularizer = FEERegularizer(
    lambda_stiffness=2.0,
    lambda_energy=0.02,
    lambda_entropy=2.0,
    entropy_band=(0.90, 0.98),
)
terms = regularizer(task_loss, states, create_graph=True)
total_loss = task_loss + warmup_scale * terms.penalty
total_loss.backward()
```

旧名 `FEEntropyRegularizer` 与实验变体 `fe_entropy` 仍作为兼容别名保留；新集成应使用
`FEERegularizer` 和 `fe_e`。

`create_graph=True` 不可省略；若梯度来自 hook 后又被 `detach()`，这些量只能用于监控，
不能训练该正则项。
