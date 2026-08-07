# 复现指南

**中文** | [English](REPRODUCIBILITY_EN.md)

## 复现层级

本仓库区分三类复现：

1. **单元测试**：验证有限元装配、熵、方向反例、双反传和MLX配置逻辑。
2. **结果重建**：从已提交的JSON/JSONL重新生成汇总、审计报告和图表。
3. **重新训练**：在PyTorch或Apple MLX上运行冻结协议，结果可能受硬件和数值后端影响。

## Python环境

需要Python 3.10+。PyTorch原型：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Apple Silicon上的MLX实验：

```bash
python -m pip install -e '.[mlx]'
PYTHONPATH=src python -m unittest tests.test_mlx_experiment tests.test_mlx_dose_response -v
```

MLX不是CUDA后端；Windows RTX 3090不能直接运行MLX入口。CUDA复核应先实现与MLX状态机
同构的PyTorch入口，并验证日志字段和首次介入前轨迹。

## 最小PyTorch实验

```bash
PYTHONPATH=src python -m fe_entropy.experiment \
  --steps 40 --layers 24 --width 32 \
  --lambda-stiffness 2 --lambda-energy 0.02 --lambda-entropy 2 \
  --entropy-lower 0.90 --entropy-upper 0.98 \
  --variants baseline,stiffness,energy,entropy,fe_e \
  --output results/example.json
```

普通Transformer使用默认的 `--depth-domain layer_index`。只有残差步长按 `O(1/L)`
缩放时，才使用 `unit_interval`。

## MLX正式入口

```bash
PYTHONPATH=src python -m fe_entropy.mlx_experiment \
  --layers 24 --steps 200 --seeds 31,47,59 \
  --variants baseline,gradient_smoothing,fe_e_always,fe_e_gated,gs_fe_e_gated
```

固定剂量—传播噪声实验：

```bash
PYTHONPATH=src python -m fe_entropy.mlx_dose_response
```

该入口默认使用128层、宽度32、seed 47、四种环境、1%/3%/5%嵌套介入率、每32步验证、
8个固定验证batch，以及99% token accuracy连续三次终点。运行前应在源码与manifest中
复核冻结参数，不要把默认值视为永远有效的正式协议。

## 从原始日志重建关键结果

```bash
python scripts/analyze_mlx_d128_s47_fee_dose_noise_acc99.py
python scripts/analyze_mlx_d192_s47_gs_sham_vs_gsf_lrshock.py
python scripts/analyze_mlx_d128_s31_adamw_fee_vs_sham.py
python scripts/analyze_mlx_d128_acc99_s3_fee_gs_adamw.py
```

脚本应从JSONL重建终点与介入事件，而不是只读取最终summary。输出位置见各脚本和
`DATA_AND_ARTIFACTS.md`。

## 数值可重复性边界

- 相同seed不保证跨PyTorch、MLX、CPU、Metal和CUDA逐位一致。
- 同一Metal设备上的运行也可能在首次介入前出现微小浮点分叉。
- 墙钟耗时受温度、系统负载、运行顺序和后端编译缓存影响。
- 因此复现重点是方向、配对分布、门控事件和终点区间，不是要求每个浮点数完全相同。

## 完整性检查

重新训练后至少确认：

- 唯一 `run_start` / `run_end`；
- 步号连续、无重复、无未解释的failure；
- 验证间隔与冻结协议一致；
- 介入步与manifest冻结表一致；
- 主终点可由验证日志重建；
- 非完整或主动停止运行没有进入正式汇总。

## 论文与附件

```bash
python scripts/build_preprint.py
python scripts/package_preprint.py
```

PDF位于 `output/pdf/`；ZIP与相邻 `.sha256` 在 `output/attachments/` 本地生成，未来
上传Release，但不进入主Git历史。校验文件验证单个归档，不代表整个Git仓库的内容哈希。
