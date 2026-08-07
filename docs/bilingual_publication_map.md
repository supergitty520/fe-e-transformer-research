# 中英文发布对应表 / Bilingual Publication Map

更新日期 / Updated: 2026-08-07

本文件定义中英文发布入口如何共享同一套证据。两种语言的结论边界相同；实验数字、代码、
原始日志和文件路径只有一份，不另建“英文结果副本”。

This file defines how the Chinese and English publication tracks share one evidence base. Both languages
must preserve the same claim boundaries. Experimental numbers, code, raw logs, and artifact paths have a
single source of truth.

## 对应文档 / Paired documents

| 内容 / Purpose | 中文 | English |
|---|---|---|
| 总入口 / Main entry | [README](../README.md) | [README](../README_EN.md) |
| 研究过程 / Research process | [中文](research_process.md) | [English](en/research_process.md) |
| 研究方法 / Methodology | [中文](methodology.md) | [English](en/methodology.md) |
| 阶段结论 / Current status | [中文](research_status.md) | [English](en/research_status.md) |
| 实验登记 / Experiment registry | [中文](experiment_registry.md) | [English](en/experiment_registry.md) |
| 复现 / Reproducibility | [中文](../REPRODUCIBILITY.md) | [English](../REPRODUCIBILITY_EN.md) |
| 数据与制品 / Data and artifacts | [中文](../DATA_AND_ARTIFACTS.md) | [English](../DATA_AND_ARTIFACTS_EN.md) |
| 授权策略 / License policy | [中文](../LICENSE_POLICY.md) | [English](../LICENSE_POLICY_EN.md) |
| 贡献规范 / Contributing | [中文](../CONTRIBUTING.md) | [English](../CONTRIBUTING_EN.md) |
| 安全说明 / Security | [中文](../SECURITY.md) | [English](../SECURITY_EN.md) |
| 私有发布清单 / Private release checklist | [中文](private_github_release_checklist.md) | [English](en/private_github_release_checklist.md) |
| 本地准备报告 / Local preparation report | [中文](../PRIVATE_REPO_PREP_REPORT.md) | [English](../PRIVATE_REPO_PREP_REPORT_EN.md) |

## 共同证据源 / Shared source of truth

- `results/`中的JSON、JSONL、manifest与结构化审计是数值证据源。
- `src/`、`tests/`和`scripts/`是实现与重建证据源。
- 详细实验报告保留原始语言；英文登记表提供等价摘要和直接链接。
- 英文预印本位于 [paper/fe_e_preprint.md](../paper/fe_e_preprint.md)；中文长文位于
  [output/markdown/刚度约束和信息熵对深度学习的反作用及引申.md](../output/markdown/刚度约束和信息熵对深度学习的反作用及引申.md)。

- JSON, JSONL, manifests, and structured analyses under `results/` are the numeric source of truth.
- `src/`, `tests/`, and `scripts/` are the implementation and reconstruction source of truth.
- Detailed experiment reports retain their original language; the English registry provides equivalent
  summaries and direct links.
- The English preprint and Chinese long-form essay are complementary editorial artifacts, not separate
  experimental records.

## 同步规则 / Synchronization rules

1. 任何主张、数字、终点定义或证据等级变化，必须同时更新对应的中英文总入口和阶段结论。
2. 方法名、变体名、seed、步数、单位、配置键和文件路径不翻译。
3. 翻译不新增原文没有的因果、显著性或可扩展性主张。
4. 若摘要与详细报告冲突，以原始日志和重建脚本为准，并修正两种语言的摘要。
5. 每次候选发布运行 `python scripts/check_github_readiness.py`，检查双语必需文件和链接。

1. Any change to a claim, number, endpoint, or evidence level must update both language entry points and
   status documents.
2. Method names, variant names, seeds, steps, units, configuration keys, and paths remain unchanged.
3. Translation must not introduce causal, statistical-significance, or scale claims absent from the source.
4. If summaries conflict with detailed reports, raw logs and reconstruction scripts take precedence; both
   language summaries must then be corrected.
5. Every candidate release runs `python scripts/check_github_readiness.py` to verify required bilingual
   files and local links.

当前状态：仅完成本地双语发布准备；尚未初始化Git、创建远端或上传。

Current status: local bilingual release preparation only; Git has not been initialized, no remote has
been created, and nothing has been uploaded.

