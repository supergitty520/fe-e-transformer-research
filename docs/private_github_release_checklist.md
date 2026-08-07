# GitHub 私有发布检查清单

**中文** | [English](en/private_github_release_checklist.md)

## A. 本地准备

- [ ] 运行 `python scripts/check_github_readiness.py`。
- [ ] 运行PyTorch单元测试；有Apple Silicon时再运行MLX测试。
- [ ] 确认 `.secrets/`、SSH脚本、`publication/`、`tmp/`、`node_modules/`和
      `local_research_archive/`均被忽略。
- [ ] 确认生成的 `output/attachments/*.zip` 与 `*.sha256` 不进入主Git历史。
- [ ] 确认没有超过50 MB的候选文件；超过100 MB必须移出Git历史。
- [ ] 阅读README顶部的当前结论，避免旧的正向结果压过最新负结果。
- [ ] 检查论文作者为 `XUEZHENG WANG`，匿名稿不含署名。
- [ ] 确认主动停止和无效运行仍有明确标签。
- [ ] 按[中英文发布对应表](bilingual_publication_map.md)逐项检查双语主张、数字和链接。

## B. 创建私有仓库

建议仓库名：`fe-e-transformer-research`。

GitHub创建时：

- Visibility选择 `Private`；
- 不自动生成README、`.gitignore`或License；仓库已经包含经过审计的分层许可证；
- 暂不开启GitHub Pages；
- 禁止forking（若组织策略允许）；
- 合作者只给最小必要权限。

本地连接命令在远程仓库创建后再执行：

```bash
git remote add origin git@github.com:<owner>/fe-e-transformer-research.git
git push -u origin main
```

不要把真实owner写入脚本或文档占位符。

## C. 首次提交建议

首次提交只包含已审计的研究快照：

```bash
git add .
git status --short
git diff --cached --stat
git commit -m "research: archive FE-E private preview v0.1"
```

提交前重点查看：

- `scripts/`中没有SSH运维脚本；
- `website/`中没有依赖和构建缓存；
- `results/`中的原始日志没有被意外改写；
- `output/`只包含有引用价值的图、PDF和小型附件。

## D. GitHub设置

- [ ] 默认分支设为 `main`。
- [ ] 保护main：合作者修改需Pull Request。
- [ ] 启用secret scanning和push protection（私有计划支持时）。
- [ ] 启用Dependabot安全更新，但不要自动合并数值依赖升级。
- [ ] 关闭不需要的Wiki、Projects和Discussions。
- [ ] 仓库描述明确写“private research preview; no general optimizer claim”。
- [ ] Topics可用：`transformer`、`finite-element`、`gradient-analysis`、`mlx`、`research`。

## E. 合作者入口

邀请合作者时要求先读：

1. `README.md`；
2. `docs/research_status.md`；
3. `docs/methodology.md`；
4. `REPRODUCIBILITY.md`；
5. `LICENSE_POLICY.md`。

英文协作者可从 `README_EN.md` 进入等价文档；两种语言共用相同的原始日志、代码和
数值证据。

新的实验提议使用Issue模板，必须预先写明假设、种子、终点、算力预算和停止条件。结果合并
必须附原始日志、分析脚本、负结果说明和首次介入前匹配审计。

## F. 转为公开前的额外门槛

- [ ] 审核专利、投稿和保密影响。
- [ ] 净化manifest中的绝对机器路径。
- [ ] 复核现有Apache-2.0 / CC BY 4.0分层许可证仍符合公开发布意图。
- [ ] 核对Gradient Smoothing等第三方实现与引用。
- [ ] 用全新clone从零复现单元测试和至少一个代表性分析。
- [ ] 明确标注单种子、合成任务和未验证7B尺度。
- [ ] 删除全部内部Issue、机器信息和发布基础设施残留。
- [ ] 完成一次中英文主张等价性审阅。

私有发布是研究协作节点，不是公开论文发布，也不是性能结论升级。
