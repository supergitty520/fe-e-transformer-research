# GitHub 私有仓库发布报告

**中文** | [English](PRIVATE_REPO_PREP_REPORT_EN.md)

日期：2026-08-07  
状态：**已发布到GitHub Private仓库。**  
仓库：<https://github.com/supergitty520/fe-e-transformer-research>

## 发布结果

- 所有者：`supergitty520`；仓库：`fe-e-transformer-research`；
- 可见性：Private；默认分支：`main`；
- 提交作者：XUEZHENG WANG，使用GitHub noreply地址；
- 452个源文件、研究文档和审计后的实验制品已进入Git历史；
- 代码按Apache-2.0授权，论文、文档、图表与结果按CC BY 4.0授权，具体边界见
  [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md)。

## 建议

现阶段适合私有协作，不适合直接公开。私有仓库可以完整保留研究路线、负结果、主动停止、
同构对照和原始日志，让少量合作者先审阅主张边界。它不应被理解为FE-E已经成为成熟
优化器，也不能替代专利或保密安排。

## 候选集

- 452个候选文件；
- 约186.2 MiB；
- 最大单文件约9 MiB；
- 正式与关键停止实验的原始JSONL约185 MiB，直接版本化，不需要Git LFS；
- 没有超过50 MiB的候选文件；
- 没有检测到私钥或常见云平台token格式；
- 可移植文档中没有残留绝对用户路径；
- Markdown本地链接检查通过。

## 纳入内容

- FE-E与Gradient Smoothing实现；
- PyTorch/MLX单元测试；
- 实验运行、分析和绘图脚本；
- 原始JSONL、manifest、运行摘要和结构化审计；
- 正式、探索、失效和主动停止报告；
- 英文预印本、匿名稿、中文文章、中英文发布入口、图表和Release附件策略；
- Apache-2.0、CC BY 4.0全文、双语适用范围、NOTICE和引用元数据；
- 互动研究网页源码，不包含依赖目录。

## 明确排除

- `.secrets/`；
- Windows SSH、管理员创建和撤销脚本；
- 博客发布副本与Cloudflare基础设施；
- `PROJECT.md`中的对话线程和本机工作区元数据；
- `tmp/`、Python缓存、`node_modules`和网站构建产物；
- 编辑器、操作系统和运行时临时文件；
- 202个冒烟、pilot、校准、被替代、协议失效或无独立结论的中间制品，约23.8 MiB；
- 由仓库内容重复生成的ZIP、SHA-256和被正式多种子结果替代的旧图。

上述中间制品没有删除，均移入被忽略的 `local_research_archive/`。主候选集只留下两项
具有独立研究意义的不完整日志：1024层冻结参数外推和128层5000步主动停止实验。

## 验证结果

- PyTorch/Gradient Smoothing：15项测试通过；
- Apple MLX配置、VJP、门控、sham和剂量调度：18项测试通过；
- 互动网页：构建成功，1项服务端渲染测试通过；
- 私有仓库就绪审计：通过；
- 直接运行系统Python全量测试会因隔离环境缺少PyTorch和Metal失败，已分别在本地Docker
  PyTorch环境和允许Metal访问的本机环境完成替代验证。

## 本次未执行

- 未发布GitHub Release；
- 未邀请合作者；
- 未更改为Public；
- 未对已生效的分层许可证作未来公开阶段的二次法律审阅。

## 需要作者决定的事项

1. 首批合作者及其权限；
2. 是否存在专利、投稿或NDA要求；
3. 未来转为公开前是否需要对现有分层授权再做专业法律审阅；
4. 是否需要把每一篇中文详细实验报告全文翻译成英文，而不只提供英文登记摘要；
5. 是否把后续大型原始日志改放私有Release附件，而不继续扩大Git历史。
