# License Scope / 许可证适用范围

Effective date / 生效日期：2026-08-07  
Copyright © 2026 XUEZHENG WANG

本仓库采用分层授权。路径规则优先于笼统描述；文件若另有明确许可证声明，以该声明为准。

This repository uses layered licensing. Path rules take precedence over general descriptions. An
explicit license notice attached to a file takes precedence for that file.

## Apache License 2.0 — software

以下内容按根目录 [LICENSE](LICENSE) 中的 Apache License 2.0 授权：

- `src/`、`tests/`、`scripts/`中的软件、测试、实验入口、分析和绘图代码；
- `website/`中的网站程序代码、样式、配置和测试；
- `.github/`协作模板，以及 `pyproject.toml`、构建配置和其他软件工程配置文件；
- 后续明确标注为 `SPDX-License-Identifier: Apache-2.0` 的文件。

The following material is licensed under the Apache License 2.0 in the root [LICENSE](LICENSE):

- software, tests, experiment entry points, analysis, and plotting code under `src/`, `tests/`, and
  `scripts/`;
- website program code, styles, configuration, and tests under `website/`;
- collaboration templates under `.github/`, plus `pyproject.toml`, build configuration, and other
  software-engineering configuration;
- files later marked `SPDX-License-Identifier: Apache-2.0`.

The Apache patent grant and termination terms apply only as stated in that license.

## CC BY 4.0 — research materials and data

以下内容按 [Creative Commons Attribution 4.0 International](LICENSES/CC-BY-4.0.txt) 授权：

- `paper/`中的论文正文与论文附件说明；
- `docs/`以及根目录README、复现、方法、数据和研究治理文档；
- `results/`中的程序生成实验日志、manifest、摘要和结构化分析；
- `output/figures/`、`output/markdown/`与 `output/pdf/`中的图表和文稿；
- `CITATION.cff`以及其他明确标注为 `CC-BY-4.0` 的研究内容。

The following material is licensed under
[Creative Commons Attribution 4.0 International](LICENSES/CC-BY-4.0.txt):

- papers and paper-attachment documentation under `paper/`;
- `docs/` and root-level READMEs, reproducibility, methodology, data, and research-governance prose;
- program-generated logs, manifests, summaries, and structured analyses under `results/`;
- figures and manuscripts under `output/figures/`, `output/markdown/`, and `output/pdf/`;
- `CITATION.cff` and other research material explicitly marked `CC-BY-4.0`.

Recommended attribution / 建议署名：

> XUEZHENG WANG, “FE-E: Finite-Element and Entropy Control of Adjoint Propagation in Deep
> Transformers,” version 0.1, 2026,
> <https://github.com/supergitty520/fe-e-transformer-research>. See `CITATION.cff`.

When sharing adapted material, identify changes and retain a link or reference to the source and the
CC BY 4.0 license.

## Mixed and excluded material / 混合与排除内容

- `website/`中的程序实现按Apache-2.0授权；从论文或研究文章复用的独立文字、图表仍按
  CC BY 4.0授权。
- 标准许可证文本本身按其发布者声明处理，本文件不对其重新授权。
- 第三方依赖继续适用各自许可证；依赖清单不改变其许可证。
- `local_research_archive/`、`.secrets/`、`publication/`及被忽略的机器本地文件不属于
  本次分发内容。
- 商标、隐私权、人格权及作者无权授予的第三方权利不因本说明而获得许可。

- Website implementation is Apache-2.0; separable prose and figures reused from research materials
  remain CC BY 4.0.
- Standard license texts remain governed by their publishers' notices and are not relicensed here.
- Third-party dependencies retain their own licenses; dependency metadata does not relicense them.
- `local_research_archive/`, `.secrets/`, `publication/`, and ignored machine-local material are not
  part of this distribution.
- Trademark, privacy, personality, and third-party rights that the author cannot grant are not licensed.

## Private visibility / 私有可见性

GitHub的Private设置只限制仓库的发现和访问，不撤销上述许可证已经授予给合法接收者的权利。
如果某项合作必须保密，应在授予访问权之前另行签署保密协议，并重新评估是否应向该接收者
分发采用开放许可证的版本。

GitHub Private visibility controls discovery and access; it does not revoke rights already granted by
the licenses to lawful recipients. If a collaboration requires confidentiality, use a separate written
agreement before granting access and reconsider whether this openly licensed distribution should be
shared with that recipient.

This scope statement is operational documentation, not legal advice.
