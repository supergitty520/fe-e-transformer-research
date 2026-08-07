# FE-E: Finite-Element and Entropy Constraints on Deep Transformer Adjoint Propagation

**English** | [中文](README.md)

FE-E is an engineering research prototype that applies one-dimensional linear finite-element stiffness
and mass operators to task gradients along a Transformer residual stream. A relative entropy coverage
term describes how gradient energy is distributed across depth.

The method is named **FE-E (Finite-Element and Entropy)**. This repository is a research record, not a
mature training recipe. The current evidence does not support FE-E as a universal optimizer or default
regularizer. In the latest 128-layer, single-seed dose-by-propagation-noise experiment, FE-E reached the
confirmed 99% token-accuracy endpoint earlier in 3 of 12 paired conditions and later in 9. Positive
results were limited to persistent middle-layer energy concentration and a low-dose, high-frequency
perturbation condition. A more defensible direction is to use stiffness, mass, and entropy first as
propagation observers, then test whether they predict future instability well enough to trigger rare,
reversible interventions.

> **Private research preview · 2026-08-07**  
> Positive findings, negative findings, invalid runs, active stops, and method changes are retained.
> No single-seed result may be extrapolated into a claimed benefit for 7B-scale language-model training.

## Repository guide

- [Bilingual publication map](docs/bilingual_publication_map.md): paired Chinese/English documents,
  shared evidence, and synchronization rules.
- [Research process](docs/en/research_process.md): how the project moved from an always-on regularizer
  to conditional observation and intervention.
- [Methodology](docs/en/methodology.md): hypotheses, controls, endpoints, logging, gating, and evidence
  levels.
- [Current research status](docs/en/research_status.md): supported, unsupported, and untested claims.
- [Experiment registry](docs/en/experiment_registry.md): formal, exploratory, stopped, and invalid runs.
- [Reproducibility](REPRODUCIBILITY_EN.md): PyTorch and MLX validation and result reconstruction.
- [Data and artifacts](DATA_AND_ARTIFACTS_EN.md): raw logs, analyses, figures, papers, and archives.
- [License scope](LICENSE_SCOPE.md): Apache-2.0 for code; CC BY 4.0 for research prose, figures, and data.
- [License policy](LICENSE_POLICY_EN.md): private visibility, reuse, patent, and third-party boundaries.
- [Private release checklist](docs/en/private_github_release_checklist.md): local and future GitHub steps.
- [Private publication report](PRIVATE_REPO_PREP_REPORT_EN.md): audited candidate set, exclusions, and publication status.
- [English preprint](paper/fe_e_preprint.md) and
  [Chinese research essay](output/markdown/刚度约束和信息熵对深度学习的反作用及引申.md).

The full mathematical audit is in the [Chinese technical report](docs/research_report.md). The early
five-seed comparison is summarized in [direct comparison results](results/direct_compare_summary.md),
and the latest conditional experiment is documented in the
[dose-by-propagation-noise report](docs/mlx_d128_s47_fee_dose_noise_acc99_report.md). The English
experiment registry links every detailed report without duplicating raw evidence.

## What FE-E constrains

Given residual-stream nodes `states = [h_0, ..., h_L]` and task loss `L_task`, FE-E acts on the adjoint
field `g_l = ∂L_task/∂h_l`:

- the stiffness term penalizes abrupt depth-wise changes in the adjoint field;
- the mass term anchors its integrated energy;
- the entropy term constrains concentration relative to finite-element quadrature weights.

Because the penalty depends on task gradients, exact parameter optimization requires second-order
automatic differentiation. Detached gradients are valid diagnostics but cannot train this regularizer.

## Current evidence in brief

- Analytic tests and small-model ablations confirm that FE-E changes roughness, energy, and depth
  concentration in the intended directions.
- Better propagation diagnostics do **not** reliably imply faster task learning.
- FE-E has not shown a general advantage over Gradient Smoothing or AdamW from 24 to 192 layers.
- A single FE-E intervention step costs about `2.24×` an ordinary step in the latest MLX experiment.
- In the latest 12 paired conditions, 3 reached the endpoint earlier and 9 later with FE-E.
- The middle-layer energy-concentration condition produced a narrow positive signal at 3% and 5% dose,
  but only for one seed.
- A 192-layer isomorphic learning-rate-shock comparison had GSF confirm 99% accuracy 250 steps later
  than GS-SHAM.

These results support continued study of FE-E as an observer or narrowly activated depth-propagation
stabilizer. They do not support production deployment yet.

## Main directories

- `src/fe_entropy/`: PyTorch and MLX implementations, Gradient Smoothing, diagnostics, and models.
- `tests/`: analytic, differentiation, configuration, gating, sham-observer, and dose tests.
- `results/`: raw JSON/JSONL evidence, manifests, summaries, and structured analyses.
- `scripts/`: experiment analysis, plotting, preprint packaging, and readiness checks.
- `docs/`: methodology, status, experiment reports, stop notes, and bilingual navigation.
- `paper/`: editable preprint sources.
- `output/`: figures, PDFs, Markdown artifacts, and the policy for locally generated Release bundles.
- `website/`: source for the interactive research explainer; dependencies and builds are excluded.

## Quick validation

Python 3.10+ and PyTorch 2.2+ are required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

A minimal PyTorch experiment:

```bash
PYTHONPATH=src python3 -m fe_entropy.experiment \
  --steps 40 --layers 24 --width 32 \
  --lambda-stiffness 2 --lambda-energy 0.02 --lambda-entropy 2 \
  --entropy-lower 0.90 --entropy-upper 0.98 \
  --variants baseline,stiffness,energy,entropy,fe_e \
  --output results/example.json
```

Use the default `--depth-domain layer_index` for an ordinary Transformer. Use `unit_interval` only when
the residual step is consistently scaled as `O(1/L)` and depth is genuinely interpreted as refinement
of a fixed continuous-time interval.

On Apple Silicon with `mlx>=0.31`:

```bash
PYTHONPATH=src python3 -m fe_entropy.mlx_experiment \
  --layers 24 --steps 200 --seeds 31,47,59 \
  --variants baseline,gradient_smoothing,fe_e_always,fe_e_gated,gs_fe_e_gated
```

See [Reproducibility](REPRODUCIBILITY_EN.md) before treating default values as a frozen protocol.

## Minimal integration

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

The compatibility aliases `FEEntropyRegularizer` and `fe_entropy` remain available. New integrations
should use `FEERegularizer` and `fe_e`.

## Authorship and release status

The signed English preprint lists **XUEZHENG WANG** as author. An anonymous Markdown version is retained
separately for blind review. Software, tests, experiment, and analysis code are Apache-2.0; papers,
research documentation, figures, and program-generated experiment data are CC BY 4.0. See
[License Scope](LICENSE_SCOPE.md). GitHub Private visibility controls access but does not revoke rights
already granted to lawful recipients by these licenses.
