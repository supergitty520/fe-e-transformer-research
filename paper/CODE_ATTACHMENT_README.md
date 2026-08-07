# FE-E code and results attachment

Author: **XUEZHENG WANG**  
Release: Preprint v0.1, August 2026

This archive accompanies *FE-E: Finite-Element and Entropy Control of Adjoint
Propagation in Deep Transformers*. It contains the exact implementation and raw
small-scale experiments reported in the preprint.

## Contents

- `src/fe_entropy/regularizer.py`: `FEERegularizer`, finite-element assembly,
  mass anchor, relative coverage, and the double-backpropagation objective.
- `src/fe_entropy/gradient_smoothing.py`: paper-aligned Gradient Smoothing
  AdamW comparator.
- `src/fe_entropy/model.py`: higher-order-autograd-friendly Pre-LN Transformer.
- `src/fe_entropy/experiment.py`: deterministic reverse-sequence experiment.
- `tests/`: analytic, numerical, compatibility, and automatic-differentiation tests.
- `results/`: raw JSON and human-readable aggregates used in the paper.
- `scripts/`: result aggregation, paper building, and container helpers.
- `paper/fe_e_preprint.md`: editable paper source.
- `MANIFEST-SHA256.txt`: SHA-256 digest of every archived file.

## Verify tests

Requires Python 3.10+ and PyTorch 2.2+:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The recorded environment uses PyTorch 2.13.0+cpu. A container command is given
in the root `README.md`.

## Reproduce the main comparison

The five confirmation files are:

```text
results/direct_compare_d24_seed31.json
results/direct_compare_d24_seed47.json
results/direct_compare_d24_seed59.json
results/direct_compare_d24_seed71.json
results/direct_compare_d24_seed89.json
```

They use 24 layers, width 32, four heads, 200 updates, AdamW learning rate
0.002, weight decay 0.01, Gradient Smoothing Standard/Proj with alpha 0.2, and
FE-E coefficients `(2, 0.02, 2)` with a ten-step warmup. See each JSON's
`config` field for the machine-readable protocol.

Rebuild the paired summary with:

```bash
python3 scripts/summarize_direct_compare.py \
  results/direct_compare_d24_seed31.json \
  results/direct_compare_d24_seed47.json \
  results/direct_compare_d24_seed59.json \
  results/direct_compare_d24_seed71.json \
  results/direct_compare_d24_seed89.json \
  --output results/direct_compare_summary.md
```

## Scope and license

The data are synthetic and generated locally. No private dataset or credential
is included. Software is Apache-2.0; papers, documentation, figures, and
program-generated experiment data are CC BY 4.0. The repository-level
`LICENSE_SCOPE.md` defines the authoritative path allocation.
