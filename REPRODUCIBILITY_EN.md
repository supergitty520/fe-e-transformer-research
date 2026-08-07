# Reproducibility Guide

[中文](REPRODUCIBILITY.md) | **English**

## Reproduction levels

This repository distinguishes three levels:

1. **Unit tests:** validate finite-element assembly, entropy, directional counterexamples, double
   backpropagation, and MLX configuration logic.
2. **Result reconstruction:** regenerate summaries, audits, and figures from committed JSON/JSONL.
3. **Retraining:** rerun frozen protocols in PyTorch or Apple MLX; results can vary by hardware and
   numerical backend.

## Python environment

Python 3.10+ is required. For the PyTorch prototype:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

For MLX on Apple Silicon:

```bash
python -m pip install -e '.[mlx]'
PYTHONPATH=src python -m unittest tests.test_mlx_experiment tests.test_mlx_dose_response -v
```

MLX is not a CUDA backend. A Windows RTX 3090 cannot run the MLX entry point directly. CUDA replication
requires an isomorphic PyTorch implementation of the MLX state machine and validation of log fields and
the trajectory before the first intervention.

## Minimal PyTorch experiment

```bash
PYTHONPATH=src python -m fe_entropy.experiment \
  --steps 40 --layers 24 --width 32 \
  --lambda-stiffness 2 --lambda-energy 0.02 --lambda-entropy 2 \
  --entropy-lower 0.90 --entropy-upper 0.98 \
  --variants baseline,stiffness,energy,entropy,fe_e \
  --output results/example.json
```

Use the default `--depth-domain layer_index` for an ordinary Transformer. Use `unit_interval` only
when the residual step scales as `O(1/L)`.

## Formal MLX entry point

```bash
PYTHONPATH=src python -m fe_entropy.mlx_experiment \
  --layers 24 --steps 200 --seeds 31,47,59 \
  --variants baseline,gradient_smoothing,fe_e_always,fe_e_gated,gs_fe_e_gated
```

The fixed-dose propagation-noise experiment is:

```bash
PYTHONPATH=src python -m fe_entropy.mlx_dose_response
```

Its current defaults are 128 layers, width 32, seed 47, four environments, nested 1%/3%/5% schedules,
validation every 32 steps on eight fixed batches, and three consecutive checks at ≥99% token accuracy.
Review source and manifest before a formal run; defaults are not permanently frozen protocols.

## Reconstructing key results

```bash
python scripts/analyze_mlx_d128_s47_fee_dose_noise_acc99.py
python scripts/analyze_mlx_d192_s47_gs_sham_vs_gsf_lrshock.py
python scripts/analyze_mlx_d128_s31_adamw_fee_vs_sham.py
python scripts/analyze_mlx_d128_acc99_s3_fee_gs_adamw.py
```

Scripts reconstruct endpoints and intervention events from JSONL instead of reading only final summaries.
Output locations are documented in [Data and Artifacts](DATA_AND_ARTIFACTS_EN.md).

## Numerical reproducibility limits

- The same seed is not bitwise identical across PyTorch, MLX, CPU, Metal, and CUDA.
- Even on one Metal device, tiny floating-point divergence can precede the first intervention.
- Wall-clock time depends on temperature, background load, execution order, and compilation caches.
- Reproduction should focus on paired directions, distributions, gate events, and endpoint intervals—not
  exact equality of every floating-point value.

## Integrity checks

After retraining, verify at minimum:

- one unique `run_start` and `run_end`;
- contiguous, non-duplicated steps and no unexplained failure;
- the frozen validation interval;
- interventions matching the manifest schedule;
- primary endpoint reconstructed from validation logs;
- incomplete and actively stopped runs excluded from formal aggregates.

## Papers and archives

```bash
python scripts/build_preprint.py
python scripts/package_preprint.py
```

PDF output is under `output/pdf/`. ZIP and adjacent `.sha256` files are generated locally under
`output/attachments/` for a future Release and remain outside the main Git history. A checksum verifies
one archive, not the entire repository.
