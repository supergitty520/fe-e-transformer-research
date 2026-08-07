# FE-E Apple MLX 24-layer experiment attachment

This attachment contains the native Apple MLX implementation and the complete
three-seed logs for the clean and learning-rate-stress experiments.

## Main command

```bash
PYTHONPATH=src python3 -m fe_entropy.mlx_experiment \
  --layers 24 --steps 200 --seeds 31,47,59 \
  --variants baseline,gradient_smoothing,fe_e_always,fe_e_periodic,fe_e_gated,gs_fe_e_gated \
  --diagnostic-every 20 --observer-probe-every 8 --evaluation-every 25 \
  --observer-calibration-steps 24 --lambda-stiffness 0.1 \
  --lambda-energy 0.02 --lambda-entropy 2.0 \
  --entropy-lower 0.5 --entropy-upper 0.98 \
  --gated-fee-gradient-ratio 0.5
```

Add the following options for the stress condition:

```bash
--stress-step 80 --stress-duration 8 --stress-lr-multiplier 5
```

Requirements: Python 3.10+, `mlx>=0.31`, and an Apple Silicon Mac. No dataset
download is required.

Each log is JSONL with one run-start record, 200 train-step records, and one
run-end record. The manifest records the full command, configuration, MLX and
system versions, Metal device information, and the SHA-256 of the experiment
source file.
