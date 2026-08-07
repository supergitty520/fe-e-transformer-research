# Experiment Registry

[中文](../experiment_registry.md) | **English**

This table separates formal evidence, exploratory results, active stops, and invalid runs. Detailed
configuration is defined by each manifest and report. Detailed reports currently retain their original
language; raw logs, code, numeric results, and paths are shared by both publication tracks.

| Stage | Scale and protocol | Primary question | Status | Main entry |
|---|---|---|---|---|
| PyTorch ablation | 24 layers, 3 seeds, 40 steps | Are stiffness, mass, and entropy complementary? | Mechanism experiment | [Technical report](../research_report.md) |
| Direct GS comparison | 24 layers, 5 seeds, 200 steps | Fixed updates versus backward-compute budget | Small formal comparison | [Summary](../../results/direct_compare_summary.md) |
| Depth audit | 48 layers, 3 seeds | Do early findings persist with depth? | Descriptive | [Summary](../../results/direct_compare_d48_summary.md) |
| MLX 24 layers | Clean/LR shock, 3 seeds | Six-way engineering comparison | Formal; periodic pulse rejected | [Report](../mlx_d24_experiment_report.md) |
| MLX 96 layers | Clean/LR shock, 3 seeds | Depth scaling and equal-time budget | Formal | [Report](../mlx_d96_experiment_report.md) |
| MLX 1024 layers | Frozen single-seed extrapolation | Ultra-deep feasibility | Actively stopped; no performance claim | [Stop note](../mlx_d1024_exploratory_stop_note.md) |
| MLX 192 layers | 250 steps, 3 seeds, 5-batch validation | Short-horizon deep stability | Formal descriptive | [Report](../mlx_d192_s3_step250_eval5_report.md) |
| MLX 128 long run | 1000 steps, 1 seed, 8 validation checks | Long-horizon AUC and transient shock | Exploratory | [Report](../mlx_d128_s1_step1000_eval8_report.md) |
| 5000-step three-way | Serial single-seed run | Loss threshold and wall-clock efficiency | Actively stopped; timing invalid | [Stop audit](../mlx_d128_s1_step5000_threecase_stop_note.md) |
| Persistent gate v3 | 128 layers, confirmed 99% endpoint | Can rare gating shorten transition? | Exploratory positive signal | [Report](../mlx_d128_acc99_persistent_gate_v3_report.md) |
| Three-seed GSF/GS/AdamW | Seeds 31/47/59, 99% endpoint | Hard-seed reliability and overall performance | Formal 3-seed summary after early stop | [Report](../mlx_d128_acc99_s3_fee_gs_adamw_report.md) |
| AdamW isomorphic sham | Seed 31 | Separate observer and FE-E effects | Single-seed isomorphic mechanism control | [Report](../mlx_d128_s31_adamw_fee_vs_sham_report.md) |
| 192-layer isomorphic shock | Seed 47, persistent LR shock | Does deep GSF beat GS? | Credible negative mechanism case | [Report](../mlx_d192_s47_acc99_gs_sham_vs_gsf_lrshock_report.md) |
| Dose × structural noise | 128 layers, 4 environments × 4 doses, seed 47 | Boundary conditions for FE-E | Latest conditional single-seed experiment | [Report](../mlx_d128_s47_fee_dose_noise_acc99_report.md) |

## Invalid-run policy

Development, smoke, missing-shock, or incomplete runs remain under `results/`, but directory presence
does not qualify them for formal aggregation. Analysis scripts use explicit inclusion lists and reports
state exclusion reasons.

