# Current Results and Claim Boundaries

[中文](../research_status.md) | **English**

Updated: 2026-08-07.

## One-sentence conclusion

FE-E demonstrably changes deep adjoint propagation, but it has not been shown to be a universal
optimizer. Its plausible role is a low-frequency stabilizer for specific abnormalities such as persistent
depth-wise energy concentration, or first as a propagation observer.

## Claim matrix

| Claim | Current evidence | Status |
|---|---|---|
| FE-E reduces inter-layer adjoint roughness | Analytic tests and 24-layer ablation | Supported |
| Mass and entropy terms change energy magnitude and distribution | Ablations and per-step diagnostics | Supported |
| Smoother propagation necessarily converges faster | Multiple negative results | Rejected |
| FE-E is generally better than GS or AdamW | 24–192-layer comparisons | Not supported |
| FE-E may have stronger per-update effects at fixed update count | Early five-seed comparison | Limited support |
| FE-E has higher compute efficiency | Double-backpropagation and timing evidence | Not supported |
| GSF can recover some difficult seeds | Seed-31 gate experiment | Candidate signal |
| Middle-layer energy concentration has a narrow dose window | Seed 47; positive 3%/5% results | Single-seed candidate |
| Stiffness, mass, and entropy predict future instability | No prospective prediction experiment | Untested |
| Results generalize to a 7B language model | No evidence at that scale | Prohibited extrapolation |

## Completed stage outputs

1. Corrected mathematical definitions: layer-index geometry, consistent FE mass, Rayleigh stiffness, and
   relative entropy coverage.
2. Exact PyTorch double-backpropagation prototype, spectral diagnostics, and analytic/counterexample tests.
3. Paper-level Gradient Smoothing and AdamW baselines.
4. Apple MLX per-step logging and gating state machine.
5. Experiments at 24, 48, 96, 128, and 192 layers; active-stop audit at 1024 layers.
6. A learning-transition endpoint: token accuracy ≥99% at three consecutive checks.
7. Isomorphic sham observer, learning-rate shock, and structural propagation-noise probes.
8. English preprint, anonymous Markdown, Chinese essay, figures, and interactive explainer.

## Most important findings

- Latest fixed-dose 128-layer study: FE-E was earlier in 3 of 12 paired conditions and later in 9.
- No-noise and global-energy-amplification environments: every FE-E dose was slower.
- Middle-layer energy concentration: 3% was 128 steps earlier and 5% was 64 steps earlier; this remains a
  single-seed screening result.
- One FE-E intervention step cost about 2.24 ordinary steps.
- Mean FE-E/task-gradient cosine was negative in all environments, indicating that FE-E usually resists
  the task gradient in exchange for regularity.
- In a 192-layer, seed-47 isomorphic learning-rate-shock comparison, GSF confirmed 250 steps later than
  GS-SHAM.

## Statements that must not appear as conclusions

- “FE-E solves exploding and vanishing gradients.”
- “FE-E already outperforms Gradient Smoothing or AdamW.”
- “A 3% intervention rate is optimal.”
- “Higher entropy means information is better preserved.”
- “A steeper short-term loss slope means the model learns faster.”
- “The small-model experiment proves a benefit for 7B models.”

## Decision threshold for further controller work

Before continuing the controller route, require a preregistered multi-seed replication of the
middle-layer concentration window; out-of-sample prediction of future transition delay; an isomorphic
sham, balanced execution order, and system-load audit; net wall-clock benefit after diagnostic and
double-backpropagation cost; and replication on at least one task closer to language modeling.

If any of these fails, observer research should take priority over increasingly complex gating.

