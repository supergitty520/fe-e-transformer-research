# FE-E Research Process

[中文](../research_process.md) | **English**

This document records how FE-E evolved from a mathematical proposal into an engineering falsification
program. It follows the order of decisions rather than retaining only favorable runs. All dates refer to
August 2026.

## 1. Initial question: does deep propagation need an explicit stabilizer?

The project began with an analogy: treat Transformer depth as a one-dimensional mesh and the gradient of
the task loss with respect to residual-stream nodes as an adjoint field on that mesh. Finite-element
stiffness can penalize abrupt inter-layer changes, the mass matrix can anchor total gradient energy, and
entropy can describe whether energy is concentrated in a few layers or directions. This produced the
name FE-E: Finite-Element and Entropy.

The initial hypothesis—that smoother, stable, less concentrated adjoint propagation should reduce
explosion, vanishing, and representation distortion—was separated into two claims:

1. FE-E changes the geometry of propagation.
2. A more regular propagation geometry makes the task learn faster.

Experiments consistently support the first claim, but not the second.

## 2. Mathematical audit: correcting the continuous-depth interpretation

An ordinary Transformer cannot automatically be mapped to a fixed interval `[0,1]`. If its residual
step does not scale as `1/L`, the more honest geometry is `z_l=l`: increasing depth lengthens the
propagation interval instead of refining a fixed mesh. The implementation therefore supports
`layer_index` and `unit_interval`, and permits the latter only with consistent residual scaling.

A second correction concerned derivative order. FE-E depends on
`g_l = ∂L_task/∂h_l`; optimizing model parameters through that penalty introduces
`∂g_l/∂θ`. Exact training therefore requires double backpropagation or an equivalent
Hessian-vector path. Gradient norms captured by detached hooks are observers, not trainable penalties.

A third correction concerned entropy. Linear finite elements assign different quadrature weights to
endpoints and interior nodes. Maximizing ordinary discrete Shannon entropy creates endpoint bias. The
implementation instead measures relative coverage against the finite-element integration measure and
uses an entropy band rather than unconditional maximization.

## 3. PyTorch prototype: proving control, not improved learning

A 24-layer, width-32 synthetic sequence-reversal task validated differentiable stiffness, mass, and
entropy terms. Stiffness substantially reduced depth-wise roughness, but by itself further attenuated
adjoint energy and hurt the task. All three terms together improved propagation diagnostics without
reliably outperforming the mass-only condition or task baseline.

A 48-layer stress test exposed the reaction of strong constraints: without warmup, the first-step
parameter-gradient peak increased sharply. Linear warmup controlled the peak but did not guarantee task
recovery. Two engineering rules followed:

- propagation diagnostics and task outcomes must be reported separately;
- any stability benefit must include the cost of double backpropagation.

## 4. Gradient Smoothing comparison: update efficiency versus compute efficiency

A paper-level Gradient Smoothing implementation was then added. In the early five-seed comparison,
FE-E showed stronger per-update control at a fixed number of parameter updates, while Gradient Smoothing
was better under an approximate backward-compute budget. “Faster descent” was therefore split into
fixed-update, fixed-wall-clock, and time-to-task-endpoint measures.

Serial wall-clock comparisons also proved fragile. Device temperature, background load, execution order,
and compilation caches changed step time. Later reports use wall-clock evidence only when system state is
comparable and always retain update-count results.

## 5. MLX depth scaling: 24, 96, 192, and 1024 layers

The Apple MLX prototype writes per-step configuration, task metrics, observer state, intervention events,
timing, and termination reasons to JSONL. The 24- and 96-layer studies compared AdamW, Gradient
Smoothing, always-on FE-E, gated FE-E, and combined variants. A fixed-period FE-E pulse showed no stable
value and was removed from the main protocol.

At 192 layers, depth conditions and learning-rate shocks became central. In the seed-47 isomorphic
observer comparison, GSF confirmed 99% token accuracy 250 steps later than GS-SHAM, providing a
relatively clean negative mechanism result.

The 1024-layer single-seed run was actively stopped because frozen parameters had not been retuned for
the ultra-deep regime. Continuing would have produced expensive but uninterpretable data. The stop
record remains part of the evidence.

## 6. Endpoint redesign: from an arbitrary loss threshold to a learning transition

The early endpoint `loss ≤ 0.001` had no stable production interpretation for language models. It was
replaced by token accuracy ≥99% on a fixed validation set, confirmed at three consecutive checkpoints.
Training was decomposed into a pre-transition plateau, a 10%-to-90% transition, and a confirmation phase.

This decomposition explains a central counterexample: FE-E can steepen the short-term loss slope inside
the transition while delaying entry into the transition by even more. The total confirmation step is
then worse. Local slope is not a sufficient endpoint.

## 7. Gate development: persistent harm instead of periodic intervention

Gate variants included a fixed four-step period, 96-step cooldown, 48-step cooldown, and persistent-harm
conditions. The final development protocol required simultaneous abnormalities in multiple propagation
metrics, persistence within a recent window, repeated sentinel-task degradation, and protection against
interrupting a rapid learning transition. Each intervention lasted one step and was followed by cooldown.

The early seed-31 combined run reached the endpoint much earlier, but tiny Metal numerical divergence was
already present before the first intervention. A later three-seed comparison found all FE-E interventions
in seed 31, where plain GS failed; seeds 47 and 59 never triggered and exactly tied GS. This supports a
possible hard-seed recovery role, not an overall advantage over AdamW.

## 8. Isomorphic sham observer: separating observation cost from intervention

An isomorphic sham observer performs the same diagnostics, sentinel probes, and gate decisions but does
not apply the FE-E update. In the seed-31 AdamW pair, gated FE-E confirmed one validation interval before
the sham. However, it matched the historical endpoint of lean AdamW. FE-E may have had a local effect
relative to sham without producing a net production benefit over the simpler baseline.

## 9. Fixed dose by structural noise: searching for boundary conditions

The final 128-layer experiment froze seed 47, validation batches, perturbation intervals, and nested
1%/3%/5% intervention schedules across four environments: no noise, high-frequency inter-layer
perturbation, global residual-energy amplification, and middle-layer energy concentration.

FE-E was earlier in 3 of 12 pairs and later in 9. Every dose was slower under no noise and global energy
amplification. The 3% and 5% doses were earlier under middle-layer concentration; only 1% was slightly
earlier under high-frequency perturbation. One intervention step cost about 2.24 ordinary steps, so an
update-count lead is not automatically a wall-clock lead.

The narrowed hypothesis is that FE-E may have a small dose window under persistent, local,
depth-distribution instability. This window has not been confirmed across seeds.

## 10. Current direction: FE-E as an observer first

FE-E is no longer described as a default optimizer. Stiffness, mass, and entropy are better treated first
as stop-gradient propagation observables. The next question is whether they predict loss degradation,
transition delay, or training failure tens of steps ahead. Intervention should return only after
predictive value, false-positive rate, and net benefit pass prospective tests.

This also motivates an educational and metacognitive analogy: smoothing, prompting, or stabilizing a
process does not imply that understanding occurs earlier. The relevant distinction is between productive
exploration and persistent instability—and whether intervention helps more than it interrupts.

## 11. Stage outputs

- trainable PyTorch and MLX FE-E prototypes;
- Gradient Smoothing, AdamW, and isomorphic sham controls;
- evidence at 24, 48, 96, 128, and 192 layers, plus an actively stopped 1024-layer run;
- per-step JSONL logs, environment manifests, analysis scripts, and PNG/SVG figures;
- an English preprint, anonymous Markdown manuscript, Chinese essay, and interactive explainer;
- preserved negative findings, invalid trials, and stop rationales.

The next stage should not add more single-seed variants. It should test prospective observer value and
replicate the middle-layer concentration dose window with fewer, stricter, preregistered experiments.

