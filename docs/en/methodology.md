# Research Methodology

[中文](../methodology.md) | **English**

## 1. Object of study

FE-E constrains neither ordinary forward hidden states nor parameter gradients directly. It acts on the
adjoint field `g_l = ∂L_task/∂h_l` at residual-stream nodes. Stiffness measures depth-wise variation,
mass anchors integrated energy, and entropy measures relative depth coverage. Exact optimization requires
a higher-order differentiation graph.

## 2. Falsifiable hypotheses

The broad proposal is divided into four testable hypotheses:

1. FE-E changes adjoint roughness, total energy, and concentration across depth.
2. Improvements in propagation metrics predict improvements in task learning.
3. FE-E has net benefit only under specific abnormality types and doses.
4. Stiffness, mass, and entropy are more valuable as observers than as continuous controllers.

Only hypothesis 1 has stable support. Hypothesis 2 fails in several environments. Hypothesis 3 has
conditional single-seed evidence. Hypothesis 4 requires prospective validation.

## 3. Method and control names

- `ADW`: lean AdamW.
- `ADF`: AdamW plus gated FE-E.
- `GS`: Gradient Smoothing.
- `GSF`: Gradient Smoothing plus gated or fixed-dose FE-E.
- `SHAM observer`: identical observation, sentinel, and gate decisions without an FE-E update.

An isomorphic sham is preferred over comparing an observer-bearing FE-E arm directly with a lean
baseline when making causal claims.

## 4. Task and model

The principal task is a program-generated short-sequence transformation used to observe learning
transitions at low cost. It is not a language-model pretraining benchmark. Experiments use small Pre-LN
Transformers, with formal or exploratory evidence from 24 to 192 layers. The 1024-layer attempt is only
an active-stop audit of an untuned extrapolation.

Results must not be extrapolated to 7B-model perplexity, throughput, memory, or stability. The small task
is useful for reproducible counterexamples and mechanism probes.

## 5. Seeds and frozen protocols

A random seed controls initialization, synthetic data, and frozen sampling schedules. Seed numbers have
no ranking and do not represent model width. Development and confirmation seeds should be separated.
Compared methods must share seeds, data, validation batches, and shock schedules.

Before a formal run, freeze:

- depth, width, head count, and residual scaling;
- optimizer and learning rate;
- validation batches and checkpoint frequency;
- perturbation type, magnitude, and active interval;
- FE-E dose, maximum gradient ratio, cooldown, and intervention table;
- primary endpoint, safety ceiling, and failure conditions.

## 6. Primary endpoint

The current primary endpoint is token accuracy ≥99% on a fixed validation set, confirmed at three
consecutive checkpoints. A numerical limit such as 8000 steps is a safety ceiling, not a performance
target.

Also report:

- first steps reaching 50%, 90%, and 99%;
- transition width from sustained 10% to first 90%;
- updates to confirmed 99%;
- fixed-horizon accuracy deficit;
- total optimizer time and mean ordinary/intervention step time.

Short-term loss slope describes only a local trajectory and must be interpreted with the transition start.

## 7. Propagation and mechanism metrics

Recorded or derived measures include normalized stiffness, FE mass energy, relative entropy coverage,
cosine between FE-E and task gradients, applied FE-E-to-task gradient ratio, intervention count and rate,
cooldown state, and ordinary/intervention wall-clock time.

These are mechanism measurements, not automatic quality labels. High-frequency variation may be noise or
necessary reorganization; energy concentration may be collapse or useful specialization.

## 8. Logging and integrity

MLX runs write JSONL with a unique start/end record, zero-based contiguous steps, configuration, task and
validation metrics, gate state, interventions, timing, failures, and termination reasons. Run directories
contain a manifest, per-run summaries, and aggregate summaries.

Analysis scripts verify step continuity, unique start/end records, frozen intervention schedules,
validation intervals, finite values, failure records, and reconstruction of the primary endpoint from raw
validation events rather than trusting only summaries.

## 9. Causal controls

Control priority is:

1. isomorphic sham observer in the same execution framework;
2. paired seed, data, validation, and perturbation schedules;
3. trajectory audit before the first intervention;
4. balanced or randomized execution order;
5. temperature, load, and step-time audit.

If trajectories diverge materially before the first intervention, report only association or exploratory
differences. If system thermal state changes, wall-clock comparison is invalid, while fixed-update
evidence may remain usable.

## 10. Evidence levels

- **Mechanism smoke test:** validates implementation and direction, not superiority.
- **Exploratory single seed:** generates candidate conditions, not stable benefit.
- **Isomorphic single-seed control:** supports local causal inference, not generalization.
- **Frozen multi-seed comparison:** estimates reliability and paired distributions.
- **Cross-task, cross-scale validation:** required before discussing engineering generality.

FE-E as a whole currently lies between exploratory and isomorphic single-seed evidence. Some early GS
comparisons used multiple seeds, but none is at realistic language-model scale.

## 11. Active stops and negative results

An active stop records its trigger, completed runs, incomplete logs, and the evidence measures that remain
valid. Invalid trials do not enter formal aggregates and are not deleted. Positive and negative results
follow the same logging and figure standards.

## 12. Minimum next experiment

The next round should answer only two questions:

1. Does the 0%/1%/3%/5% dose pattern replicate under middle-layer energy concentration across at least
   five preregistered seeds?
2. Can stop-gradient stiffness, mass, and entropy predict transition delay or validation degradation
   32–128 steps ahead?

Only if the second result has stable out-of-sample predictive value should a low-dose, one-step causal
intervention with sham and rollback windows be designed.

