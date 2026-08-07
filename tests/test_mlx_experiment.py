import unittest
from pathlib import Path
import tempfile

import mlx.core as mx

from fe_entropy.mlx_experiment import (
    ExperimentConfig,
    FEEObserver,
    TinyTransformer,
    _variant_flags,
    fe_terms,
    hidden_adjoint_field,
    parse_args,
    reverse_sequence_loss,
    summarize,
    window_smooth,
)


class MLXFEEExperimentTests(unittest.TestCase):
    def test_adamw_sham_observer_matches_gated_fee_except_intervention(self):
        self.assertEqual(_variant_flags("adamw_observer_control"), (False, False, True))
        self.assertEqual(_variant_flags("fe_e_gated"), (False, True, True))

    def test_evaluation_batches_cli_argument(self):
        args = parse_args(["--evaluation-batches", "5"])
        self.assertEqual(args.evaluation_batches, 5)

    def test_summary_without_adamw_uses_first_method_as_time_reference(self):
        config = ExperimentConfig(steps=10)

        def result(variant, seconds):
            return {
                "variant": variant,
                "seed": 31,
                "status": "completed",
                "evaluation_loss": 3.0,
                "evaluation_accuracy": 0.1,
                "mean_step_seconds": seconds,
                "timed_training_seconds": 10.0 * seconds,
                "regularized_fraction": 0.0,
                "probe_fraction": 0.0,
                "peak_memory_bytes": 1024,
                "max_parameter_gradient_norm": 1.0,
                "final_stiffness_normalized": 1.0,
                "final_mass_energy": 1.0,
                "final_relative_coverage": 1.0,
                "evaluation_history": [],
            }

        with tempfile.TemporaryDirectory() as directory:
            summary = summarize(
                config,
                [result("gradient_smoothing", 2.0), result("fe_e_always", 3.0)],
                Path(directory),
            )
        self.assertEqual(summary["time_reference_variant"], "gradient_smoothing")
        self.assertEqual(summary["methods"]["gradient_smoothing"]["time_ratio"]["mean"], 1.0)
        self.assertEqual(summary["methods"]["fe_e_always"]["time_ratio"]["mean"], 1.5)

    def test_window_smoothing_boundaries(self):
        values = [mx.array([0.0]), mx.array([2.0]), mx.array([4.0])]
        smoothed = window_smooth(values, 0.2)
        mx.eval(smoothed)
        self.assertAlmostEqual(float(smoothed[0].item()), 0.2, places=6)
        self.assertAlmostEqual(float(smoothed[1].item()), 2.0, places=6)
        self.assertAlmostEqual(float(smoothed[2].item()), 3.8, places=6)

    def test_constant_field_has_zero_stiffness_and_full_coverage(self):
        gradients = [mx.ones((2, 3, 4)) for _ in range(5)]
        terms = fe_terms(gradients)
        mx.eval(terms)
        self.assertAlmostEqual(float(terms["stiffness_raw"].item()), 0.0, places=7)
        self.assertAlmostEqual(float(terms["relative_coverage"].item()), 1.0, places=6)

    def test_persistent_harm_gate_ignores_first_anomaly_then_schedules_next_step(self):
        config = ExperimentConfig(
            steps=10,
            observer_calibration_steps=2,
            diagnostic_every=1,
            observer_on_threshold=1.0,
            observer_persistence_window=2,
            observer_persistence_required=2,
            observer_metric_votes_required=2,
            observer_harm_consecutive=2,
        )
        observer = FEEObserver(config)
        observer.observe(0, 0.1, 1.0, 0.95, 1.0, 1.0, 0.50)
        self.assertFalse(observer.should_apply())
        first = observer.observe(2, 10.0, 0.01, 0.20, 10.0, 1.1, 0.45)
        self.assertTrue(first["observer_propagation_abnormal"])
        self.assertFalse(first["observer_confirmed_harm"])
        self.assertFalse(observer.should_apply())
        second = observer.observe(3, 12.0, 0.01, 0.20, 12.0, 1.2, 0.40)
        self.assertTrue(second["observer_confirmed_harm"])
        self.assertTrue(observer.should_apply())
        self.assertEqual(observer.state, "CONFIRMED_HARM")
        self.assertAlmostEqual(observer.intervention_reference_mass, 1.0)

    def test_persistent_propagation_without_task_damage_does_not_trigger(self):
        config = ExperimentConfig(
            steps=10,
            observer_calibration_steps=2,
            observer_on_threshold=0.1,
            observer_off_threshold=0.05,
            observer_persistence_window=2,
            observer_persistence_required=2,
            observer_metric_votes_required=2,
            observer_harm_consecutive=2,
        )
        observer = FEEObserver(config)
        observer.observe(0, 0.1, 1.0, 0.95, 1.0, 1.0, 0.50)
        observer.observe(2, 10.0, 0.01, 0.20, 10.0, 0.9, 0.55)
        result = observer.observe(3, 12.0, 0.01, 0.20, 12.0, 0.8, 0.60)
        self.assertTrue(result["observer_persistent_propagation"])
        self.assertFalse(result["observer_persistent_damage"])
        self.assertFalse(observer.should_apply())

    def test_recovered_current_probe_does_not_trigger_from_stale_history(self):
        config = ExperimentConfig(
            steps=10,
            observer_calibration_steps=2,
            observer_on_threshold=1.0,
            observer_persistence_window=3,
            observer_persistence_required=2,
            observer_metric_votes_required=2,
            observer_harm_consecutive=2,
        )
        observer = FEEObserver(config)
        observer.observe(0, 0.1, 1.0, 0.95, 1.0, 1.0, 0.50)
        observer.observe(2, 10.0, 0.01, 0.20, 10.0, 1.1, 0.45)
        observer.observe(3, 12.0, 0.01, 0.20, 12.0, 1.2, 0.40)
        result = observer.observe(4, 0.1, 1.0, 0.95, 1.0, 1.3, 0.35)
        self.assertTrue(result["observer_persistent_propagation"])
        self.assertFalse(result["observer_propagation_abnormal"])
        self.assertTrue(result["observer_persistent_damage"])
        self.assertFalse(result["observer_confirmed_harm"])
        self.assertFalse(observer.should_apply())

    def test_benign_drift_updates_rolling_baseline(self):
        config = ExperimentConfig(
            steps=10,
            observer_calibration_steps=2,
            observer_on_threshold=1.0,
            observer_baseline_window=4,
            observer_adaptive_baseline=True,
        )
        observer = FEEObserver(config)
        observer.observe(0, 0.1, 1.0, 0.95, 1.0, 1.0, 0.50)
        shifted = observer.observe(
            2, 10.0, 0.01, 0.20, 10.0, 0.99, 0.51
        )
        adapted = observer.observe(
            3, 10.0, 0.01, 0.20, 10.0, 0.98, 0.52
        )
        self.assertTrue(shifted["observer_propagation_abnormal"])
        self.assertTrue(shifted["observer_baseline_adapted"])
        self.assertFalse(adapted["observer_propagation_abnormal"])

    def test_damaging_probe_is_excluded_from_rolling_baseline(self):
        config = ExperimentConfig(
            steps=10,
            observer_calibration_steps=2,
            observer_on_threshold=1.0,
            observer_baseline_window=4,
            observer_adaptive_baseline=True,
        )
        observer = FEEObserver(config)
        observer.observe(0, 0.1, 1.0, 0.95, 1.0, 1.0, 0.50)
        result = observer.observe(
            2, 10.0, 0.01, 0.20, 10.0, 1.1, 0.45
        )
        self.assertTrue(result["observer_damage_event"])
        self.assertFalse(result["observer_baseline_adapted"])
        self.assertEqual(result["observer_baseline_samples"], 1)

    def test_fast_learning_phase_suppresses_otherwise_confirmed_harm(self):
        config = ExperimentConfig(
            steps=20,
            observer_calibration_steps=2,
            observer_probe_every=1,
            observer_on_threshold=1.0,
            observer_persistence_window=2,
            observer_persistence_required=2,
            observer_metric_votes_required=2,
            observer_harm_consecutive=2,
            observer_phase_improvement_tolerance=0.02,
        )
        observer = FEEObserver(config)
        observer.cheap_update(3.0, 1.0)
        observer.observe(0, 0.1, 1.0, 0.95, 1.0, 1.0, 0.50)
        observer.cheap_update(2.0, 10.0)
        observer.observe(2, 10.0, 0.01, 0.20, 10.0, 1.1, 0.45)
        observer.cheap_update(1.0, 12.0)
        result = observer.observe(3, 12.0, 0.01, 0.20, 12.0, 1.2, 0.40)
        self.assertTrue(result["observer_persistent_propagation"])
        self.assertTrue(result["observer_persistent_damage"])
        self.assertTrue(result["observer_phase_improving"])
        self.assertFalse(result["observer_confirmed_harm"])
        self.assertFalse(observer.should_apply())

    def test_manual_reverse_vjp_matches_suffix_gradients(self):
        config = ExperimentConfig(
            steps=2,
            layers=2,
            width=8,
            heads=2,
            sequence_length=4,
            batch_size=2,
        )
        mx.random.seed(5)
        model = TinyTransformer(config)
        tokens = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        states = model.forward_states(tokens)
        manual = hidden_adjoint_field(model, states, tokens)
        reference = []
        for start in range(config.layers + 1):
            def suffix_loss(hidden, start=start):
                for block in model.blocks[start:]:
                    hidden = block(hidden)
                return reverse_sequence_loss(model.logits_from_hidden(hidden), tokens)

            reference.append(mx.grad(suffix_loss)(states[start]))
        errors = [mx.max(mx.abs(left - right)) for left, right in zip(manual, reference)]
        mx.eval(errors)
        self.assertLess(max(float(error.item()) for error in errors), 1e-5)


if __name__ == "__main__":
    unittest.main()
