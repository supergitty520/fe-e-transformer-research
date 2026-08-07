import unittest

import mlx.core as mx

from fe_entropy.mlx_dose_response import (
    forward_states_profiled,
    hidden_adjoint_field_profiled,
    intervention_schedules,
    residual_profile,
)
from fe_entropy.mlx_experiment import (
    ExperimentConfig,
    TinyTransformer,
    reverse_sequence_loss,
)


class MLXDoseResponseTests(unittest.TestCase):
    def test_intervention_schedules_are_nested_and_exact_per_full_block(self):
        schedules = intervention_schedules(800, 20260806)
        self.assertTrue(schedules["gsf_q01"] < schedules["gsf_q03"])
        self.assertTrue(schedules["gsf_q03"] < schedules["gsf_q05"])
        self.assertNotIn(0, schedules["gsf_q05"])
        for start in range(0, 800, 100):
            stop = start + 100
            counts = [
                sum(start <= step < stop for step in schedules[variant])
                for variant in ("gsf_q01", "gsf_q03", "gsf_q05")
            ]
            self.assertEqual(counts, [1, 3, 5])

    def test_high_frequency_profile(self):
        profile = residual_profile(128, "high_frequency")
        mx.eval(profile)
        self.assertAlmostEqual(float(mx.mean(profile).item()), 1.0, places=7)
        self.assertAlmostEqual(float(profile[0].item()), 1.25, places=7)
        self.assertAlmostEqual(float(profile[1].item()), 0.75, places=7)

    def test_energy_profile(self):
        profile = residual_profile(128, "energy")
        mx.eval(profile)
        self.assertAlmostEqual(float(mx.min(profile).item()), 1.5, places=7)
        self.assertAlmostEqual(float(mx.max(profile).item()), 1.5, places=7)

    def test_concentration_profile(self):
        profile = residual_profile(128, "concentration")
        mx.eval(profile)
        values = [float(value.item()) for value in profile]
        self.assertEqual(sum(abs(value - 2.5) < 1e-7 for value in values), 16)
        self.assertAlmostEqual(values[0], 88 / 112, places=6)
        self.assertAlmostEqual(float(mx.mean(profile).item()), 1.0, places=6)

    def test_unit_profile_matches_model_forward(self):
        config = ExperimentConfig(
            steps=2,
            layers=3,
            width=8,
            heads=2,
            sequence_length=4,
            batch_size=2,
        )
        mx.random.seed(5)
        model = TinyTransformer(config)
        tokens = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        direct = model.forward_states(tokens)
        profiled = forward_states_profiled(model, tokens, mx.ones((config.layers,)))
        errors = [mx.max(mx.abs(left - right)) for left, right in zip(direct, profiled)]
        mx.eval(errors)
        # The two algebraically equivalent residual expressions can differ by
        # one float32 rounding unit on Metal.
        self.assertLess(max(float(error.item()) for error in errors), 1e-6)

    def test_profiled_reverse_vjp_matches_suffix_gradients(self):
        config = ExperimentConfig(
            steps=2,
            layers=3,
            width=8,
            heads=2,
            sequence_length=4,
            batch_size=2,
        )
        mx.random.seed(7)
        model = TinyTransformer(config)
        tokens = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        profile = mx.array([1.25, 0.75, 1.5])
        states = forward_states_profiled(model, tokens, profile)
        manual = hidden_adjoint_field_profiled(model, states, tokens, profile)
        reference = []
        for start in range(config.layers + 1):
            def suffix_loss(hidden, start=start):
                for index in range(start, config.layers):
                    candidate = model.blocks[index](hidden)
                    hidden = hidden + profile[index] * (candidate - hidden)
                logits = model.logits_from_hidden(hidden)
                return reverse_sequence_loss(logits, tokens)

            reference.append(mx.grad(suffix_loss)(states[start]))
        errors = [mx.max(mx.abs(left - right)) for left, right in zip(manual, reference)]
        mx.eval(errors)
        self.assertLess(max(float(error.item()) for error in errors), 1e-5)


if __name__ == "__main__":
    unittest.main()
