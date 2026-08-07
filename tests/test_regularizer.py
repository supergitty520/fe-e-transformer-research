import math
import unittest

import torch

from fe_entropy.regularizer import FEERegularizer, FEEntropyRegularizer, fe_entropy_terms
from fe_entropy.spectral import spectral_metrics


class FiniteElementAssemblyTests(unittest.TestCase):
    def test_constant_field_has_zero_stiffness_and_exact_mass(self) -> None:
        gradients = [torch.full((2, 3, 4), 2.0) for _ in range(5)]
        terms = fe_entropy_terms(gradients)
        self.assertAlmostEqual(float(terms.stiffness_raw), 0.0, places=7)
        self.assertAlmostEqual(float(terms.mass_energy), 4.0, places=6)
        self.assertAlmostEqual(float(terms.relative_coverage), 1.0, places=6)

    def test_linear_field_matches_continuum_energy(self) -> None:
        positions = torch.linspace(0.0, 1.0, 9)
        gradients = [torch.full((1, 1, 2), float(1.0 + 2.0 * z)) for z in positions]
        terms = fe_entropy_terms(gradients, positions=positions)
        expected_mass = 1.0 + 2.0 + 4.0 / 3.0
        self.assertAlmostEqual(float(terms.stiffness_raw), 4.0, places=5)
        self.assertAlmostEqual(float(terms.mass_energy), expected_mass, places=5)

    def test_mesh_refinement_preserves_continuum_energies(self) -> None:
        measurements = []
        for elements in (4, 8, 16, 32):
            positions = torch.linspace(0.0, 1.0, elements + 1)
            gradients = [
                torch.full((1, 1, 2), float(1.0 + 2.0 * z)) for z in positions
            ]
            terms = fe_entropy_terms(gradients, positions=positions)
            measurements.append(
                (float(terms.stiffness_raw), float(terms.mass_energy))
            )
        for stiffness, mass in measurements:
            self.assertAlmostEqual(stiffness, 4.0, places=4)
            self.assertAlmostEqual(mass, 13.0 / 3.0, places=4)

    def test_vector_mode_sees_direction_flip_but_norm_proxy_does_not(self) -> None:
        positive = torch.ones(1, 1, 4)
        negative = -positive
        vector = fe_entropy_terms([positive, negative], stiffness_mode="vector")
        proxy = fe_entropy_terms([positive, negative], stiffness_mode="norm")
        self.assertGreater(float(vector.stiffness_raw), 1.0)
        self.assertAlmostEqual(float(proxy.stiffness_raw), 0.0, places=7)

    def test_concentration_reduces_relative_coverage(self) -> None:
        uniform = fe_entropy_terms([torch.ones(1, 1, 2) for _ in range(7)])
        concentrated_nodes = [torch.zeros(1, 1, 2) for _ in range(7)]
        concentrated_nodes[3] = torch.ones(1, 1, 2)
        concentrated = fe_entropy_terms(concentrated_nodes)
        self.assertGreater(
            float(uniform.relative_coverage), float(concentrated.relative_coverage)
        )

    def test_mask_excludes_padding_tokens(self) -> None:
        mask = torch.tensor([[1.0, 0.0]])
        left = torch.tensor([[[1.0, 1.0], [100.0, 100.0]]])
        right = torch.tensor([[[1.0, 1.0], [-100.0, -100.0]]])
        terms = fe_entropy_terms([left, right], mask=mask)
        self.assertAlmostEqual(float(terms.stiffness_raw), 0.0, places=6)
        self.assertAlmostEqual(float(terms.mass_energy), 1.0, places=6)


class AutogradTests(unittest.TestCase):
    def test_legacy_regularizer_name_is_compatible(self) -> None:
        self.assertIs(FEEntropyRegularizer, FEERegularizer)

    def test_active_regularizer_double_backpropagates(self) -> None:
        torch.manual_seed(1)
        layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        hidden = torch.randn(2, 4, requires_grad=True)
        states = [hidden]
        for layer in layers:
            hidden = hidden + torch.tanh(layer(hidden))
            states.append(hidden)
        task_loss = hidden.square().mean()
        regularizer = FEEntropyRegularizer(
            lambda_stiffness=1e-3,
            lambda_energy=1e-3,
            lambda_entropy=1e-3,
        )
        output = regularizer(task_loss, states, create_graph=True)
        (task_loss + output.penalty).backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in layers.parameters()
            )
        )

    def test_detached_active_regularizer_is_rejected(self) -> None:
        hidden = torch.randn(2, 4, requires_grad=True)
        output = hidden.square()
        loss = output.mean()
        regularizer = FEEntropyRegularizer(lambda_stiffness=1.0)
        with self.assertRaisesRegex(ValueError, "create_graph=True"):
            regularizer(loss, [hidden, output], create_graph=False)


class SpectralTests(unittest.TestCase):
    def test_entropy_is_scale_free_but_isometry_error_is_not(self) -> None:
        identity = torch.eye(4)
        small = 0.001 * identity
        identity_metrics = spectral_metrics(identity)
        small_metrics = spectral_metrics(small)
        self.assertAlmostEqual(
            float(identity_metrics["spectral_entropy_normalized"]),
            float(small_metrics["spectral_entropy_normalized"]),
            places=5,
        )
        self.assertGreater(float(small_metrics["log_isometry_error"]), 10.0)
        self.assertAlmostEqual(float(identity_metrics["log_isometry_error"]), 0.0)


if __name__ == "__main__":
    unittest.main()
