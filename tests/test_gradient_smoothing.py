import copy
import unittest

import torch

from fe_entropy.gradient_smoothing import (
    GradientSmoothingAdamW,
    window_smooth_tensors,
)


class WindowSmoothingTests(unittest.TestCase):
    def test_exact_boundary_and_interior_weights(self) -> None:
        values = [torch.tensor(float(value)) for value in (0.0, 10.0, 20.0)]
        smoothed = window_smooth_tensors(values, alpha=0.2)
        expected = (1.0, 10.0, 19.0)
        for actual, target in zip(smoothed, expected, strict=True):
            self.assertAlmostEqual(float(actual), target, places=6)

    def test_constant_depth_signal_is_preserved(self) -> None:
        values = [torch.ones(2, 3) for _ in range(5)]
        smoothed = window_smooth_tensors(values, alpha=0.4)
        self.assertTrue(all(torch.equal(value, torch.ones(2, 3)) for value in smoothed))


class TinyRepeatedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Linear(3, 3)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(3),
                    torch.nn.Linear(3, 3),
                    torch.nn.Tanh(),
                    torch.nn.Linear(3, 3),
                )
                for _ in range(3)
            ]
        )
        self.output = torch.nn.Linear(3, 2)

    def forward(self, value):
        value = self.input(value)
        for block in self.blocks:
            value = value + block(value)
        return self.output(value)


class OptimizerTests(unittest.TestCase):
    def test_alpha_zero_matches_torch_adamw(self) -> None:
        torch.manual_seed(4)
        reference = TinyRepeatedModel()
        candidate = copy.deepcopy(reference)
        torch_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=3e-4, weight_decay=0.01
        )
        smoothing_optimizer = GradientSmoothingAdamW(
            candidate.parameters(),
            blocks=candidate.blocks,
            lr=3e-4,
            weight_decay=0.01,
            alpha=0.0,
            scope="proj",
        )
        for step in range(4):
            generator = torch.Generator().manual_seed(100 + step)
            inputs = torch.randn(5, 3, generator=generator)
            targets = torch.randn(5, 2, generator=generator)
            for model, optimizer in (
                (reference, torch_optimizer),
                (candidate, smoothing_optimizer),
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = (model(inputs) - targets).square().mean()
                loss.backward()
                optimizer.step()
        for expected, actual in zip(
            reference.parameters(), candidate.parameters(), strict=True
        ):
            self.assertTrue(torch.allclose(expected, actual, atol=2e-7, rtol=2e-6))

    def test_norm_variant_preserves_each_block_update_norm(self) -> None:
        torch.manual_seed(5)
        model = TinyRepeatedModel()
        optimizer = GradientSmoothingAdamW(
            model.parameters(),
            blocks=model.blocks,
            alpha=0.2,
            variant="norm",
            scope="proj",
        )
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertAlmostEqual(
            optimizer.last_metrics["update_raw_mean_block_norm"],
            optimizer.last_metrics["update_applied_mean_block_norm"],
            places=5,
        )

    def test_standard_smoothing_reduces_update_roughness(self) -> None:
        torch.manual_seed(6)
        model = TinyRepeatedModel()
        optimizer = GradientSmoothingAdamW(
            model.parameters(),
            blocks=model.blocks,
            alpha=0.2,
            variant="standard",
            scope="proj",
        )
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertLessEqual(
            optimizer.last_metrics["update_applied_roughness"],
            optimizer.last_metrics["update_raw_roughness"] + 1e-7,
        )


if __name__ == "__main__":
    unittest.main()

