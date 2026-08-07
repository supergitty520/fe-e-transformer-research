"""FE-E diagnostics and regularization for deep residual networks.

PyTorch and Apple MLX may live in different Python environments.  The public
PyTorch symbols are therefore imported lazily so importing the MLX experiment
does not require PyTorch to be installed in the same interpreter.
"""

from importlib import import_module

__all__ = [
    "FEERegularizer",
    "FEEntropyRegularizer",
    "GradientSmoothingAdamW",
    "fe_entropy_terms",
    "spectral_metrics",
]


_LAZY_IMPORTS = {
    "FEERegularizer": (".regularizer", "FEERegularizer"),
    "FEEntropyRegularizer": (".regularizer", "FEEntropyRegularizer"),
    "fe_entropy_terms": (".regularizer", "fe_entropy_terms"),
    "spectral_metrics": (".spectral", "spectral_metrics"),
    "GradientSmoothingAdamW": (".gradient_smoothing", "GradientSmoothingAdamW"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
