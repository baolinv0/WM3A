from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Any
import numpy as np


class FusionBackend(ABC):
    # Set to True for recurrent backends whose output depends on acquisition
    # order rather than only on the selected exposure set.
    order_sensitive: bool = False

    @abstractmethod
    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        raise NotImplementedError


def make_backend(cfg: dict[str, Any]) -> FusionBackend:
    kind = cfg.get("type", "linear_radiance")
    if kind == "linear_radiance":
        from .simple_weighted import LinearRadianceFusion
        return LinearRadianceFusion(**cfg.get("params", {}))
    if kind == "mertens":
        from .mertens import MertensFusion
        return MertensFusion(**cfg.get("params", {}))
    if kind == "freemef":
        from .freemef_backend import FreeMEFBackend
        return FreeMEFBackend(**cfg.get("params", {}))
    if kind == "python_callable":
        target = cfg["target"]
        module_name, attr = target.split(":", 1)
        fn = getattr(importlib.import_module(module_name), attr)
        return CallableFusionBackend(fn)
    raise ValueError(f"Unknown fusion backend type={kind!r}")


class CallableFusionBackend(FusionBackend):
    def __init__(self, fn):
        self.fn = fn

    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        return np.asarray(self.fn(exposures, selected_evs), dtype=np.float32)
