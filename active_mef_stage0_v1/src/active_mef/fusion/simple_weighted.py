from __future__ import annotations

import numpy as np
from active_mef.sim.camera import mu_tonemap
from .base import FusionBackend


class LinearRadianceFusion(FusionBackend):
    """Fast variable-frame radiance fusion for Stage-0 oracle enumeration.

    LDRs are approximately linearized, divided by exposure multiplier, and fused
    with well-exposedness weights. The result is mu-law tone-mapped to [0,1].
    This backend is deliberately small and deterministic; it is not a replacement
    for learned HDR fusion in final paper experiments.
    """

    def __init__(self, gamma: float = 2.2, mu: float = 5000.0, sigma: float = 0.2):
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        if not selected_evs:
            raise ValueError("selected_evs must not be empty")
        rad_sum = None
        w_sum = None
        for ev in selected_evs:
            img = np.clip(exposures[float(ev)], 0.0, 1.0)
            linear_sensor = np.power(img, self.gamma)
            radiance = linear_sensor / (2.0 ** float(ev))
            lum = img.mean(axis=2, keepdims=True)
            w = np.exp(-0.5 * ((lum - 0.5) / max(self.sigma, 1e-6)) ** 2)
            w *= ((lum > 0.02) & (lum < 0.98)).astype(np.float32) + 1e-3
            if rad_sum is None:
                rad_sum = radiance * w
                w_sum = w
            else:
                rad_sum += radiance * w
                w_sum += w
        hdr = rad_sum / np.maximum(w_sum, 1e-8)
        out = mu_tonemap(np.maximum(hdr, 0.0), self.mu)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
