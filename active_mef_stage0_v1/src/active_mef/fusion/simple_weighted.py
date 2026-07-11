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

    def _accumulate(
        self, exposures: dict[float, np.ndarray], selected_evs: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rad_sum, w_sum) accumulators — shared by fuse() and fuse_with_state()."""
        if not selected_evs:
            raise ValueError("selected_evs must not be empty")
        rad_sum: np.ndarray | None = None
        w_sum:   np.ndarray | None = None
        for ev in selected_evs:
            img = np.clip(exposures[float(ev)], 0.0, 1.0)
            linear_sensor = np.power(img, self.gamma)
            radiance = linear_sensor / (2.0 ** float(ev))
            lum = img.mean(axis=2, keepdims=True)
            w = np.exp(-0.5 * ((lum - 0.5) / max(self.sigma, 1e-6)) ** 2)
            w *= ((lum > 0.02) & (lum < 0.98)).astype(np.float32) + 1e-3
            if rad_sum is None:
                rad_sum, w_sum = radiance * w, w.copy()
            else:
                rad_sum += radiance * w
                w_sum  += w
        return rad_sum, w_sum  # type: ignore[return-value]

    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        rad_sum, w_sum = self._accumulate(exposures, selected_evs)
        hdr = rad_sum / np.maximum(w_sum, 1e-8)
        out = mu_tonemap(np.maximum(hdr, 0.0), self.mu)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def fuse_with_state(
        self, exposures: dict[float, np.ndarray], selected_evs: list[float]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Fuse and return the fusion-native intermediate state for Kill Test 2.

        The state dict contains:
          ``S``  — weighted radiance accumulator  (H, W, 3), un-normalised
          ``W``  — weight accumulator              (H, W, 1)
          ``cov_under`` — fraction of pixels with luminance < 0.05 per channel (H, W, 1)
          ``cov_over``  — fraction of pixels with luminance > 0.95 per channel (H, W, 1)

        These four maps capture **what information the current set has accumulated**
        and **which radiance regions remain under- or over-exposed**, without
        leaking the final fused image that L2 uses.
        """
        rad_sum, w_sum = self._accumulate(exposures, selected_evs)
        hdr = rad_sum / np.maximum(w_sum, 1e-8)
        out = mu_tonemap(np.maximum(hdr, 0.0), self.mu)
        fused = np.clip(out, 0.0, 1.0).astype(np.float32)

        # Coverage maps: mean across selected frames per pixel
        lum_stack = np.stack([
            np.clip(exposures[float(ev)], 0.0, 1.0).mean(axis=2)
            for ev in selected_evs
        ], axis=0)                             # (n_frames, H, W)
        cov_under = (lum_stack < 0.05).mean(axis=0, keepdims=False)[..., None]  # (H, W, 1)
        cov_over  = (lum_stack > 0.95).mean(axis=0, keepdims=False)[..., None]

        state = {
            "S":         rad_sum.astype(np.float32),          # (H, W, 3)
            "W":         w_sum.astype(np.float32),             # (H, W, 1)
            "cov_under": cov_under.astype(np.float32),        # (H, W, 1)
            "cov_over":  cov_over.astype(np.float32),         # (H, W, 1)
        }
        return fused, state
