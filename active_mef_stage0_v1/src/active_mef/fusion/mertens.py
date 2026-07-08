from __future__ import annotations

import cv2
import numpy as np
from .base import FusionBackend


class MertensFusion(FusionBackend):
    def __init__(self, contrast_weight: float = 1.0, saturation_weight: float = 1.0, exposure_weight: float = 1.0):
        self.merge = cv2.createMergeMertens(
            contrast_weight=float(contrast_weight),
            saturation_weight=float(saturation_weight),
            exposure_weight=float(exposure_weight),
        )

    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        imgs_bgr = []
        for ev in selected_evs:
            rgb = np.clip(exposures[float(ev)], 0.0, 1.0).astype(np.float32)
            imgs_bgr.append(rgb[..., ::-1])
        fused_bgr = self.merge.process(imgs_bgr)
        fused_rgb = fused_bgr[..., ::-1]
        return np.clip(fused_rgb, 0.0, 1.0).astype(np.float32)
