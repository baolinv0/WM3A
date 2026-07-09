from __future__ import annotations

import math
import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Standard linear-domain PSNR."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-14:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def psnr_mu(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    mu: float = 5000.0,
) -> float:
    """Mu-law tone-mapped PSNR (standard HDR quality metric).

    Both images are mapped to the mu-law tone domain before computing PSNR,
    which weights perceptually relevant mid-tone errors more heavily than
    highlight/shadow extremes.  This is the correct metric when the target is
    already in mu-tone domain (as produced by CameraSimulator) because errors
    in over-/under-exposed regions are compressed in proportion to perception.
    """
    def _mu_tonemap(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, data_range)
        return np.log1p(mu * x / data_range) / np.log1p(mu)

    return psnr(_mu_tonemap(pred), _mu_tonemap(target), data_range=1.0)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(target, pred, channel_axis=2, data_range=data_range))


def metric_fn(name: str):
    name = name.lower()
    if name == "psnr":
        return psnr
    if name in {"mu_psnr", "psnr_mu"}:
        return psnr_mu
    if name == "ssim":
        return ssim
    raise ValueError(f"Unsupported metric: {name!r}. Choose from: psnr, mu_psnr, ssim")
