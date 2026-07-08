from __future__ import annotations

import math
import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-14:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def psnr_mu(
    pred: np.ndarray,
    target: np.ndarray,
    mu: float = 5000.0,
    data_range: float = 1.0,
) -> float:
    """Compute PSNR after mu-law tone mapping.

    Use this only when ``pred`` and ``target`` are linear HDR/radiance-domain
    arrays. If a fusion backend already returns mu-tone-mapped display-domain
    images, use plain ``psnr`` to avoid double tone mapping.
    """
    from active_mef.sim.camera import mu_tonemap

    p = mu_tonemap(np.clip(pred, 0.0, data_range), mu)
    t = mu_tonemap(np.clip(target, 0.0, data_range), mu)
    return psnr(p, t, data_range=1.0)


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
    raise ValueError(f"Unsupported metric: {name}")
