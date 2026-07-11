from __future__ import annotations

import math
import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Standard PSNR in the arrays' current domain."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-14:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def _mu_tonemap_linear(x: np.ndarray, mu: float, data_range: float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, data_range)
    return np.log1p(mu * x / data_range) / np.log1p(mu)


def psnr_mu(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """PSNR for inputs that are already in the mu-tone-mapped domain.

    The Stage-0 HDR simulator and ``LinearRadianceFusion`` both produce
    display-domain/mu-tone-mapped arrays. Applying mu-law a second time would
    silently evaluate a different metric. Therefore ``mu_psnr`` measures PSNR
    directly in that already mapped domain.
    """
    return psnr(pred, target, data_range=data_range)


def psnr_mu_from_linear(
    pred_linear: np.ndarray,
    target_linear: np.ndarray,
    data_range: float = 1.0,
    mu: float = 5000.0,
) -> float:
    """Mu-law PSNR for explicitly linear-radiance inputs."""
    pred_mu = _mu_tonemap_linear(pred_linear, mu=mu, data_range=data_range)
    target_mu = _mu_tonemap_linear(target_linear, mu=mu, data_range=data_range)
    return psnr(pred_mu, target_mu, data_range=1.0)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(target, pred, channel_axis=2, data_range=data_range))


def metric_fn(name: str):
    name = name.lower()
    if name == "psnr":
        return psnr
    if name in {"mu_psnr", "psnr_mu"}:
        return psnr_mu
    if name in {"mu_psnr_linear", "psnr_mu_linear"}:
        return psnr_mu_from_linear
    if name == "ssim":
        return ssim
    raise ValueError(
        f"Unsupported metric: {name!r}. Choose from: "
        "psnr, mu_psnr, mu_psnr_linear, ssim"
    )
