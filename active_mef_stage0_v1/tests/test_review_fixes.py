from __future__ import annotations

import numpy as np

from active_mef.data.manifest import ExposurePoolSample
from active_mef.fusion.simple_weighted import LinearRadianceFusion
from active_mef.metrics import psnr, psnr_mu
from active_mef.oracle import SceneEvaluator
from active_mef.sim.camera import CameraSimulator


def _sample() -> ExposurePoolSample:
    h = w = 24
    hdr = np.ones((h, w, 3), np.float32) * 0.08
    hdr[:, w // 2 :] = 2.0
    sim = CameraSimulator(motion_mode="static", seed=7)
    pool = sim.make_pool(hdr, [-2, 0, 2])
    target = np.clip(np.log1p(5000 * sim.normalize_hdr(hdr)) / np.log1p(5000), 0, 1)
    return ExposurePoolSample("toy", pool, target.astype(np.float32), {})


def test_oracle_sequence_returns_exact_budget_when_available():
    sample = _sample()
    evaluator = SceneEvaluator(sample, LinearRadianceFusion(), "psnr")
    selected, score = evaluator.oracle_sequence(3, 0.0)
    assert len(selected) == 3
    assert np.isfinite(score)


def test_psnr_mu_is_not_silent_alias_for_psnr():
    pred = np.full((8, 8, 3), 0.25, np.float32)
    target = np.full((8, 8, 3), 0.5, np.float32)
    assert abs(psnr(pred, target) - psnr_mu(pred, target)) > 1e-6
