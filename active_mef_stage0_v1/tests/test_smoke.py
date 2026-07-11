from __future__ import annotations

import numpy as np
from active_mef.sim.camera import CameraSimulator
from active_mef.data.manifest import ExposurePoolSample
from active_mef.fusion.simple_weighted import LinearRadianceFusion
from active_mef.oracle import SceneEvaluator


def test_oracle_smoke():
    h = w = 32
    hdr = np.ones((h, w, 3), np.float32) * 0.1
    hdr[:, w//2:] = 3.0
    sim = CameraSimulator(motion_mode="static", seed=1)
    pool = sim.make_pool(hdr, [-2, 0, 2])
    target = np.clip(np.log1p(5000 * sim.normalize_hdr(hdr)) / np.log1p(5000), 0, 1)
    sample = ExposurePoolSample("toy", pool, target.astype(np.float32), {})
    ev = SceneEvaluator(sample, LinearRadianceFusion(), "psnr")
    gset, gscore = ev.oracle_greedy(2, 0.0)
    sset, sscore = ev.oracle_sequence(2, 0.0)
    assert len(gset) == 2
    assert len(sset) == 2
    assert np.isfinite(gscore)
    assert sscore + 1e-6 >= gscore
