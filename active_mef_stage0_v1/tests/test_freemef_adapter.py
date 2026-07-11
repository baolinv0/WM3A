from __future__ import annotations

import numpy as np

from active_mef.data.manifest import ExposurePoolSample
from active_mef.fusion.base import FusionBackend
from active_mef.fusion.freemef_backend import FreeMEFBackend
from active_mef.oracle import SceneEvaluator


def test_freemef_padding_matches_factor_and_unpad_shape():
    import torch

    backend = object.__new__(FreeMEFBackend)
    backend.pad_factor = 8
    base = torch.zeros((1, 3, 15, 17), dtype=torch.float32)
    others = torch.zeros((1, 2, 3, 15, 17), dtype=torch.float32)
    padded_base, padded_others, original_h, original_w = backend._pad(base, others)
    assert (original_h, original_w) == (15, 17)
    assert padded_base.shape == (1, 3, 16, 24)
    assert padded_others.shape == (1, 2, 3, 16, 24)
    assert padded_base[..., :original_h, :original_w].shape[-2:] == (15, 17)


class _OrderSensitiveBackend(FusionBackend):
    order_sensitive = True

    def fuse(self, exposures, selected_evs):
        # Encode acquisition order into a constant prediction. The sequence
        # [0, 1, -1] is optimal against the all-one target.
        score = 1.0 if selected_evs == [0.0, 1.0, -1.0] else 0.0
        return np.full((4, 4, 3), score, dtype=np.float32)


def test_oracle_sequence_enumerates_permutations_for_recurrent_backend():
    exposures = {
        -1.0: np.zeros((4, 4, 3), np.float32),
        0.0: np.zeros((4, 4, 3), np.float32),
        1.0: np.zeros((4, 4, 3), np.float32),
    }
    sample = ExposurePoolSample(
        scene_id="ordered",
        exposures=exposures,
        target=np.ones((4, 4, 3), np.float32),
        metadata={},
    )
    evaluator = SceneEvaluator(sample, _OrderSensitiveBackend(), metric="psnr")
    selected, score = evaluator.oracle_sequence(3, 0.0)
    assert selected == [0.0, 1.0, -1.0]
    assert score == 100.0
