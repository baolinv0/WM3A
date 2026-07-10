from __future__ import annotations

import pandas as pd

from active_mef.kt2.rollout import headroom_recovery


def test_headroom_recovery_handles_numeric_csv_scene_ids():
    rollout = pd.DataFrame([
        {"scene_id": "1", "budget": 3, "level": "L4", "policy_score": 31.5},
        {"scene_id": "2", "budget": 3, "level": "L4", "policy_score": 31.5},
    ])
    # Mimic pandas CSV inference: numeric-looking scene IDs become integers.
    per_scene = pd.DataFrame([
        {"scene_id": 1, "budget": 3, "method": "best_fixed", "score": 31.0},
        {"scene_id": 2, "budget": 3, "method": "best_fixed", "score": 31.0},
        {"scene_id": 1, "budget": 3, "method": "oracle_greedy", "score": 31.5},
        {"scene_id": 2, "budget": 3, "method": "oracle_greedy", "score": 31.5},
    ])
    result = headroom_recovery(rollout, per_scene, "best_fixed", "oracle_greedy")
    assert len(result) == 1
    assert float(result.iloc[0].eta_global) == 1.0
