"""Closed-loop policy rollout and oracle-headroom recovery for Kill Test 2."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from .dataset import ValueTensorDataset, canonical_state_key
from .features import encode_action, encode_state_evs


def rollout_policy(
    dataset: ValueTensorDataset,
    level: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    budgets: list[int],
    base_ev: float = 0.0,
) -> pd.DataFrame:
    """Roll out a learned one-step value policy on held-out scenes.

    The rollout stays entirely inside the oracle-value tensor state graph. This is
    deliberate: it tests decision quality without recomputing fusion or labels.
    """
    feature_matrix = dataset.cache.matrix(level)
    key_to_idx = dataset.cache.index()
    table: dict[tuple[str, tuple[float, ...]], list[dict]] = defaultdict(list)
    for row in dataset.rows:
        if dataset.scene_split.get(str(row["scene_id"])) != "test":
            continue
        key = (str(row["scene_id"]), tuple(sorted(float(x) for x in row["current"])))
        table[key].append(row)

    rows: list[dict] = []
    test_scenes = sorted(scene for scene, split in dataset.scene_split.items() if split == "test")
    for scene in test_scenes:
        for budget in budgets:
            current = [float(base_ev)]
            final_score: float | None = None
            actions_taken: list[float] = []
            if budget <= 1:
                candidates = table.get((scene, tuple(current)), [])
                if not candidates:
                    raise KeyError(f"missing base state for scene={scene}")
                final_score = float(candidates[0]["score_before"])

            while len(current) < budget:
                state_tuple = tuple(sorted(current))
                candidates = table.get((scene, state_tuple), [])
                if not candidates:
                    raise KeyError(f"missing rollout state scene={scene}, current={state_tuple}")
                state_key = canonical_state_key(scene, current)
                idx = key_to_idx.get(state_key)
                if idx is None:
                    raise KeyError(f"cache missing rollout state {state_key}")
                state_feature = feature_matrix[idx]
                state_meta = encode_state_evs(current, dataset.max_abs_ev)
                x = []
                for candidate in candidates:
                    action = encode_action(candidate["action"], current, dataset.max_abs_ev)
                    x.append(np.concatenate([state_feature, state_meta, action]).astype(np.float32))
                pred = predict_fn(np.stack(x))
                choice = int(np.argmax(pred))
                selected = candidates[choice]
                action = float(selected["action"])
                actions_taken.append(action)
                final_score = float(selected["score_after"])
                current = sorted(current + [action])

            if final_score is None:
                raise RuntimeError("rollout failed to produce a score")
            rows.append({
                "scene_id": str(scene),
                "budget": int(budget),
                "level": level,
                "policy_score": final_score,
                "selected_actions": actions_taken,
                "final_set": current,
            })
    return pd.DataFrame(rows)


def headroom_recovery(
    rollout: pd.DataFrame,
    per_scene_results: pd.DataFrame,
    baseline_method: str,
    oracle_method: str = "oracle_greedy",
    denominator_eps: float = 1e-6,
) -> pd.DataFrame:
    """Compute global oracle-headroom recovery eta for each level and budget.

    Global eta uses the aggregate mean headroom. Per-scene median eta is reported
    only for scenes with strictly positive oracle headroom, because zero/negative
    denominators do not represent recoverable opportunity.
    """
    required = {"scene_id", "budget", "method", "score"}
    missing = required - set(per_scene_results.columns)
    if missing:
        raise ValueError(f"per_scene results missing columns: {sorted(missing)}")

    # CSV readers often infer numeric-looking SICE scene IDs as integers. Cast
    # both sides explicitly so rollout/object IDs join deterministically.
    rollout = rollout.copy()
    scores = per_scene_results.copy()
    rollout["scene_id"] = rollout["scene_id"].astype(str)
    scores["scene_id"] = scores["scene_id"].astype(str)
    rollout["budget"] = rollout["budget"].astype(int)
    scores["budget"] = scores["budget"].astype(int)

    base = scores[scores.method == baseline_method][["scene_id", "budget", "score"]].rename(columns={"score": "baseline_score"})
    oracle = scores[scores.method == oracle_method][["scene_id", "budget", "score"]].rename(columns={"score": "oracle_score"})
    merged = rollout.merge(base, on=["scene_id", "budget"]).merge(oracle, on=["scene_id", "budget"])
    if merged.empty:
        raise RuntimeError("rollout did not match baseline/oracle result rows")

    records: list[dict] = []
    for (level, budget), group in merged.groupby(["level", "budget"]):
        denom = float(group.oracle_score.mean() - group.baseline_score.mean())
        numer = float(group.policy_score.mean() - group.baseline_score.mean())
        eta = numer / denom if denom > denominator_eps else float("nan")
        scene_denom = group.oracle_score - group.baseline_score
        valid = scene_denom > denominator_eps
        scene_eta = (group.loc[valid, "policy_score"] - group.loc[valid, "baseline_score"]) / scene_denom[valid]
        records.append({
            "level": level,
            "budget": int(budget),
            "mean_policy_score": float(group.policy_score.mean()),
            "mean_baseline_score": float(group.baseline_score.mean()),
            "mean_oracle_score": float(group.oracle_score.mean()),
            "eta_global": float(eta),
            "eta_scene_median_positive_headroom": float(scene_eta.median()) if len(scene_eta) else float("nan"),
            "positive_headroom_scenes": int(valid.sum()),
            "num_scenes": int(len(group)),
        })
    return pd.DataFrame(records)
