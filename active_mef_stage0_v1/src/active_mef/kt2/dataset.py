"""Kill Test 2 — value tensor dataset with scene-level split.

Design principle: **never split rows randomly**.  All rows from the same
scene must land in the same fold, otherwise the predictor can memorise scene
content and Spearman / regret will be optimistically biased.

Usage
-----
ds = ValueTensorDataset.from_jsonl(
    tensor_path="results/kill_test_sice/oracle_value_tensor.jsonl",
    feature_cache="results/kill_test_sice/kt2_features.npz",
    level="L4",
)
X_train, y_train, meta_train = ds.split("train")
X_val,   y_val,   meta_val   = ds.split("val")
X_test,  y_test,  meta_test  = ds.split("test")
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import numpy as np


Split = Literal["train", "val", "test"]


class ValueTensorDataset:
    """Loads oracle_value_tensor.jsonl and attaches pre-computed features.

    Parameters
    ----------
    rows : list[dict]
        Raw rows from the JSONL (scene_id, current, action, gain, …).
    features : dict[str, np.ndarray]
        Mapping from string key ``"{scene_id}||{sorted_ev_tuple}"`` to
        feature vectors produced by run_kt2_features.py.
    level : "L1" | "L2" | "L4"
        Which feature level to use as the state representation.
    action_features : dict[str, np.ndarray]
        Mapping from same key to action encoding vectors.
    scene_split : dict[str, Split]
        Mapping from scene_id to split name.
    """

    def __init__(
        self,
        rows: list[dict],
        features: dict[str, np.ndarray],
        level: str,
        action_features: dict[str, np.ndarray],
        scene_split: dict[str, Split],
    ) -> None:
        self.rows = rows
        self.features = features
        self.level = level
        self.action_features = action_features
        self.scene_split = scene_split

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_jsonl(
        cls,
        tensor_path: str | Path,
        feature_cache: str | Path,
        level: str = "L4",
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
        seed: int = 42,
    ) -> "ValueTensorDataset":
        """Load dataset from JSONL tensor + pre-computed feature cache.

        The feature cache is an .npz file produced by run_kt2_features.py.
        It stores arrays keyed by ``"L1/{key}"``, ``"L2/{key}"``, ``"L4/{key}"``
        and ``"act/{key}"``, where ``key = scene_id + "||" + ev_tuple_str``.
        """
        # Load tensor rows
        rows = []
        with Path(tensor_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        # Load feature cache
        cache = np.load(Path(feature_cache), allow_pickle=True)

        def _lookup(prefix: str, scene_id: str, current: list[float]) -> np.ndarray | None:
            key = f"{prefix}/{scene_id}||{_ev_key(current)}"
            return cache[key] if key in cache else None

        # Build feature dicts (keyed by canonical string)
        features:        dict[str, np.ndarray] = {}
        action_features: dict[str, np.ndarray] = {}
        for row in rows:
            sid     = row["scene_id"]
            current = row["current"]
            action  = row["action"]
            k_state  = _state_key(sid, current)
            k_action = _action_key(sid, current, action)

            if k_state not in features:
                feat = _lookup(level, sid, current)
                if feat is not None:
                    features[k_state] = feat

            if k_action not in action_features:
                act_feat = _lookup("act", sid, current)
                if act_feat is not None:
                    # action_features cache stores a dict per state; index by action
                    # (stored as structured array or separate per action)
                    pass   # handled via full_action_key below

        # Reload: action features stored as "act__{scene}||{ev_tuple}||{action}"
        action_features2: dict[str, np.ndarray] = {}
        for key in cache.files:
            if key.startswith("act__"):
                action_features2[key[5:]] = cache[key]   # strip "act__"

        # Scene-level split
        all_scenes = sorted({r["scene_id"] for r in rows})
        rng = random.Random(seed)
        rng.shuffle(all_scenes)
        n = len(all_scenes)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        scene_split: dict[str, Split] = {}
        for i, sid in enumerate(all_scenes):
            if i < n_train:
                scene_split[sid] = "train"
            elif i < n_train + n_val:
                scene_split[sid] = "val"
            else:
                scene_split[sid] = "test"

        return cls(rows, features, level, action_features2, scene_split)

    # ------------------------------------------------------------------ #
    # Split accessor                                                       #
    # ------------------------------------------------------------------ #

    def split(
        self, split_name: Split
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Return (X, y, meta) for the requested split.

        X shape: (n_samples, state_dim + act_dim)
        y shape: (n_samples,)   — ΔQ gain values
        meta:    list of dicts with scene_id, current, action keys
        """
        X_parts, y_parts, meta = [], [], []
        for row in self.rows:
            sid     = row["scene_id"]
            if self.scene_split.get(sid) != split_name:
                continue
            current = row["current"]
            action  = row["action"]

            state_key  = _state_key(sid, current)
            action_key = _full_action_key(sid, current, action)

            state_feat  = self.features.get(state_key)
            action_feat = self.action_features.get(action_key)

            if state_feat is None or action_feat is None:
                continue   # feature cache miss — skip

            X_parts.append(np.concatenate([state_feat, action_feat]))
            y_parts.append(float(row["gain"]))
            meta.append({"scene_id": sid, "current": current, "action": float(action)})

        if not X_parts:
            return np.zeros((0, 1), dtype=np.float32), np.zeros(0), []
        return (
            np.stack(X_parts).astype(np.float32),
            np.array(y_parts, dtype=np.float32),
            meta,
        )

    def n_scenes(self, split_name: Split) -> int:
        return sum(1 for v in self.scene_split.values() if v == split_name)


# ------------------------------------------------------------------ #
# Key helpers                                                          #
# ------------------------------------------------------------------ #

def _ev_key(evs: list[float]) -> str:
    return "_".join(f"{e:.1f}" for e in sorted(evs))


def _state_key(scene_id: str, current: list[float]) -> str:
    return f"{scene_id}||{_ev_key(current)}"


def _action_key(scene_id: str, current: list[float], action: float) -> str:
    return f"{scene_id}||{_ev_key(current)}||{action:.1f}"


def _full_action_key(scene_id: str, current: list[float], action: float) -> str:
    return _action_key(scene_id, current, action)
