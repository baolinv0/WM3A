"""Kill Test 2 oracle-label dataset and strict scene-level splitting."""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from active_mef.utils import sha256_file
from .features import encode_action, encode_state_evs

Split = Literal["train", "val", "test"]


def canonical_state_key(
    scene_id: str,
    current: Sequence[float],
    order_sensitive: bool = False,
) -> str:
    values = [float(x) for x in current]
    if not order_sensitive:
        values = sorted(values)
    evs = ",".join(f"{value:.8g}" for value in values)
    prefix = "seq" if order_sensitive else "set"
    return f"{scene_id}||{prefix}||{evs}"


def load_tensor_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            required = {"scene_id", "current", "action", "gain", "score_before", "score_after"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields: {sorted(missing)}")
            row["scene_id"] = str(row["scene_id"])
            row["order_sensitive"] = bool(row.get("order_sensitive", False))
            current = [float(x) for x in row["current"]]
            row["current"] = current if row["order_sensitive"] else sorted(current)
            row["action"] = float(row["action"])
            row["gain"] = float(row["gain"])
            row["score_before"] = float(row["score_before"])
            row["score_after"] = float(row["score_after"])
            numeric = [row["action"], row["gain"], row["score_before"], row["score_after"], *row["current"]]
            if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
                raise ValueError(f"{path}:{line_no} contains non-finite numeric values")
            if row["action"] in row["current"]:
                raise ValueError(f"{path}:{line_no} action already present in current state")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no rows loaded from {path}")
    return rows


def make_scene_split(
    scene_ids: Iterable[str],
    train_frac: float = 2.0 / 3.0,
    val_frac: float = 1.0 / 6.0,
    seed: int = 42,
) -> dict[str, Split]:
    scenes = sorted(set(str(x) for x in scene_ids))
    if len(scenes) < 3:
        raise ValueError("at least 3 scenes are required for train/val/test splitting")
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("invalid split fractions")
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    # Reserve at least one scene for both validation and test before rounding.
    n_train = min(max(1, int(round(n * train_frac))), n - 2)
    n_val = min(max(1, int(round(n * val_frac))), n - n_train - 1)
    split: dict[str, Split] = {}
    for idx, scene in enumerate(scenes):
        if idx < n_train:
            split[scene] = "train"
        elif idx < n_train + n_val:
            split[scene] = "val"
        else:
            split[scene] = "test"
    if set(split.values()) != {"train", "val", "test"}:
        raise RuntimeError("scene split failed to create all three partitions")
    return split


def save_scene_split(path: str | Path, split: dict[str, Split]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2, sort_keys=True)


def load_scene_split(path: str | Path) -> dict[str, Split]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result: dict[str, Split] = {}
    for key, value in raw.items():
        if value not in {"train", "val", "test"}:
            raise ValueError(f"invalid split value for {key}: {value}")
        result[str(key)] = value
    return result


@dataclass
class FeatureCache:
    state_keys: np.ndarray
    l1: np.ndarray
    l2: np.ndarray
    l4: np.ndarray
    encoder_name: str
    tensor_sha256: str
    manifest_sha256: str
    stage0_config_sha256: str
    encoder_signature: str
    git_commit: str

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCache":
        with np.load(Path(path), allow_pickle=False) as data:
            required = {
                "state_keys", "l1", "l2", "l4", "encoder_name",
                "tensor_sha256", "manifest_sha256", "stage0_config_sha256",
                "encoder_signature", "git_commit",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"feature cache missing provenance/features: {sorted(missing)}. "
                    "Rebuild the cache with scripts/build_killtest2_cache.py."
                )
            state_keys = data["state_keys"].astype(str)
            l1 = data["l1"].astype(np.float32)
            l2 = data["l2"].astype(np.float32)
            l4 = data["l4"].astype(np.float32)
            encoder_name = str(data["encoder_name"].item())
            tensor_sha256 = str(data["tensor_sha256"].item())
            manifest_sha256 = str(data["manifest_sha256"].item())
            stage0_config_sha256 = str(data["stage0_config_sha256"].item())
            encoder_signature = str(data["encoder_signature"].item())
            git_commit = str(data["git_commit"].item())
        if not (len(state_keys) == len(l1) == len(l2) == len(l4)):
            raise ValueError("feature cache arrays have inconsistent lengths")
        if len(state_keys) != len(set(state_keys.tolist())):
            raise ValueError("feature cache contains duplicate state keys")
        if l2.shape[1] != l4.shape[1]:
            raise ValueError(f"L2/L4 feature dimensions differ: {l2.shape[1]} vs {l4.shape[1]}")
        for name, matrix in (("L1", l1), ("L2", l2), ("L4", l4)):
            if matrix.ndim != 2:
                raise ValueError(f"{name} feature matrix must be 2D, got {matrix.shape}")
            if not np.isfinite(matrix).all():
                raise ValueError(f"{name} feature matrix contains non-finite values")
        return cls(
            state_keys, l1, l2, l4, encoder_name,
            tensor_sha256, manifest_sha256, stage0_config_sha256,
            encoder_signature, git_commit,
        )

    def matrix(self, level: str) -> np.ndarray:
        name = level.upper()
        if name == "L0":
            return np.zeros((len(self.state_keys), 0), dtype=np.float32)
        if name == "L1":
            return self.l1
        if name == "L2":
            return self.l2
        if name == "L4":
            return self.l4
        raise ValueError(f"unknown level {level}; expected L0/L1/L2/L4")

    def index(self) -> dict[str, int]:
        return {str(key): idx for idx, key in enumerate(self.state_keys.tolist())}


@dataclass
class ValueTensorDataset:
    rows: list[dict]
    cache: FeatureCache
    scene_split: dict[str, Split]
    max_abs_ev: float

    @classmethod
    def create(
        cls,
        tensor_path: str | Path,
        cache_path: str | Path,
        split_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        stage0_config_path: str | Path | None = None,
        train_frac: float = 2.0 / 3.0,
        val_frac: float = 1.0 / 6.0,
        seed: int = 42,
    ) -> "ValueTensorDataset":
        tensor_path = Path(tensor_path)
        rows = load_tensor_rows(tensor_path)
        cache = FeatureCache.load(cache_path)

        actual_tensor_sha = sha256_file(tensor_path)
        if cache.tensor_sha256 != actual_tensor_sha:
            raise ValueError(
                "feature cache was built from a different oracle tensor: "
                f"cache={cache.tensor_sha256}, current={actual_tensor_sha}. Rebuild the cache."
            )
        if manifest_path is not None:
            actual_manifest_sha = sha256_file(manifest_path)
            if cache.manifest_sha256 != actual_manifest_sha:
                raise ValueError(
                    "feature cache was built from a different manifest: "
                    f"cache={cache.manifest_sha256}, current={actual_manifest_sha}. Rebuild the cache."
                )
        if stage0_config_path is not None:
            actual_config_sha = sha256_file(stage0_config_path)
            if cache.stage0_config_sha256 != actual_config_sha:
                raise ValueError(
                    "feature cache was built from a different Stage-0 config: "
                    f"cache={cache.stage0_config_sha256}, current={actual_config_sha}. Rebuild the cache."
                )

        scenes = {str(row["scene_id"]) for row in rows}
        if split_path is not None and Path(split_path).exists():
            split = load_scene_split(split_path)
            missing = scenes - set(split)
            if missing:
                raise ValueError(f"split file missing {len(missing)} scenes")
            extra = set(split) - scenes
            if extra:
                raise ValueError(f"split file contains {len(extra)} scenes absent from tensor")
        else:
            split = make_scene_split(scenes, train_frac, val_frac, seed)
            if split_path is not None:
                save_scene_split(split_path, split)
        all_evs = [abs(float(row["action"])) for row in rows]
        for row in rows:
            all_evs.extend(abs(float(x)) for x in row["current"])
        max_abs_ev = max(max(all_evs, default=1.0), 1.0)
        return cls(rows, cache, split, max_abs_ev)

    def arrays(self, level: str, split_name: Split) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        feature_matrix = self.cache.matrix(level)
        key_to_idx = self.cache.index()
        x_parts: list[np.ndarray] = []
        y: list[float] = []
        meta: list[dict] = []
        for row in self.rows:
            scene = str(row["scene_id"])
            if self.scene_split.get(scene) != split_name:
                continue
            key = canonical_state_key(scene, row["current"], row["order_sensitive"])
            idx = key_to_idx.get(key)
            if idx is None:
                raise KeyError(f"feature cache missing state {key}")
            state_feature = feature_matrix[idx]
            state_meta = encode_state_evs(row["current"], self.max_abs_ev)
            action = encode_action(row["action"], row["current"], self.max_abs_ev)
            x_parts.append(np.concatenate([state_feature, state_meta, action]).astype(np.float32))
            y.append(float(row["gain"]))
            meta.append({
                "scene_id": scene,
                "current": list(row["current"]),
                "state_key": key,
                "action": float(row["action"]),
                "score_before": float(row["score_before"]),
                "score_after": float(row["score_after"]),
                "order_sensitive": bool(row["order_sensitive"]),
            })
        if not x_parts:
            raise RuntimeError(f"no samples for split={split_name}")
        x_array = np.stack(x_parts).astype(np.float32)
        y_array = np.asarray(y, dtype=np.float32)
        if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
            raise FloatingPointError(f"non-finite data found in split={split_name}, level={level}")
        return x_array, y_array, meta

    def scene_counts(self) -> dict[str, int]:
        return {
            split: sum(1 for value in self.scene_split.values() if value == split)
            for split in ("train", "val", "test")
        }
