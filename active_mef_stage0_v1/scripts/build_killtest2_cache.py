#!/usr/bin/env python3
"""Build unique-state features for Kill Test 2 exactly once.

The script reads the oracle value tensor only to discover current states. It
never uses candidate/future frames while constructing L1/L2/L4 state features.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from active_mef.config import load_yaml
from active_mef.data import ManifestExposurePoolDataset
from active_mef.fusion import make_backend
from active_mef.kt2.dataset import canonical_state_key, load_tensor_rows
from active_mef.kt2.features import (
    FrozenResNet18Encoder,
    l1_global_statistics,
    l2_grid_feature,
    l4_grid_feature,
    make_l2_dual_feature,
    make_l4_dual_feature,
    state_aux_grid_feature,
)
from active_mef.utils import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", required=True, help="oracle_value_tensor.jsonl")
    parser.add_argument("--manifest", required=True, help="dataset manifest matching the tensor")
    parser.add_argument("--stage0-config", required=True, help="Stage-0 config defining the fusion backend")
    parser.add_argument("--output", required=True, help="output .npz feature cache")
    parser.add_argument("--encoder", choices=["resnet18", "grid_stats"], default="resnet18")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resnet-weights", default=None, help="optional local ResNet-18 state_dict")
    parser.add_argument("--weights-mode", choices=["imagenet1k_v1", "none"], default="imagenet1k_v1")
    parser.add_argument("--max-image-size", type=int, default=None)
    return parser.parse_args()


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        count = int((~np.isfinite(array)).sum())
        raise FloatingPointError(f"{name} contains {count} non-finite values")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    tensor_path = Path(args.tensor)
    manifest_path = Path(args.manifest)
    config_path = Path(args.stage0_config)
    rows = load_tensor_rows(tensor_path)
    states_by_scene: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    for row in rows:
        states_by_scene[str(row["scene_id"])].add(tuple(float(x) for x in row["current"]))

    cfg = load_yaml(config_path)
    if cfg.get("simulator", {}).get("enabled", False):
        raise ValueError(
            "Kill Test 2 cache builder currently expects precomputed exposure stacks. "
            "For simulated HDR pairs, build a deterministic exposure-pool manifest first."
        )
    max_image_size = args.max_image_size
    if max_image_size is None:
        max_image_size = cfg.get("dataset", {}).get("max_image_size")
    candidate_evs = [float(x) for x in cfg["experiment"]["candidate_evs"]]
    dataset = ManifestExposurePoolDataset(
        manifest_path,
        candidate_evs=candidate_evs,
        simulator=None,
        limit=None,
        max_image_size=max_image_size,
    )
    scene_to_index: dict[str, int] = {}
    for idx, record in enumerate(dataset.records):
        scene_id = str(record.get("scene_id", idx))
        if scene_id in scene_to_index:
            raise ValueError(f"duplicate scene_id in manifest: {scene_id}")
        scene_to_index[scene_id] = idx
    missing_scenes = set(states_by_scene) - set(scene_to_index)
    if missing_scenes:
        preview = sorted(missing_scenes)[:5]
        raise KeyError(f"manifest missing {len(missing_scenes)} tensor scenes, e.g. {preview}")

    backend = make_backend(cfg["fusion"])
    if not hasattr(backend, "fuse_with_state"):
        raise TypeError(
            f"backend {type(backend).__name__} lacks fuse_with_state(); "
            "Kill Test 2 L4 requires a backend-native accumulation state"
        )

    rgb_encoder = None
    if args.encoder == "resnet18":
        rgb_encoder = FrozenResNet18Encoder(
            3,
            device=args.device,
            weights=args.weights_mode,
            weights_path=args.resnet_weights,
        )

    all_keys: list[str] = []
    all_l1: list[np.ndarray] = []
    all_l2: list[np.ndarray] = []
    all_l4: list[np.ndarray] = []

    for scene_id in tqdm(sorted(states_by_scene), desc="KT2 feature cache"):
        sample = dataset[scene_to_index[scene_id]]
        if sample.evs != sorted(candidate_evs):
            raise RuntimeError(
                f"scene {scene_id} action pool {sample.evs} does not match configured pool {sorted(candidate_evs)}"
            )
        scene_states = sorted(states_by_scene[scene_id], key=lambda state: (len(state), state))
        rgb_images: list[np.ndarray] = []
        aux_features: list[np.ndarray] = []
        scene_l1: list[np.ndarray] = []
        scene_l2_grid: list[np.ndarray] = []
        scene_l4_grid: list[np.ndarray] = []
        scene_keys: list[str] = []

        for current in scene_states:
            unavailable = [ev for ev in current if ev not in sample.exposures]
            if unavailable:
                raise KeyError(
                    f"scene {scene_id} tensor contains EVs outside the configured action pool: {unavailable}. "
                    "Regenerate oracle_value_tensor.jsonl after the action-pool fix."
                )
            fused, state = backend.fuse_with_state(sample.exposures, list(current))
            scene_keys.append(canonical_state_key(scene_id, current))
            scene_l1.append(l1_global_statistics(sample.exposures, current))
            if args.encoder == "resnet18":
                rgb_images.append(fused)
                aux_features.append(state_aux_grid_feature(state))
            else:
                scene_l2_grid.append(l2_grid_feature(fused))
                scene_l4_grid.append(l4_grid_feature(fused, state))

        if args.encoder == "resnet18":
            rgb_features = rgb_encoder.encode(rgb_images, batch_size=args.batch_size)
            scene_l2 = np.stack([make_l2_dual_feature(feature) for feature in rgb_features])
            scene_l4 = np.stack([
                make_l4_dual_feature(feature, aux)
                for feature, aux in zip(rgb_features, aux_features)
            ])
        else:
            scene_l2 = np.stack(scene_l2_grid)
            scene_l4 = np.stack(scene_l4_grid)

        all_keys.extend(scene_keys)
        all_l1.extend(scene_l1)
        all_l2.extend(scene_l2)
        all_l4.extend(scene_l4)

    state_keys = np.asarray(all_keys, dtype=str)
    if len(state_keys) != len(set(state_keys.tolist())):
        raise RuntimeError("duplicate state keys in cache construction")
    l1 = np.stack(all_l1).astype(np.float32)
    l2 = np.stack(all_l2).astype(np.float32)
    l4 = np.stack(all_l4).astype(np.float32)
    if not (len(state_keys) == len(l1) == len(l2) == len(l4)):
        raise RuntimeError("feature cache arrays have inconsistent lengths")
    if l2.shape[1] != l4.shape[1]:
        raise RuntimeError(f"L2/L4 feature dimensions differ: {l2.shape[1]} vs {l4.shape[1]}")
    _require_finite("L1 features", l1)
    _require_finite("L2 features", l2)
    _require_finite("L4 features", l4)

    tensor_sha = sha256_file(tensor_path)
    manifest_sha = sha256_file(manifest_path)
    config_sha = sha256_file(config_path)
    if args.resnet_weights:
        encoder_signature = f"local:{sha256_file(args.resnet_weights)}"
    else:
        encoder_signature = f"torchvision:{args.weights_mode}"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state_keys=state_keys,
        l1=l1,
        l2=l2,
        l4=l4,
        encoder_name=np.asarray(args.encoder),
        tensor_sha256=np.asarray(tensor_sha),
        manifest_sha256=np.asarray(manifest_sha),
        stage0_config_sha256=np.asarray(config_sha),
        encoder_signature=np.asarray(encoder_signature),
        git_commit=np.asarray(_git_commit()),
    )
    metadata = {
        "tensor": str(tensor_path.resolve()),
        "tensor_sha256": tensor_sha,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "stage0_config": str(config_path.resolve()),
        "stage0_config_sha256": config_sha,
        "encoder": args.encoder,
        "encoder_signature": encoder_signature,
        "git_commit": _git_commit(),
        "candidate_evs": candidate_evs,
        "num_states": int(len(state_keys)),
        "l0_dim": 0,
        "l1_dim": int(l1.shape[1]),
        "l2_dim": int(l2.shape[1]),
        "l4_dim": int(l4.shape[1]),
        "num_scenes": int(len(states_by_scene)),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
