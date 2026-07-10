#!/usr/bin/env python3
"""Build unique-state features for Kill Test 2 exactly once.

The script reads the oracle value tensor only to discover current states. It
never uses candidate/future frames while constructing L1/L2/L4 state features.
"""
from __future__ import annotations

import argparse
import json
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
    state_to_map,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", required=True, help="oracle_value_tensor.jsonl")
    p.add_argument("--manifest", required=True, help="dataset manifest matching the tensor")
    p.add_argument("--stage0-config", required=True, help="Stage-0 config defining the fusion backend")
    p.add_argument("--output", required=True, help="output .npz feature cache")
    p.add_argument("--encoder", choices=["resnet18", "grid_stats"], default="resnet18")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--resnet-weights", default=None, help="optional local ResNet-18 state_dict")
    p.add_argument("--weights-mode", choices=["imagenet1k_v1", "none"], default="imagenet1k_v1")
    p.add_argument("--max-image-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_tensor_rows(args.tensor)
    states_by_scene: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    for row in rows:
        states_by_scene[str(row["scene_id"])].add(tuple(sorted(float(x) for x in row["current"])))

    cfg = load_yaml(args.stage0_config)
    if cfg.get("simulator", {}).get("enabled", False):
        raise ValueError(
            "Kill Test 2 cache builder currently expects precomputed exposure stacks. "
            "For simulated HDR pairs, build a deterministic exposure-pool manifest first."
        )
    max_image_size = args.max_image_size
    if max_image_size is None:
        max_image_size = cfg.get("dataset", {}).get("max_image_size")
    dataset = ManifestExposurePoolDataset(
        args.manifest,
        candidate_evs=[float(x) for x in cfg["experiment"]["candidate_evs"]],
        simulator=None,
        limit=None,
        max_image_size=max_image_size,
    )
    scene_to_index = {str(record.get("scene_id", idx)): idx for idx, record in enumerate(dataset.records)}
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

    if args.encoder == "resnet18":
        l2_encoder = FrozenResNet18Encoder(
            3,
            device=args.device,
            weights=args.weights_mode,
            weights_path=args.resnet_weights,
        )
        l4_encoder = FrozenResNet18Encoder(
            6,
            device=args.device,
            weights=args.weights_mode,
            weights_path=args.resnet_weights,
        )
    else:
        l2_encoder = l4_encoder = None

    all_keys: list[str] = []
    all_l1: list[np.ndarray] = []
    all_l2: list[np.ndarray] = []
    all_l4: list[np.ndarray] = []

    for scene_id in tqdm(sorted(states_by_scene), desc="KT2 feature cache"):
        sample = dataset[scene_to_index[scene_id]]
        scene_states = sorted(states_by_scene[scene_id], key=lambda x: (len(x), x))
        l2_images: list[np.ndarray] = []
        l4_maps: list[np.ndarray] = []
        scene_l1: list[np.ndarray] = []
        scene_l2_grid: list[np.ndarray] = []
        scene_l4_grid: list[np.ndarray] = []
        scene_keys: list[str] = []

        for current in scene_states:
            unavailable = [ev for ev in current if ev not in sample.exposures]
            if unavailable:
                raise KeyError(f"scene {scene_id} missing EVs {unavailable} for state {current}")
            fused, state = backend.fuse_with_state(sample.exposures, list(current))
            scene_keys.append(canonical_state_key(scene_id, current))
            scene_l1.append(l1_global_statistics(sample.exposures, current))
            if args.encoder == "resnet18":
                l2_images.append(fused)
                l4_maps.append(state_to_map(fused, state))
            else:
                scene_l2_grid.append(l2_grid_feature(fused))
                scene_l4_grid.append(l4_grid_feature(fused, state))

        if args.encoder == "resnet18":
            scene_l2 = l2_encoder.encode(l2_images, batch_size=args.batch_size)
            scene_l4 = l4_encoder.encode(l4_maps, batch_size=args.batch_size)
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state_keys=state_keys,
        l1=l1,
        l2=l2,
        l4=l4,
        encoder_name=np.asarray(args.encoder),
    )
    metadata = {
        "tensor": str(Path(args.tensor).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "stage0_config": str(Path(args.stage0_config).resolve()),
        "encoder": args.encoder,
        "num_states": int(len(state_keys)),
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
