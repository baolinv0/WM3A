from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from active_mef.kt2.dataset import ValueTensorDataset, canonical_state_key
from active_mef.utils import sha256_file


def _write_tensor(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for scene in ("a", "b", "c"):
            for action, gain in ((-1.0, 0.1), (1.0, 0.2)):
                handle.write(json.dumps({
                    "scene_id": scene,
                    "current": [0.0],
                    "action": action,
                    "score_before": 10.0,
                    "score_after": 10.0 + gain,
                    "gain": gain,
                    "order_sensitive": False,
                }) + "\n")


def _write_cache(path: Path, tensor: Path, manifest: Path, config: Path) -> None:
    keys = np.asarray([canonical_state_key(scene, [0.0]) for scene in ("a", "b", "c")])
    n = len(keys)
    np.savez_compressed(
        path,
        state_keys=keys,
        l1=np.zeros((n, 4), np.float32),
        l2=np.zeros((n, 8), np.float32),
        l4=np.zeros((n, 8), np.float32),
        encoder_name=np.asarray("grid_stats"),
        tensor_sha256=np.asarray(sha256_file(tensor)),
        manifest_sha256=np.asarray(sha256_file(manifest)),
        stage0_config_sha256=np.asarray(sha256_file(config)),
        encoder_signature=np.asarray("test"),
        git_commit=np.asarray("test"),
    )


def test_cache_validates_tensor_manifest_and_config(tmp_path: Path):
    tensor = tmp_path / "tensor.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    config = tmp_path / "stage0.yaml"
    cache = tmp_path / "cache.npz"
    _write_tensor(tensor)
    manifest.write_text("manifest-v1\n", encoding="utf-8")
    config.write_text("config-v1\n", encoding="utf-8")
    _write_cache(cache, tensor, manifest, config)

    dataset = ValueTensorDataset.create(
        tensor,
        cache,
        manifest_path=manifest,
        stage0_config_path=config,
        seed=1,
    )
    assert dataset.scene_counts() == {"train": 1, "val": 1, "test": 1}

    manifest.write_text("manifest-v2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different manifest"):
        ValueTensorDataset.create(
            tensor,
            cache,
            manifest_path=manifest,
            stage0_config_path=config,
            seed=1,
        )
