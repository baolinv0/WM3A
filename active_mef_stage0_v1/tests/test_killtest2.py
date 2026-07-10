from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
import pandas as pd

from active_mef.fusion.simple_weighted import LinearRadianceFusion
from active_mef.kt2.dataset import FeatureCache, ValueTensorDataset, canonical_state_key, make_scene_split
from active_mef.kt2.evaluation import evaluate_decisions
from active_mef.kt2.features import FrozenResNet18Encoder, state_to_map
from active_mef.kt2.rollout import headroom_recovery, rollout_policy


def test_scene_split_has_no_leakage():
    split = make_scene_split([f"s{i}" for i in range(30)], seed=7)
    assert set(split.values()) == {"train", "val", "test"}
    assert len(split) == 30


def test_state_map_is_l2_superset():
    rng = np.random.default_rng(0)
    exposures = {
        -1.0: rng.random((16, 16, 3), dtype=np.float32),
        0.0: rng.random((16, 16, 3), dtype=np.float32),
    }
    backend = LinearRadianceFusion()
    fused, state = backend.fuse_with_state(exposures, [-1.0, 0.0])
    state_map = state_to_map(fused, state)
    assert state_map.shape == (16, 16, 9)
    np.testing.assert_allclose(state_map[..., :3], fused, atol=0, rtol=0)
    assert np.isfinite(state_map).all()
    # L4 must carry explicit S/W accumulation information, not only Y_t.
    assert np.any(state_map[..., 3:6] > 0)
    assert np.any(state_map[..., 6:7] > 0)


def test_decision_regret_known_case():
    meta = [
        {"scene_id": "a", "state_key": "a||0", "current": [0], "action": -1},
        {"scene_id": "a", "state_key": "a||0", "current": [0], "action": 1},
        {"scene_id": "b", "state_key": "b||0", "current": [0], "action": -1},
        {"scene_id": "b", "state_key": "b||0", "current": [0], "action": 1},
    ]
    y_true = np.array([1.0, 0.5, 0.2, 0.9], np.float32)
    y_pred = np.array([0.1, 0.2, 0.1, 0.9], np.float32)
    summary, frame = evaluate_decisions(y_true, y_pred, meta)
    assert abs(summary.mean_regret - 0.25) < 1e-6
    assert summary.top1_accuracy == 0.5
    assert len(frame) == 2


def test_resnet_encoder_shapes_without_pretrained_weights():
    import pytest

    if os.environ.get("RUN_KT2_RESNET_TEST") != "1":
        pytest.skip("set RUN_KT2_RESNET_TEST=1 for slower ResNet encoder smoke test")
    rgb = [np.zeros((32, 32, 3), np.float32), np.ones((32, 32, 3), np.float32)]
    state = [np.zeros((32, 32, 9), np.float32), np.ones((32, 32, 9), np.float32)]
    enc3 = FrozenResNet18Encoder(3, device="cpu", weights="none", image_size=64)
    enc9 = FrozenResNet18Encoder(9, device="cpu", weights="none", image_size=64)
    assert enc3.encode(rgb, batch_size=2).shape == (2, 512)
    assert enc9.encode(state, batch_size=2).shape == (2, 512)


def _make_end_to_end_fixture(root: Path, n_scenes: int = 9):
    img_root = root / "images"
    img_root.mkdir(parents=True)
    manifest = root / "manifest.jsonl"
    tensor = root / "tensor.jsonl"
    backend = LinearRadianceFusion()
    rng = np.random.default_rng(4)
    with manifest.open("w", encoding="utf-8") as mf, tensor.open("w", encoding="utf-8") as tf:
        for s in range(n_scenes):
            scene = f"s{s:02d}"
            exposures = {}
            exp_recs = []
            base = rng.random((32, 32, 3), dtype=np.float32) * (0.15 + 0.03 * s)
            base[:, 16:] += 0.5 + 0.02 * s
            for ev in [-1.0, 0.0, 1.0]:
                img = np.clip(base * (2.0 ** ev), 0, 1).astype(np.float32)
                path = img_root / f"{scene}_{ev:+.0f}.png"
                iio.imwrite(path, (img * 255).astype(np.uint8))
                exposures[ev] = img
                exp_recs.append({"ev": ev, "path": str(path)})
            gt = backend.fuse(exposures, [-1.0, 0.0, 1.0])
            gt_path = img_root / f"{scene}_gt.png"
            iio.imwrite(gt_path, (gt * 255).astype(np.uint8))
            mf.write(json.dumps({
                "scene_id": scene,
                "kind": "precomputed",
                "gt": str(gt_path),
                "exposures": exp_recs,
            }) + "\n")
            for current in ([0.0], [-1.0, 0.0], [0.0, 1.0]):
                before = backend.fuse(exposures, list(current))
                before_score = -float(np.mean((before - gt) ** 2))
                for action in [-1.0, 0.0, 1.0]:
                    if action in current:
                        continue
                    after = backend.fuse(exposures, sorted(list(current) + [action]))
                    after_score = -float(np.mean((after - gt) ** 2))
                    tf.write(json.dumps({
                        "scene_id": scene,
                        "current": list(current),
                        "action": action,
                        "score_before": before_score,
                        "score_after": after_score,
                        "gain": after_score - before_score,
                    }) + "\n")
    cfg = root / "stage0.yaml"
    cfg.write_text(
        "simulator:\n  enabled: false\nfusion:\n  type: linear_radiance\n  params:\n"
        "    gamma: 2.2\n    mu: 5000.0\n    sigma: 0.2\ndataset:\n  max_image_size: 64\n"
        "experiment:\n  candidate_evs: [-1, 0, 1]\n",
        encoding="utf-8",
    )
    return manifest, tensor, cfg


def test_end_to_end_grid_cache_and_training(tmp_path: Path):
    manifest, tensor, stage0 = _make_end_to_end_fixture(tmp_path)
    cache = tmp_path / "cache.npz"
    sub_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run([
        sys.executable,
        "scripts/build_killtest2_cache.py",
        "--tensor", str(tensor),
        "--manifest", str(manifest),
        "--stage0-config", str(stage0),
        "--output", str(cache),
        "--encoder", "grid_stats",
    ], cwd=Path(__file__).parents[1], check=True, env=sub_env)

    cfg = tmp_path / "kt2.yaml"
    cfg.write_text(f"""
seed: 42
device: cpu
levels: [L1, L2, L4]
data:
  tensor: {tensor.as_posix()}
  feature_cache: {cache.as_posix()}
split:
  train_frac: 0.6666667
  val_frac: 0.1666667
training:
  hidden_dims: [32, 16]
  dropout: 0.0
  learning_rate: 0.003
  weight_decay: 0.0
  batch_size: 32
  max_epochs: 4
  patience: 2
evaluation:
  bootstrap_repeats: 100
""", encoding="utf-8")
    out = tmp_path / "out"
    subprocess.run([
        sys.executable,
        "scripts/run_killtest2.py",
        "--config", str(cfg),
        "--output", str(out),
    ], cwd=Path(__file__).parents[1], check=True, env=sub_env)
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary.level) == {"L1", "L2", "L4"}
    assert (summary.num_states > 0).all()
    assert (out / "representation_comparison.json").exists()


def test_rollout_and_headroom_recovery():
    scenes = ["a", "b"]
    rows = []
    keys = []
    for scene in scenes:
        for current in ([0.0], [-1.0, 0.0], [0.0, 1.0]):
            keys.append(canonical_state_key(scene, current))
        rows += [
            {"scene_id": scene, "current": [0.0], "action": -1.0, "gain": 0.1, "score_before": 30.0, "score_after": 30.1},
            {"scene_id": scene, "current": [0.0], "action": 1.0, "gain": 1.0, "score_before": 30.0, "score_after": 31.0},
            {"scene_id": scene, "current": [0.0, 1.0], "action": -1.0, "gain": 0.5, "score_before": 31.0, "score_after": 31.5},
            {"scene_id": scene, "current": [-1.0, 0.0], "action": 1.0, "gain": 1.4, "score_before": 30.1, "score_after": 31.5},
        ]
    keys_arr = np.asarray(keys, dtype=str)
    cache = FeatureCache(
        keys_arr,
        np.zeros((len(keys_arr), 4), np.float32),
        np.zeros((len(keys_arr), 4), np.float32),
        np.zeros((len(keys_arr), 4), np.float32),
        "grid",
    )
    ds = ValueTensorDataset(rows, cache, {"a": "test", "b": "test"}, 1.0)

    def predictor(x):
        return x[:, -3]

    rollout = rollout_policy(ds, "L1", predictor, budgets=[2, 3], base_ev=0.0)
    assert np.allclose(rollout[rollout.budget == 2].policy_score, 31.0)
    assert np.allclose(rollout[rollout.budget == 3].policy_score, 31.5)

    per_scene = pd.DataFrame(
        [{"scene_id": s, "budget": 2, "method": "best_fixed", "score": 30.5} for s in scenes]
        + [{"scene_id": s, "budget": 2, "method": "oracle_greedy", "score": 31.0} for s in scenes]
        + [{"scene_id": s, "budget": 3, "method": "best_fixed", "score": 31.0} for s in scenes]
        + [{"scene_id": s, "budget": 3, "method": "oracle_greedy", "score": 31.5} for s in scenes]
    )
    recovery = headroom_recovery(rollout, per_scene, "best_fixed", "oracle_greedy")
    assert np.allclose(recovery.eta_global, 1.0)
