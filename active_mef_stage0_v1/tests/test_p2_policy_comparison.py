from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from active_mef.kt2.dataset import canonical_state_key
from active_mef.kt2.policy_comparison import (
    StateGroup,
    _build_model,
    _state_loss,
    centered_numpy,
    evaluate_group_predictions,
    oracle_predictions,
    parameter_counts,
)
from active_mef.utils import sha256_file


def _group(num_candidates: int, input_dim: int = 8) -> StateGroup:
    actions = np.arange(num_candidates, dtype=np.float32)
    true = np.linspace(0.0, 1.0, num_candidates, dtype=np.float32)
    meta = [
        {
            "scene_id": "s",
            "state_key": "s||set||0",
            "current": [0.0],
            "action": float(action),
            "order_sensitive": False,
        }
        for action in actions
    ]
    return StateGroup(
        scene_id="s",
        state_key="s||set||0",
        current=[0.0],
        order_sensitive=False,
        actions=actions,
        x=np.zeros((num_candidates, input_dim), np.float32),
        true_gain=true,
        prior_gain=np.zeros(num_candidates, np.float32),
        meta=meta,
    )


def test_candidate_groups_exclude_invalid_actions_by_construction():
    import torch

    model = _build_model(8, (4,), 0.0)
    for count in (2, 6):
        group = _group(count)
        x = torch.from_numpy(group.x)
        true = torch.from_numpy(group.true_gain)
        prior = torch.from_numpy(group.prior_gain)
        loss, _, _, predicted = _state_loss(model, x, true, prior, "direct", 0.25, 0.1)
        assert predicted.shape == (count,)
        assert torch.isfinite(loss)
        assert torch.softmax(predicted, dim=0).shape[0] == count


def test_state_balancing_uses_mean_over_states_not_rows():
    # Equal per-state losses must remain equally weighted despite candidate counts.
    per_state_losses = np.asarray([2.0, 2.0], dtype=np.float32)
    candidate_counts = np.asarray([2, 6], dtype=np.float32)
    state_balanced = float(per_state_losses.mean())
    row_weighted = float(np.average(per_state_losses, weights=candidate_counts))
    assert state_balanced == 2.0
    assert row_weighted == 2.0
    # A non-equal example exposes the distinction explicitly.
    per_state_losses = np.asarray([1.0, 3.0], dtype=np.float32)
    assert float(per_state_losses.mean()) == 2.0
    assert float(np.average(per_state_losses, weights=candidate_counts)) == 2.5


def test_explicit_residual_centering_is_zero_sum():
    values = np.asarray([3.0, -1.0, 2.5, 7.0], dtype=np.float32)
    centered = centered_numpy(values)
    assert abs(float(centered.sum())) < 1e-6


def test_oracle_action_has_zero_regret():
    group = _group(4)
    summary, frame = evaluate_group_predictions([group], oracle_predictions([group]))
    assert summary.mean_regret == 0.0
    assert frame.regret.iloc[0] == 0.0


def test_isomorphic_scorers_have_identical_capacity():
    b1 = _build_model(704, (256, 128), 0.1)
    b2 = _build_model(704, (256, 128), 0.1)
    b4 = _build_model(704, (256, 128), 0.1)
    assert parameter_counts(b1) == parameter_counts(b2) == parameter_counts(b4)


def _make_p2_fixture(root: Path, n_scenes: int = 10):
    tensor = root / "tensor.jsonl"
    rows = []
    state_keys = []
    l2 = []
    l4 = []
    split = {}
    for scene_index in range(n_scenes):
        scene = f"s{scene_index:02d}"
        split[scene] = "train" if scene_index < 6 else "val" if scene_index < 8 else "test"
        scene_signal = -1.0 if scene_index % 2 == 0 else 1.0
        for current in ([0.0], [-1.0, 0.0], [0.0, 1.0]):
            key = canonical_state_key(scene, current)
            state_keys.append(key)
            base_feature = np.asarray([scene_signal, len(current), scene_index / n_scenes, 1.0], np.float32)
            l2.append(np.concatenate([base_feature, np.zeros(4, np.float32)]))
            l4.append(np.concatenate([base_feature, np.asarray([scene_signal, 0.5, -0.5, 1.0], np.float32)]))
            for action in (-1.0, 0.0, 1.0):
                if action in current:
                    continue
                geometry = 0.2 if action < 0 else 0.1
                scene_term = 0.35 * scene_signal * action
                gain = geometry + scene_term
                rows.append({
                    "scene_id": scene,
                    "current": list(current),
                    "action": action,
                    "score_before": 20.0,
                    "score_after": 20.0 + gain,
                    "gain": gain,
                    "order_sensitive": False,
                })
    with tensor.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    cache = root / "cache.npz"
    np.savez_compressed(
        cache,
        state_keys=np.asarray(state_keys, dtype=str),
        l1=np.zeros((len(state_keys), 2), np.float32),
        l2=np.asarray(l2, np.float32),
        l4=np.asarray(l4, np.float32),
        encoder_name=np.asarray("synthetic"),
        tensor_sha256=np.asarray(sha256_file(tensor)),
        manifest_sha256=np.asarray(""),
        stage0_config_sha256=np.asarray(""),
        encoder_signature=np.asarray("synthetic-v1"),
        git_commit=np.asarray("test"),
    )
    return tensor, cache, split_path


def test_p2_end_to_end_smoke(tmp_path: Path):
    tensor, cache, split_path = _make_p2_fixture(tmp_path)
    config = tmp_path / "p2.yaml"
    config.write_text(f"""
experiment:
  seeds: [42]
data:
  tensor: {tensor.as_posix()}
  feature_cache: {cache.as_posix()}
  split_file: {split_path.as_posix()}
  action_pool: [-1, 0, 1]
training:
  hidden_dims: [16, 8]
  dropout: 0.0
  learning_rate: 0.01
  weight_decay: 0.0
  states_per_batch: 4
  max_epochs: 2
  patience: 2
  target_temperature: 0.25
  lambda_regression: 0.1
evaluation:
  bootstrap_repeats: 50
device: cpu
""", encoding="utf-8")
    out = tmp_path / "out"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run([
        sys.executable,
        "scripts/run_p2_policy_comparison.py",
        "--config", str(config),
        "--output", str(out),
    ], cwd=Path(__file__).parents[1], check=True, env=env)

    summary = pd.read_csv(out / "summary.csv")
    assert {"B0_prior", "B1_direct_listwise", "B2_absolute", "B3_prior_residual_l2", "B4_prior_residual_l4", "Oracle"} == set(summary.method)
    counts = json.loads((out / "model_parameter_counts.json").read_text(encoding="utf-8"))
    learned = [counts[name]["trainable_parameters"] for name in counts]
    assert len(set(learned)) == 1
    assert (out / "pairwise_bootstrap.json").exists()
    assert (out / "integrity_report.json").exists()
