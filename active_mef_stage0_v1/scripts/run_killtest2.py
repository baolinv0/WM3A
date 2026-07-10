#!/usr/bin/env python3
"""Train and evaluate L1/L2/L4 scalar value predictors for Kill Test 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from active_mef.kt2.dataset import ValueTensorDataset
from active_mef.kt2.evaluation import evaluate_decisions, scene_bootstrap_regret_difference
from active_mef.kt2.models import predict, save_bundle, train_scalar_mlp
from active_mef.kt2.rollout import headroom_recovery, rollout_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--with-rollout", action="store_true", help="run closed-loop eta evaluation after offline gate")
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    split_path = out / "scene_split.json"
    dataset = ValueTensorDataset.create(
        tensor_path=cfg["data"]["tensor"],
        cache_path=cfg["data"]["feature_cache"],
        split_path=split_path,
        train_frac=float(cfg["split"].get("train_frac", 2.0 / 3.0)),
        val_frac=float(cfg["split"].get("val_frac", 1.0 / 6.0)),
        seed=int(cfg.get("seed", 42)),
    )
    print("Scene counts:", dataset.scene_counts())

    train_cfg = cfg.get("training", {})
    levels = [str(x).upper() for x in cfg.get("levels", ["L1", "L2", "L4"])]
    summaries: list[dict] = []
    state_frames: dict[str, pd.DataFrame] = {}
    bundles: dict[str, dict] = {}

    for level in levels:
        x_train, y_train, _ = dataset.arrays(level, "train")
        x_val, y_val, _ = dataset.arrays(level, "val")
        x_test, y_test, meta_test = dataset.arrays(level, "test")
        bundle = train_scalar_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dims=tuple(int(x) for x in train_cfg.get("hidden_dims", [256, 128])),
            dropout=float(train_cfg.get("dropout", 0.1)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            batch_size=int(train_cfg.get("batch_size", 512)),
            max_epochs=int(train_cfg.get("max_epochs", 120)),
            patience=int(train_cfg.get("patience", 15)),
            seed=int(cfg.get("seed", 42)),
            device=str(cfg.get("device", "cuda")),
        )
        bundles[level] = bundle
        y_pred = predict(bundle, x_test)
        summary, state_frame = evaluate_decisions(y_test, y_pred, meta_test)
        state_frames[level] = state_frame
        row = {
            "level": level,
            **summary.as_dict(),
            "best_epoch": bundle["best_epoch"],
            "best_val_loss": bundle["best_val_loss"],
        }
        summaries.append(row)

        level_dir = out / level
        level_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "scene_id": [m["scene_id"] for m in meta_test],
            "state_key": [m["state_key"] for m in meta_test],
            "current": [json.dumps(m["current"]) for m in meta_test],
            "action": [m["action"] for m in meta_test],
            "true_gain": y_test,
            "pred_gain": y_pred,
        }).to_csv(level_dir / "predictions.csv", index=False)
        state_frame.to_csv(level_dir / "per_state_metrics.csv", index=False)
        pd.DataFrame(bundle["history"]).to_csv(level_dir / "training_history.csv", index=False)
        save_bundle(level_dir / "model.pt", bundle, level)
        with (level_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
        print(level, json.dumps(row, indent=2))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "summary.csv", index=False)

    comparisons = []
    if "L2" in state_frames and "L4" in state_frames:
        comp = scene_bootstrap_regret_difference(
            reference=state_frames["L2"],
            challenger=state_frames["L4"],
            n_boot=int(cfg.get("evaluation", {}).get("bootstrap_repeats", 5000)),
            seed=int(cfg.get("seed", 42)),
        )
        comparisons.append({"reference": "L2", "challenger": "L4", **comp})
    if "L1" in state_frames and "L4" in state_frames:
        comp = scene_bootstrap_regret_difference(
            reference=state_frames["L1"],
            challenger=state_frames["L4"],
            n_boot=int(cfg.get("evaluation", {}).get("bootstrap_repeats", 5000)),
            seed=int(cfg.get("seed", 42)),
        )
        comparisons.append({"reference": "L1", "challenger": "L4", **comp})
    with (out / "representation_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparisons, handle, indent=2)

    if args.with_rollout:
        rollout_cfg = cfg.get("rollout", {})
        budgets = [int(x) for x in rollout_cfg.get("budgets", [2, 3, 4])]
        rollout_frames = []
        for level in levels:
            bundle = bundles[level]
            frame = rollout_policy(
                dataset,
                level,
                predict_fn=lambda x, b=bundle: predict(b, x),
                budgets=budgets,
                base_ev=float(rollout_cfg.get("base_ev", 0.0)),
            )
            rollout_frames.append(frame)
        rollout_all = pd.concat(rollout_frames, ignore_index=True)
        rollout_all.to_csv(out / "rollout_per_scene.csv", index=False)
        per_scene = pd.read_csv(rollout_cfg["per_scene_results"])
        recovery = headroom_recovery(
            rollout_all,
            per_scene,
            baseline_method=str(rollout_cfg.get("baseline_method", "best_fixed")),
            oracle_method=str(rollout_cfg.get("oracle_method", "oracle_greedy")),
        )
        recovery.to_csv(out / "rollout_headroom_recovery.csv", index=False)
        print("\nClosed-loop headroom recovery")
        print(recovery.to_string(index=False))


if __name__ == "__main__":
    main()
