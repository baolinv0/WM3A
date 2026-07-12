#!/usr/bin/env python3
"""Run the frozen P2 novelty-isolation experiment on a fixed 7-EV tensor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml

from active_mef.kt2.dataset import ValueTensorDataset
from active_mef.kt2.policy_comparison import (
    build_mean_gain_prior,
    correction_corruption,
    evaluate_group_predictions,
    group_split,
    oracle_predictions,
    pairwise_gate,
    parameter_counts,
    predict_groups,
    prediction_frame,
    prior_predictions,
    save_policy_bundle,
    save_prior,
    subset_groups_by_scene,
    train_grouped_scorer,
)
from active_mef.utils import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, help="override configured training seeds")
    parser.add_argument("--smoke-scenes", type=int, default=None, help="limit each split to N scenes")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _validate_action_pool(dataset: ValueTensorDataset, expected: list[float]) -> None:
    expected_set = {float(x) for x in expected}
    per_scene: dict[str, set[float]] = {}
    for row in dataset.rows:
        scene = str(row["scene_id"])
        per_scene.setdefault(scene, set()).update(float(x) for x in row["current"])
        per_scene[scene].add(float(row["action"]))
    bad = {scene: sorted(values) for scene, values in per_scene.items() if values != expected_set}
    if bad:
        first = next(iter(bad.items()))
        raise ValueError(
            f"action-pool mismatch in {len(bad)} scenes; first={first}, expected={sorted(expected_set)}"
        )


def _method_specs() -> dict[str, tuple[str, str]]:
    return {
        "B1_direct_listwise": ("L4", "direct"),
        "B2_absolute": ("L4", "absolute"),
        "B3_prior_residual_l2": ("L2", "prior_residual"),
        "B4_prior_residual_l4": ("L4", "prior_residual"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    data_cfg = cfg["data"]
    split_path = Path(data_cfg["split_file"])
    if not split_path.exists():
        raise FileNotFoundError(
            f"P2 requires a frozen split file; missing {split_path}. "
            "Do not regenerate the split from a training seed."
        )
    shutil.copy2(split_path, out / "scene_split.json")
    dataset = ValueTensorDataset.create(
        tensor_path=data_cfg["tensor"],
        cache_path=data_cfg["feature_cache"],
        split_path=split_path,
        manifest_path=data_cfg.get("manifest"),
        stage0_config_path=data_cfg.get("stage0_config"),
        train_frac=float(cfg.get("split", {}).get("train_frac", 2.0 / 3.0)),
        val_frac=float(cfg.get("split", {}).get("val_frac", 1.0 / 6.0)),
        seed=int(cfg.get("split", {}).get("seed", 42)),
    )
    expected_pool = [float(x) for x in data_cfg.get("action_pool", [-3, -2, -1, 0, 1, 2, 3])]
    _validate_action_pool(dataset, expected_pool)

    prior = build_mean_gain_prior(dataset)
    prior_path = out / "mean_gain_prior.json"
    save_prior(prior_path, prior)

    grouped: dict[tuple[str, str], list] = {}
    for level in ("L2", "L4"):
        for split in ("train", "val", "test"):
            groups = group_split(dataset, level, split, prior)
            groups = subset_groups_by_scene(groups, args.smoke_scenes)
            grouped[(level, split)] = groups

    # The cache contract must make L2/L4 candidate input dimensions identical.
    l2_dim = grouped[("L2", "train")][0].x.shape[1]
    l4_dim = grouped[("L4", "train")][0].x.shape[1]
    if l2_dim != l4_dim:
        raise ValueError(f"L2/L4 input dimensions differ: {l2_dim} vs {l4_dim}")

    seeds = args.seeds if args.seeds else [int(x) for x in cfg["experiment"].get("seeds", [42, 123, 2026])]
    train_cfg = cfg.get("training", {})
    evaluation_cfg = cfg.get("evaluation", {})
    n_boot = int(evaluation_cfg.get("bootstrap_repeats", 5000))

    all_summary: list[dict] = []
    all_states: list[pd.DataFrame] = []
    all_correction: list[dict] = []
    all_comparisons: list[dict] = []
    parameter_report: dict[str, dict] = {}

    for seed in seeds:
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        test_groups = grouped[("L4", "test")]

        prior_pred = prior_predictions(test_groups)
        prior_summary, prior_state = evaluate_group_predictions(test_groups, prior_pred)
        oracle_summary, oracle_state = evaluate_group_predictions(test_groups, oracle_predictions(test_groups))
        if oracle_summary.mean_regret > 1e-8:
            raise AssertionError(f"oracle regret must be zero, got {oracle_summary.mean_regret}")

        expected_prior = evaluation_cfg.get("expected_prior_regret_db")
        if args.smoke_scenes is None and expected_prior is not None:
            tolerance = float(evaluation_cfg.get("prior_regret_tolerance_db", 5e-4))
            if abs(prior_summary.mean_regret - float(expected_prior)) > tolerance:
                raise AssertionError(
                    f"B0 prior reproduction failed: got {prior_summary.mean_regret:.8f}, "
                    f"expected {float(expected_prior):.8f} ± {tolerance}"
                )

        state_frames: dict[str, pd.DataFrame] = {
            "B0_prior": prior_state,
            "Oracle": oracle_state,
        }
        for method, summary, state_frame, predictions in (
            ("B0_prior", prior_summary, prior_state, prior_pred),
            ("Oracle", oracle_summary, oracle_state, oracle_predictions(test_groups)),
        ):
            method_dir = seed_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame(test_groups, predictions).to_csv(method_dir / "predictions.csv", index=False)
            state_frame.to_csv(method_dir / "per_state_metrics.csv", index=False)
            row = {"seed": seed, "method": method, **summary.as_dict(), "best_epoch": None}
            all_summary.append(row)
            with (method_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(row, handle, indent=2)

        for method, (level, formulation) in _method_specs().items():
            bundle = train_grouped_scorer(
                grouped[(level, "train")],
                grouped[(level, "val")],
                formulation=formulation,
                hidden_dims=tuple(int(x) for x in train_cfg.get("hidden_dims", [256, 128])),
                dropout=float(train_cfg.get("dropout", 0.1)),
                learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
                weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
                states_per_batch=int(train_cfg.get("states_per_batch", 32)),
                max_epochs=int(train_cfg.get("max_epochs", 16)),
                patience=int(train_cfg.get("patience", 8)),
                target_temperature=float(train_cfg.get("target_temperature", 0.25)),
                lambda_regression=float(train_cfg.get("lambda_regression", 0.1)),
                seed=seed,
                device=str(cfg.get("device", "cuda")),
            )
            predictions = predict_groups(bundle, grouped[(level, "test")])
            summary, state_frame = evaluate_group_predictions(grouped[(level, "test")], predictions)
            state_frames[method] = state_frame
            row = {
                "seed": seed,
                "method": method,
                "level": level,
                "formulation": formulation,
                **summary.as_dict(),
                "best_epoch": int(bundle["best_epoch"]),
                "input_dim": int(bundle["input_dim"]),
                "trainable_parameters": int(bundle["trainable_parameters"]),
            }
            all_summary.append(row)
            parameter_report.setdefault(method, {
                "input_dim": int(bundle["input_dim"]),
                "hidden_dims": list(bundle["hidden_dims"]),
                **parameter_counts(bundle["model"]),
            })

            method_dir = seed_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            prediction_frame(grouped[(level, "test")], predictions).to_csv(
                method_dir / "predictions.csv", index=False
            )
            state_frame.to_csv(method_dir / "per_state_metrics.csv", index=False)
            pd.DataFrame(bundle["history"]).to_csv(method_dir / "training_history.csv", index=False)
            save_policy_bundle(method_dir / "model.pt", bundle, method)
            with (method_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(row, handle, indent=2)

        learned_counts = [parameter_report[name]["trainable_parameters"] for name in _method_specs()]
        if len(set(learned_counts)) != 1:
            raise AssertionError(f"capacity matching failed: {dict(zip(_method_specs(), learned_counts))}")

        for method, frame in state_frames.items():
            tagged = frame.copy()
            tagged.insert(0, "seed", seed)
            tagged.insert(1, "method", method)
            all_states.append(tagged)
            if method not in {"B0_prior", "Oracle"}:
                all_correction.append({
                    "seed": seed,
                    "method": method,
                    **correction_corruption(prior_state, frame),
                })

        for gate_name, reference_name in (
            ("P2-A", "B1_direct_listwise"),
            ("P2-B", "B2_absolute"),
            ("P2-C", "B3_prior_residual_l2"),
        ):
            comparison = pairwise_gate(
                state_frames[reference_name],
                state_frames["B4_prior_residual_l4"],
                n_boot=n_boot,
                seed=seed,
            )
            all_comparisons.append({
                "seed": seed,
                "gate": gate_name,
                "reference": reference_name,
                "challenger": "B4_prior_residual_l4",
                **comparison,
            })

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(out / "summary.csv", index=False)
    states_df = pd.concat(all_states, ignore_index=True)
    states_df.to_csv(out / "per_state_metrics.csv", index=False)
    states_df.groupby(["seed", "method", "scene_id"], as_index=False).regret.mean().to_csv(
        out / "per_scene_metrics.csv", index=False
    )
    pd.DataFrame(all_correction).to_csv(out / "correction_corruption.csv", index=False)
    with (out / "pairwise_bootstrap.json").open("w", encoding="utf-8") as handle:
        json.dump(all_comparisons, handle, indent=2)
    with (out / "representation_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump([row for row in all_comparisons if row["gate"] == "P2-C"], handle, indent=2)
    with (out / "model_parameter_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(parameter_report, handle, indent=2)

    integrity = {
        "tensor_sha256": sha256_file(data_cfg["tensor"]),
        "feature_cache_sha256": sha256_file(data_cfg["feature_cache"]),
        "prior_sha256": sha256_file(prior_path),
        "split_sha256": sha256_file(split_path),
        "config_sha256": sha256_file(args.config),
        "cache_tensor_sha256": dataset.cache.tensor_sha256,
        "cache_manifest_sha256": dataset.cache.manifest_sha256,
        "cache_stage0_config_sha256": dataset.cache.stage0_config_sha256,
        "encoder_signature": dataset.cache.encoder_signature,
        "git_commit_at_cache_build": dataset.cache.git_commit,
        "seeds": seeds,
        "candidate_evs": expected_pool,
        "scene_counts": dataset.scene_counts(),
        "smoke_scenes": args.smoke_scenes,
    }
    with (out / "integrity_report.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)

    gates_df = pd.DataFrame(all_comparisons)
    gate_summary = {
        gate: {
            "all_seeds_pass": bool(frame["pass"].all()),
            "mean_improvement_db_across_seeds": float(frame.mean_improvement_db.mean()),
            "std_improvement_db_across_seeds": float(frame.mean_improvement_db.std(ddof=0)),
        }
        for gate, frame in gates_df.groupby("gate")
    }
    with (out / "gate_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(gate_summary, handle, indent=2)

    print(summary_df.to_string(index=False))
    print(json.dumps(gate_summary, indent=2))


if __name__ == "__main__":
    main()
