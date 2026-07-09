#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import json
import random
import platform
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from active_mef.config import load_yaml
from active_mef.data import ManifestExposurePoolDataset
from active_mef.sim.camera import CameraSimulator
from active_mef.fusion import make_backend
from active_mef.oracle import SceneEvaluator
from active_mef.policies import HistogramCoverageHeuristic, rollout_heuristic, rollout_random
from active_mef.analysis import summarize_results, paired_bootstrap_delta
from active_mef.io import write_json, append_jsonl


def nearest_base(evs: list[float], requested: float) -> float:
    return float(min(evs, key=lambda e: abs(float(e) - requested)))


def best_fixed_sets(train_ds, backend, metric: str, budgets: list[int], base_requested: float, max_scenes: int | None = None):
    datasets = train_ds
    scores = {b: {} for b in budgets}
    for idx, sample in enumerate(tqdm(datasets, desc="search best fixed")):
        if max_scenes is not None and idx >= max_scenes:
            break
        base = nearest_base(sample.evs, base_requested)
        evs = [e for e in sample.evs if e != base]
        evaluator = SceneEvaluator(sample, backend, metric)
        for b in budgets:
            if b == 1:
                combos = [tuple()]
            else:
                combos = combinations(evs, min(b - 1, len(evs)))
            for combo in combos:
                subset = tuple(sorted((base,) + tuple(float(e) for e in combo)))
                scores[b].setdefault(subset, []).append(evaluator.evaluate(list(subset)))
    best = {}
    for b in budgets:
        if not scores[b]:
            continue
        subset = max(scores[b], key=lambda k: float(np.mean(scores[b][k])))
        best[b] = list(subset)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved_config.json", cfg)
    write_json(out / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp": time.time(),
    })

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    sim_cfg = cfg.get("simulator", {})
    simulator = CameraSimulator(**sim_cfg.get("params", {})) if sim_cfg.get("enabled", False) else None
    candidate_evs = [float(x) for x in cfg["experiment"]["candidate_evs"]]

    data_cfg = cfg["dataset"]
    max_image_size = data_cfg.get("max_image_size", None)
    if max_image_size is not None:
        max_image_size = int(max_image_size)

    test_ds = ManifestExposurePoolDataset(
        data_cfg["test_manifest"], candidate_evs, simulator,
        data_cfg.get("limit_test"), max_image_size=max_image_size,
    )
    backend = make_backend(cfg["fusion"])
    metric = cfg["experiment"].get("metric", "psnr")
    budgets = [int(x) for x in cfg["experiment"].get("budgets", [1, 2, 3])]
    base_requested = float(cfg["experiment"].get("base_ev", 0.0))

    fixed_sets = {int(k): [float(x) for x in v] for k, v in cfg["experiment"].get("fixed_sets", {}).items()}
    train_manifest = data_cfg.get("train_manifest")
    if train_manifest:
        train_ds = ManifestExposurePoolDataset(
            train_manifest, candidate_evs, simulator,
            data_cfg.get("limit_train"), max_image_size=max_image_size,
        )
        fixed_sets.update(best_fixed_sets(
            train_ds, backend, metric, budgets, base_requested, data_cfg.get("fixed_search_scenes")
        ))
    write_json(out / "fixed_sets.json", fixed_sets)

    heuristic = HistogramCoverageHeuristic(**cfg.get("heuristic", {}))
    random_repeats = int(cfg["experiment"].get("random_repeats", 5))
    enumerate_values = bool(cfg["experiment"].get("enumerate_value_tensor", True))
    max_subset_size = int(cfg["experiment"].get("value_tensor_max_subset_size", 2))
    tensor_path = out / "oracle_value_tensor.jsonl"
    if tensor_path.exists():
        tensor_path.unlink()

    rows = []
    gaps = []
    action_rows = []
    for sample in tqdm(test_ds, desc="stage0 scenes"):
        base = nearest_base(sample.evs, base_requested)
        evaluator = SceneEvaluator(sample, backend, metric)

        if enumerate_values:
            for rec in evaluator.enumerate_value_tensor(max_subset_size, base):
                append_jsonl(tensor_path, rec)

        for budget in budgets:
            budget = min(budget, len(sample.evs))
            # Best fixed chosen on train split or configured explicitly.
            if budget in fixed_sets:
                # Keep only EVs that exist in this sample (train/test EV sets
                # may differ when using a simulator).
                subset = [e for e in fixed_sets[budget] if e in sample.exposures]
                if base not in subset:
                    subset = [base] + subset
                subset = subset[:budget]
                # If fewer frames matched than the budget, fill up to budget
                # with the nearest unused EVs so best_fixed is always evaluated
                # at the correct budget size.
                if len(subset) < budget:
                    unused = sorted(
                        [e for e in sample.evs if e not in subset],
                        key=lambda e: abs(e - base),
                    )
                    subset = subset + unused[: budget - len(subset)]
            else:
                # Conservative symmetric fallback around base.
                ordered = sorted(sample.evs, key=lambda e: (abs(e - base), e))
                subset = sorted(ordered[:budget])
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "best_fixed", "score": evaluator.evaluate(subset), "selected": json.dumps(subset)})

            hset = rollout_heuristic(sample.exposures, budget, base, heuristic)
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "strong_heuristic", "score": evaluator.evaluate(hset), "selected": json.dumps(hset)})

            gset, gscore = evaluator.oracle_greedy(budget, base)
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "oracle_greedy", "score": gscore, "selected": json.dumps(gset)})

            sset, sscore = evaluator.oracle_sequence(budget, base)
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "oracle_sequence", "score": sscore, "selected": json.dumps(sset)})
            gaps.append({"scene_id": sample.scene_id, "budget": budget, "greedy_score": gscore, "sequence_score": sscore, "gap": sscore - gscore})
            if len(gset) > 1:
                action_rows.append({"scene_id": sample.scene_id, "budget": budget, "first_oracle_action": gset[1]})

            random_scores = []
            for r in range(random_repeats):
                rset = rollout_random(
                    sample.exposures, budget, base,
                    random.Random(seed + 7919 * r + hash(sample.scene_id) % 100003),
                )
                r_score = evaluator.evaluate(rset)
                random_scores.append(r_score)
                # Store each repeat as its own row so downstream analysis can
                # compute per-repeat variance and do proper paired bootstrap.
                rows.append({
                    "scene_id": sample.scene_id,
                    "budget": budget,
                    "method": f"random_r{r}",
                    "score": float(r_score),
                    "selected": json.dumps(rset),
                })
            # Also store the mean repeat as "random" for easy summary lookup.
            rows.append({
                "scene_id": sample.scene_id,
                "budget": budget,
                "method": "random",
                "score": float(np.mean(random_scores)),
                "selected": "mean_over_repeats",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_scene.csv", index=False)
    summary = summarize_results(df)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(out / "greedy_sequence_gap.csv", index=False)
    pd.DataFrame(action_rows).to_csv(out / "oracle_action_distribution.csv", index=False)

    boot = []
    for b in budgets:
        for a, c in [("oracle_greedy", "strong_heuristic"), ("oracle_greedy", "best_fixed"), ("oracle_sequence", "oracle_greedy")]:
            stat = paired_bootstrap_delta(df, a, c, b, n_boot=int(cfg["experiment"].get("bootstrap_repeats", 1000)), seed=seed)
            boot.append({"budget": b, "method_a": a, "method_b": c, **stat})
    pd.DataFrame(boot).to_csv(out / "paired_bootstrap.csv", index=False)

    print("\n=== Summary (main methods) ===")
    # random_r{i} rows exist for variance analysis but clutter the console.
    main_methods = {"best_fixed", "strong_heuristic", "oracle_greedy", "oracle_sequence", "random"}
    print(summary[summary.method.isin(main_methods)].to_string(index=False))
    print(f"\nRaw results: {out.resolve()}")


if __name__ == "__main__":
    main()
