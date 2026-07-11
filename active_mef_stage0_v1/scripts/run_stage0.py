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
from active_mef.utils import stable_int_hash


def nearest_base(evs: list[float], requested: float) -> float:
    return float(min(evs, key=lambda e: abs(float(e) - requested)))


def best_fixed_sets(
    train_ds,
    backend,
    metric: str,
    budgets: list[int],
    base_requested: float,
    max_scenes: int | None = None,
) -> dict[int, list[float]]:
    """Search the strongest non-adaptive bracket on training scenes only."""
    scores = {b: {} for b in budgets}
    for idx, sample in enumerate(tqdm(train_ds, desc="search best fixed")):
        if max_scenes is not None and idx >= max_scenes:
            break
        base = nearest_base(sample.evs, base_requested)
        remaining = [e for e in sample.evs if e != base]
        evaluator = SceneEvaluator(sample, backend, metric)
        for budget in budgets:
            effective_budget = min(int(budget), len(sample.evs))
            k = max(0, effective_budget - 1)
            for combo in combinations(remaining, k):
                subset = tuple(sorted((base,) + tuple(float(e) for e in combo)))
                scores[budget].setdefault(subset, []).append(evaluator.evaluate(list(subset)))
    best: dict[int, list[float]] = {}
    for budget in budgets:
        if not scores[budget]:
            continue
        subset = max(scores[budget], key=lambda key: float(np.mean(scores[budget][key])))
        best[int(budget)] = list(subset)
    return best


def resolve_fixed_subset(
    requested: list[float],
    sample_evs: list[float],
    budget: int,
    base: float,
) -> list[float]:
    """Resolve a fixed bracket without silently changing the requested budget."""
    subset = [float(ev) for ev in requested if float(ev) in sample_evs]
    if base not in subset:
        subset = [base] + subset
    # Preserve uniqueness while keeping configured order.
    deduped: list[float] = []
    for ev in subset:
        if ev not in deduped:
            deduped.append(ev)
    subset = deduped[:budget]
    if len(subset) < budget:
        unused = sorted(
            [ev for ev in sample_evs if ev not in subset],
            key=lambda ev: (abs(ev - base), ev),
        )
        subset.extend(unused[: budget - len(subset)])
    if len(subset) != budget:
        raise RuntimeError(
            f"could not construct fixed subset of size {budget}; "
            f"requested={requested}, available={sample_evs}"
        )
    return subset


def main() -> None:
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
    max_image_size = data_cfg.get("max_image_size")
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

    standard_fixed_sets = {
        int(k): [float(x) for x in v]
        for k, v in cfg["experiment"].get("fixed_sets", {}).items()
    }
    learned_fixed_sets: dict[int, list[float]] = {}
    train_manifest = data_cfg.get("train_manifest")
    if train_manifest:
        train_ds = ManifestExposurePoolDataset(
            train_manifest, candidate_evs, simulator,
            data_cfg.get("limit_train"), max_image_size=max_image_size,
        )
        learned_fixed_sets = best_fixed_sets(
            train_ds,
            backend,
            metric,
            budgets,
            base_requested,
            data_cfg.get("fixed_search_scenes"),
        )
    write_json(out / "fixed_sets.json", {
        "standard_fixed": standard_fixed_sets,
        "best_fixed_train_selected": learned_fixed_sets,
    })

    heuristic = HistogramCoverageHeuristic(**cfg.get("heuristic", {}))
    random_repeats = int(cfg["experiment"].get("random_repeats", 5))
    enumerate_values = bool(cfg["experiment"].get("enumerate_value_tensor", True))
    max_subset_size = int(cfg["experiment"].get("value_tensor_max_subset_size", 2))
    tensor_path = out / "oracle_value_tensor.jsonl"
    if tensor_path.exists():
        tensor_path.unlink()

    rows: list[dict] = []
    gaps: list[dict] = []
    action_rows: list[dict] = []
    for sample in tqdm(test_ds, desc="stage0 scenes"):
        if sample.evs != sorted(candidate_evs):
            raise RuntimeError(
                f"scene {sample.scene_id} action pool {sample.evs} does not match "
                f"configured candidate_evs {sorted(candidate_evs)}"
            )
        base = nearest_base(sample.evs, base_requested)
        evaluator = SceneEvaluator(sample, backend, metric)

        if enumerate_values:
            for rec in evaluator.enumerate_value_tensor(max_subset_size, base):
                append_jsonl(tensor_path, rec)

        for requested_budget in budgets:
            budget = min(requested_budget, len(sample.evs))

            if budget in standard_fixed_sets:
                subset = resolve_fixed_subset(standard_fixed_sets[budget], sample.evs, budget, base)
            else:
                ordered = sorted(sample.evs, key=lambda ev: (abs(ev - base), ev))
                subset = sorted(ordered[:budget])
            rows.append({
                "scene_id": sample.scene_id,
                "budget": budget,
                "method": "standard_fixed",
                "score": evaluator.evaluate(subset),
                "selected": json.dumps(subset),
            })

            if budget in learned_fixed_sets:
                learned_subset = resolve_fixed_subset(learned_fixed_sets[budget], sample.evs, budget, base)
                rows.append({
                    "scene_id": sample.scene_id,
                    "budget": budget,
                    "method": "best_fixed",
                    "score": evaluator.evaluate(learned_subset),
                    "selected": json.dumps(learned_subset),
                })

            hset = rollout_heuristic(sample.exposures, budget, base, heuristic)
            rows.append({
                "scene_id": sample.scene_id,
                "budget": budget,
                "method": "strong_heuristic",
                "score": evaluator.evaluate(hset),
                "selected": json.dumps(hset),
            })

            gset, gscore = evaluator.oracle_greedy(budget, base)
            rows.append({
                "scene_id": sample.scene_id,
                "budget": budget,
                "method": "oracle_greedy",
                "score": gscore,
                "selected": json.dumps(gset),
            })

            sset, sscore = evaluator.oracle_sequence(budget, base)
            rows.append({
                "scene_id": sample.scene_id,
                "budget": budget,
                "method": "oracle_sequence",
                "score": sscore,
                "selected": json.dumps(sset),
            })
            gaps.append({
                "scene_id": sample.scene_id,
                "budget": budget,
                "greedy_score": gscore,
                "sequence_score": sscore,
                "gap": sscore - gscore,
            })
            if len(gset) > 1:
                action_rows.append({
                    "scene_id": sample.scene_id,
                    "budget": budget,
                    "first_oracle_action": gset[1],
                })

            random_scores: list[float] = []
            scene_seed = stable_int_hash(sample.scene_id, modulo=100_003)
            for repeat in range(random_repeats):
                rset = rollout_random(
                    sample.exposures,
                    budget,
                    base,
                    random.Random(seed + 7919 * repeat + scene_seed),
                )
                r_score = evaluator.evaluate(rset)
                random_scores.append(r_score)
                rows.append({
                    "scene_id": sample.scene_id,
                    "budget": budget,
                    "method": f"random_r{repeat}",
                    "score": float(r_score),
                    "selected": json.dumps(rset),
                })
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

    methods = set(df.method.unique())
    comparisons = [
        ("oracle_greedy", "strong_heuristic"),
        ("oracle_greedy", "standard_fixed"),
        ("oracle_greedy", "best_fixed"),
        ("oracle_sequence", "oracle_greedy"),
    ]
    boot: list[dict] = []
    for budget in budgets:
        for method_a, method_b in comparisons:
            if method_a not in methods or method_b not in methods:
                continue
            stat = paired_bootstrap_delta(
                df,
                method_a,
                method_b,
                budget,
                n_boot=int(cfg["experiment"].get("bootstrap_repeats", 1000)),
                seed=seed,
            )
            boot.append({"budget": budget, "method_a": method_a, "method_b": method_b, **stat})
    pd.DataFrame(boot).to_csv(out / "paired_bootstrap.csv", index=False)

    print("\n=== Summary (main methods) ===")
    main_methods = {
        "standard_fixed", "best_fixed", "strong_heuristic",
        "oracle_greedy", "oracle_sequence", "random",
    }
    print(summary[summary.method.isin(main_methods)].to_string(index=False))
    print(f"\nRaw results: {out.resolve()}")


if __name__ == "__main__":
    main()
