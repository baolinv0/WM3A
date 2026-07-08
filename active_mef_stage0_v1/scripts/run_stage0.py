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
import zlib

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


def resolve_fixed_subset(
    requested: list[float],
    available: list[float],
    base: float,
    budget: int,
) -> list[float]:
    """Resolve a train-selected fixed set on a possibly different test EV pool.

    Exact matches are kept first. Missing slots are filled with unused exposures
    nearest to the base exposure so the baseline is never silently evaluated
    with fewer frames than the requested exact budget.
    """
    subset = [float(e) for e in requested if float(e) in available]
    if base not in subset:
        subset = [float(base)] + subset

    # Remove duplicates while preserving order.
    subset = list(dict.fromkeys(subset))[:budget]
    if len(subset) < budget:
        remaining = [float(e) for e in available if float(e) not in subset]
        remaining = sorted(remaining, key=lambda e: (abs(e - base), e))
        subset.extend(remaining[: budget - len(subset)])
    return subset[:budget]


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
    test_ds = ManifestExposurePoolDataset(
        data_cfg["test_manifest"], candidate_evs, simulator, data_cfg.get("limit_test")
    )
    backend = make_backend(cfg["fusion"])
    metric = cfg["experiment"].get("metric", "psnr")
    budgets = [int(x) for x in cfg["experiment"].get("budgets", [1, 2, 3])]
    base_requested = float(cfg["experiment"].get("base_ev", 0.0))

    fixed_sets = {int(k): [float(x) for x in v] for k, v in cfg["experiment"].get("fixed_sets", {}).items()}
    train_manifest = data_cfg.get("train_manifest")
    if train_manifest:
        train_ds = ManifestExposurePoolDataset(
            train_manifest, candidate_evs, simulator, data_cfg.get("limit_train")
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
                subset = resolve_fixed_subset(fixed_sets[budget], sample.evs, base, budget)
            else:
                # Conservative symmetric fallback around base.
                ordered = sorted(sample.evs, key=lambda e: (abs(e - base), e))
                subset = sorted(ordered[:budget])
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "best_fixed", "repeat": None, "score": evaluator.evaluate(subset), "selected": json.dumps(subset)})

            hset = rollout_heuristic(sample.exposures, budget, base, heuristic)
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "strong_heuristic", "repeat": None, "score": evaluator.evaluate(hset), "selected": json.dumps(hset)})

            gset, gscore = evaluator.oracle_greedy(budget, base)
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "oracle_greedy", "repeat": None, "score": gscore, "selected": json.dumps(gset)})

            sset, sscore = evaluator.oracle_sequence(budget, base)
            rows.append({"scene_id": sample.scene_id, "budget": budget, "method": "oracle_sequence", "repeat": None, "score": sscore, "selected": json.dumps(sset)})
            gaps.append({"scene_id": sample.scene_id, "budget": budget, "greedy_score": gscore, "sequence_score": sscore, "gap": sscore - gscore})
            if len(gset) > 1:
                action_rows.append({"scene_id": sample.scene_id, "budget": budget, "first_oracle_action": gset[1]})

            scene_seed = zlib.crc32(sample.scene_id.encode("utf-8")) & 0xFFFFFFFF
            for r in range(random_repeats):
                rng = random.Random(seed + 7919 * r + scene_seed)
                rset = rollout_random(sample.exposures, budget, base, rng)
                rows.append({
                    "scene_id": sample.scene_id,
                    "budget": budget,
                    "method": "random",
                    "repeat": r,
                    "score": float(evaluator.evaluate(rset)),
                    "selected": json.dumps(rset),
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

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nRaw results: {out.resolve()}")


if __name__ == "__main__":
    main()
