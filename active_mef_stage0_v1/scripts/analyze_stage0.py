#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key_subset(xs) -> tuple[float, ...]:
    return tuple(sorted(float(x) for x in xs))


def analyze_value_structure(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    by_scene = {}
    for r in rows:
        scene = str(r["scene_id"])
        current = key_subset(r["current"])
        action = float(r["action"])
        gain = float(r["gain"])
        by_scene.setdefault(scene, {})[(current, action)] = gain

    violation_rows = []
    action_change_rates = []
    positive_gain_rates = []
    for scene, table in by_scene.items():
        currents = sorted({k[0] for k in table}, key=lambda x: (len(x), x))
        # Best-action diversity across states of the same scene.
        best_actions = []
        gains_all = []
        for current in currents:
            candidates = [(a, g) for (c, a), g in table.items() if c == current]
            if not candidates:
                continue
            best_actions.append(max(candidates, key=lambda x: x[1])[0])
            gains_all.extend([g for _, g in candidates])
        if best_actions:
            action_change_rates.append({
                "scene_id": scene,
                "num_states": len(best_actions),
                "unique_best_actions": len(set(best_actions)),
                "best_action_change_rate": 1.0 - max(best_actions.count(a) for a in set(best_actions)) / len(best_actions),
            })
        if gains_all:
            positive_gain_rates.append(float(np.mean(np.asarray(gains_all) > 0)))

        # Approximate diminishing-return test: A subset B, same candidate action a.
        for A, B in combinations(currents, 2):
            setA, setB = set(A), set(B)
            if setA.issubset(setB):
                small, large = A, B
            elif setB.issubset(setA):
                small, large = B, A
            else:
                continue
            actions = {a for (c, a) in table if c == small} & {a for (c, a) in table if c == large}
            for a in actions:
                g_small = table[(small, a)]
                g_large = table[(large, a)]
                violation = g_small + 1e-12 < g_large
                violation_rows.append({
                    "scene_id": scene,
                    "small_subset": json.dumps(small),
                    "large_subset": json.dumps(large),
                    "action": a,
                    "gain_small": g_small,
                    "gain_large": g_large,
                    "violation": bool(violation),
                    "violation_magnitude": max(0.0, g_large - g_small),
                })

    violations = pd.DataFrame(violation_rows)
    changes = pd.DataFrame(action_change_rates)
    summary = {
        "num_scenes": len(by_scene),
        "mean_positive_gain_rate": float(np.mean(positive_gain_rates)) if positive_gain_rates else None,
        "mean_best_action_change_rate": float(changes.best_action_change_rate.mean()) if not changes.empty else None,
        "mean_unique_best_actions_per_scene": float(changes.unique_best_actions.mean()) if not changes.empty else None,
        "diminishing_return_violation_ratio": float(violations.violation.mean()) if not violations.empty else None,
        "mean_violation_magnitude": float(violations.violation_magnitude.mean()) if not violations.empty else None,
        "num_diminishing_return_comparisons": int(len(violations)),
    }
    return violations, {"summary": summary, "state_action_diversity": changes.to_dict(orient="records")}


def quality_cost_auc(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    rows = []
    for method, g in df.groupby("method"):
        g = g.sort_values("budget")
        x = g["budget"].to_numpy(dtype=float)
        y = g["mean_score"].to_numpy(dtype=float)
        auc = float(np.trapezoid(y, x)) if len(x) > 1 else 0.0
        rows.append({"method": method, "quality_cost_auc": auc, "min_budget": int(x.min()), "max_budget": int(x.max())})
    return pd.DataFrame(rows).sort_values("quality_cost_auc", ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True)
    args = p.parse_args()
    root = Path(args.result_dir)

    auc = quality_cost_auc(root / "summary.csv")
    auc.to_csv(root / "quality_cost_auc.csv", index=False)

    tensor_path = root / "oracle_value_tensor.jsonl"
    if tensor_path.exists():
        rows = load_jsonl(tensor_path)
        violations, payload = analyze_value_structure(rows)
        violations.to_csv(root / "diminishing_returns.csv", index=False)
        with (root / "utility_structure.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print("\nQuality-cost AUC")
    print(auc.to_string(index=False))


if __name__ == "__main__":
    main()
