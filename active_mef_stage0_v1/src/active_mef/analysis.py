from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "budget"]
    rows = []
    for keys, g in df.groupby(group_cols):
        method, budget = keys
        vals = g["score"].to_numpy(dtype=float)
        rows.append({
            "method": method,
            "budget": int(budget),
            "mean_score": float(np.mean(vals)),
            "std_score": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values(["budget", "mean_score"], ascending=[True, False])


def paired_bootstrap_delta(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    budget: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    a = df[(df.method == method_a) & (df.budget == budget)][["scene_id", "score"]].rename(columns={"score": "a"})
    b = df[(df.method == method_b) & (df.budget == budget)][["scene_id", "score"]].rename(columns={"score": "b"})
    m = a.merge(b, on="scene_id")
    d = (m.a - m.b).to_numpy(dtype=float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(len(d)),
    }
