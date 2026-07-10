"""Decision-level evaluation for Kill Test 2."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class DecisionSummary:
    mean_regret: float
    median_regret: float
    p90_regret: float
    frac_regret_lt_01: float
    frac_regret_lt_025: float
    mean_spearman: float
    top1_accuracy: float
    top2_recall: float
    num_states: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_decisions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: list[dict],
    tie_eps: float = 1e-8,
) -> tuple[DecisionSummary, pd.DataFrame]:
    """Evaluate candidate ranking per current acquisition state.

    Main metric is Decision Regret:
      max_a v*(a|C) - v*(argmax_a v_hat(a|C)|C)
    """
    if not (len(y_true) == len(y_pred) == len(meta)):
        raise ValueError("y_true, y_pred, meta lengths differ")
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(meta):
        groups[str(item["state_key"])].append(idx)

    rows: list[dict] = []
    for state_key, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        true = y_true[idx]
        pred = y_pred[idx]
        actions = np.asarray([float(meta[i]["action"]) for i in idx], dtype=np.float32)
        best_true = float(true.max())
        true_best_mask = true >= best_true - tie_eps
        chosen_local = int(np.argmax(pred))
        chosen_true = float(true[chosen_local])
        regret = max(0.0, best_true - chosen_true)
        exact_top1 = bool(true_best_mask[chosen_local])
        pred_order = np.argsort(-pred)
        top2 = pred_order[: min(2, len(pred_order))]
        top2_hit = bool(np.any(true_best_mask[top2]))
        if len(true) >= 2 and np.std(true) > 1e-12 and np.std(pred) > 1e-12:
            rho = float(spearmanr(true, pred).statistic)
        else:
            rho = float("nan")
        first = meta[int(idx[0])]
        rows.append({
            "scene_id": str(first["scene_id"]),
            "state_key": state_key,
            "context_size": len(first["current"]),
            "num_candidates": len(indices),
            "best_true_gain": best_true,
            "chosen_true_gain": chosen_true,
            "predicted_action": float(actions[chosen_local]),
            "regret": regret,
            "spearman": rho,
            "top1": float(exact_top1),
            "top2": float(top2_hit),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no decision groups were evaluated")
    summary = DecisionSummary(
        mean_regret=float(frame.regret.mean()),
        median_regret=float(frame.regret.median()),
        p90_regret=float(frame.regret.quantile(0.9)),
        frac_regret_lt_01=float((frame.regret < 0.1).mean()),
        frac_regret_lt_025=float((frame.regret < 0.25).mean()),
        mean_spearman=float(frame.spearman.dropna().mean()) if frame.spearman.notna().any() else float("nan"),
        top1_accuracy=float(frame.top1.mean()),
        top2_recall=float(frame.top2.mean()),
        num_states=int(len(frame)),
    )
    return summary, frame


def scene_bootstrap_regret_difference(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Bootstrap scene-level regret improvement: reference - challenger.

    Positive values mean the challenger has lower decision regret.
    """
    a = reference.groupby("scene_id", as_index=False).regret.mean().rename(columns={"regret": "ref"})
    b = challenger.groupby("scene_id", as_index=False).regret.mean().rename(columns={"regret": "challenger"})
    merged = a.merge(b, on="scene_id", validate="one_to_one")
    diff = (merged.ref - merged.challenger).to_numpy(dtype=np.float64)
    if diff.size == 0:
        raise RuntimeError("no overlapping scenes for bootstrap")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[indices].mean(axis=1)
    return {
        "mean_improvement_db": float(diff.mean()),
        "ci95_low_db": float(np.percentile(means, 2.5)),
        "ci95_high_db": float(np.percentile(means, 97.5)),
        "num_scenes": int(len(diff)),
    }
