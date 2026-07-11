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
    mean_spearman_valid: float
    spearman_valid_fraction: float
    constant_prediction_state_rate: float
    constant_true_state_rate: float
    top1_accuracy: float
    top2_recall: float
    num_states: int
    num_rankable_states: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_decisions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: list[dict],
    tie_eps: float = 1e-8,
) -> tuple[DecisionSummary, pd.DataFrame]:
    """Evaluate candidate ranking per current acquisition state.

    ``mean_spearman`` is zero-filled over rankable states: a constant predictor
    on a state with non-constant oracle gains contributes zero rather than being
    silently dropped. ``mean_spearman_valid`` is also reported for conventional
    comparison, together with the valid fraction.
    """
    if not (len(y_true) == len(y_pred) == len(meta)):
        raise ValueError("y_true, y_pred, meta lengths differ")
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(meta):
        groups[str(item["state_key"])].append(idx)

    rows: list[dict] = []
    for state_key, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        true = np.asarray(y_true[idx], dtype=np.float64)
        pred = np.asarray(y_pred[idx], dtype=np.float64)
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

        rankable = bool(len(true) >= 2 and np.std(true) > 1e-12)
        pred_constant = bool(np.std(pred) <= 1e-12)
        if rankable and not pred_constant:
            rho = float(spearmanr(true, pred).statistic)
            if not np.isfinite(rho):
                rho = 0.0
            rho_zero_filled = rho
            spearman_valid = True
        elif rankable:
            rho = float("nan")
            rho_zero_filled = 0.0
            spearman_valid = False
        else:
            rho = float("nan")
            rho_zero_filled = float("nan")
            spearman_valid = False

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
            "rankable": float(rankable),
            "pred_constant": float(pred_constant),
            "spearman_valid": float(spearman_valid),
            "spearman": rho,
            "spearman_zero_filled": rho_zero_filled,
            "top1": float(exact_top1),
            "top2": float(top2_hit),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no decision groups were evaluated")
    rankable_frame = frame[frame.rankable > 0.5]
    valid_frame = rankable_frame[rankable_frame.spearman_valid > 0.5]
    summary = DecisionSummary(
        mean_regret=float(frame.regret.mean()),
        median_regret=float(frame.regret.median()),
        p90_regret=float(frame.regret.quantile(0.9)),
        frac_regret_lt_01=float((frame.regret < 0.1).mean()),
        frac_regret_lt_025=float((frame.regret < 0.25).mean()),
        mean_spearman=float(rankable_frame.spearman_zero_filled.mean()) if len(rankable_frame) else float("nan"),
        mean_spearman_valid=float(valid_frame.spearman.mean()) if len(valid_frame) else float("nan"),
        spearman_valid_fraction=float(len(valid_frame) / len(rankable_frame)) if len(rankable_frame) else float("nan"),
        constant_prediction_state_rate=float(rankable_frame.pred_constant.mean()) if len(rankable_frame) else float("nan"),
        constant_true_state_rate=float(1.0 - len(rankable_frame) / len(frame)),
        top1_accuracy=float(frame.top1.mean()),
        top2_recall=float(frame.top2.mean()),
        num_states=int(len(frame)),
        num_rankable_states=int(len(rankable_frame)),
    )
    return summary, frame


def scene_bootstrap_regret_difference(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Bootstrap scene-level regret improvement: reference - challenger."""
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
