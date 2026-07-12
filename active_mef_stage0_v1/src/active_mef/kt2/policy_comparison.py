"""Grouped-state P2 policy/value comparison for active multi-exposure acquisition.

The module implements the frozen P2 contract:
- B0: train mean-gain prior;
- B1: direct listwise candidate policy;
- B2: absolute value learning;
- B3/B4: fixed prior plus explicitly centered residual prediction.

All learned methods use the same candidate-wise scalar MLP. Training is balanced
per acquisition state rather than per state-action row.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import random
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .dataset import Split, ValueTensorDataset
from .evaluation import evaluate_decisions, scene_bootstrap_regret_difference
from .models import Standardizer

Formulation = Literal["direct", "absolute", "prior_residual"]


def _float_key(value: float) -> str:
    return f"{float(value):.8g}"


def prior_state_key(current: Iterable[float], order_sensitive: bool = False) -> str:
    values = [float(x) for x in current]
    if not order_sensitive:
        values = sorted(values)
    prefix = "seq" if order_sensitive else "set"
    return f"{prefix}||{','.join(_float_key(x) for x in values)}"


def prior_lookup_key(current: Iterable[float], action: float, order_sensitive: bool = False) -> str:
    return f"{prior_state_key(current, order_sensitive)}||a={_float_key(action)}"


@dataclass
class StateGroup:
    scene_id: str
    state_key: str
    current: list[float]
    order_sensitive: bool
    actions: np.ndarray
    x: np.ndarray
    true_gain: np.ndarray
    prior_gain: np.ndarray
    meta: list[dict]

    @property
    def num_candidates(self) -> int:
        return int(len(self.actions))


def build_mean_gain_prior(dataset: ValueTensorDataset) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in dataset.rows:
        scene = str(row["scene_id"])
        if dataset.scene_split.get(scene) != "train":
            continue
        key = prior_lookup_key(row["current"], row["action"], bool(row.get("order_sensitive", False)))
        buckets.setdefault(key, []).append(float(row["gain"]))
    if not buckets:
        raise RuntimeError("cannot build prior: no training rows")
    return {key: float(np.mean(values)) for key, values in buckets.items()}


def save_prior(path: str | Path, prior: dict[str, float]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(prior, handle, indent=2, sort_keys=True)


def group_split(
    dataset: ValueTensorDataset,
    level: str,
    split_name: Split,
    prior: dict[str, float],
) -> list[StateGroup]:
    x, y, meta = dataset.arrays(level, split_name)
    grouped: dict[str, list[int]] = {}
    for idx, item in enumerate(meta):
        grouped.setdefault(str(item["state_key"]), []).append(idx)

    result: list[StateGroup] = []
    for state_key in sorted(grouped):
        indices = grouped[state_key]
        first = meta[indices[0]]
        actions = np.asarray([float(meta[i]["action"]) for i in indices], dtype=np.float32)
        priors: list[float] = []
        for action in actions:
            key = prior_lookup_key(first["current"], float(action), bool(first["order_sensitive"]))
            if key not in prior:
                raise KeyError(f"train prior missing candidate {key} for split={split_name}")
            priors.append(float(prior[key]))
        group = StateGroup(
            scene_id=str(first["scene_id"]),
            state_key=state_key,
            current=[float(v) for v in first["current"]],
            order_sensitive=bool(first["order_sensitive"]),
            actions=actions,
            x=np.asarray(x[indices], dtype=np.float32),
            true_gain=np.asarray(y[indices], dtype=np.float32),
            prior_gain=np.asarray(priors, dtype=np.float32),
            meta=[meta[i] for i in indices],
        )
        if group.x.shape[0] != group.num_candidates or group.num_candidates < 1:
            raise ValueError(f"invalid candidate grouping for {state_key}")
        result.append(group)
    if not result:
        raise RuntimeError(f"no state groups for split={split_name}, level={level}")
    return result


def subset_groups_by_scene(groups: list[StateGroup], max_scenes: int | None) -> list[StateGroup]:
    if max_scenes is None:
        return groups
    scenes = sorted({group.scene_id for group in groups})[: int(max_scenes)]
    allowed = set(scenes)
    return [group for group in groups if group.scene_id in allowed]


def centered_numpy(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("centering expects a non-empty 1D vector")
    return (arr - arr.mean()).astype(np.float32)


def _build_model(input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
    import torch.nn as nn

    layers: list[nn.Module] = []
    dim = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(dim, int(hidden)), nn.ReLU(inplace=True)])
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        dim = int(hidden)
    final = nn.Linear(dim, 1)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


def parameter_counts(model) -> dict[str, int]:
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_scaler(groups: list[StateGroup]) -> Standardizer:
    rows = np.concatenate([group.x for group in groups], axis=0)
    return Standardizer.fit(rows)


def _target_distribution(true_gain, temperature: float):
    import torch

    if temperature <= 0:
        raise ValueError("target_temperature must be positive")
    return torch.softmax(true_gain / float(temperature), dim=0)


def _state_loss(
    model,
    x,
    true_gain,
    prior_gain,
    formulation: Formulation,
    target_temperature: float,
    lambda_regression: float,
):
    import torch
    import torch.nn.functional as F

    raw = model(x).squeeze(1)
    target_prob = _target_distribution(true_gain, target_temperature)
    if formulation == "direct":
        predicted_value = raw
        auxiliary = raw.new_zeros(())
    elif formulation == "absolute":
        predicted_value = raw
        auxiliary = F.smooth_l1_loss(raw, true_gain)
    elif formulation == "prior_residual":
        centered = raw - raw.mean()
        predicted_value = prior_gain + centered
        target_residual = true_gain - prior_gain
        target_residual = target_residual - target_residual.mean()
        auxiliary = F.smooth_l1_loss(centered, target_residual)
    else:
        raise ValueError(f"unknown formulation {formulation}")
    rank_loss = -(target_prob * torch.log_softmax(predicted_value, dim=0)).sum()
    total = rank_loss + float(lambda_regression) * auxiliary
    return total, rank_loss, auxiliary, predicted_value


def _prepare_groups(groups: list[StateGroup], scaler: Standardizer, device):
    import torch

    prepared = []
    for group in groups:
        x = torch.from_numpy(scaler.transform(group.x)).to(device)
        true_gain = torch.from_numpy(group.true_gain).to(device)
        prior_gain = torch.from_numpy(group.prior_gain).to(device)
        prepared.append((group, x, true_gain, prior_gain))
    return prepared


def _predict_prepared(model, prepared, formulation: Formulation) -> list[np.ndarray]:
    import torch

    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for _, x, _, prior_gain in prepared:
            raw = model(x).squeeze(1)
            if formulation == "prior_residual":
                raw = prior_gain + (raw - raw.mean())
            predictions.append(raw.detach().cpu().numpy().astype(np.float32))
    return predictions


def flatten_group_predictions(
    groups: list[StateGroup], predictions: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if len(groups) != len(predictions):
        raise ValueError("group/prediction counts differ")
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    meta: list[dict] = []
    for group, pred in zip(groups, predictions):
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != group.true_gain.shape:
            raise ValueError(f"prediction shape mismatch for {group.state_key}: {pred.shape}")
        y_true.append(group.true_gain)
        y_pred.append(pred)
        meta.extend(group.meta)
    return np.concatenate(y_true), np.concatenate(y_pred), meta


def evaluate_group_predictions(groups: list[StateGroup], predictions: list[np.ndarray]):
    y_true, y_pred, meta = flatten_group_predictions(groups, predictions)
    return evaluate_decisions(y_true, y_pred, meta)


def prior_predictions(groups: list[StateGroup]) -> list[np.ndarray]:
    return [group.prior_gain.copy() for group in groups]


def oracle_predictions(groups: list[StateGroup]) -> list[np.ndarray]:
    return [group.true_gain.copy() for group in groups]


def train_grouped_scorer(
    train_groups: list[StateGroup],
    val_groups: list[StateGroup],
    formulation: Formulation,
    *,
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    states_per_batch: int = 32,
    max_epochs: int = 16,
    patience: int = 8,
    target_temperature: float = 0.25,
    lambda_regression: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    import torch

    if not train_groups or not val_groups:
        raise ValueError("training and validation groups must be non-empty")
    if states_per_batch <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("states_per_batch, max_epochs, and patience must be positive")
    input_dim = int(train_groups[0].x.shape[1])
    if any(group.x.shape[1] != input_dim for group in train_groups + val_groups):
        raise ValueError("inconsistent candidate input dimensions")

    _set_seed(seed)
    selected_device = device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    dev = torch.device(selected_device)
    scaler = _fit_scaler(train_groups)
    train_prepared = _prepare_groups(train_groups, scaler, dev)
    val_prepared = _prepare_groups(val_groups, scaler, dev)
    model = _build_model(input_dim, hidden_dims, dropout).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    best_state = copy.deepcopy(model.state_dict())
    best_tuple = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(train_prepared))
        train_losses: list[float] = []
        for start in range(0, len(order), states_per_batch):
            batch_indices = order[start : start + states_per_batch]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for idx in batch_indices:
                _, x, true_gain, prior_gain = train_prepared[int(idx)]
                total, _, _, _ = _state_loss(
                    model, x, true_gain, prior_gain, formulation,
                    target_temperature, lambda_regression,
                )
                losses.append(total)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for _, x, true_gain, prior_gain in val_prepared:
                total, _, _, _ = _state_loss(
                    model, x, true_gain, prior_gain, formulation,
                    target_temperature, lambda_regression,
                )
                val_losses.append(float(total.detach().cpu()))
        val_predictions = _predict_prepared(model, val_prepared, formulation)
        val_summary, _ = evaluate_group_predictions(val_groups, val_predictions)
        val_loss = float(np.mean(val_losses))
        selection_tuple = (
            float(val_summary.mean_regret),
            float(val_summary.p90_regret),
            -float(val_summary.top1_accuracy),
            val_loss,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_regret": float(val_summary.mean_regret),
            "val_p90_regret": float(val_summary.p90_regret),
            "val_top1": float(val_summary.top1_accuracy),
        })
        if selection_tuple < best_tuple:
            best_tuple = selection_tuple
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "x_scaler": scaler,
        "formulation": formulation,
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_tuple": list(best_tuple),
        "device": str(dev),
        "input_dim": input_dim,
        "hidden_dims": tuple(int(x) for x in hidden_dims),
        "dropout": float(dropout),
        "target_temperature": float(target_temperature),
        "lambda_regression": float(lambda_regression),
        **parameter_counts(model),
    }


def predict_groups(bundle: dict, groups: list[StateGroup]) -> list[np.ndarray]:
    import torch

    dev = next(bundle["model"].parameters()).device
    prepared = _prepare_groups(groups, bundle["x_scaler"], dev)
    return _predict_prepared(bundle["model"], prepared, bundle["formulation"])


def save_policy_bundle(path: str | Path, bundle: dict, method: str) -> None:
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": bundle["model"].state_dict(),
        "method": method,
        "formulation": bundle["formulation"],
        "x_mean": bundle["x_scaler"].mean,
        "x_std": bundle["x_scaler"].std,
        "input_dim": bundle["input_dim"],
        "hidden_dims": list(bundle["hidden_dims"]),
        "dropout": bundle["dropout"],
        "target_temperature": bundle["target_temperature"],
        "lambda_regression": bundle["lambda_regression"],
        "best_epoch": bundle["best_epoch"],
        "best_selection_tuple": bundle["best_selection_tuple"],
    }, out)


def prediction_frame(groups: list[StateGroup], predictions: list[np.ndarray]) -> pd.DataFrame:
    rows: list[dict] = []
    for group, pred in zip(groups, predictions):
        for idx, value in enumerate(np.asarray(pred, dtype=np.float32)):
            rows.append({
                "scene_id": group.scene_id,
                "state_key": group.state_key,
                "current": json.dumps(group.current),
                "action": float(group.actions[idx]),
                "true_gain": float(group.true_gain[idx]),
                "prior_gain": float(group.prior_gain[idx]),
                "pred_gain": float(value),
            })
    return pd.DataFrame(rows)


def correction_corruption(prior_frame: pd.DataFrame, method_frame: pd.DataFrame) -> dict:
    columns = ["scene_id", "state_key", "top1", "regret", "predicted_action"]
    left = prior_frame[columns].rename(columns={
        "top1": "prior_top1", "regret": "prior_regret", "predicted_action": "prior_action"
    })
    right = method_frame[columns].rename(columns={
        "top1": "method_top1", "regret": "method_regret", "predicted_action": "method_action"
    })
    merged = left.merge(right, on=["scene_id", "state_key"], validate="one_to_one")
    prior_wrong = merged.prior_top1 < 0.5
    prior_correct = ~prior_wrong
    corrected = prior_wrong & (merged.method_top1 > 0.5)
    corrupted = prior_correct & (merged.method_top1 < 0.5)
    return {
        "num_states": int(len(merged)),
        "action_change_rate": float((merged.prior_action != merged.method_action).mean()),
        "correction_rate": float(corrected.sum() / max(1, prior_wrong.sum())),
        "corruption_rate": float(corrupted.sum() / max(1, prior_correct.sum())),
        "correction_gain_sum_db": float(merged.loc[corrected, "prior_regret"].sum()),
        "corruption_loss_sum_db": float(merged.loc[corrupted, "method_regret"].sum()),
        "net_quality_gain_db": float(
            merged.loc[corrected, "prior_regret"].sum()
            - merged.loc[corrupted, "method_regret"].sum()
        ),
    }


def pairwise_gate(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    result = scene_bootstrap_regret_difference(reference, challenger, n_boot=n_boot, seed=seed)
    result["pass"] = bool(
        result["mean_improvement_db"] > 0.0 and result["ci95_low_db"] > 0.0
    )
    return result
