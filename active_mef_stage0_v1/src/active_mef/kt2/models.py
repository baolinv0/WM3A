"""Small scalar value predictor used by Kill Test 2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import random

import numpy as np


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        if x.ndim != 2 or len(x) == 0:
            raise ValueError(f"x must be a non-empty 2D matrix, got {x.shape}")
        if not np.isfinite(x).all():
            raise FloatingPointError("x contains non-finite values")
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        transformed = ((x - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise FloatingPointError("standardized features contain non-finite values")
        return transformed


class ScalarMLP:
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128), dropout: float = 0.1):
        import torch.nn as nn

        layers: list[nn.Module] = []
        dim = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(dim, hidden), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.module = nn.Sequential(*layers)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_xy(name: str, x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError(f"{name}: expected x=2D/y=1D, got x={x.shape}, y={y.shape}")
    if len(x) != len(y) or len(x) == 0:
        raise ValueError(f"{name}: inconsistent or empty sample counts x={len(x)}, y={len(y)}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError(f"{name}: non-finite feature or target values")


def train_scalar_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    meta_val: list[dict] | None = None,
    selection_metric: str = "decision_regret",
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    max_epochs: int = 120,
    patience: int = 15,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _validate_xy("train", x_train, y_train)
    _validate_xy("val", x_val, y_val)
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("train/val feature dimensions differ")
    if batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    selection_metric = str(selection_metric).lower()
    if selection_metric not in {"decision_regret", "val_loss"}:
        raise ValueError("selection_metric must be 'decision_regret' or 'val_loss'")
    if selection_metric == "decision_regret":
        if meta_val is None or len(meta_val) != len(y_val):
            raise ValueError("decision_regret checkpoint selection requires meta_val aligned with y_val")

    _set_seed(seed)
    selected_device = device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    dev = torch.device(selected_device)

    x_scaler = Standardizer.fit(x_train)
    x_train_s = x_scaler.transform(x_train)
    x_val_s = x_scaler.transform(x_val)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0
    y_train_s = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val_s = ((y_val - y_mean) / y_std).astype(np.float32)

    wrapper = ScalarMLP(x_train.shape[1], hidden_dims, dropout)
    model = wrapper.module.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=1.0)

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(y_train_s[:, None]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    x_val_t = torch.from_numpy(x_val_s).to(dev)
    y_val_t = torch.from_numpy(y_val_s[:, None]).to(dev)

    best_state = copy.deepcopy(model.state_dict())
    best_selection = float("inf")
    best_val_loss = float("inf")
    best_val_regret = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            val_pred_s_t = model(x_val_t).squeeze(1)
            val_loss = float(loss_fn(val_pred_s_t[:, None], y_val_t).detach().cpu())
            val_pred_s = val_pred_s_t.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
        val_pred = val_pred_s * y_std + y_mean

        val_regret = float("nan")
        val_spearman = float("nan")
        val_top1 = float("nan")
        if meta_val is not None:
            from .evaluation import evaluate_decisions

            decision_summary, _ = evaluate_decisions(y_val, val_pred, meta_val)
            val_regret = float(decision_summary.mean_regret)
            val_spearman = float(decision_summary.mean_spearman)
            val_top1 = float(decision_summary.top1_accuracy)

        selection_value = val_regret if selection_metric == "decision_regret" else val_loss
        if not np.isfinite(selection_value):
            raise FloatingPointError(
                f"non-finite checkpoint selection value at epoch {epoch}: {selection_metric}={selection_value}"
            )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_regret": val_regret,
            "val_spearman": val_spearman,
            "val_top1": val_top1,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
        })

        if selection_value < best_selection - 1e-8:
            best_selection = selection_value
            best_val_loss = val_loss
            best_val_regret = val_regret
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_value": best_selection,
        "selection_metric": selection_metric,
        "best_val_loss": best_val_loss,
        "best_val_regret": best_val_regret,
        "device": str(dev),
        "hidden_dims": tuple(hidden_dims),
        "dropout": float(dropout),
    }


def predict(bundle: dict, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    import torch

    if x.ndim != 2:
        raise ValueError(f"prediction features must be 2D, got {x.shape}")
    model = bundle["model"]
    dev = next(model.parameters()).device
    x_s = bundle["x_scaler"].transform(x)
    outputs: list[np.ndarray] = []
    for start in range(0, len(x_s), batch_size):
        xb = torch.from_numpy(x_s[start : start + batch_size]).to(dev)
        with torch.inference_mode():
            pred = model(xb).squeeze(1).detach().cpu().numpy()
        outputs.append(pred.astype(np.float32))
    pred_s = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
    result = pred_s * float(bundle["y_std"]) + float(bundle["y_mean"])
    if not np.isfinite(result).all():
        raise FloatingPointError("predictor produced non-finite outputs")
    return result


def save_bundle(path: str | Path, bundle: dict, level: str) -> None:
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": bundle["model"].state_dict(),
        "input_dim": int(bundle["x_scaler"].mean.size),
        "hidden_dims": list(bundle["hidden_dims"]),
        "dropout": float(bundle["dropout"]),
        "x_mean": bundle["x_scaler"].mean,
        "x_std": bundle["x_scaler"].std,
        "y_mean": float(bundle["y_mean"]),
        "y_std": float(bundle["y_std"]),
        "level": level,
        "best_epoch": int(bundle["best_epoch"]),
        "selection_metric": bundle["selection_metric"],
        "best_selection_value": float(bundle["best_selection_value"]),
        "best_val_loss": float(bundle["best_val_loss"]),
        "best_val_regret": float(bundle["best_val_regret"]),
    }
    torch.save(payload, out)
