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
        mean = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


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


def train_scalar_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
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
    best_val = float("inf")
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
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
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
        "best_val_loss": best_val,
        "device": str(dev),
        "hidden_dims": tuple(hidden_dims),
        "dropout": float(dropout),
    }


def predict(bundle: dict, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    import torch

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
    return pred_s * float(bundle["y_std"]) + float(bundle["y_mean"])


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
        "best_val_loss": float(bundle["best_val_loss"]),
    }
    torch.save(payload, out)
