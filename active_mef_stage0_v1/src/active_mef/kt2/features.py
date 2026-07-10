"""Kill Test 2 representation features and encoders.

L1: global exposure statistics, intentionally no spatial layout.
L2: frozen feature of the current fused output Y_t.
L4: frozen feature of a strict superset of Y_t: [Y_t, confidence, under, over].

The production path uses ImageNet-pretrained ResNet-18 for L2/L4. A deterministic
``grid_stats`` path is provided only for CPU smoke tests and fast pipeline checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

HIST_BINS = 32
MAX_CONTEXT = 4
GRID_ROWS = 4
GRID_COLS = 4
ACT_DIM = 3
STATE_META_DIM = 2 * MAX_CONTEXT + 1


def _safe_scale(max_abs_ev: float) -> float:
    return max(float(max_abs_ev), 1.0)


def encode_action(action_ev: float, current_evs: Sequence[float], max_abs_ev: float) -> np.ndarray:
    """Candidate query encoding shared by all representation levels."""
    if not current_evs:
        raise ValueError("current_evs must be non-empty")
    scale = _safe_scale(max_abs_ev)
    cur = [float(x) for x in current_evs]
    a = float(action_ev)
    return np.asarray([
        a / scale,
        (a - min(cur)) / (2.0 * scale),
        (a - max(cur)) / (2.0 * scale),
    ], dtype=np.float32)


def encode_state_evs(
    current_evs: Sequence[float],
    max_abs_ev: float,
    max_context: int = MAX_CONTEXT,
) -> np.ndarray:
    """Encode acquisition-set metadata identically for L1/L2/L4.

    Layout: sorted normalized EV values, validity mask, normalized frame count.
    Padding values are masked, so EV=0 is not ambiguous with padding.
    """
    cur = sorted(float(x) for x in current_evs)
    if len(cur) > max_context:
        raise ValueError(f"context size {len(cur)} exceeds max_context={max_context}")
    scale = _safe_scale(max_abs_ev)
    values = np.zeros(max_context, dtype=np.float32)
    mask = np.zeros(max_context, dtype=np.float32)
    if cur:
        values[: len(cur)] = np.asarray(cur, dtype=np.float32) / scale
        mask[: len(cur)] = 1.0
    count = np.asarray([len(cur) / max(max_context, 1)], dtype=np.float32)
    return np.concatenate([values, mask, count])


def l1_global_statistics(
    exposures: dict[float, np.ndarray],
    current_evs: Sequence[float],
    hist_bins: int = HIST_BINS,
    max_context: int = MAX_CONTEXT,
) -> np.ndarray:
    """Strong global-statistics baseline.

    Per selected frame (EV-sorted): luminance histogram, shadow ratio, saturation
    ratio, mean, and standard deviation. No spatial coordinates are retained.
    """
    cur = sorted(float(x) for x in current_evs)
    if len(cur) > max_context:
        raise ValueError(f"context size {len(cur)} exceeds max_context={max_context}")
    per_frame_dim = hist_bins + 4
    parts: list[np.ndarray] = []
    for ev in cur:
        img = np.clip(np.asarray(exposures[ev], dtype=np.float32), 0.0, 1.0)
        lum = img.mean(axis=2).reshape(-1)
        hist, _ = np.histogram(lum, bins=hist_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32) / max(int(lum.size), 1)
        stats = np.asarray([
            float(np.mean(lum < 0.05)),
            float(np.mean(lum > 0.95)),
            float(lum.mean()),
            float(lum.std()),
        ], dtype=np.float32)
        parts.append(np.concatenate([hist, stats]))
    while len(parts) < max_context:
        parts.append(np.zeros(per_frame_dim, dtype=np.float32))
    return np.concatenate(parts, dtype=np.float32)


def _grid_stats(img: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"expected HWC array, got shape={img.shape}")
    h, w, c = img.shape
    parts: list[np.ndarray] = []
    for r in range(rows):
        r0, r1 = r * h // rows, (r + 1) * h // rows
        for col in range(cols):
            c0, c1 = col * w // cols, (col + 1) * w // cols
            patch = img[r0:r1, c0:c1].reshape(-1, c)
            parts.extend([patch.mean(axis=0), patch.std(axis=0)])
    return np.concatenate(parts).astype(np.float32)


def state_to_map(fused: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    """Build the L4 six-channel map [Y_t RGB, confidence, under, over].

    The first three channels are exactly the fused output used by L2. Therefore
    L4 is a strict information superset of L2 under the same fusion backend.
    """
    fused = np.clip(np.asarray(fused, dtype=np.float32), 0.0, 1.0)
    w = np.asarray(state["W"], dtype=np.float32)
    under = np.asarray(state["cov_under"], dtype=np.float32)
    over = np.asarray(state["cov_over"], dtype=np.float32)
    if w.ndim == 2:
        w = w[..., None]
    if under.ndim == 2:
        under = under[..., None]
    if over.ndim == 2:
        over = over[..., None]
    if fused.shape[:2] != w.shape[:2] or fused.shape[:2] != under.shape[:2] or fused.shape[:2] != over.shape[:2]:
        raise ValueError("fused/state maps must share spatial shape")

    conf = np.log1p(np.maximum(w, 0.0))
    denom = float(conf.max())
    if denom > 1e-8:
        conf = conf / denom
    else:
        conf = np.zeros_like(conf)
    return np.concatenate([
        fused,
        np.clip(conf, 0.0, 1.0),
        np.clip(under, 0.0, 1.0),
        np.clip(over, 0.0, 1.0),
    ], axis=2).astype(np.float32)


def l2_grid_feature(fused: np.ndarray) -> np.ndarray:
    return _grid_stats(np.asarray(fused, dtype=np.float32))


def l4_grid_feature(fused: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    return _grid_stats(state_to_map(fused, state))


@dataclass
class FrozenResNet18Encoder:
    """Frozen ResNet-18 global feature encoder with configurable input channels."""

    in_channels: int
    device: str = "cuda"
    weights: str = "imagenet1k_v1"
    weights_path: str | None = None
    image_size: int = 224

    def __post_init__(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18

        self.torch = torch
        chosen_device = self.device
        if chosen_device.startswith("cuda") and not torch.cuda.is_available():
            chosen_device = "cpu"
        self._device = torch.device(chosen_device)

        if self.weights_path:
            model = resnet18(weights=None)
            raw = torch.load(Path(self.weights_path), map_location="cpu")
            if isinstance(raw, dict) and "state_dict" in raw:
                raw = raw["state_dict"]
            if not isinstance(raw, dict):
                raise ValueError("weights_path must contain a state_dict-like mapping")
            state_dict = {str(k).removeprefix("module."): v for k, v in raw.items()}
            model.load_state_dict(state_dict, strict=True)
        elif self.weights.lower() in {"imagenet1k_v1", "default"}:
            try:
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception as exc:  # pragma: no cover - network/cache dependent
                raise RuntimeError(
                    "Could not load ImageNet ResNet-18 weights. Pre-download the torchvision "
                    "weights or pass --resnet-weights PATH. Random fallback is intentionally disabled."
                ) from exc
        elif self.weights.lower() == "none":
            model = resnet18(weights=None)
        else:
            raise ValueError(f"unsupported weights mode: {self.weights}")

        if self.in_channels != 3:
            old = model.conv1
            new = nn.Conv2d(
                self.in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            with torch.no_grad():
                new.weight[:, :3].copy_(old.weight)
                extra = old.weight.mean(dim=1, keepdim=True)
                for channel in range(3, self.in_channels):
                    new.weight[:, channel : channel + 1].copy_(extra)
            model.conv1 = new

        model.fc = nn.Identity()
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @property
    def output_dim(self) -> int:
        return 512

    def _normalize(self, x):
        torch = self.torch
        means = [0.485, 0.456, 0.406] + [0.5] * max(0, self.in_channels - 3)
        stds = [0.229, 0.224, 0.225] + [0.25] * max(0, self.in_channels - 3)
        mean = torch.tensor(means, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(stds, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return (x - mean) / std

    def encode(self, images: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F

        if not images:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            arr = np.stack([np.asarray(img, dtype=np.float32).transpose(2, 0, 1) for img in chunk])
            if arr.shape[1] != self.in_channels:
                raise ValueError(f"expected {self.in_channels} channels, got {arr.shape[1]}")
            x = torch.from_numpy(arr).to(self._device)
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = self._normalize(x)
            with torch.inference_mode():
                feat = self.model(x)
            outputs.append(feat.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)
