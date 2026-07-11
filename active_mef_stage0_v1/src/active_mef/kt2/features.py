"""Kill Test 2 representation features and encoders.

L0: EV-set metadata + candidate action only.
L1: global exposure statistics, intentionally no spatial layout.
L2: frozen RGB feature of the current fused output Y_t, followed by a zero
    auxiliary block.
L4: the same frozen RGB feature, followed by fixed spatial statistics of the
    LinearRadiance accumulation state (S, W, under, over).

The dual-path design avoids feeding non-RGB state maps through frozen ImageNet
BatchNorm statistics. L2 and L4 have exactly the same feature dimensionality and
therefore use exactly the same scalar predictor capacity.
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
STATE_AUX_CHANNELS = 6  # compressed S RGB, W, under, over
STATE_AUX_GRID_DIM = GRID_ROWS * GRID_COLS * 2 * STATE_AUX_CHANNELS


def _safe_scale(max_abs_ev: float) -> float:
    return max(float(max_abs_ev), 1.0)


def encode_action(action_ev: float, current_evs: Sequence[float], max_abs_ev: float) -> np.ndarray:
    """Candidate query encoding shared by all representation levels."""
    if not current_evs:
        raise ValueError("current_evs must be non-empty")
    scale = _safe_scale(max_abs_ev)
    cur = [float(x) for x in current_evs]
    action = float(action_ev)
    return np.asarray([
        action / scale,
        (action - min(cur)) / (2.0 * scale),
        (action - max(cur)) / (2.0 * scale),
    ], dtype=np.float32)


def encode_state_evs(
    current_evs: Sequence[float],
    max_abs_ev: float,
    max_context: int = MAX_CONTEXT,
) -> np.ndarray:
    """Encode acquisition-set metadata identically for L0/L1/L2/L4."""
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
    """Global-statistics baseline without spatial coordinates."""
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
    h, w, channels = img.shape
    parts: list[np.ndarray] = []
    for row in range(rows):
        r0, r1 = row * h // rows, (row + 1) * h // rows
        for col in range(cols):
            c0, c1 = col * w // cols, (col + 1) * w // cols
            patch = img[r0:r1, c0:c1].reshape(-1, channels)
            parts.extend([patch.mean(axis=0), patch.std(axis=0)])
    return np.concatenate(parts).astype(np.float32)


def state_aux_map(state: dict[str, np.ndarray]) -> np.ndarray:
    """Return six current-state channels [compressed S RGB, W, under, over]."""
    radiance_sum = np.asarray(state["S"], dtype=np.float32)
    weights = np.asarray(state["W"], dtype=np.float32)
    under = np.asarray(state["cov_under"], dtype=np.float32)
    over = np.asarray(state["cov_over"], dtype=np.float32)
    if weights.ndim == 2:
        weights = weights[..., None]
    if under.ndim == 2:
        under = under[..., None]
    if over.ndim == 2:
        over = over[..., None]
    if radiance_sum.ndim != 3 or radiance_sum.shape[2] != 3:
        raise ValueError(f"state S must be HWC RGB, got {radiance_sum.shape}")
    spatial = radiance_sum.shape[:2]
    if any(arr.shape[:2] != spatial for arr in (weights, under, over)):
        raise ValueError("state maps must share spatial shape")

    # Fixed monotonic compression; no per-state normalization that could erase
    # absolute state differences.
    s_comp = np.log1p(np.maximum(radiance_sum, 0.0)) / np.log1p(64.0)
    confidence = np.maximum(weights, 0.0) / float(MAX_CONTEXT)
    aux = np.concatenate([
        np.clip(s_comp, 0.0, 1.0),
        np.clip(confidence, 0.0, 1.0),
        np.clip(under, 0.0, 1.0),
        np.clip(over, 0.0, 1.0),
    ], axis=2).astype(np.float32)
    if aux.shape[2] != STATE_AUX_CHANNELS:
        raise RuntimeError(f"unexpected auxiliary channel count: {aux.shape}")
    return aux


def state_to_map(fused: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    """Nine-channel visualization/debug map [Y_t RGB, state auxiliary maps]."""
    fused = np.clip(np.asarray(fused, dtype=np.float32), 0.0, 1.0)
    aux = state_aux_map(state)
    if fused.shape[:2] != aux.shape[:2]:
        raise ValueError("fused output and state maps must share spatial shape")
    return np.concatenate([fused, aux], axis=2).astype(np.float32)


def state_aux_grid_feature(state: dict[str, np.ndarray]) -> np.ndarray:
    feature = _grid_stats(state_aux_map(state))
    if feature.size != STATE_AUX_GRID_DIM:
        raise RuntimeError(f"unexpected state auxiliary feature dimension: {feature.size}")
    return feature


def make_l2_dual_feature(rgb_feature: np.ndarray) -> np.ndarray:
    rgb_feature = np.asarray(rgb_feature, dtype=np.float32).reshape(-1)
    return np.concatenate([rgb_feature, np.zeros(STATE_AUX_GRID_DIM, dtype=np.float32)])


def make_l4_dual_feature(rgb_feature: np.ndarray, aux_feature: np.ndarray) -> np.ndarray:
    rgb_feature = np.asarray(rgb_feature, dtype=np.float32).reshape(-1)
    aux_feature = np.asarray(aux_feature, dtype=np.float32).reshape(-1)
    if aux_feature.size != STATE_AUX_GRID_DIM:
        raise ValueError(f"expected aux feature dim {STATE_AUX_GRID_DIM}, got {aux_feature.size}")
    return np.concatenate([rgb_feature, aux_feature]).astype(np.float32)


def l2_grid_feature(fused: np.ndarray) -> np.ndarray:
    return make_l2_dual_feature(_grid_stats(np.asarray(fused, dtype=np.float32)))


def l4_grid_feature(fused: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    return make_l4_dual_feature(
        _grid_stats(np.asarray(fused, dtype=np.float32)),
        state_aux_grid_feature(state),
    )


@dataclass
class FrozenResNet18Encoder:
    """Frozen ImageNet ResNet-18 RGB global feature encoder."""

    in_channels: int = 3
    device: str = "cuda"
    weights: str = "imagenet1k_v1"
    weights_path: str | None = None
    image_size: int = 224

    def __post_init__(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18

        if self.in_channels != 3:
            raise ValueError(
                "Kill Test 2 production encoder is RGB-only. "
                "State maps use a separate fixed-statistics path."
            )
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
            state_dict = {str(key).removeprefix("module."): value for key, value in raw.items()}
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

        model.fc = nn.Identity()
        model.eval().to(self._device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @property
    def output_dim(self) -> int:
        return 512

    def _normalize(self, tensor):
        torch = self.torch
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def encode(self, images: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F

        if not images:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            array = np.stack([
                np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
                for image in chunk
            ])
            if array.shape[1] != 3:
                raise ValueError(f"expected RGB images, got {array.shape[1]} channels")
            tensor = torch.from_numpy(array).to(self._device)
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            tensor = self._normalize(tensor)
            with torch.inference_mode():
                feature = self.model(tensor)
            outputs.append(feature.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)
