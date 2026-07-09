"""Kill Test 2 — state feature extractors.

Three levels for the representation ablation:

  L1  Histogram state   — global luminance histograms, no spatial structure
  L2  Fused-output feature — spatial statistics of the current fused image Y_t
  L4  Fusion-native state  — spatial statistics of [S, W, cov_under, cov_over]
                             from LinearRadianceFusion.fuse_with_state()

All extractors return fixed-length float32 numpy vectors so they can be fed
directly into scikit-learn estimators.

Grid size: 4×4 spatial cells.  This gives enough spatial resolution to capture
regional under/over-exposure without requiring a neural backbone.
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HIST_BINS    = 32
MAX_CONTEXT  = 4      # max frames in a context set (base + up to 3 others)
GRID_ROWS    = 4
GRID_COLS    = 4
N_CELLS      = GRID_ROWS * GRID_COLS   # 16

# Derived dimensions
L1_DIM  = MAX_CONTEXT * (HIST_BINS + 2)   # 4 × 34 = 136
L2_DIM  = N_CELLS * 3 * 2                # 16 × 3ch × (mean+std) = 96
L4_DIM  = N_CELLS * 4 * 2                # 16 × 4maps × (mean+std) = 128
ACT_DIM = 3                               # [ev_norm, delta_from_min, delta_from_max]


# ─────────────────────────────────────────────────────────────────────────────
# Action encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_action(action_ev: float, current_evs: list[float], max_ev: float = 4.0) -> np.ndarray:
    """Encode a candidate EV as a 3-dim vector.

    Components:
      - ev_norm:           action EV normalised to [-1, 1]
      - delta_from_min:    how much darker the action is relative to darkest current frame
      - delta_from_max:    how much brighter the action is relative to brightest current frame
    """
    ev_norm = float(action_ev) / max(max_ev, 1e-6)
    delta_min = float(action_ev) - float(min(current_evs))
    delta_max = float(action_ev) - float(max(current_evs))
    # normalise deltas by max_ev range
    return np.array([ev_norm, delta_min / (2 * max_ev), delta_max / (2 * max_ev)],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial grid utilities
# ─────────────────────────────────────────────────────────────────────────────

def _grid_stats(img: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """Compute mean + std of each grid cell for an (H, W, C) image.

    Returns a flat vector of length rows × cols × C × 2.
    """
    H, W, C = img.shape
    stats = []
    for r in range(rows):
        r0, r1 = r * H // rows, (r + 1) * H // rows
        for c in range(cols):
            c0, c1 = c * W // cols, (c + 1) * W // cols
            patch = img[r0:r1, c0:c1, :]          # (ph, pw, C)
            flat  = patch.reshape(-1, C)            # (ph*pw, C)
            stats.append(flat.mean(axis=0))         # (C,)
            stats.append(flat.std(axis=0) + 1e-8)   # (C,)
    return np.concatenate(stats).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# L1: Histogram state
# ─────────────────────────────────────────────────────────────────────────────

def l1_histogram(
    exposures: dict[float, np.ndarray],
    current_evs: list[float],
) -> np.ndarray:
    """L1 feature: per-frame luminance histogram + extreme ratios.

    Frames are sorted by EV (darkest first) and zero-padded to MAX_CONTEXT.
    Shape: (MAX_CONTEXT × (HIST_BINS + 2),) = (136,)
    """
    sorted_evs = sorted(current_evs)
    parts = []
    for ev in sorted_evs[:MAX_CONTEXT]:
        img = np.clip(exposures[float(ev)], 0.0, 1.0)
        lum = img.mean(axis=2).ravel()
        hist, _ = np.histogram(lum, bins=HIST_BINS, range=(0.0, 1.0))
        hist = hist.astype(np.float32) / max(lum.size, 1)
        shadow_frac = float(np.mean(lum < 0.05))
        satur_frac  = float(np.mean(lum > 0.95))
        parts.append(np.concatenate([hist, [shadow_frac, satur_frac]]).astype(np.float32))

    # Zero-pad to MAX_CONTEXT frames
    while len(parts) < MAX_CONTEXT:
        parts.append(np.zeros(HIST_BINS + 2, dtype=np.float32))

    return np.concatenate(parts)  # (136,)


# ─────────────────────────────────────────────────────────────────────────────
# L2: Fused-output feature
# ─────────────────────────────────────────────────────────────────────────────

def l2_fused_output(fused_image: np.ndarray) -> np.ndarray:
    """L2 feature: spatial grid statistics of the current fused image Y_t.

    This is the critical baseline: given the current best reconstruction,
    can we predict what's still missing — without seeing the internal
    fusion state?

    Input:  fused_image  (H, W, 3) float32 in [0,1]
    Output: flat vector  (96,)
    """
    return _grid_stats(fused_image)   # N_CELLS × 3ch × 2 = 96


# ─────────────────────────────────────────────────────────────────────────────
# L4: Fusion-native state
# ─────────────────────────────────────────────────────────────────────────────

def l4_fusion_state(state: dict[str, np.ndarray]) -> np.ndarray:
    """L4 feature: spatial grid statistics of [S, W, cov_under, cov_over].

    The fusion-native state encodes *how* the current radiance estimate was
    accumulated, not just *what it looks like*.  Specifically:
      - S (weighted radiance sum):  encodes regional radiance magnitude
      - W (weight sum):             encodes how much reliable information
                                    has been collected per region
      - cov_under / cov_over:       fraction of frames that under/over-expose
                                    each region — directly flags missing info

    Input:  state dict from LinearRadianceFusion.fuse_with_state()
    Output: flat vector  (128,)
    """
    S         = state["S"]          # (H, W, 3)
    W         = state["W"]          # (H, W, 1)
    cov_under = state["cov_under"]  # (H, W, 1)
    cov_over  = state["cov_over"]   # (H, W, 1)

    # Normalise S by W to get radiance estimate magnitude (avoid scale issues)
    R_est = S / np.maximum(W, 1e-8)   # (H, W, 3)

    # Use luminance of R_est as a single channel to keep dims symmetric
    R_lum = R_est.mean(axis=2, keepdims=True)    # (H, W, 1)
    R_lum = np.clip(R_lum / (R_lum.max() + 1e-8), 0.0, 1.0)

    # Stack into (H, W, 4) for the grid stats
    state_map = np.concatenate([R_lum, W / (W.max() + 1e-8), cov_under, cov_over], axis=2)

    return _grid_stats(state_map)   # N_CELLS × 4 maps × 2 = 128
