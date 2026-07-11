from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import imageio.v3 as iio


def read_image(path: str | Path, max_size: int | None = None) -> np.ndarray:
    """Read LDR/HDR image as float32 RGB HWC.

    Supported directly: common image formats, .hdr/.exr through imageio plugins,
    .npy, and .npz (first array). Integer inputs are normalized by dtype range.
    Float inputs are preserved except NaN/Inf sanitization.

    Args:
        path: Image file path.
        max_size: If set, downscale so the longest edge is at most this many
            pixels (area-average, aspect-ratio preserved).  No upscaling.
            Useful for high-resolution datasets like SICE (3456×5184) where
            full-resolution oracle loops would be prohibitively slow.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".npz":
        z = np.load(p)
        if not z.files:
            raise ValueError(f"Empty npz: {p}")
        arr = z[z.files[0]]
    else:
        arr = iio.imread(p)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got {arr.shape} from {p}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got {arr.shape} from {p}")
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / float(info.max)
    else:
        arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if max_size is not None:
        h, w = arr.shape[:2]
        longest = max(h, w)
        if longest > max_size:
            scale = max_size / longest
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            # Area-average downscale via block-mean (no extra deps).
            # Works by reshaping into blocks and averaging — exact only when
            # dimensions are divisible; otherwise falls back to bilinear-style
            # via OpenCV if available, else nearest-block approximation.
            try:
                import cv2  # noqa: PLC0415
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except ImportError:
                from PIL import Image  # noqa: PLC0415
                pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr


def write_json(path: str | Path, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str | Path, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
