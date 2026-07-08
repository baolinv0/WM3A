from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import imageio.v3 as iio


def read_image(path: str | Path) -> np.ndarray:
    """Read LDR/HDR image as float32 RGB HWC.

    Supported directly: common image formats, .hdr/.exr through imageio plugins,
    .npy, and .npz (first array). Integer inputs are normalized by dtype range.
    Float inputs are preserved except NaN/Inf sanitization.
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
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


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
