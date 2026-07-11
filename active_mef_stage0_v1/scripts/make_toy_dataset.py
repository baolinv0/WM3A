#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import imageio.v3 as iio


def make_scene(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 0.03 + 0.25 * (x / max(w - 1, 1)) + 0.15 * (y / max(h - 1, 1))
    hdr = np.stack([base * 1.1, base, base * 0.85], axis=-1)
    # bright window
    hdr[h//6:h//2, w//2:5*w//6] += 4.0 + seed * 0.2
    # dark textured object
    mask = (x - w*0.3)**2 + (y - h*0.7)**2 < (min(h,w)*0.15)**2
    texture = 0.02 + 0.02 * rng.random((h, w))
    hdr[mask] = np.stack([texture[mask], texture[mask]*0.8, texture[mask]*0.7], axis=-1)
    return hdr.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--scenes", type=int, default=8)
    p.add_argument("--size", type=int, default=96)
    args = p.parse_args()
    root = Path(args.output)
    hdr_root = root / "hdr_video"
    hdr_root.mkdir(parents=True, exist_ok=True)
    manifest = root / "toy_hdr.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for s in range(args.scenes):
            seq = hdr_root / f"scene_{s:03d}"
            seq.mkdir(parents=True, exist_ok=True)
            hdr0 = make_scene(args.size, args.size, s)
            hdr1 = np.roll(hdr0, shift=(s % 5) + 1, axis=1)
            p0, p1 = seq / "0000.npy", seq / "0001.npy"
            np.save(p0, hdr0)
            np.save(p1, hdr1)
            rec = {
                "scene_id": f"scene_{s:03d}",
                "kind": "hdr_pair",
                "hdr": str(p0.resolve()),
                "hdr_next": str(p1.resolve()),
                "frame_dt": 1/30,
            }
            f.write(json.dumps(rec) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
