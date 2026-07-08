from __future__ import annotations

from pathlib import Path
import json
import math
import re
import warnings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
HDR_EXTS = {".hdr", ".exr", ".npy", ".npz", ".tif", ".tiff"}


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def _image_files(folder: Path, exts: set[str] = IMAGE_EXTS) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts], key=_natural_key)


def _parse_exposure_sidecar(seq_dir: Path, n: int) -> list[float] | None:
    for name in ("exposure.txt", "exposures.txt", "Exposure.txt"):
        p = seq_dir / name
        if not p.exists():
            continue
        vals = []
        for token in p.read_text(encoding="utf-8", errors="ignore").replace(",", " ").split():
            try:
                vals.append(float(token))
            except ValueError:
                pass
        if len(vals) != n:
            continue
        arr = vals
        if all(v > 0 for v in arr) and max(arr) / max(min(arr), 1e-12) > 2.0:
            mid = sorted(arr)[len(arr) // 2]
            return [math.log2(v / mid) for v in arr]
        med = sorted(arr)[len(arr) // 2]
        return [v - med for v in arr]
    return None


def _ordinal_evs(n: int, step: float = 1.0) -> list[float]:
    center = (n - 1) / 2.0
    return [(i - center) * step for i in range(n)]


def _has_duplicate_evs(evs: list[float], decimals: int = 8) -> bool:
    rounded = [round(float(ev), decimals) for ev in evs]
    return len(rounded) != len(set(rounded))


def build_sequence_pool_manifest(
    input_root: str | Path,
    gt_root: str | Path,
    output_jsonl: str | Path,
    ordinal_step: float = 1.0,
    gt_in_subdir: bool = False,
) -> int:
    input_root = Path(input_root)
    gt_root = Path(gt_root)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for seq_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        frames = _image_files(seq_dir)
        if len(frames) < 2:
            continue
        evs = _parse_exposure_sidecar(seq_dir, len(frames))
        ev_source = "sidecar"
        if evs is None:
            evs = _ordinal_evs(len(frames), step=ordinal_step)
            ev_source = "ordinal"

        if _has_duplicate_evs(evs):
            warnings.warn(
                f"Duplicate EVs in scene {seq_dir.name}: {evs}; skipping scene to avoid dict overwrite.",
                RuntimeWarning,
            )
            continue

        if gt_in_subdir:
            gt_candidates = _image_files(gt_root / seq_dir.name)
        else:
            gt_candidates = [p for p in gt_root.glob(f"{seq_dir.name}.*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not gt_candidates:
            continue
        rec = {
            "scene_id": seq_dir.name,
            "kind": "precomputed",
            "gt": str(gt_candidates[0].resolve()),
            "ev_source": ev_source,
            "exposures": [{"ev": float(ev), "path": str(path.resolve())} for ev, path in zip(evs, frames)],
        }
        records.append(rec)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def build_hdr_video_manifest(
    hdr_root: str | Path,
    output_jsonl: str | Path,
    frame_dt: float = 1.0 / 30.0,
    stride: int = 1,
) -> int:
    root = Path(hdr_root)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    dirs = [root] + [p for p in root.rglob("*") if p.is_dir()]
    used = set()
    for seq_dir in dirs:
        frames = _image_files(seq_dir, HDR_EXTS)
        if len(frames) < 2:
            continue
        key = tuple(str(p.resolve()) for p in frames)
        if key in used:
            continue
        used.add(key)
        for i in range(0, len(frames) - 1, max(1, stride)):
            rec = {
                "scene_id": f"{seq_dir.name}_{i:06d}",
                "kind": "hdr_pair",
                "hdr": str(frames[i].resolve()),
                "hdr_next": str(frames[i + 1].resolve()),
                "frame_dt": float(frame_dt),
            }
            records.append(rec)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)
