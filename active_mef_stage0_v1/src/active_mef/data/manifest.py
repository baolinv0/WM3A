from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterator
import numpy as np

from active_mef.io import read_image
from active_mef.sim.camera import CameraSimulator, mu_tonemap


@dataclass
class ExposurePoolSample:
    scene_id: str
    exposures: dict[float, np.ndarray]
    target: np.ndarray
    metadata: dict

    @property
    def evs(self) -> list[float]:
        return sorted(self.exposures.keys())


class ManifestExposurePoolDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        candidate_evs: list[float] | None = None,
        simulator: CameraSimulator | None = None,
        limit: int | None = None,
        max_image_size: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records: list[dict] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if limit is not None:
            self.records = self.records[: int(limit)]
        if not self.records:
            raise RuntimeError(f"No records in manifest: {self.manifest_path}")
        self.candidate_evs = candidate_evs
        self.simulator = simulator
        self.max_image_size = max_image_size

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ExposurePoolSample]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> ExposurePoolSample:
        rec = self.records[index]
        kind = rec.get("kind", "precomputed")
        scene_id = str(rec.get("scene_id", index))
        if kind == "precomputed":
            exposures: dict[float, np.ndarray] = {}
            for item in rec["exposures"]:
                ev = float(item["ev"])
                exposures[ev] = np.clip(
                    read_image(item["path"], max_size=self.max_image_size), 0.0, 1.0
                )
            target = np.clip(
                read_image(rec["gt"], max_size=self.max_image_size), 0.0, 1.0
            )
            # Guard against EXIF-orientation inconsistency between GT and
            # exposure frames (common in SICE where Label and sequence images
            # were processed independently and may carry different EXIF tags).
            # If target is the exact transpose of the exposure grid, rotate it.
            if exposures:
                ref_shape = next(iter(exposures.values())).shape  # (H, W, C)
                t_h, t_w = target.shape[:2]
                r_h, r_w = ref_shape[:2]
                if (t_h == r_w and t_w == r_h) and (t_h != t_w):
                    # Target is transposed relative to exposures — rotate 90°.
                    target = np.ascontiguousarray(np.rot90(target))
            return ExposurePoolSample(scene_id, exposures, target, rec)

        if kind == "hdr_pair":
            if self.simulator is None or self.candidate_evs is None:
                raise ValueError("hdr_pair requires simulator and candidate_evs")
            hdr = np.maximum(read_image(rec["hdr"], max_size=self.max_image_size), 0.0)
            hdr_next = None
            if rec.get("hdr_next"):
                hdr_next = np.maximum(
                    read_image(rec["hdr_next"], max_size=self.max_image_size), 0.0
                )
            frame_dt = float(rec.get("frame_dt", self.simulator.frame_dt))
            exposures = self.simulator.make_pool(
                hdr=hdr,
                evs=self.candidate_evs,
                hdr_next=hdr_next,
                frame_dt=frame_dt,
                scene_seed=abs(hash(scene_id)) % (2**31 - 1),
            )
            target = mu_tonemap(self.simulator.normalize_hdr(hdr), self.simulator.mu)
            target = np.clip(target, 0.0, 1.0).astype(np.float32)
            return ExposurePoolSample(scene_id, exposures, target, rec)

        raise ValueError(f"Unsupported manifest kind={kind!r} for scene={scene_id}")
