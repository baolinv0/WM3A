from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterator
import numpy as np

from active_mef.io import read_image
from active_mef.sim.camera import CameraSimulator, mu_tonemap
from active_mef.utils import stable_int_hash


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
        ev_match_tolerance: float = 1e-6,
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
        if candidate_evs is None:
            self.candidate_evs = None
        else:
            values = [float(x) for x in candidate_evs]
            if len(values) != len(set(values)):
                raise ValueError(f"candidate_evs contains duplicates: {values}")
            self.candidate_evs = values
        self.simulator = simulator
        self.max_image_size = max_image_size
        self.ev_match_tolerance = float(ev_match_tolerance)
        if self.ev_match_tolerance < 0:
            raise ValueError("ev_match_tolerance must be non-negative")

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ExposurePoolSample]:
        for i in range(len(self)):
            yield self[i]

    def _precomputed_items(self, rec: dict, scene_id: str) -> list[tuple[float, dict]]:
        raw = [(float(item["ev"]), item) for item in rec["exposures"]]
        raw_evs = [ev for ev, _ in raw]
        if len(raw_evs) != len(set(raw_evs)):
            raise ValueError(f"scene {scene_id} has duplicate EV entries: {raw_evs}")
        if self.candidate_evs is None:
            return raw

        selected: list[tuple[float, dict]] = []
        for requested in self.candidate_evs:
            matches = [(actual, item) for actual, item in raw if abs(actual - requested) <= self.ev_match_tolerance]
            if not matches:
                raise ValueError(
                    f"scene {scene_id} is missing configured candidate EV {requested}; "
                    f"available={sorted(raw_evs)}"
                )
            if len(matches) > 1:
                raise ValueError(f"scene {scene_id} has ambiguous matches for candidate EV {requested}")
            _, item = matches[0]
            # Use the configured EV as the dictionary key so every scene exposes
            # exactly the same action space, even if a sidecar contains tiny
            # floating-point deviations.
            selected.append((float(requested), item))
        return selected

    def __getitem__(self, index: int) -> ExposurePoolSample:
        rec = self.records[index]
        kind = rec.get("kind", "precomputed")
        scene_id = str(rec.get("scene_id", index))
        if kind == "precomputed":
            exposures: dict[float, np.ndarray] = {}
            for ev, item in self._precomputed_items(rec, scene_id):
                exposures[ev] = np.clip(
                    read_image(item["path"], max_size=self.max_image_size), 0.0, 1.0
                )
            if not exposures:
                raise RuntimeError(f"scene {scene_id} has no usable exposures")
            shapes = {img.shape for img in exposures.values()}
            if len(shapes) != 1:
                raise ValueError(f"scene {scene_id} exposure shapes differ: {sorted(shapes)}")

            target = np.clip(
                read_image(rec["gt"], max_size=self.max_image_size), 0.0, 1.0
            )
            # Guard against EXIF-orientation inconsistency between GT and
            # exposure frames (common in SICE where Label and sequence images
            # were processed independently and may carry different EXIF tags).
            ref_shape = next(iter(exposures.values())).shape  # (H, W, C)
            t_h, t_w = target.shape[:2]
            r_h, r_w = ref_shape[:2]
            if (t_h == r_w and t_w == r_h) and (t_h != t_w):
                target = np.ascontiguousarray(np.rot90(target))
            if target.shape != ref_shape:
                raise ValueError(
                    f"scene {scene_id} target shape {target.shape} does not match exposure shape {ref_shape}"
                )
            metadata = dict(rec)
            metadata["effective_candidate_evs"] = sorted(exposures)
            return ExposurePoolSample(scene_id, exposures, target, metadata)

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
                scene_seed=stable_int_hash(scene_id),
            )
            target = mu_tonemap(self.simulator.normalize_hdr(hdr), self.simulator.mu)
            target = np.clip(target, 0.0, 1.0).astype(np.float32)
            metadata = dict(rec)
            metadata["effective_candidate_evs"] = sorted(exposures)
            return ExposurePoolSample(scene_id, exposures, target, metadata)

        raise ValueError(f"Unsupported manifest kind={kind!r} for scene={scene_id}")
