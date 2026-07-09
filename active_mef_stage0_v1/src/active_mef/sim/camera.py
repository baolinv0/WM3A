from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def mu_tonemap(x: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.log1p(mu * x) / np.log1p(mu)


def mu_inverse(y: np.ndarray, mu: float = 5000.0) -> np.ndarray:
    y = np.maximum(y, 0.0)
    return np.expm1(y * np.log1p(mu)) / mu


@dataclass
class CameraSimulator:
    """Lightweight Stage-0 exposure simulator.

    This is paper-inspired, not a bit-exact reproduction of AdaptiveAE. It keeps
    the same factorization principle: temporal integration/blur first, then
    exposure-dependent shot/read/ADC noise and sensor clipping.

    For serious dynamic-scene results, replace motion_mode='linear_tmo' with
    precomputed RIFE-based blur pools through the same manifest interface.
    """

    mu: float = 5000.0
    frame_dt: float = 1.0 / 30.0
    base_shutter: float = 1.0 / 120.0
    base_iso: float = 200.0
    full_well_e: float = 12000.0
    read_noise_e: float = 3.0
    adc_noise_dn: float = 1.0 / 4096.0
    gamma: float = 2.2
    reference_percentile: float = 90.0
    reference_level: float = 0.7
    motion_mode: str = "linear_tmo"
    temporal_samples: int = 8
    seed: int = 0

    def normalize_hdr(self, hdr: np.ndarray) -> np.ndarray:
        p = float(np.percentile(hdr[np.isfinite(hdr)], self.reference_percentile))
        scale = self.reference_level / max(p, 1e-8)
        return np.maximum(hdr * scale, 0.0).astype(np.float32)

    def _temporal_integrate(
        self,
        hdr: np.ndarray,
        hdr_next: np.ndarray | None,
        shutter: float,
        frame_dt: float,
    ) -> np.ndarray:
        if hdr_next is None or self.motion_mode == "static" or shutter <= 0:
            return hdr
        alpha_max = min(max(shutter / max(frame_dt, 1e-8), 0.0), 1.0)
        if alpha_max <= 1e-4:
            return hdr
        t0 = mu_tonemap(hdr, self.mu)
        t1 = mu_tonemap(hdr_next, self.mu)
        n = max(2, int(self.temporal_samples))
        acc = np.zeros_like(t0, dtype=np.float32)
        for u in np.linspace(0.0, alpha_max, n, dtype=np.float32):
            acc += (1.0 - u) * t0 + u * t1
        acc /= float(n)
        return mu_inverse(acc, self.mu).astype(np.float32)

    def _simulate_normalized(
        self,
        hdr0: np.ndarray,
        ev: float,
        hdr1: np.ndarray | None,
        frame_dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Core simulation that operates on already-normalized HDR frames.

        Separated from ``simulate`` so that ``make_pool`` can normalize once
        rather than once per EV.
        """
        exposure_mult = 2.0 ** float(ev)
        shutter = self.base_shutter * exposure_mult
        integrated = self._temporal_integrate(hdr0, hdr1, shutter, frame_dt)

        sensor_linear = np.maximum(integrated * exposure_mult, 0.0)
        electrons = np.clip(sensor_linear, 0.0, 1.0) * self.full_well_e
        shot = rng.normal(0.0, np.sqrt(np.maximum(electrons, 0.0)), size=electrons.shape)
        read = rng.normal(0.0, self.read_noise_e, size=electrons.shape)
        noisy_e = electrons + shot + read
        noisy = noisy_e / self.full_well_e
        noisy += rng.normal(0.0, self.adc_noise_dn, size=noisy.shape)
        noisy = np.clip(noisy, 0.0, 1.0)
        ldr = np.power(noisy, 1.0 / self.gamma)
        return ldr.astype(np.float32)

    def simulate(
        self,
        hdr: np.ndarray,
        ev: float,
        hdr_next: np.ndarray | None = None,
        frame_dt: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Public API: normalise the raw HDR then delegate to _simulate_normalized."""
        frame_dt = self.frame_dt if frame_dt is None else float(frame_dt)
        rng = rng or np.random.default_rng(self.seed)
        hdr0 = self.normalize_hdr(hdr)
        hdr1 = self.normalize_hdr(hdr_next) if hdr_next is not None else None
        return self._simulate_normalized(hdr0, ev, hdr1, frame_dt, rng)

    def make_pool(
        self,
        hdr: np.ndarray,
        evs: list[float],
        hdr_next: np.ndarray | None = None,
        frame_dt: float | None = None,
        scene_seed: int = 0,
    ) -> dict[float, np.ndarray]:
        """Simulate the full exposure pool for one scene.

        ``normalize_hdr`` is called once here, not once per EV, so all
        simulated frames share the same radiometric scale.
        """
        frame_dt = self.frame_dt if frame_dt is None else float(frame_dt)
        hdr0 = self.normalize_hdr(hdr)
        hdr1 = self.normalize_hdr(hdr_next) if hdr_next is not None else None
        pool: dict[float, np.ndarray] = {}
        for i, ev in enumerate(evs):
            # Each EV gets a deterministic but independent RNG stream.
            rng = np.random.default_rng(self.seed + int(scene_seed) + 1009 * i)
            pool[float(ev)] = self._simulate_normalized(hdr0, ev, hdr1, frame_dt, rng)
        return pool
