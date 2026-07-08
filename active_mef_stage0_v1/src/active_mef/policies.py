from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np


@dataclass
class HistogramCoverageHeuristic:
    gamma: float = 2.2
    reliable_low: float = 0.04
    reliable_high: float = 0.96
    redundancy_penalty: float = 0.05

    def _predict_from_reference(self, ref: np.ndarray, ref_ev: float, cand_ev: float) -> np.ndarray:
        lin = np.power(np.clip(ref, 0, 1), self.gamma)
        pred_lin = lin * (2.0 ** (cand_ev - ref_ev))
        return np.power(np.clip(pred_lin, 0, 1), 1.0 / self.gamma)

    def choose(self, exposures: dict[float, np.ndarray], selected: list[float], remaining: list[float]) -> float:
        if not remaining:
            raise ValueError("No remaining actions")
        ref_ev = min(selected, key=lambda e: abs(e))
        ref = exposures[float(ref_ev)]
        current_reliable = np.zeros(ref.shape[:2], dtype=bool)
        for ev in selected:
            img = exposures[float(ev)].mean(axis=2)
            current_reliable |= (img > self.reliable_low) & (img < self.reliable_high)
        best_ev, best_score = None, -float("inf")
        for ev in remaining:
            pred = self._predict_from_reference(ref, ref_ev, ev).mean(axis=2)
            cand_rel = (pred > self.reliable_low) & (pred < self.reliable_high)
            gain = np.mean((~current_reliable) & cand_rel)
            overlap = np.mean(current_reliable & cand_rel)
            score = float(gain - self.redundancy_penalty * overlap)
            if score > best_score:
                best_ev, best_score = ev, score
        return float(best_ev)


def rollout_heuristic(exposures, budget: int, base_ev: float, heuristic: HistogramCoverageHeuristic) -> list[float]:
    selected = [float(base_ev)]
    while len(selected) < budget:
        remaining = [e for e in sorted(exposures) if e not in selected]
        if not remaining:
            break
        selected.append(heuristic.choose(exposures, selected, remaining))
    return selected


def rollout_random(exposures, budget: int, base_ev: float, rng: random.Random) -> list[float]:
    remaining = [e for e in sorted(exposures) if e != base_ev]
    rng.shuffle(remaining)
    return [float(base_ev)] + [float(e) for e in remaining[: max(0, budget - 1)]]
