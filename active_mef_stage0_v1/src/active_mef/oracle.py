from __future__ import annotations

from itertools import combinations, permutations
import numpy as np

from active_mef.data.manifest import ExposurePoolSample
from active_mef.fusion.base import FusionBackend
from active_mef.metrics import metric_fn


class SceneEvaluator:
    def __init__(self, sample: ExposurePoolSample, backend: FusionBackend, metric: str = "psnr"):
        self.sample = sample
        self.backend = backend
        self.metric = metric_fn(metric)
        self.order_sensitive = bool(getattr(backend, "order_sensitive", False))
        self._cache: dict[tuple[float, ...], tuple[float, np.ndarray]] = {}

    def _key(self, selected_evs: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        values = tuple(float(e) for e in selected_evs)
        return values if self.order_sensitive else tuple(sorted(values))

    def evaluate(self, selected_evs: list[float] | tuple[float, ...]) -> float:
        key = self._key(selected_evs)
        if key not in self._cache:
            pred = self.backend.fuse(self.sample.exposures, list(key))
            score = float(self.metric(pred, self.sample.target))
            self._cache[key] = (score, pred)
        return self._cache[key][0]

    def prediction(self, selected_evs: list[float] | tuple[float, ...]) -> np.ndarray:
        key = self._key(selected_evs)
        self.evaluate(key)
        return self._cache[key][1]

    def marginal_gain(self, current: list[float], action: float) -> float:
        before = self.evaluate(current)
        after = self.evaluate(current + [float(action)])
        return after - before

    def oracle_greedy(self, budget: int, base_ev: float) -> tuple[list[float], float]:
        selected = [float(base_ev)]
        while len(selected) < budget:
            remaining = [e for e in self.sample.evs if e not in selected]
            if not remaining:
                break
            gains = [(self.marginal_gain(selected, e), e) for e in remaining]
            _, best_ev = max(gains, key=lambda x: x[0])
            selected.append(float(best_ev))
        return selected, self.evaluate(selected)

    def oracle_sequence(self, budget: int, base_ev: float) -> tuple[list[float], float]:
        """Exhaustive optimal subset/sequence search for exactly ``budget`` frames.

        For order-independent backends this enumerates combinations. For a
        recurrent/order-sensitive backend it enumerates acquisition-order
        permutations after the fixed base exposure.
        """
        if budget <= 1:
            return [float(base_ev)], self.evaluate([base_ev])
        remaining = [e for e in self.sample.evs if e != base_ev]
        k = min(budget - 1, len(remaining))
        if k == 0:
            return [float(base_ev)], self.evaluate([base_ev])
        candidates = permutations(remaining, k) if self.order_sensitive else combinations(remaining, k)
        best_set: list[float] = []
        best_score = -float("inf")
        for combo in candidates:
            selected = [float(base_ev)] + [float(e) for e in combo]
            score = self.evaluate(selected)
            if score > best_score:
                best_set, best_score = selected, score
        return best_set, float(best_score)

    def enumerate_value_tensor(self, max_subset_size: int, base_ev: float) -> list[dict]:
        rows = []
        others = [e for e in self.sample.evs if e != base_ev]
        max_size = min(max_subset_size, len(self.sample.evs) - 1)
        for extra_size in range(0, max_size + 1):
            current_iter = (
                permutations(others, extra_size)
                if self.order_sensitive
                else combinations(others, extra_size)
            )
            for combo in current_iter:
                current = [float(base_ev)] + [float(e) for e in combo]
                current_score = self.evaluate(current)
                for action in self.sample.evs:
                    if action in current:
                        continue
                    next_score = self.evaluate(current + [action])
                    rows.append({
                        "scene_id": self.sample.scene_id,
                        "current": list(current) if self.order_sensitive else sorted(current),
                        "action": float(action),
                        "score_before": current_score,
                        "score_after": next_score,
                        "gain": next_score - current_score,
                        "order_sensitive": self.order_sensitive,
                    })
        return rows
