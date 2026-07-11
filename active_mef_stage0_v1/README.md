# Active MEF Stage-0 v1

This repository implements the minimum evidence chain for adaptive multi-exposure acquisition:

```text
Common Exposure Pool
→ Oracle Value Tensor
→ Oracle vs Strong Non-adaptive Baselines
→ Greedy vs Global Search
→ L0/L1/L2/L4 Value Predictability
→ Optional Closed-loop Rollout
```

It is not yet the full paper method. RL, world models, dense spatial ECF prediction, and FreeMEF-conditioned training remain deferred until the cheap gates pass.

## 1. Dataset interface

All datasets are converted to JSONL manifests.

### Precomputed exposure stack

```json
{"scene_id":"s1","kind":"precomputed","gt":"/abs/gt.png","exposures":[{"ev":-2,"path":"/abs/a.png"},{"ev":0,"path":"/abs/b.png"},{"ev":2,"path":"/abs/c.png"}]}
```

### HDR pair for simulated acquisition

```json
{"scene_id":"v1_001","kind":"hdr_pair","hdr":"/abs/0001.exr","hdr_next":"/abs/0002.exr","frame_dt":0.0333}
```

For precomputed data, `candidate_evs` is now enforced strictly. Every scene must contain every configured action, and exposures outside that common pool are removed. This prevents mixed 7-frame/9-frame SICE scenes from using different action spaces.

## 2. Recommended data roles

| Role | Dataset | Purpose |
|---|---|---|
| Static feasibility | SICE | Oracle headroom, value structure, cheap KT2 |
| Learned-fusion check | Kalantari/FreeMEF-compatible data | backend dependence |
| Dynamic primary study | Real-HDRV HDR-GT | blur/noise/motion-aware acquisition |
| Dynamic cross-dataset | DeepHDRVideo HDR-GT | generalization |

The included simulator is paper-inspired, not a bit-exact AdaptiveAE reproduction.

## 3. Installation

```bash
cd active_mef_stage0_v1
pip install -e ".[kt2]"
```

## 4. Clean SICE Stage-0 run

The current SICE protocol uses the common ordinal action pool:

```text
[-3, -2, -1, 0, 1, 2, 3]
```

Part1 selects the strongest fixed bracket; Part2 remains the evaluation set.

```bash
python scripts/run_stage0.py \
  --config configs/stage0_sice_mertens.yaml \
  --output results/kill_test_sice

python scripts/analyze_stage0.py \
  --result-dir results/kill_test_sice
```

Main Stage-0 methods:

- `standard_fixed`: configured camera-style bracket;
- `best_fixed`: train-selected non-adaptive bracket, frozen on test;
- `strong_heuristic`: current-frame coverage heuristic;
- `random`: repeated random action selection;
- `oracle_greedy`: exact one-step marginal-gain selection;
- `oracle_sequence`: exhaustive global subset search for order-independent backends, or permutation search for order-sensitive backends.

Main outputs:

```text
per_scene.csv
summary.csv
paired_bootstrap.csv
oracle_value_tensor.jsonl
greedy_sequence_gap.csv
utility_structure.json
fixed_sets.json
resolved_config.json
environment.json
```

Previously committed SICE results were produced before the common-action-pool fix and must be regenerated before being used as final evidence.

## 5. Kill Test 2

See [`KILL_TEST_2.md`](./KILL_TEST_2.md) for the authoritative protocol.

Representations:

- `L0`: current EV metadata + candidate action only;
- `L1`: global per-frame exposure statistics;
- `L2`: frozen RGB feature of the current fused output plus a zero auxiliary block;
- `L4`: the same RGB feature plus fixed spatial statistics of the accumulation state `(S, W, under, over)`.

The decisive gate is:

\[
L4 > L2
\]

on held-out-scene Decision Regret.

Build the cache:

```bash
python scripts/build_killtest2_cache.py \
  --tensor results/kill_test_sice/oracle_value_tensor.jsonl \
  --manifest data/manifests/sice_test.jsonl \
  --stage0-config configs/stage0_sice_mertens.yaml \
  --output results/kill_test_sice/kt2_features_resnet18.npz \
  --encoder resnet18 \
  --device cuda
```

Train and evaluate:

```bash
python scripts/run_killtest2.py \
  --config configs/killtest2_sice.yaml \
  --output outputs/killtest2_sice
```

Run rollout only after the offline gate is healthy:

```bash
python scripts/run_killtest2.py \
  --config configs/killtest2_sice.yaml \
  --output outputs/killtest2_sice_rollout \
  --with-rollout
```

## 6. Reproducibility safeguards

- deterministic scene hashing replaces Python's process-randomized `hash()`;
- scene-level split only;
- cache provenance includes tensor, manifest, config, encoder, and commit signatures;
- training aborts when the cache tensor hash does not match;
- L2/L4 have identical feature dimensionality and scalar-MLP capacity;
- validation Decision Regret selects checkpoints;
- constant-prediction states are not silently removed from Spearman reporting.

## 7. FreeMEF adapter

The external FreeMEF adapter now:

- pads inputs to the official factor-of-eight requirement and unpads outputs;
- keeps the configured base exposure as the main frame;
- preserves auxiliary-frame acquisition order;
- marks itself order-sensitive so exhaustive search uses permutations rather than set combinations.

No FreeMEF source or checkpoint is bundled here.

## 8. Scope of the Greedy result

The current SICE result can support only:

> Greedy is near the globally best static, order-independent exposure subset under the tested backend.

It does not prove that dynamic recurrent acquisition never needs multi-step planning.
