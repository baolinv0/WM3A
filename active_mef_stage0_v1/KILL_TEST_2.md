# Kill Test 2 — Value Predictability

This stage tests one question only:

> Does the fusion accumulation state contain candidate-value information beyond the current fused output?

The minimum comparison is:

- **L1** — global exposure statistics + EV metadata
- **L2** — frozen feature of the current fused output `Y_t`
- **L4** — frozen feature of `[Y_t, confidence, under-exposure map, over-exposure map]`

All three predictors receive identical state EV metadata and candidate-action features. The default split is scene-level only: approximately **140 / 35 / 35** scenes for the 210-scene SICE experiment.

## 1. Install

```bash
cd active_mef_stage0_v1
pip install -e ".[kt2]"
```

Equivalent:

```bash
pip install -r requirements.txt
pip install -r requirements-kt2.txt
pip install -e .
```

## 2. Build the unique-state feature cache

Production run with frozen ImageNet ResNet-18 features:

```bash
python scripts/build_killtest2_cache.py \
  --tensor results/kill_test_sice/oracle_value_tensor.jsonl \
  --manifest data/manifests/sice_test.jsonl \
  --stage0-config configs/stage0_sice_mertens.yaml \
  --output results/kill_test_sice/kt2_features_resnet18.npz \
  --encoder resnet18 \
  --device cuda \
  --batch-size 32
```

The Stage-0 config used here **must define the same LinearRadiance fusion backend that generated the oracle value tensor**. If the existing SICE config points to a different fusion backend, create a matching config before building the cache.

For a fast CPU pipeline smoke test only:

```bash
python scripts/build_killtest2_cache.py \
  --tensor results/kill_test_sice/oracle_value_tensor.jsonl \
  --manifest data/manifests/sice_test.jsonl \
  --stage0-config <matching-linear-stage0-config.yaml> \
  --output results/kill_test_sice/kt2_features_grid.npz \
  --encoder grid_stats \
  --device cpu
```

`grid_stats` is not the primary paper baseline; it exists for debugging and fast pipeline verification.

## 3. Train the three scalar predictors

Update `configs/killtest2_sice.yaml` so `data.feature_cache` points to the cache produced above, then run:

```bash
python scripts/run_killtest2.py \
  --config configs/killtest2_sice.yaml \
  --output outputs/killtest2_sice
```

Primary outputs:

```text
outputs/killtest2_sice/
├── scene_split.json
├── summary.csv
├── representation_comparison.json
├── L1/
│   ├── predictions.csv
│   ├── per_state_metrics.csv
│   ├── training_history.csv
│   └── model.pt
├── L2/
└── L4/
```

`summary.csv` reports:

- mean / median / P90 Decision Regret
- fraction with regret below 0.1 dB and 0.25 dB
- mean per-state Spearman rank correlation
- Top-1 candidate accuracy
- Top-2 recall

`representation_comparison.json` performs a scene-level bootstrap comparison of **L4 vs L2** and **L4 vs L1**. Positive regret improvement means the challenger has lower regret.

## 4. Optional closed-loop rollout and oracle-headroom recovery

After the offline ranking/regret experiment is healthy:

```bash
python scripts/run_killtest2.py \
  --config configs/killtest2_sice.yaml \
  --output outputs/killtest2_sice_rollout \
  --with-rollout
```

This writes:

```text
rollout_per_scene.csv
rollout_headroom_recovery.csv
```

Global headroom recovery is:

\[
\eta = \frac{Q_{policy}-Q_{fixed}}{Q_{oracle}-Q_{fixed}}.
\]

The rollout uses only states already represented in `oracle_value_tensor.jsonl`. With a maximum budget of 4, the tensor must contain current subsets up to size 3.

## 5. Kill Test interpretation

The current decisive gate is **not** merely `L4 > Histogram`.

The core gate is:

\[
\boxed{L4 > L2}
\]

where the comparison is made on held-out scenes with Decision Regret as the primary metric.

Suggested internal interpretation:

- `L4 ≈ L2`: fusion-native state has no demonstrated unique value; stop or reframe.
- `L4 < L2`: state representation is actively worse; diagnose before any method expansion.
- `L4 > L2`, but tiny bootstrap interval overlapping zero: promising but not decisive.
- `L4 > L2` with meaningful scene-level regret reduction and stable ranking gain: proceed to Stage 1.

Only after this gate passes should the project add heavier experiments such as generic exposure-set encoders, FreeMEF-conditioned label enumeration, or dense spatial complementarity prediction.

## 6. Leakage and fairness constraints

- Split by `scene_id`, never by individual state-action rows.
- L1/L2/L4 use the same scene split, candidate actions, state EV metadata, MLP capacity, optimizer, and training schedule.
- L2 and L4 are cached once per unique current state.
- L4 uses no candidate image and no future measurement; it is built only from the currently selected exposure set.
- The first three channels of the L4 map are exactly the L2 fused output, making L4 an explicit information superset rather than a different reconstruction target.
- SICE ordinal exposure ranks are not physical EV values; do not use this experiment to claim continuous physical-action zero-shot generalization.
