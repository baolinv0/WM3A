# Kill Test 2 — Candidate Value Predictability

This document is the authoritative Kill Test 2 protocol. `Experiment.md` is retained only as a short historical pointer.

## 1. Research question

The current gate is:

> Does an accumulation-aware acquisition state predict candidate exposure value better than the current fused image alone?

The decisive comparison is:

\[
\boxed{L4 > L2}
\]

measured on held-out scenes using **Decision Regret**.

This experiment validates a cheap LinearRadiance accumulation state. It does not yet validate the learned FreeMEF recurrent hidden state.

## 2. Mandatory clean rerun

The original SICE results were generated before `candidate_evs` was strictly enforced. Nine-frame scenes could therefore use ordinal actions `-4` and `+4`, while seven-frame scenes used only `[-3,3]`.

After the fix, every SICE scene exposes exactly:

```text
[-3, -2, -1, 0, 1, 2, 3]
```

Consequently, the following files must be regenerated before KT2 training:

```text
results/kill_test_sice/per_scene.csv
results/kill_test_sice/summary.csv
results/kill_test_sice/paired_bootstrap.csv
results/kill_test_sice/oracle_value_tensor.jsonl
results/kill_test_sice/utility_structure.json
results/kill_test_sice/kt2_features_resnet18.npz
```

The cache builder now rejects a tensor containing states outside the configured action pool.

## 3. Stage-0 baselines

Two fixed baselines are reported separately:

- `standard_fixed`: configured camera-style brackets such as `[-2,0,+2]`;
- `best_fixed`: the strongest non-adaptive bracket searched on SICE Part1 and frozen on Part2.

Do not call the configured standard bracket `best_fixed`.

Run:

```bash
python scripts/run_stage0.py \
  --config configs/stage0_sice_mertens.yaml \
  --output results/kill_test_sice

python scripts/analyze_stage0.py \
  --result-dir results/kill_test_sice
```

## 4. Representation hierarchy

All predictors receive the same current-EV metadata and candidate-action encoding.

### L0 — action prior only

```text
current EV set + candidate EV + context size
```

No image information.

### L1 — global exposure statistics

For each currently acquired frame:

- luminance histogram;
- shadow and saturation ratios;
- mean and standard deviation.

No spatial layout.

### L2 — current fused output

```text
Frozen ResNet-18(Y_t) + zero auxiliary block
```

### L4 — dual-path accumulation-aware state

```text
Frozen ResNet-18(Y_t)
+
fixed 4×4 grid statistics of [compressed S, W, under, over]
```

L2 and L4 have identical feature dimensionality and identical scalar-MLP capacity. The auxiliary block is zero for L2 and populated for L4. This avoids passing non-RGB state maps through frozen ImageNet BatchNorm statistics.

No candidate image or future measurement is used by any representation.

## 5. Build the feature cache

Install:

```bash
pip install -e ".[kt2]"
```

Build production features:

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

The cache stores and validates:

- tensor SHA-256;
- manifest SHA-256;
- Stage-0 config SHA-256;
- encoder signature;
- git commit when available.

Training aborts if the tensor hash does not match the cache provenance.

A CPU-only smoke path is available with `--encoder grid_stats`, but it is not the primary experiment.

## 6. Train the predictors

```bash
python scripts/run_killtest2.py \
  --config configs/killtest2_sice.yaml \
  --output outputs/killtest2_sice
```

The fast Kill Test uses a strict scene-level split of approximately `140 / 35 / 35` on the 210 Part2 scenes. No state-action row from one scene may cross partitions.

The MLP is trained with SmoothL1 regression, but checkpoint selection and early stopping use **validation Decision Regret**, matching the primary scientific metric.

## 7. Metrics

Primary:

\[
R(C_t)=\max_a v^*(C_t,a)-v^*(C_t,\arg\max_a \hat v(C_t,a)).
\]

Report:

- mean, median, and P90 Decision Regret;
- fraction with regret below 0.1 dB and 0.25 dB;
- Top-1 and Top-2 candidate accuracy;
- zero-filled Spearman over rankable states;
- valid-only Spearman;
- Spearman valid fraction;
- constant-prediction state rate.

A constant predictor on a rankable state contributes zero Spearman instead of being silently dropped.

Representation differences are bootstrapped at the **scene level**, not the state-action-row level.

## 8. Optional closed-loop rollout

Run only after the offline gate is healthy:

```bash
python scripts/run_killtest2.py \
  --config configs/killtest2_sice.yaml \
  --output outputs/killtest2_sice_rollout \
  --with-rollout
```

Headroom recovery is:

\[
\eta=\frac{Q_{policy}-Q_{best\_fixed}}{Q_{oracle}-Q_{best\_fixed}}.
\]

The tensor must contain current states up to three frames for a maximum rollout budget of four. With a fixed base exposure, `value_tensor_max_subset_size: 2` provides those states.

## 9. Interpretation

- `L4 ≈ L2`: accumulation state has no demonstrated unique value; stop or reframe.
- `L4 < L2`: diagnose representation failure before expanding the method.
- `L4 > L2` but the scene-bootstrap interval overlaps zero: promising but inconclusive.
- `L4 > L2` with lower regret, stable ranking, and better rollout headroom recovery: proceed to Stage 1.

The expected hierarchy is:

\[
L4 > L2 > L1 > L0.
\]

Only after this gate passes should the project add FreeMEF-state extraction, generic exposure-set encoders, or a dense spatial complementarity field.

## 10. Scope of Kill Test 3

The SICE Greedy-vs-Sequence result supports only this statement:

> In the current static, order-independent SICE subset-construction setting, one-step greedy selection is empirically near the globally best subset.

It does not prove that a world model is unnecessary for dynamic recurrent acquisition. That question must be revisited on a dynamic capture simulator or real burst data if the project expands in that direction.
