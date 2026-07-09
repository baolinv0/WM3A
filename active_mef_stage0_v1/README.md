# Active MEF Stage-0 v1

First-pass code for the experiment chain:

`Exposure Pool -> Oracle Value Tensor -> Oracle vs Strong Heuristic -> Greedy vs Sequence -> Gate A`

The code is intentionally **not** a full method implementation. It answers whether the research problem has enough headroom before training a value predictor, spatial complementarity field, RL policy, or world model.

## 1. Why the dataset layer is manifest-first

The two closest prior-work families use different data regimes:

- **AdaptiveAE-style dynamic HDR**: HDR video ground truth is used to synthesize candidate LDR measurements with exposure-dependent blur/noise. For Stage 0, use Real-HDRV and DeepHDRVideo HDR-GT frames as primary/cross-dataset sources after reorganizing the HDR GT into per-sequence folders.
- **FreeMEF-style flexible fusion**: SICE and Kalantari-style folders provide precomputed exposure stacks and GT images. These are useful for static sanity checks and direct compatibility with an external FreeMEF checkpoint.

Instead of hard-coding one dataset layout, every dataset is converted to JSONL.

### Precomputed pool record

```json
{"scene_id":"s1","kind":"precomputed","gt":"/abs/gt.png","exposures":[{"ev":-2,"path":"/abs/a.png"},{"ev":0,"path":"/abs/b.png"},{"ev":2,"path":"/abs/c.png"}]}
```

### HDR pair record

```json
{"scene_id":"v1_001","kind":"hdr_pair","hdr":"/abs/0001.exr","hdr_next":"/abs/0002.exr","frame_dt":0.0333}
```

The second format lazily synthesizes a candidate EV pool.

## 2. Recommended experimental dataset matrix

| Role | Dataset family | Purpose |
|---|---|---|
| Static sanity | SICE 5-frame | verify subset enumeration and FreeMEF integration |
| Static learned-fusion check | Kalantari-style MEF data | test fusion-backend dependence |
| Primary Stage 0 | Real-HDRV HDR-GT source | dynamic scene headroom and motion sensitivity |
| Cross dataset | DeepHDRVideo HDR-GT source | verify that headroom is not dataset-specific |

Important: the lightweight simulator in this repository is **paper-inspired, not a bit-exact AdaptiveAE reproduction**. Its default `linear_tmo` temporal integration is a cheap fallback. For final dynamic-scene paper experiments, precompute RIFE-based intermediate frames/blur pools or plug in a more faithful simulator, then keep the same manifest and Stage-0 evaluation code.

## 3. Installation

```bash
cd active_mef_stage0_v1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For the optional FreeMEF backend, separately install the official FreeMEF environment and provide absolute paths in `configs/stage0_sice_freemef.yaml`.

## 4. Build manifests

### SICE-style sequence pool

```bash
python scripts/build_manifest.py sequence_pool \
  --input-root /data/SICE/input_5frame \
  --gt-root /data/SICE/gt_resize \
  --gt-in-subdir \
  --output data/manifests/sice_test.jsonl
```

### Kalantari-style sequence pool

```bash
python scripts/build_manifest.py sequence_pool \
  --input-root /data/Kalantari_MEF/Testing_input \
  --gt-root /data/Kalantari_MEF/Testing_gt \
  --output data/manifests/kalantari_test.jsonl
```

When `exposure.txt`/`exposures.txt` is present, the builder derives relative EVs from it. Otherwise it assigns ordinal EV ranks and records `ev_source=ordinal`; do not overclaim physical continuous-EV generalization on ordinal-only data.

### HDR video GT source

Reorganize linear HDR GT frames as:

```text
/hdr_gt_root/
  scene_001/000000.exr 000001.exr ...
  scene_002/000000.exr 000001.exr ...
```

Then:

```bash
python scripts/build_manifest.py hdr_video \
  --hdr-root /hdr_gt_root \
  --output data/manifests/real_hdrv_test.jsonl \
  --frame-dt 0.0333333
```

Use official train/test splits when available; do not select the best fixed bracket on the test split.

## 5. Smoke test on synthetic toy data

```bash
python scripts/make_toy_dataset.py --output data/toy --scenes 8
cp data/toy/toy_hdr.jsonl data/toy/train.jsonl
cp data/toy/toy_hdr.jsonl data/toy/test.jsonl
```

Copy `configs/stage0_hdr_video.yaml` and point both manifests to those toy files, then:

```bash
python scripts/run_stage0.py \
  --config configs/stage0_toy.yaml \
  --output outputs/toy_stage0
```

## 6. Run Stage 0

```bash
python scripts/run_stage0.py \
  --config configs/stage0_hdr_video.yaml \
  --output outputs/real_hdrv_stage0

python scripts/analyze_stage0.py \
  --result-dir outputs/real_hdrv_stage0
```

Outputs:

- `per_scene.csv`: raw scene-level scores, no summary-only reporting
- `summary.csv`: method/budget mean and std
- `oracle_value_tensor.jsonl`: `V*(scene, current_subset, candidate_action)`
- `greedy_sequence_gap.csv`: horizon diagnosis
- `oracle_action_distribution.csv`: action diversity
- `paired_bootstrap.csv`: paired bootstrap deltas and 95% intervals
- `quality_cost_auc.csv`: integrated quality-cost comparison
- `utility_structure.json`: state/action diversity and approximate diminishing-return summary
- `diminishing_returns.csv`: raw nested-subset marginal-gain comparisons
- `fixed_sets.json`: train-selected fixed brackets
- `resolved_config.json`, `environment.json`: reproducibility records

## 7. Methods included in Stage 0

- **best_fixed**: selected on the train manifest, then frozen for test
- **random**: mean over repeated draws with identical action pool/budget
- **strong_heuristic**: current-image exposure-coverage heuristic using action metadata, without peeking at unobserved candidate frames
- **oracle_greedy**: exact one-step marginal fusion gain
- **oracle_sequence**: exhaustive best subset for the budget

## 8. Decision gates

### STOP

Stop if oracle greedy is close to the strong heuristic and best fixed policy across the quality-cost frontier, with no meaningful frame-count/latency advantage.

### GO

Proceed to scalar value prediction only if:

1. oracle headroom over the strong heuristic is meaningful;
2. optimal actions vary by scene/current subset;
3. greedy-sequence gap is small enough for one-step valuation, or a clear horizon gap justifies re-refinement.

## 9. Deliberate omissions in v1

- no RL;
- no world model;
- no HDR generation model;
- no spatial ECF predictor yet;
- no claim that the lightweight simulator reproduces AdaptiveAE exactly;
- no claim that ordinal SICE/Kalantari frame ranks are physical EVs.

The next code milestone is a scalar predictor only after Gate A passes.


