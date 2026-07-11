# Historical results — rerun required

The CSV/JSON files currently committed in this directory were generated before the common-action-pool correction.

At that time, seven-frame SICE scenes used ordinal actions `[-3,3]`, while nine-frame scenes could also use `-4` and `+4`, despite the configuration declaring `candidate_evs: [-3,-2,-1,0,1,2,3]`.

They must not be used as final GO/STOP evidence for the revised protocol.

Regenerate this directory with:

```bash
python scripts/run_stage0.py \
  --config configs/stage0_sice_mertens.yaml \
  --output results/kill_test_sice

python scripts/analyze_stage0.py \
  --result-dir results/kill_test_sice
```

The clean rerun will also report both:

- `standard_fixed`: configured camera-style bracket;
- `best_fixed`: Part1-selected bracket frozen on Part2.

After the oracle tensor is regenerated, rebuild the KT2 feature cache before training predictors.
