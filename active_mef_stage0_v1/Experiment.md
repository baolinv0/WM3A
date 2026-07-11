# Active MEF Experiment Notes

This file is no longer the authoritative Kill Test 2 protocol.

Use:

- [`KILL_TEST_2.md`](./KILL_TEST_2.md) for the current L0/L1/L2/L4 experiment;
- [`configs/stage0_sice_mertens.yaml`](./configs/stage0_sice_mertens.yaml) for the clean SICE Stage-0 rerun;
- [`configs/killtest2_sice.yaml`](./configs/killtest2_sice.yaml) for predictor training and evaluation.

## Current mandatory protocol

1. Every SICE scene is filtered to the common ordinal action pool:

   ```text
   [-3, -2, -1, 0, 1, 2, 3]
   ```

2. `standard_fixed` reports configured camera-style brackets.

3. `best_fixed` is searched on SICE Part1 and frozen on Part2.

4. The old mixed 7/9-frame Stage-0 results and oracle tensor must be regenerated.

5. Kill Test 2 compares:

   ```text
   L0: EV/action metadata only
   L1: global exposure statistics
   L2: current fused output feature
   L4: current fused output + accumulation-state feature
   ```

6. The primary metric is held-out-scene Decision Regret. Checkpoints are selected by validation Decision Regret.

7. The decisive gate is:

   \[
   L4 > L2.
   \]

## Scope of the previous Greedy-vs-Sequence result

The previous SICE result shows that greedy selection is near the globally best **static, order-independent exposure subset** under the tested LinearRadiance backend.

It does not prove that multi-step planning or a world model is unnecessary for dynamic, order-sensitive recurrent acquisition. That question remains deferred until a dynamic capture experiment is run.

## Historical results

Previously committed files under `results/kill_test_sice/` were generated before the common-action-pool fix. They are retained only as historical artifacts and must not be used as the final GO/STOP evidence for the revised protocol.
