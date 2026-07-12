# RESEARCH STATE

## Topic

Fusion-state-conditioned counterfactual exposure valuation for adaptive multi-exposure acquisition.

## Current Stage

BUILD / novelty isolation (P2).

## Problem Anchor

Determine whether a fusion-native accumulation state predicts which candidate exposure provides complementary marginal value beyond both the current fused output and strong action-geometry baselines.

## Existing Evidence

- The calibrated 5-EV static protocol is a valid low-headroom negative control: oracle residual and tiny-set overfit controls pass, while held-out residual scale selection chooses alpha=0 for L2 and L4.
- Restoring the common 7-EV action pool raises train mean-gain-prior Decision Regret from 0.0438 dB to 0.2433 dB.
- On the 7-EV prior-residual PROBE, regret is 0.2433 dB for the prior, 0.1223 dB for Prior+L2, and 0.0806 dB for Prior+L4.
- L4 improves over L2 by approximately 0.042 dB with scene-bootstrap CI95 [0.015, 0.073]. The representation hypothesis is SUPPORTED_IN_SCOPE.

## What Evidence Does Not Yet Prove

- Prior-residual valuation is better than a capacity-matched direct policy.
- Prior decomposition is necessary relative to strong absolute-value learning.
- One-step regret improvements translate into end-to-end rollout quality.
- Results generalize beyond the current static SICE backend/metric.

## Dominant Claim

When sufficient acquisition headroom exists, an accumulation-aware fusion state predicts candidate-specific marginal value better than the current fused output alone.

## Anti-Claim

The observed gain comes only from a stronger representation; a capacity-matched direct policy or absolute-value scorer can achieve the same result without prior-residual decomposition.

## P2 Minimum Decisive Experiment

Primary methods:

- B0: train mean-gain prior.
- B1: L4 direct candidate policy with state-balanced listwise supervision.
- B2: L4 absolute value prediction with listwise plus absolute regression.
- B3: fixed prior plus explicitly centered L2 residual.
- B4: fixed prior plus explicitly centered L4 residual.
- Oracle: evaluation-only upper bound.

All learned methods use the same candidate-wise scalar MLP, state metadata, action encoding, split, optimization budget, and seeds. B3 uses zero-padded L2 features so its input dimension and parameter count match B4.

## P2 Gates

- P2-A: B4 regret < B1 regret with paired scene-bootstrap CI95 lower bound > 0.
- P2-B: B4 regret < B2 regret with paired scene-bootstrap CI95 lower bound > 0.
- P2-C: B4 regret < B3 regret with paired scene-bootstrap CI95 lower bound > 0.
- Net quality gate: correction gain exceeds corruption loss.

## Frozen Constraints

- Prior is computed from train scenes only and is not an input to learned residual heads.
- Residual predictions are explicitly centered within every candidate set.
- Candidate groups are trained as complete states; loss is balanced per state.
- Target temperature is 0.25, model temperature is 1, and auxiliary regression weight is 0.1 for the first run.
- Seeds 42, 123, and 2026 change initialization/minibatch order only; the scene split and prior remain fixed.
- Bootstrap sampling is paired at scene level.

## Next Gate

Does B4 outperform B1, B2, and B3 under the frozen capacity-matched protocol?

## Kill / Scope Revision Conditions

- If B4 beats B3 but not B1, retain the accumulation-state representation claim and drop prior-residual necessity.
- If B4 beats B1/B2 but not B3, retain prior-residual formulation and drop accumulation-state uniqueness.
- If B1 beats B4, the explicit counterfactual decomposition is not necessary under this setting.
