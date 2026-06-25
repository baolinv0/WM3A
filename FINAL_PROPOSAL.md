# Final Proposal: Predictive Camera State Modeling for Anticipatory Auto Exposure

**Date**: 2026-06-24
**Target venue**: CVPR 2027
**Status**: Ready for E0/E1 experiments

---

## Problem Anchor

> Given only the current frame at time t, predict **Scene Imaging Attributes** (SIA) at time t+k — dynamic range, highlight risk, shadow risk, subject importance — without observing the future frame. Use this prediction to make an exposure decision that maximizes information capture at t+k.

Traditional AE is reactive: it reacts to what it sees now. This paper makes AE anticipatory: it plans based on what the scene *will require*.

---

## Core Insight

PPISP proves that Camera State can be *estimated* from a single observed image.

Corollary: if Camera State can be estimated from any image (including a future one), then a temporal predictor trained on video can *predict* future Camera State from current observations — enabling anticipatory exposure control without ever seeing the future frame.

---

## Method

### Module 1 — Scene Imaging Attribute (SIA) Estimator

```
Input:  Frame_i (any exposure level)
Output: SIA_i = {DR, HighlightRatio, ShadowRatio, FaceScore}
Supervision: HDR bracket at time i → compute GT attributes analytically
Architecture: ResNet-18 + 4-head regression output
Loss: L_sia = MSE(SIA_pred, SIA_gt)
```

SIA is **scene-side only** (not camera settings). It answers: "What imaging challenges does this scene pose?"

### Module 2 — Temporal SIA Predictor (TSP)

```
Input:  SIA_t (and optionally SIA_{t-1}, SIA_{t-2} for context)
Output: SIA_pred_{t+k}
Architecture: 4-layer Transformer or GRU
Loss: L_tsp = MSE(SIA_pred_{t+k}, SIA_gt_{t+k})
Training: video sequences with HDR brackets at both t and t+k
```

### Module 3 — AE Planner

```
Input:  [SIA_t ; SIA_pred_{t+k}]
Output: EV_t+1
Architecture: 2-layer MLP head
Supervision: pseudo-label from reference AE, or HDR reconstruction quality as auxiliary loss
```

### Training protocol

1. Train Module 1 independently on HDR bracket data.
2. Freeze Module 1. Train Module 2 on video sequences.
3. Fine-tune Module 3 end-to-end (Modules 1+2 frozen or low-LR).

---

## Title Candidates

1. "Predictive Camera State Modeling for Anticipatory Auto Exposure"
2. "Future Imaging State Prediction for Proactive Exposure Control"
3. "Scene Imaging Attribute Forecasting for Auto Exposure"

---

## Claims

**C1 (Main)**: Anticipatory AE using predicted future SIA reduces highlight/shadow clipping vs. reactive AE on scenes with rapid illumination change.

**C2 (Ablation)**: Predicted future SIA (State_{t+k}) provides strictly more useful information for EV selection than current SIA (State_t) alone.

**C3 (Secondary)**: Predictive AE achieves better temporal EV stability (lower EV variance) than reactive AE.

---

## Differentiation from Related Work

| Work | Difference |
|------|-----------|
| Time-Aware AWB (ICCV 2025) | That work stabilizes WB using past frames (reactive). This work predicts future scene attributes (predictive) and controls EV (not just WB). |
| PPISP | Estimates current Camera State from image. Does not predict future state, does not make AE decisions. |
| DreamerV3 / V-JEPA 2 | General visual world models. Not applied to camera ISP, no imaging-specific state definition, no exposure planning. |
| CLIP-Guided AE (2026) | Single-frame exposure correction. No temporal prediction, no planning. |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| HDR bracket data scarce | Simulate brackets from RAW video via exposure scaling; or use paired video+HDR datasets |
| SIA estimator accuracy insufficient | Expand bracket stack, add perceptual loss |
| Temporal predictor overfits to slow scenes | Augment training with fast-motion / illumination-change clips |
| Reviewer: "World Model is overselling" | Frame as "Temporal Imaging State Predictor" — never use "World Model" in the paper |
| Reviewer: "Just run PPISP on future frame" | Future frame not available at inference. The predictor approximates PPISP applied to unseen future. |
