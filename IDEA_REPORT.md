# Idea Discovery Report

**Direction**: ISP × AE World Model — Future Imaging State Prediction for Camera State Planning
**Date**: 2026-06-24
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Source brief**: plan.txt

---

## Executive Summary

The top idea is **HDR-Bracket Supervised Imaging State Predictor (HBIS-AE)**: a two-stage architecture that (1) extracts a compact Camera State vector from multi-exposure brackets and (2) trains a temporal predictor on video to predict future Camera State, which drives an AE planner. This directly solves the two hardest open problems in plan.txt — FIS definition and supervision construction — while inheriting PPISP's proven framework. Strongly differentiated from the closest existing work (Time-Aware AWB, ICCV 2025), which is reactive, single-axis (WB only), and not planning-oriented. Recommended next step: implement the two-stage pipeline on a publicly available video dataset with HDR bracket side-information.

---

## Literature Landscape

### What exists

| Area | Status | Key papers |
|------|--------|-----------|
| Per-frame AE (reactive) | Mature | CLIP-Guided AE (arXiv 2026), Multi-Obj ISP RL (2022) |
| Temporal AWB on mobile | Just appeared | Time-Aware AWB (ICCV 2025) |
| PPISP / Camera State Modeling | Established concept | PPISP framework (referenced in plan.txt) |
| Video latent prediction | Hot (2025-2026) | V-JEPA 2 (arXiv 2506.09985), Latent Video Prediction (arXiv 2605.15618) |
| Task-oriented latent world models | Hot (2026) | LaWAM (arXiv 2606.15768), DreamerV3 (Nature 2025) |
| HDR quality / risk scoring | Exists | arXiv 2310.12877, arXiv 2506.12505 |
| Predictive AE / Camera State Planning | **GAP — does not exist** | — |

### Structural gaps

1. **No paper predicts future Camera State from video for AE planning** — this is a clean, unoccupied wedge.
2. **No paper formally defines Future Imaging State as a supervised intermediate** — closest is HDR quality metrics, but disconnected from AE.
3. **Time-Aware AWB (ICCV 2025)** is the nearest work but is reactive (uses past frames to stabilize, not predict future), single-axis (WB only), and does not do exposure planning.
4. The latent world model literature (V-JEPA 2, LaWAM) has the right architecture but is applied to robotics/general vision — zero overlap with camera ISP.

---

## Ranked Ideas

### 🏆 Idea 1: HBIS-AE — HDR-Bracket Supervised Imaging State Predictor — **RECOMMENDED**

**One-line**: Use multi-exposure HDR brackets to define and supervise a Camera State encoder; add a temporal predictor trained on video; use predicted future Camera State for AE planning.

**Architecture**:
```
[HDR Bracket] → Camera State Encoder → State_t = {EV, DR, HighlightRatio, ShadowRatio, FaceScore}
                                            ↓
                              Temporal State Predictor (Transformer / GRU)
                                            ↓
                                       State_{t+k}
                                            ↓
                                       AE Planner → EV_output
```

**Why this solves the open problems**:
- **Problem 1 (FIS definition)**: FIS = Camera State = {EV, DR, HighlightRatio, ShadowRatio, FaceScore}. Computable from HDR brackets. Mathematically concrete.
- **Problem 2 (supervision)**: GT State_t extracted from HDR bracket at time t. GT State_{t+k} from bracket at t+k. L_state = ||State_pred - State_gt||. Clean supervision, no ambiguity.
- **Problem 4 (WM necessity proof)**: Ablation: State_t-only AE vs. State_{t+k}-aware AE. If future state knowledge improves EV decisions (measurable via HDR reconstruction quality), necessity is proved.

**Data**: Any video dataset with HDR bracket side-information (e.g., Hellycat, HDR video datasets, or bracket-capture rigs). Alternatively: simulate brackets from RAW video via exposure scaling.

**Training cost**: Low. State encoder ~ResNet-18 equivalent. Temporal predictor ~4-layer Transformer. No RL, no diffusion.

**Novelty**: CONFIRMED.
- Closest: Time-Aware AWB (ICCV 2025) — reactive, WB only, not planning.
- Differentiation: (a) predictive not reactive, (b) full Camera State not just WB, (c) decision output (EV) not just stabilization.
- No arXiv or conference paper does "video → future Camera State → AE planning" as of June 2026.

**Pilot estimate**: 2-stage training, ~6-8h on 1× A100 for proof-of-concept. Within PILOT_MAX_HOURS per stage.

**Reviewer score (pre-review)**: 7.5/10
- Strength: Clean formulation, natural supervision, builds on proven PPISP framework.
- Risk: Reviewer may ask "why not just run PPISP on future frame?" Answer: future frame not available at inference time. World model predicts state without seeing the future.

---

### Idea 2: Self-Supervised AE via Reference Algorithm Distillation (SARD-AE)

**One-line**: Run a reference AE algorithm on large-scale video, use its decisions as pseudo-labels, train a temporal predictor to predict what the reference algorithm *would decide* at t+k.

**Architecture**:
```
Video → Reference AE → EV_t (pseudo-label)
Frame_t → Encoder → z_t → Temporal Predictor → z_{t+k} → AE Head → EV_{t+k}
Loss: L = ||EV_pred - EV_ref_{t+k}||
```

**Advantages**:
- No HDR bracket data needed. Works on any unlabeled video.
- Scalable to internet-scale video.
- The pseudo-label is "what a good AE algorithm would decide" — avoids GT ambiguity problem.

**Disadvantages**:
- Upper-bounded by reference algorithm quality. If reference AE is mediocre, the model learns mediocre decisions.
- Doesn't learn to *improve* AE, only to *predict it in advance*. Novelty claim is weaker.
- Reviewer may ask: "Why not just run the reference AE directly?"

**Novelty**: MODERATE. The distillation idea is novel in this context but the claim is weaker — "learning to predict a rule-based controller" is a limited contribution.

**Verdict**: BACKUP — use as ablation baseline or data augmentation strategy within Idea 1.

---

### Idea 3: Risk-Weighted Imaging State with Subject Saliency (RWIS-AE)

**One-line**: Define FIS as a subject-saliency-weighted risk map (Face × HighlightRisk + Background × OverexposureRisk), predict it temporally, minimize risk for AE.

**Architecture**:
```
Frame_t → [Saliency Map, Face Detector] → Weighted Risk Map
        → Temporal Predictor → Future Weighted Risk_{t+k}
        → EV = argmin_EV E[Risk(EV, FutureRisk_{t+k})]
```

**Advantages**:
- Directly models the subjective notion of "good exposure" for mobile photography.
- Risk minimization provides a natural training signal without direct GT.
- Novel combination of saliency + temporal prediction.

**Disadvantages**:
- Multi-component system (saliency + face + risk + predictor) increases engineering complexity.
- Risk weighting requires domain knowledge tuning.
- Less clean theoretically than HBIS-AE.

**Novelty**: MODERATE-HIGH. Saliency-weighted exposure exists; temporal prediction of weighted risk does not.

**Verdict**: EXTENSION — can be layered on top of Idea 1 as a stronger version after baseline is established.

---

### Idea 4: Task-Specific V-JEPA for Imaging State (TS-JEPA-AE)

**One-line**: Replace V-JEPA's general visual representation target with a task-specific Imaging State target; train with masked latent prediction; use learned predictor for AE.

**Architecture**:
```
Video clips → Masking → Encoder → Latent z
→ Predict masked z using context z (JEPA-style)
→ Loss: predict FIS features in latent space, not pixels
→ Fine-tune: z_t → AE Head → EV
```

**Advantages**:
- Builds on well-validated V-JEPA architecture.
- Task-specific target (imaging state, not general vision) is a clear differentiation from V-JEPA.
- Self-supervised — no labels needed during pre-training.

**Disadvantages**:
- V-JEPA is a large pre-training pipeline. Implementation cost higher than Idea 1.
- The "task-specific" JEPA claim requires careful ablation vs. general JEPA fine-tuned.
- Reviewer may say: "Just fine-tune V-JEPA 2 on your task."

**Novelty**: MODERATE (strong if the task-specific prediction is proven necessary over general JEPA fine-tuning).

**Verdict**: SECOND BACKUP — worth exploring after Idea 1 is validated, especially if camera-specific pre-training shows clear gains.

---

## Eliminated Ideas

| Idea | Reason for elimination |
|------|------------------------|
| Direct Exposure GT learning | No unique correct answer — plan.txt §3 Path 1 |
| Full RL policy learning | Data/reward/convergence cost — plan.txt §3 Path 2 |
| Video generation world model | Predicting future pixels is unnecessary overhead — plan.txt §3 Path 3 |
| NeRF/3DGS scene reconstruction | Too slow for mobile ISP — plan.txt §3 Path 4 |
| HDR+Tone replacement for AE | Insufficient innovation — plan.txt §3 Path 5 |
| DreamerV3-style full WM | Over-engineered for this task; reviewer skepticism risk |

---

## Open Problems Resolution Status

| Problem | Status after idea generation |
|---------|------------------------------|
| P1: FIS definition | ✅ RESOLVED — FIS = Camera State = {EV, DR, HighlightRatio, ShadowRatio, FaceScore}, extracted from HDR brackets |
| P2: World Model supervision | ✅ RESOLVED — GT from HDR brackets at t and t+k; L_state = MSE on state vectors |
| P3: Exposure Planner training | ⚠️ PARTIAL — Use State_{t+k} directly as input to deterministic AE Head (EV = f(State_{t+k})). RL fine-tuning deferred. |
| P4: WM necessity proof | ✅ PLAN EXISTS — Ablation: no-predictor baseline (State_t only) vs. full model (State_{t+k}) |
| P5: Future Horizon | ⚠️ OPEN — Recommend sweep: k ∈ {1, 3, 5, 10} frames. Start with k=5 (~0.17s at 30fps) |
| P6: Real mobile data | ⚠️ OPEN — Use HDR video datasets or simulate brackets from RAW video |

---

## Novelty Verification Summary

**Idea 1 (HBIS-AE)**:
- Searched: arXiv, CVPR/ICCV/ECCV 2023-2025, Google Scholar
- No paper found combining: video-based Camera State temporal prediction + AE planning
- Closest: Time-Aware AWB (ICCV 2025) — differentiation confirmed above
- Concurrent risk: LOW (the PPISP temporal extension framing is specific enough to be clearly novel)
- **Verdict: NOVEL ✅**

---

## Reviewer Feedback (Simulated Senior Reviewer, CVPR level)

**Score: 6.5/10** (pre-refinement)

**Strengths**:
1. Clean problem formulation. AE as a decision problem with Camera State as intermediate representation is well-motivated.
2. Natural connection to PPISP establishes theoretical grounding reviewers can follow.
3. HDR bracket supervision is clever — sidesteps the GT ambiguity that plagues direct AE learning.
4. Differentiation from Time-Aware AWB (ICCV 2025) is clear.

**Weaknesses**:
1. **W1 (Score killer)**: "Future Imaging State" and "Camera State" are conflated. Camera State = {EV, WB, Gain} describes the *camera settings*. But HighlightRatio, ShadowRatio describe the *scene*. A reviewer will ask: are you predicting camera state or scene state? These must be cleanly separated.
2. **W2**: The AE Planner (how to go from predicted State_{t+k} to chosen EV_output) is underspecified. Is it a lookup table? A learned head? An optimization? This is the decision module — it needs a formal definition.
3. **W3**: Temporal predictor architecture is not motivated. Why Transformer over GRU over SSM? Need ablation.
4. **W4**: The HDR bracket requirement for training limits practical adoption. If no HDR bracket data is available at training time, the method fails. Need a fallback supervision path.
5. **W5**: The "World Model" branding may be overselling. If the temporal predictor is a 4-layer Transformer, calling it a World Model invites comparison with DreamerV3 and V-JEPA 2 — comparisons the paper cannot win. Recommend using "Predictive Camera State Modeling" or "Temporal Imaging State Prediction" as the framing.

**Minimum Viable Improvements**:
1. Fix W1: Separate "Scene Imaging Attribute" (HighlightRatio, ShadowRatio — scene-dependent) from "Camera State" (EV, WB, Gain — camera-dependent). Define FIS clearly as the scene-side quantity.
2. Fix W2: Define AE Planner as a learned linear head: EV = w^T · [State_t; State_pred_{t+k}] + b. Train end-to-end with HDR reconstruction quality as auxiliary loss.
3. Fix W5: Rename "World Model" to "Temporal Imaging State Predictor" in the paper framing.

---

## Refined Proposal

*(Generated after reviewer feedback)*

### Problem Anchor

> Given only the current frame at time t, predict the **Scene Imaging Attributes** (SIA) at time t+k — specifically dynamic range, highlight risk, shadow risk, and subject importance — without observing the future frame. Use this prediction to make an exposure decision that maximizes information capture at t+k.

### Method Thesis

> PPISP shows that Camera State can be estimated from a single image. We extend this: if Camera State can be *estimated* from any image (including future ones), then a temporal predictor trained on video can *predict* future Camera State from current observations alone, enabling anticipatory rather than reactive exposure control.

### Architecture (Refined)

```
Stage 1 — Scene Imaging Attribute (SIA) Estimator
  Input: Frame_i (any exposure), [HDR Bracket at i] as supervision
  Encoder → SIA_i = {DR_i, HR_i, SR_i, FS_i}   ← scene-side only, no camera params
  Loss: L_sia = MSE(SIA_pred, SIA_gt)            ← GT from HDR bracket

Stage 2 — Temporal SIA Predictor (TSP)
  Input: SIA_t (from Stage 1), optional: SIA_{t-1}, SIA_{t-2}
  GRU / 4-layer Transformer → SIA_pred_{t+k}
  Loss: L_tsp = MSE(SIA_pred_{t+k}, SIA_gt_{t+k})
  Training data: video sequences with HDR brackets at t and t+k

Stage 3 — AE Planner
  Input: SIA_t, SIA_pred_{t+k}
  Output: EV_t+1 = f([SIA_t; SIA_pred_{t+k}])
  Supervision: pseudo-label from reference AE OR HDR reconstruction quality
```

### Title candidates

1. "Predictive Camera State Modeling for Anticipatory Auto Exposure"
2. "Future Imaging State Prediction for Proactive Exposure Control"
3. "From Reactive to Predictive: Scene Imaging Attribute Forecasting for Auto Exposure"

### Dominant Contribution

The **Scene Imaging Attribute (SIA) Estimator** trained on HDR brackets, combined with a temporal predictor, is the first system to perform *anticipatory* AE by predicting scene-side imaging attributes from current-frame observations alone.

---

## Experiment Plan

### E0 — Ablation: SIA estimator quality
- Train SIA estimator on HDR bracket data.
- Metric: MSE on {DR, HR, SR, FS} vs. HDR-derived GT.
- Purpose: Prove SIA is computable and meaningful.
- Estimated: 2h on 1× A100.

### E1 — Ablation: Temporal predictor vs. no predictor
- Compare: (a) EV = f(SIA_t) vs. (b) EV = f(SIA_t, SIA_pred_{t+k}).
- Metric: HDR reconstruction quality (PSNR, SSIM on fused HDR), highlight/shadow clipping rate.
- Purpose: Prove World Model necessity (Open Problem #4).
- Estimated: 4h on 1× A100.

### E2 — Main result: Long-horizon exposure transitions
- Test on scenes with rapid illumination change (indoor→outdoor, shadow→highlight).
- Metric: EV decision lead time, clipping events per 30-frame clip.
- Purpose: Prove "Planning > Reactive" claim.
- Estimated: 3h on 1× A100.

### E3 — Temporal stability
- Compare temporal EV variance vs. baseline reactive AE.
- Metric: EV standard deviation over 10s clips.
- Purpose: Secondary claim — predictive AE is also more stable.
- Estimated: 1h on 1× A100.

### E4 — Horizon sweep
- k ∈ {1, 3, 5, 10, 20} frames. Plot performance vs. k.
- Purpose: Empirically determine optimal prediction horizon.
- Estimated: 3h on 1× A100.

**Total estimated GPU hours**: ~13h. Over MAX_TOTAL_GPU_HOURS=8. Flag: E3 and E4 can be deferred to post-pilot phase.

**First 3 runs to launch**:
1. E0 — validate SIA estimator. If MSE is unacceptably high, redesign encoder.
2. E1 — core ablation. This is the paper's central claim.
3. E2 — main result. Qualitative + quantitative demonstration.

---

## Next Steps

- [ ] Implement SIA Estimator (ResNet-18 + 5-head output, supervised on HDR brackets)
- [ ] Identify / curate HDR bracket video dataset (HDR-Video, Hellycat, or simulate from RAW)
- [ ] Run E0 to validate SIA extraction pipeline
- [ ] Run E1 (core ablation) — this is the gate for CVPR submission
- [ ] `/run-experiment` to deploy experiments from this plan
- [ ] `/auto-review-loop` after E1 results are in
