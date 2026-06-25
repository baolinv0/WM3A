# Experiment Plan: HBIS-AE

**Date**: 2026-06-24
**Method**: Predictive Camera State Modeling for Anticipatory AE
**Run order**: E0 → E1 → E2 → (E3, E4 deferred)

---

## E0 — SIA Estimator Validation (Gate experiment)

**Goal**: Prove Scene Imaging Attributes are computable and meaningful from single frames.

**Setup**:
- Model: ResNet-18 + 4 regression heads (DR, HighlightRatio, ShadowRatio, FaceScore)
- Training data: HDR bracket pairs → compute GT attributes analytically
  - DR = log2(max_luminance / min_luminance)
  - HighlightRatio = fraction of pixels > 0.9 in fused HDR
  - ShadowRatio = fraction of pixels < 0.1 in fused HDR
  - FaceScore = face detection confidence × face area fraction
- Loss: MSE on 4-dim SIA vector
- Estimated GPU: 2h × 1 A100

**Pass criterion**: MSE < 0.05 on held-out bracket pairs. If fail: redesign encoder or SIA definition.

---

## E1 — Core Ablation: Temporal Prediction Necessity (Paper's central claim)

**Goal**: Prove that predicted future SIA improves EV selection vs. current SIA alone.

**Variants**:
- **Baseline**: AE Head input = SIA_t only
- **Ours**: AE Head input = [SIA_t ; SIA_pred_{t+k}], k=5 frames

**Metric**:
- Highlight clipping rate (% frames with HighlightRatio > 0.15)
- Shadow clipping rate (% frames with ShadowRatio > 0.15)
- HDR reconstruction PSNR/SSIM (fused from predicted-EV captures)

**Pass criterion**: Ours reduces clipping rate by ≥ 10% vs. Baseline. If not, check prediction horizon k.

**Estimated GPU**: 4h × 1 A100

---

## E2 — Main Result: Illumination Transition Scenes

**Goal**: Demonstrate anticipatory AE advantage on challenging scenes.

**Test cases**:
- Indoor → outdoor transition
- Shadow → direct sunlight
- Night → day (time-lapse accelerated)
- Moving subject from shadow into highlight

**Metric**:
- EV decision lead time (how many frames *before* the scene change does EV start adjusting?)
- Clipping events per 30-frame clip
- Qualitative comparison: side-by-side video frames

**Estimated GPU**: 3h × 1 A100

---

## E3 — Temporal EV Stability (Deferred)

**Goal**: Confirm predictive AE reduces EV oscillation.

**Metric**: EV standard deviation over 10s clips with slow/no scene change.

**Estimated GPU**: 1h × 1 A100. Run after E1 confirmed.

---

## E4 — Prediction Horizon Sweep (Deferred)

**Goal**: Find optimal k.

**Setup**: k ∈ {1, 3, 5, 10, 20} frames. Plot clipping rate vs. k.

**Expected**: U-shaped curve — too short = no advantage, too long = scene too unpredictable.

**Estimated GPU**: 3h × 1 A100. Run after E1 confirmed.

---

## Dataset Notes

**Preferred**: Any video dataset with multi-exposure / HDR bracket side-information.
- Option A: Simulate brackets from RAW video by rescaling pixel values (fast, approx)
- Option B: Capture real brackets with a mobile phone on a bracket-rig
- Option C: Use existing HDR video datasets (e.g., HDR-Eye, LIVE-HDR)

**Minimum requirement for E0**: ~500 HDR bracket pairs.
**Minimum requirement for E1**: ~50 video clips × 30 frames with bracket at t and t+5.
