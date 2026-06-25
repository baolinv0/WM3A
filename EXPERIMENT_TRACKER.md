# Experiment Tracker

| ID | Name | Status | GPU hrs | Key metric | Result | Notes |
|----|------|--------|---------|-----------|--------|-------|
| E0 | SIA Estimator Validation | ⬜ PENDING | est. 2h | MSE < 0.05 | — | Gate for E1 |
| E1 | Core Ablation (temporal necessity) | ⬜ PENDING | est. 4h | Clipping rate −10% | — | Paper's central claim |
| E2 | Illumination Transition Scenes | ⬜ PENDING | est. 3h | Clipping events, lead time | — | Main result |
| E3 | Temporal EV Stability | ⬜ DEFERRED | est. 1h | EV std dev | — | Run after E1 |
| E4 | Horizon Sweep k ∈ {1,3,5,10,20} | ⬜ DEFERRED | est. 3h | Clipping vs k curve | — | Run after E1 |

**Total estimated (E0-E2)**: 9h on 1× A100
**Total estimated (all)**: 13h on 1× A100
