# Stage-0 Review Fixes

This branch incorporates the first code-review pass before real Kill Tests.

Fixed:

1. `oracle_sequence` now enforces exact-budget semantics when enough candidates exist and no longer compares exact-budget candidates against a one-frame initialization.
2. `mu_psnr`/`psnr_mu` now perform actual mu-law mapping instead of silently calling linear PSNR.
3. Mertens fusion now passes float32 images directly to OpenCV instead of quantizing to uint8 first.
4. Exposure-pool simulation normalizes the HDR pair once per scene and reuses the same scene scale across candidate EVs.
5. Best-fixed fallback fills missing EVs to the requested budget instead of silently evaluating fewer frames.
6. Random-policy repeats are preserved as separate rows (`method=random`, `repeat=r`) rather than averaged away before statistics.
7. Duplicate EVs are detected during manifest building and the affected scene is skipped with a warning.
8. Histogram coverage heuristic now chooses a candidate-dependent reference frame instead of always using the frame closest to EV 0.
9. Paired bootstrap sampling is vectorized.

Regression tests were added for exact-budget oracle behavior and the `psnr_mu` alias bug.

Note: the current Stage-0 protocol still distinguishes exact-budget selection from future stop-aware / at-most-budget acquisition. The latter should be evaluated explicitly before making claims about STOP or capture sufficiency.
