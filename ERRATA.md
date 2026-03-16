# Errata & Notes for Writing Agent

## 1. Expand Ratio Differs Between Experiments

- **Experiment B (crop method comparison, from notebook):** uses `expand_ratio=0.25`
- **Experiment A (FPS comparison):** uses `expand_ratio=0.3` to match the reference thesis pipeline

Do NOT compare 86.1% (Exp B) to 92.6% (Exp A) directly. Present them as separate experiments with separate configs. Suggested wording:

> "Experiment 1 (Section 3.1) isolates the effect of SAM2 segmentation using an expansion ratio of 0.25. Experiment 2 (Section 3.2) evaluates frame rate sensitivity using the reference pipeline's expansion ratio of 0.3. Results should be compared within each experiment, not across them."

## 2. Player Count Differs Between Experiments at Same FPS

At 10fps on the same 30-second clip:
- **Experiment B (no tracking):** 3,594 players
- **Experiment A (with ByteTrack):** 3,107 players

ByteTrack filters duplicates and short-lived tracks, reducing the count. Mention this:

> "The FPS comparison includes ByteTrack tracking, which suppresses duplicates and short-lived tracks, accounting for the lower player count (3,107 vs 3,594) compared to the crop comparison experiment which uses raw per-frame detections."

## 3. No Figures Exist for the FPS Comparison (Main Experiment)

All 8 figures in `export/` are from Experiment B (crop comparison). **You need to generate at least one figure for Experiment A.** Critical one:

### Detection Rate vs Frame Rate
- Bar or line chart
- X-axis: 2, 5, 10, 25 fps
- Y-axis: detection rate (%)
- Horizontal dashed line at 94.2% labeled "Reference (5min, 25fps)"
- Data: 2fps=93.0%, 5fps=94.0%, 10fps=92.6%, 25fps=89.1%
- Raw data in `fps_comparison_results.json`
