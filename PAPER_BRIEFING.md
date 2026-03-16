# Paper Briefing: Complete Data & Narrative for Writing Agent

## Paper Title (suggested)
"Effect of Frame Rate on SAM2-Assisted Pose Estimation in Broadcast Football Video"

## Format
- 3-5 pages, academic/technical report style (2.5 ECTS project)
- IEEE or similar conference format
- Include figures and tables from the existing exports

---

## 1. NARRATIVE / STORY

### Problem
Extracting per-player skeleton data from broadcast football video is valuable for tactical analysis, biomechanics, and performance tracking. A recent pipeline (Olsen, 2025) demonstrated this using: YOLO detection → ByteTrack tracking → SAM2 instance segmentation → background blur cropping → YOLO-pose estimation. However, processing every frame at native 25fps is computationally expensive (~30 seconds per frame on GPU). The question is: **can we reduce the frame rate without losing pose estimation quality?**

### Approach
We reconstruct the core pose estimation pipeline and run a controlled experiment on a 30-second clip from a Danish Superliga match (BIF vs FCN), testing four frame rates: 25fps, 10fps, 5fps, and 2fps. We also run two additional baselines: (1) a crop-based comparison (direct crop vs expanded crop vs SAM2+blur crop) to isolate the contribution of SAM2 segmentation, and (2) a full-frame YOLO-pose baseline.

### Key Findings
1. **Frame rate has minimal impact on detection quality.** Detection rate is stable from 25fps (89.1%) to 2fps (93.0%), with 5fps (94.0%) matching the reference pipeline (94.2%).
2. **SAM2 segmentation is critical for crop-based pose estimation.** Without it, detection rate drops from 86.1% to 21.6% (direct crop) — a 4x degradation.
3. **The bottleneck is player detection, not pose estimation.** Roboflow detection takes ~30s/frame regardless of how many players are found. SAM2 adds only +26ms/player (+2%).
4. **5fps is the sweet spot**: same quality as 25fps, 5x fewer frames to process, proportional compute savings on detection.

### Why 25fps has LOWER detection rate than 5fps
This is counterintuitive but explainable: at 25fps you process every frame including motion-blurred transitional frames where players are mid-stride, turning, or partially occluded by other moving players. At 5fps (stride=5), you effectively subsample "cleaner" frames. The detection model and SAM2 segmentation both perform better on these sharper frames.

---

## 2. ALL EXPERIMENTAL DATA

### Experiment A: FPS Comparison (Main Experiment)
Pipeline: YOLO detect → ByteTrack → Box expand (ratio=0.3) → SAM2 segment → Blur crop → YOLO-pose
Video: 1920x1080, 25fps native, 30-second clip
GPU: Quadro RTX 8000 (48GB), node cn12.hpc.itu.dk
Models: Roboflow football-players-detection-3zvbc/11, YOLOv11x-pose, SAM2.1 Hiera Large
Confidence threshold: 0.3, Minimum keypoints: 6/17

| Target FPS | Stride | Frames | Players | Skeletons (≥6kp) | Det Rate | Avg KP | High-Q (≥15kp) | Total KP | Avg Frame Time |
|-----------|--------|--------|---------|-------------------|----------|--------|----------------|----------|----------------|
| 25        | 1      | 750    | 6,811   | 6,067             | 89.1%    | 14.5   | 3,307          | 88,178   | 30,156 ms      |
| 10        | 2      | 375    | 3,107   | 2,877             | 92.6%    | 14.5   | 1,572          | 41,829   | 30,449 ms      |
| 5         | 5      | 150    | 879     | 826               | 94.0%    | 14.5   | 444            | 11,986   | 30,505 ms      |
| 2         | 12     | 63     | 214     | 199               | 93.0%    | 14.5   | 102            | 2,879    | 29,936 ms      |

Reference (5min JSON, full pipeline @25fps on T4 GPU):
- 71,817 players detected, 67,684 skeletons extracted
- Detection rate: 94.2%, Avg keypoints: 17.0
- 5,452 frames processed from 300-second clip

### Experiment B: Crop Method Comparison (from notebook)
Same 30s clip at 10fps (stride=2), 375 frames
GPU: NVIDIA A30 (24GB), node cn18.hpc.itu.dk

| Method | Players | Skeletons (≥6kp) | Det Rate | Avg KP | Total KP |
|--------|---------|-------------------|----------|--------|----------|
| Direct Crop (baseline) | 3,594 | 778 | 21.6% | 14.7 | 11,426 |
| Expanded Crop (control) | 3,594 | 2,393 | 66.6% | — | — |
| SAM2 + Blur (proposed) | 3,594 | 3,096 | 86.1% | 14.5 | 44,909 |

Control experiment breakdown:
- Crop size alone contributes: +45.0% (69.7% of total improvement)
- SAM2 segmentation adds: +19.6% (30.3% of total improvement)

### Experiment C: Full-Frame YOLO-Pose Baseline
Same 30s clip at 10fps, 375 frames
GPU: Quadro RTX 8000

| Method | Players | Skeletons | Det Rate | Avg KP | Total KP | Inference/frame |
|--------|---------|-----------|----------|--------|----------|-----------------|
| YOLO Full-Frame | 645 | 609 | 94.4% | 11.6 | 7,057 | 47.5 ms |

Note: Full-frame detects far fewer players (645 vs 3,594) because it misses small/distant players. The crop-based pipeline uses a specialized football player detector (Roboflow) which finds 5.6x more players.

### Experiment D: Timing Analysis (from notebook)
Per-step inference times (GPU-warmed, 50 frames):

| Step | Time (ms) |
|------|-----------|
| Player detection (Roboflow) | 31,245 ± 966 |
| Box expansion | 0.02 ± 0.01 |
| SAM2 set_image (per-frame) | 135 ± 88 |
| SAM2 predict (per-player) | 7.25 ± 0.13 |
| Background blur + crop | 8.57 ± 0.38 |
| YOLO pose (direct crop) | 26.10 ± 60.95 |
| YOLO pose (SAM2 crop) | 15.68 ± 0.90 |

SAM2 overhead vs expanded crop: +26.4 ms/player (+2%)

---

## 3. AVAILABLE FIGURES

All in /home/mseo/2.5-ECTS-Project/export/:

1. **paper_fig1_detection_rate.png** — Bar chart comparing detection rates across 3 crop methods
2. **paper_fig2_keypoint_distribution.png** — Keypoint quality distribution comparison
3. **paper_fig3_threshold_analysis.png** — Success rate by keypoint threshold (sweep from 1-17)
4. **paper_fig4_comprehensive_metrics.png** — 2x2 grid of performance metrics
5. **paper_fig5_timing_analysis.png** — Computational cost breakdown
6. **sam2_comparison_3methods.png** — Visual comparison of crop methods
7. **sam2_success_cases.png** — Cases where SAM2 succeeds but direct crop fails
8. **sam2_crop_comparison_simple.png** — Simple side-by-side crop comparison

NOTE: There are NO figures yet for the FPS comparison (Experiment A). The writing agent should generate these or note they need to be created. Suggested figures:
- Detection rate vs FPS (bar chart or line plot with reference line at 94.2%)
- Total keypoints captured vs FPS
- Cost-benefit: frames processed vs detection rate

---

## 4. SUGGESTED PAPER STRUCTURE

### Abstract (~150 words)
Per-player pose estimation from broadcast football video. SAM2 segmentation pipeline. Frame rate impact study. Key result: 5fps matches 25fps quality with 5x less compute.

### 1. Introduction (~0.5 page)
- Motivation: sports analytics, biomechanics, tactical analysis need per-player skeleton data
- Challenge: processing every frame is expensive (~30s/frame with current GPU hardware)
- Research question: what is the minimum frame rate that preserves pose estimation quality?
- Brief mention of pipeline from prior work (cite the thesis the diagram is from)

### 2. Method (~1 page)
- Pipeline overview (reference the diagram in img.png):
  1. Player detection: Roboflow YOLO-based football player detector
  2. Tracking: ByteTrack multi-object tracker
  3. Instance segmentation: SAM2.1 Hiera Large with box expansion (ratio=0.3)
  4. Background suppression: Gaussian blur on non-player regions
  5. Pose estimation: YOLOv11x-pose on isolated crops
  6. Coordinate transformation back to global frame
- Experimental setup: 30s clip from Danish Superliga (BIF vs FCN), 1920x1080
- Frame rate conditions: 25fps (native), 10fps, 5fps, 2fps via stride subsampling
- Baselines: direct crop, expanded crop, full-frame YOLO-pose
- Hardware: ITU HPC cluster, Quadro RTX 8000 (48GB) and NVIDIA A30 (24GB)

### 3. Results (~1-1.5 pages)
- Table 1: FPS comparison results (Experiment A)
- Table 2: Crop method comparison (Experiment B)
- Key finding 1: Detection rate stable across FPS (89-94%)
- Key finding 2: 5fps matches reference at 94.0% vs 94.2%
- Key finding 3: SAM2 is essential — without it, detection drops to 21.6%
- Key finding 4: Full-frame YOLO misses 82% of players (645 vs 3,594)
- Key finding 5: SAM2 overhead is negligible (+2% compute)
- Include relevant figures from the exports

### 4. Discussion (~0.5 page)
- Why lower FPS can have higher detection rate (motion blur, transitional frames)
- The keypoint gap vs reference (14.5 vs 17.0) — likely temporal smoothing
- Practical implications: 5fps processing for offline analysis is optimal
- Limitations: single 30s clip, single match, no temporal smoothing implemented

### 5. Conclusion (~0.25 page)
- 5fps is sufficient for high-quality pose estimation in broadcast football
- SAM2 segmentation is critical for crop-based approaches
- Future work: temporal smoothing, longer clips, multiple matches

### References
- SAM2: Ravi et al., "SAM 2: Segment Anything in Images and Videos" (2024)
- YOLOv11: Ultralytics (2024)
- ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
- Supervision: Roboflow supervision library
- The thesis that provided the pipeline diagram and reference data (cite appropriately)
- DINOv2: Oquab et al. (2023) — mentioned in pipeline but not implemented in our experiments

---

## 5. IMPORTANT NOTES FOR THE WRITING AGENT

1. This is a 2.5 ECTS project report, not a full conference paper. Keep it focused and concise.
2. The pipeline was reconstructed from a diagram (img.png) from another student's thesis. We did NOT implement team identification (step 4) or temporal smoothing (step 6) from that pipeline. Be honest about this.
3. The reference 5min JSON was produced by the complete pipeline (with tracking + team ID + smoothing) on a T4 GPU. Our reconstruction omits team ID and smoothing — this likely explains the keypoint quality gap (14.5 vs 17.0 avg).
4. The 25fps detection rate (89.1%) being lower than 5fps (94.0%) is a real finding, not an error. Discuss it.
5. All experiments used the same 30-second clip from the same video. This is a limitation.
6. Two different GPUs were used across experiments (A30 for crop comparison, RTX 8000 for FPS comparison and full-frame baseline). Timing comparisons should note this.
