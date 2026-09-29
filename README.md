# Frame rate vs. pose estimation in broadcast football video

2.5 ECTS project at ITU (2026). How much can you lower the frame rate of a SAM2-assisted pose estimation pipeline before the skeletons get worse?

The pipeline follows Olsen (2025): YOLO player detection, ByteTrack tracking, SAM2 segmentation, background-blurred crops, then YOLOv11x-pose. I rebuilt it and ran it on a 30-second clip from a Danish Superliga match (BIF vs FCN) at 25, 10, 5 and 2 fps.

## Results

| fps | Frames | Players | Skeletons (≥6 kp) | Detection rate |
|----:|-------:|--------:|------------------:|---------------:|
| 25 | 750 | 6,811 | 6,067 | 89.1% |
| 10 | 375 | 3,107 | 2,877 | 92.6% |
| 5 | 150 | 879 | 826 | 94.0% |
| 2 | 63 | 214 | 199 | 93.0% |

- 5 fps gives the same quality as 25 fps with a fifth of the frames.
- SAM2 is what makes crop-based pose work: without it the detection rate drops from 86.1% to 21.6%.
- The slow part is player detection (about 30 s per frame), not pose estimation. SAM2 adds about 26 ms per player.

Raw numbers are in `fps_comparison_results.json` and the `skeleton_data_*fps.json` files. `PAPER_BRIEFING.md` has the full write-up and `ERRATA.md` the corrections.

## Running it

- `run_fps_comparison.sh` runs the frame-rate experiment
- `run_fullframe.sh` runs the full-frame YOLO-pose baseline
