"""
Run YOLO pose on full frames at 10fps — baseline comparison.
Same 30s clip, same thresholds as SAM2 experiment.
"""
import cv2
import json
import time
import numpy as np
from ultralytics import YOLO

SOURCE_VIDEO_PATH = "./export/bif_fcn_20min.mp4"
TARGET_FPS = 10
TEST_DURATION_SECONDS = 30
CONF_THRESHOLD = 0.3
MIN_KEYPOINTS = 6

print("Loading YOLOv11x-pose...")
model = YOLO("./export/yolo11x-pose.pt")

cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
native_fps = cap.get(cv2.CAP_PROP_FPS)
stride = max(1, round(native_fps / TARGET_FPS))
test_frames = int(native_fps * TEST_DURATION_SECONDS)

print(f"Video FPS: {native_fps}, stride: {stride}, effective FPS: {native_fps/stride:.1f}")
print(f"Processing {TEST_DURATION_SECONDS}s...")

total_players = 0
total_with_skeleton = 0
total_keypoints = 0
high_quality = 0
kp_counts = []
timings = []
processed = 0

frame_idx = 0
while frame_idx < test_frames:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % stride != 0:
        frame_idx += 1
        continue

    t0 = time.time()
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    t1 = time.time()
    timings.append(t1 - t0)

    result = results[0]
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        n_det = len(keypoints)
        total_players += n_det

        for i in range(n_det):
            kps = keypoints[i]
            if confs is not None:
                valid = int(np.sum(confs[i] >= CONF_THRESHOLD))
            else:
                valid = int(np.sum((kps[:, 0] > 0) | (kps[:, 1] > 0)))
            kp_counts.append(valid)

            if valid >= MIN_KEYPOINTS:
                total_with_skeleton += 1
                total_keypoints += valid
                if valid >= 15:
                    high_quality += 1

    processed += 1
    if processed % 50 == 0:
        print(f"  {processed} frames done...")

    frame_idx += 1

cap.release()

det_rate = total_with_skeleton / total_players * 100 if total_players else 0
avg_kp = total_keypoints / total_with_skeleton if total_with_skeleton else 0
avg_time = np.mean(timings) * 1000

print(f"\n{'='*70}")
print(f"YOLO FULL-FRAME POSE @ {TARGET_FPS}fps  (30s clip)")
print(f"{'='*70}")
print(f"Frames processed:              {processed}")
print(f"Total persons detected:        {total_players}")
print(f"Skeletons (>={MIN_KEYPOINTS} kp):            {total_with_skeleton}")
print(f"Detection rate:                {det_rate:.1f}%")
print(f"High-quality (>=15 kp):        {high_quality}")
print(f"Avg keypoints/skeleton:        {avg_kp:.1f}")
print(f"Total keypoints:               {total_keypoints}")
print(f"Avg inference time/frame:      {avg_time:.1f} ms")
print(f"{'='*70}")
print(f"\nCOMPARISON (same 30s clip, {TARGET_FPS}fps):")
print(f"{'Method':<30} {'Det Rate':>10} {'Avg KP':>8} {'Total KP':>10}")
print(f"{'-'*60}")
print(f"{'Direct Crop':<30} {'21.6%':>10} {'14.7':>8} {'11,426':>10}")
print(f"{'Expanded Crop':<30} {'66.6%':>10} {'--':>8} {'--':>10}")
print(f"{'SAM2 + Blur':<30} {'86.1%':>10} {'14.5':>8} {'44,909':>10}")
print(f"{'YOLO Full-Frame':<30} {f'{det_rate:.1f}%':>10} {f'{avg_kp:.1f}':>8} {f'{total_keypoints:,}':>10}")

output = {
    "method": "YOLO full-frame pose",
    "config": {
        "model": "yolo11x-pose",
        "target_fps": TARGET_FPS,
        "stride": stride,
        "conf_threshold": CONF_THRESHOLD,
        "min_keypoints": MIN_KEYPOINTS,
        "test_duration_s": TEST_DURATION_SECONDS,
    },
    "results": {
        "frames_processed": processed,
        "total_players": total_players,
        "skeletons_extracted": total_with_skeleton,
        "detection_rate": round(det_rate, 2),
        "high_quality_15plus": high_quality,
        "avg_keypoints": round(avg_kp, 2),
        "total_keypoints": total_keypoints,
        "avg_inference_ms": round(avg_time, 1),
    },
    "kp_distribution": kp_counts,
}

with open("yolo_fullframe_10fps.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to yolo_fullframe_10fps.json")
