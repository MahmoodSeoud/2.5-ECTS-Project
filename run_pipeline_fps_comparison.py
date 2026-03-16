"""
Reconstruct the skeleton extraction pipeline from the thesis diagram:
  YOLO detect → ByteTrack → Box expand (0.3) → SAM2 segment → Blur crop → YOLO pose

Run at multiple FPS rates and compare to the 5min JSON ground truth.
"""
import os
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

import sys
import cv2
import json
import time
import numpy as np
from tqdm import tqdm
import supervision as sv
from inference import get_model
from ultralytics import YOLO

# SAM2
sys.path.insert(0, './segment-anything-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ─── Configuration (matched to thesis diagram) ───
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
if not ROBOFLOW_API_KEY:
    raise ValueError("Set ROBOFLOW_API_KEY")

DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
SOURCE_VIDEO_PATH = "./export/bif_fcn_20min.mp4"
DEVICE = 'cuda'
PLAYER_ID = 2
EXPAND_RATIO = 0.3          # Thesis uses 0.3
CONF_THRESHOLD = 0.3
MIN_KEYPOINTS = 6
TEST_DURATION_SECONDS = 30   # Same 30s clip for all FPS rates

FPS_RATES = [25, 10, 5, 2]  # Test these target FPS rates

# ─── Helper functions (matched to thesis) ───

def expand_box(box, frame_shape, expand_ratio=0.3):
    """Expand bounding box. Thesis uses ratio=0.3, vertical expansion 1.5x."""
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    expand_x = box_w * expand_ratio
    expand_y = box_h * expand_ratio * 1.5
    return np.array([
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(w, x2 + expand_x),
        min(h, y2 + expand_y)
    ])


def create_isolated_crop(frame, mask, min_padding=30, max_padding=60):
    """Create isolated player crop with blurred background."""
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None, None

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    player_height = y_max - y_min
    if player_height < 80:
        padding = max_padding
    elif player_height < 150:
        padding = (min_padding + max_padding) // 2
    else:
        padding = min_padding

    h, w = frame.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    crop = frame[y_min:y_max, x_min:x_max].copy()
    mask_crop = mask[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None, None, None

    blurred = cv2.GaussianBlur(crop, (71, 71), 0)
    mask_3ch = np.stack([mask_crop] * 3, axis=-1)
    isolated = np.where(mask_3ch > 0, crop, blurred)
    return isolated, x_min, y_min


# ─── Load models once ───
print("Loading models...")
detection_model = get_model(model_id=DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
pose_model = YOLO("./export/yolo11x-pose.pt")

SAM2_CHECKPOINT = "./export/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("All models loaded.\n")

# Get video info
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
native_fps = video_info.fps
print(f"Video: {video_info.width}x{video_info.height} @ {native_fps}fps")
print(f"Test duration: {TEST_DURATION_SECONDS}s")
print(f"FPS rates to test: {FPS_RATES}\n")


def run_pipeline(target_fps):
    """Run the full pipeline at a given FPS rate. Returns results dict."""
    stride = max(1, round(native_fps / target_fps))
    effective_fps = native_fps / stride
    test_frames = int(native_fps * TEST_DURATION_SECONDS)
    effective_frames = len(range(0, test_frames, stride))

    print(f"\n{'='*70}")
    print(f"Running pipeline @ {target_fps}fps (stride={stride}, effective={effective_fps:.1f}fps)")
    print(f"Frames to process: {effective_frames}")
    print(f"{'='*70}")

    # ByteTrack tracker — fresh instance per run
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(effective_fps)
    )

    total_players = 0
    skeletons_detected = 0
    total_keypoints = 0
    high_quality = 0
    kp_per_skeleton = []
    frames_data = []
    frame_timings = []

    frame_generator = sv.get_video_frames_generator(
        SOURCE_VIDEO_PATH, stride=stride, end=test_frames
    )

    for frame_idx, frame in enumerate(tqdm(
        frame_generator, total=effective_frames,
        desc=f'{target_fps}fps'
    )):
        t_frame_start = time.time()
        actual_frame_num = frame_idx * stride

        # 1. Detect players
        result = detection_model.infer(frame, confidence=CONF_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result)
        players = detections[detections.class_id == PLAYER_ID]

        # 2. Track with ByteTrack
        players = tracker.update_with_detections(players)

        if len(players) == 0:
            frame_timings.append(time.time() - t_frame_start)
            continue

        total_players += len(players)

        # 3. SAM2 — set image once per frame
        sam2_predictor.set_image(frame)

        frame_players = []

        for i, box in enumerate(players.xyxy):
            tracker_id = int(players.tracker_id[i]) if players.tracker_id is not None else -1

            # 4. Expand box (ratio=0.3, matching thesis)
            expanded_box = expand_box(box, frame.shape, expand_ratio=EXPAND_RATIO)

            # 5. SAM2 segmentation
            mask, _, _ = sam2_predictor.predict(box=expanded_box, multimask_output=False)
            if mask is None or len(mask) == 0:
                continue

            player_mask = mask[0].astype(np.uint8)

            # 6. Background blur + crop
            isolated_crop, crop_x, crop_y = create_isolated_crop(frame, player_mask)
            if isolated_crop is None:
                continue

            # 7. YOLO pose on isolated crop
            pose_result = pose_model(isolated_crop, verbose=False, conf=CONF_THRESHOLD)

            if not pose_result or len(pose_result) == 0:
                continue
            if pose_result[0].keypoints is None or len(pose_result[0].keypoints) == 0:
                continue

            kpts_data = pose_result[0].keypoints[0]
            kpts_xy = kpts_data.xy.cpu().numpy()[0]   # (17, 2)
            confs = kpts_data.conf.cpu().numpy()[0]    # (17,)

            valid_kpts = int(np.sum(confs >= CONF_THRESHOLD))

            if valid_kpts >= MIN_KEYPOINTS:
                skeletons_detected += 1
                total_keypoints += valid_kpts
                kp_per_skeleton.append(valid_kpts)
                if valid_kpts >= 15:
                    high_quality += 1

                # 8. Transform keypoints back to global coordinates
                global_kpts = kpts_xy.copy()
                global_kpts[:, 0] += crop_x
                global_kpts[:, 1] += crop_y

                frame_players.append({
                    "tracker_id": tracker_id,
                    "bounding_box": box.tolist(),
                    "skeleton": {
                        "keypoints": global_kpts.tolist(),
                        "confidences": confs.tolist()
                    },
                    "valid_keypoints": valid_kpts
                })

        frames_data.append({
            "frame_number": actual_frame_num,
            "players": frame_players
        })

        frame_timings.append(time.time() - t_frame_start)

    det_rate = skeletons_detected / total_players * 100 if total_players else 0
    avg_kp = np.mean(kp_per_skeleton) if kp_per_skeleton else 0
    avg_frame_time = np.mean(frame_timings) * 1000 if frame_timings else 0

    results = {
        "target_fps": target_fps,
        "effective_fps": effective_fps,
        "stride": stride,
        "frames_processed": effective_frames,
        "total_players_detected": total_players,
        "skeletons_extracted": skeletons_detected,
        "detection_rate": round(det_rate, 2),
        "high_quality_15plus": high_quality,
        "avg_keypoints": round(float(avg_kp), 2),
        "total_keypoints": total_keypoints,
        "avg_frame_time_ms": round(avg_frame_time, 1),
        "kp_distribution": kp_per_skeleton,
        "frames": frames_data,
    }

    print(f"\n  Players detected:      {total_players}")
    print(f"  Skeletons extracted:   {skeletons_detected}")
    print(f"  Detection rate:        {det_rate:.1f}%")
    print(f"  Avg keypoints:         {avg_kp:.1f}")
    print(f"  High-quality (>=15):   {high_quality}")
    print(f"  Avg frame time:        {avg_frame_time:.1f} ms")

    return results


# ─── Run at all FPS rates ───
all_results = {}
for fps in FPS_RATES:
    all_results[fps] = run_pipeline(fps)


# ─── Load 5min JSON for comparison ───
print(f"\n\n{'='*70}")
print("COMPARISON TABLE")
print(f"{'='*70}")

ref_rate = None
try:
    with open("skeleton_data_5min.json") as f:
        ref = json.load(f)
    ref_stats = ref["statistics"]
    ref_total = ref_stats["total_players_detected"]
    ref_skel = ref_stats["total_skeletons_extracted"]
    ref_rate = ref_skel / ref_total * 100
    print(f"{'Reference (5min JSON @25fps)':<35} {'Det Rate':>10}  {'Avg KP':>8}  {'Total KP':>10}  {'Players':>8}")
    print(f"{'─'*80}")
    ref_rate_str = f"{ref_rate:.1f}%"
    ref_total_str = f"{ref_total:,}"
    print(f"{'5min JSON (ground truth)':<35} {ref_rate_str:>10}  {'17.0':>8}  {'1,150,628':>10}  {ref_total_str:>8}")
except FileNotFoundError:
    print("(5min JSON not found for comparison)")

print(f"{'─'*80}")
print(f"{'Our pipeline (30s clip)':<35} {'Det Rate':>10}  {'Avg KP':>8}  {'Total KP':>10}  {'Players':>8}")
print(f"{'─'*80}")

for fps in FPS_RATES:
    r = all_results[fps]
    label = f"  @ {fps}fps (stride={r['stride']})"
    det_str = f"{r['detection_rate']:.1f}%"
    kp_str = f"{r['avg_keypoints']:.1f}"
    tkp_str = f"{r['total_keypoints']:,}"
    tp_str = f"{r['total_players_detected']:,}"
    print(f"{label:<35} {det_str:>10}  {kp_str:>8}  {tkp_str:>10}  {tp_str:>8}")

print(f"{'='*70}")


# ─── Save all results ───
output = {
    "experiment": "FPS comparison - reconstructed pipeline",
    "config": {
        "video": SOURCE_VIDEO_PATH,
        "test_duration_s": TEST_DURATION_SECONDS,
        "expand_ratio": EXPAND_RATIO,
        "conf_threshold": CONF_THRESHOLD,
        "min_keypoints": MIN_KEYPOINTS,
        "detection_model": DETECTION_MODEL_ID,
        "pose_model": "yolo11x-pose",
        "sam2_model": "sam2.1_hiera_large",
    },
    "results": {
        str(fps): {k: v for k, v in all_results[fps].items() if k != "frames"}
        for fps in FPS_RATES
    },
    "reference_5min_detection_rate": ref_rate,
}

with open("fps_comparison_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSummary saved to fps_comparison_results.json")

# Save full frame-level data per FPS rate
for fps in FPS_RATES:
    fname = f"skeleton_data_{fps}fps.json"
    with open(fname, "w") as f:
        json.dump({
            "video_info": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": native_fps,
                "target_fps": fps,
                "stride": all_results[fps]["stride"],
            },
            "statistics": {
                "total_players_detected": all_results[fps]["total_players_detected"],
                "total_skeletons_extracted": all_results[fps]["skeletons_extracted"],
                "frames_processed": all_results[fps]["frames_processed"],
            },
            "frames": all_results[fps]["frames"],
        }, f)
    print(f"  Frame data saved to {fname}")

print("\nDone.")
