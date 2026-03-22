#!/usr/bin/env python3
"""Test frame extraction from clerk.mp4 with resize"""

import cv2
from pathlib import Path

video_path = Path("input_videos/isaacsim/guide/IMG_0507/clerk.mp4")

if not video_path.exists():
    print(f"ERROR: {video_path} not found")
    exit(1)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"ERROR: Could not open {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration_sec = frame_count / fps if fps > 0 else 0

# Calculate sampling at 2Hz
sampling_hz = 2.0
target_count = int(duration_sec * sampling_hz)

print(f"Video: {video_path}")
print(f"  FPS: {fps}")
print(f"  Frame Count: {frame_count}")
print(f"  Duration: {duration_sec:.2f} seconds")
print(f"  Sampling at {sampling_hz}Hz → {target_count} frames")

# Extract sample frames with resize
sample_frames = []
resize_size = (640, 480)

for i in range(min(5, target_count)):  # Extract first 5 frames as sample
    target_sec = i / sampling_hz
    cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
    ok, frame = cap.read()
    if ok:
        frame_resized = cv2.resize(frame, resize_size)
        sample_frames.append(frame_resized)
        print(f"  ✓ Frame {i} @ {target_sec:.2f}s: original {frame.shape} → resized {frame_resized.shape}")
    else:
        print(f"  ✗ Frame {i}: Failed to read")

cap.release()

print(f"\nSuccess: Extracted {len(sample_frames)} sample frames from {video_path}")
print(f"Total frames to extract at 2Hz: {target_count}")

