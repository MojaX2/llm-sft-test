#!/usr/bin/env python3
"""Extract frames from a video at a target sampling rate.

Example:
    python extract_frames_sample.py --video input.mp4 --output output_frames

Dependency:
    pip install opencv-python
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_hz: float = 2.0,
    num_frames: int | None = None,
) -> list[Path]:
    if target_hz <= 0:
        raise ValueError("target_hz must be greater than 0")
    if num_frames is not None and num_frames <= 0:
        raise ValueError("num_frames must be greater than 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine FPS from the input video")

    # Sample at exact timestamps across the full video: 0.0, 1/Hz, 2/Hz, ...
    saved_paths: list[Path] = []

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = frame_count / fps if frame_count > 0 else None
    if duration_sec is None:
        cap.release()
        raise RuntimeError("Could not determine video duration from metadata")

    max_frames_by_duration = int(duration_sec * target_hz)
    target_count = max_frames_by_duration if num_frames is None else min(num_frames, max_frames_by_duration)

    try:
        for i in range(target_count):
            target_sec = i / target_hz

            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break

            out_path = output_dir / f"frame_{i:02d}.jpg"
            wrote = cv2.imwrite(str(out_path), frame)
            if not wrote:
                raise RuntimeError(f"Failed to write frame: {out_path}")
            saved_paths.append(out_path)
    finally:
        cap.release()

    if num_frames is not None and len(saved_paths) < num_frames:
        raise RuntimeError(
            f"Only extracted {len(saved_paths)} frame(s). "
            f"Needed {num_frames}. Try a longer video."
        )

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frames_2hz"),
        help="Output directory (default: frames_2hz)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=2.0,
        help="Sampling rate in Hz across the whole video (default: 2.0)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Optional max number of frames to extract (default: all video duration)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        target_hz=args.hz,
        num_frames=args.num_frames,
    )
    print("Extracted frames:")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()
