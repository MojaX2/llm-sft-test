#!/usr/bin/env python3
"""Create per-scene horizontal 1Hz strip images for thirdperson and clerk videos.

For each scene directory that contains both `thirdperson.mp4` and `clerk.mp4`,
this script samples frames at fixed Hz on a shared timeline and writes a
comparison image:
- top row: thirdperson frames (left-to-right in time)
- bottom row: clerk frames (left-to-right in time)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import textwrap
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_caption_text(text: str) -> str:
    return " ".join(str(text).split())


def read_caption_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        return records

    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]

    return []


def load_caption_map(source: Path) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    if not source.exists():
        logger.warning("Caption source does not exist: %s", source)
        return caption_map

    if source.is_file():
        records = read_caption_records(source)
    else:
        records = []
        json_files = sorted(p for p in source.glob("*.json") if p.is_file())
        for p in json_files:
            records.extend(read_caption_records(p))

        teacher_jsonl = source / "teacher_raw.jsonl"
        if teacher_jsonl.exists():
            records.extend(read_caption_records(teacher_jsonl))

    for rec in records:
        video_id = rec.get("video_id")
        caption = rec.get("caption")
        if not isinstance(video_id, str) or not video_id.strip():
            continue
        if not isinstance(caption, str) or not caption.strip():
            continue
        caption_map[video_id.strip()] = normalize_caption_text(caption)

    logger.info("Loaded %s caption(s) from %s", len(caption_map), source)
    return caption_map


def build_caption_lookup_keys(scene_id: str, clerk_path: Path) -> list[str]:
    keys: list[str] = [scene_id]

    parts = scene_id.split("_")
    if len(parts) >= 3 and parts[1].upper() == "IMG":
        keys.append(f"clerk_{parts[0]}_{parts[2]}")

    scenario = clerk_path.parent.parent.name
    img_dir = clerk_path.parent.name
    if img_dir.startswith("IMG_"):
        clip_id = img_dir.split("_", 1)[1]
        keys.append(f"clerk_{scenario}_{clip_id}")

    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def find_caption_for_scene(scene_id: str, clerk_path: Path, caption_map: dict[str, str]) -> str | None:
    for key in build_caption_lookup_keys(scene_id, clerk_path):
        if key in caption_map:
            return caption_map[key]
    return None


def discover_scenes(input_root: Path) -> list[tuple[str, Path, Path]]:
    scenes: list[tuple[str, Path, Path]] = []
    for clerk_path in sorted(input_root.rglob("clerk.mp4")):
        scene_dir = clerk_path.parent
        third_path = scene_dir / "thirdperson.mp4"
        if not third_path.exists():
            continue

        scene_id = str(scene_dir.relative_to(input_root)).replace("/", "_")
        scenes.append((scene_id, third_path, clerk_path))

    return scenes


def get_video_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid fps/frame_count: {video_path}")
        return frame_count / fps
    finally:
        cap.release()


def build_timestamps(duration_sec: float, sampling_hz: float) -> list[float]:
    count = max(1, int(math.floor(duration_sec * sampling_hz)))
    return [i / sampling_hz for i in range(count)]


def sample_frames(
    video_path: Path,
    timestamps_sec: list[float],
    tile_width: int,
    tile_height: int,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    fallback = np.full((tile_height, tile_width, 3), 30, dtype=np.uint8)
    try:
        for ts in timestamps_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok:
                frames.append(fallback.copy())
                continue

            frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()

    return frames


def add_time_overlay(frame: np.ndarray, ts: float) -> np.ndarray:
    out = frame.copy()
    label = f"t={ts:.1f}s"
    cv2.rectangle(out, (8, 8), (110, 34), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_row(label: str, frames: list[np.ndarray], timestamps_sec: list[float], gap_px: int) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to build row")

    h, w, _ = frames[0].shape
    label_w = 150
    label_panel = np.full((h, label_w, 3), 245, dtype=np.uint8)
    cv2.putText(label_panel, label, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)

    parts: list[np.ndarray] = [label_panel]
    gap = np.full((h, gap_px, 3), 255, dtype=np.uint8)
    parts.append(gap)

    for idx, frame in enumerate(frames):
        parts.append(add_time_overlay(frame, timestamps_sec[idx]))
        if idx != len(frames) - 1:
            parts.append(gap)

    return cv2.hconcat(parts)


def build_header(width: int, scene_id: str, n_frames: int, sampling_hz: float) -> np.ndarray:
    header_h = 70
    header = np.full((header_h, width, 3), 250, dtype=np.uint8)
    cv2.putText(header, f"Scene: {scene_id}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        header,
        f"Sampling: {sampling_hz:.2f} Hz | Frames per row: {n_frames}",
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return header


def build_caption_panel(width: int, gpt_caption: str | None, cosmos_caption: str | None) -> np.ndarray:
    entries = [
        ("GPT-5.4", gpt_caption),
        ("Cosmos-Reason2", cosmos_caption),
    ]

    max_chars = max(40, (width - 40) // 10)
    wrapped_blocks: list[tuple[str, list[str]]] = []
    total_lines = 0

    for label, text in entries:
        body = normalize_caption_text(text) if text else "(caption not found)"
        lines = textwrap.wrap(
            body,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not lines:
            lines = ["(empty)"]
        wrapped_blocks.append((label, lines))
        total_lines += 1 + len(lines)

    top_margin = 16
    bottom_margin = 16
    line_h = 22
    block_gap = 12
    panel_h = top_margin + bottom_margin + total_lines * line_h + (len(wrapped_blocks) - 1) * block_gap
    panel = np.full((panel_h, width, 3), 246, dtype=np.uint8)

    y = top_margin + line_h
    for idx, (label, lines) in enumerate(wrapped_blocks):
        cv2.putText(panel, f"{label}:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
        y += line_h
        for line in lines:
            cv2.putText(panel, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA)
            y += line_h
        if idx != len(wrapped_blocks) - 1:
            y += block_gap

    return panel


def render_scene(
    scene_id: str,
    third_path: Path,
    clerk_path: Path,
    output_dir: Path,
    gpt_caption: str | None,
    cosmos_caption: str | None,
    sampling_hz: float,
    tile_width: int,
    tile_height: int,
    gap_px: int,
) -> Path:
    dur_third = get_video_duration_sec(third_path)
    dur_clerk = get_video_duration_sec(clerk_path)
    shared_duration = min(dur_third, dur_clerk)

    timestamps_sec = build_timestamps(shared_duration, sampling_hz)
    third_frames = sample_frames(third_path, timestamps_sec, tile_width, tile_height)
    clerk_frames = sample_frames(clerk_path, timestamps_sec, tile_width, tile_height)

    row_third = build_row("thirdperson", third_frames, timestamps_sec, gap_px)
    row_clerk = build_row("clerk", clerk_frames, timestamps_sec, gap_px)

    width = max(row_third.shape[1], row_clerk.shape[1])
    if row_third.shape[1] < width:
        pad = np.full((row_third.shape[0], width - row_third.shape[1], 3), 255, dtype=np.uint8)
        row_third = cv2.hconcat([row_third, pad])
    if row_clerk.shape[1] < width:
        pad = np.full((row_clerk.shape[0], width - row_clerk.shape[1], 3), 255, dtype=np.uint8)
        row_clerk = cv2.hconcat([row_clerk, pad])

    spacer = np.full((10, width, 3), 255, dtype=np.uint8)
    caption_spacer = np.full((14, width, 3), 255, dtype=np.uint8)
    header = build_header(width, scene_id, len(timestamps_sec), sampling_hz)
    caption_panel = build_caption_panel(width, gpt_caption, cosmos_caption)
    canvas = cv2.vconcat([header, spacer, row_third, spacer, row_clerk, caption_spacer, caption_panel])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scene_id}_strip.png"
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Failed to write image: {out_path}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-scene strip images from thirdperson/clerk videos")
    parser.add_argument("--input-root", type=Path, default=Path("input_videos/isaacsim"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_frames/scene_strips_1hz"))
    parser.add_argument("--sampling-hz", type=float, default=1.0)
    parser.add_argument("--tile-width", type=int, default=240)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--gap-px", type=int, default=4)
    parser.add_argument("--gpt-caption-source", type=Path, default=Path("data/ground_truth/raw"))
    parser.add_argument("--cosmos-caption-source", type=Path, default=Path("data/cosmos-reason2-2b"))
    parser.add_argument("--limit-scenes", type=int, default=0, help="0 means all scenes")
    args = parser.parse_args()

    if args.sampling_hz <= 0:
        raise ValueError("sampling_hz must be > 0")

    scenes = discover_scenes(args.input_root)
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]

    gpt_caption_map = load_caption_map(args.gpt_caption_source)
    cosmos_caption_map = load_caption_map(args.cosmos_caption_source)

    logger.info("Discovered %s scenes", len(scenes))
    if not scenes:
        raise RuntimeError(f"No scenes found under: {args.input_root}")

    generated = 0
    for scene_id, third_path, clerk_path in scenes:
        try:
            gpt_caption = find_caption_for_scene(scene_id, clerk_path, gpt_caption_map)
            cosmos_caption = find_caption_for_scene(scene_id, clerk_path, cosmos_caption_map)
            out_path = render_scene(
                scene_id=scene_id,
                third_path=third_path,
                clerk_path=clerk_path,
                output_dir=args.output_dir,
                gpt_caption=gpt_caption,
                cosmos_caption=cosmos_caption,
                sampling_hz=args.sampling_hz,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                gap_px=args.gap_px,
            )
            generated += 1
            logger.info("Generated: %s", out_path)
        except Exception as e:
            logger.exception("Failed scene %s: %s", scene_id, e)

    logger.info("Done. generated=%s/%s", generated, len(scenes))


if __name__ == "__main__":
    main()
