#!/usr/bin/env python3
"""
Generate caption for each video using cosmos-reason2-2b.

Reads video_manifest.csv, extracts frames, encodes as base64,
and calls an OpenAI-compatible API endpoint to generate teacher data (caption only).

Output:
  - Individual JSON files in data/cosmos-reason2-2b/{video_id}.json
  - Aggregated JSONL in data/cosmos-reason2-2b/teacher_raw.jsonl
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load active rows from manifest CSV."""
    videos: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("active", "").lower() in ("true", "1", "yes"):
                videos.append(row)
    logger.info("Loaded %s active video(s) from %s", len(videos), manifest_path)
    return videos


def extract_frames_from_video(
    video_path: Path,
    max_frames: int | None = None,
    sampling_hz: float = 2.0,
    resize_size: tuple[int, int] = (640, 480),
) -> list[bytes]:
    """Extract JPEG frame bytes from a video at a target sampling rate."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS or frame count for {video_path}")

    duration_sec = frame_count / fps
    max_frames_by_duration = int(duration_sec * sampling_hz)
    target_count = max_frames_by_duration if max_frames is None else min(max_frames, max_frames_by_duration)

    frames_bytes: list[bytes] = []
    try:
        for i in range(target_count):
            target_sec = i / sampling_hz
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ok, frame = cap.read()
            if not ok:
                logger.warning("Could not read frame at %.2fs", target_sec)
                break

            frame_resized = cv2.resize(frame, resize_size)
            ok, encoded = cv2.imencode(".jpg", frame_resized)
            if not ok:
                logger.warning("Could not encode frame at %.2fs", target_sec)
                continue

            frames_bytes.append(encoded.tobytes())
    finally:
        cap.release()

    logger.info("Extracted %s frame(s) from %s", len(frames_bytes), video_path)
    return frames_bytes


def encode_frames_to_base64(frames_bytes: list[bytes]) -> list[str]:
    return [base64.b64encode(f).decode("utf-8") for f in frames_bytes]


def extract_response_text(message: Any) -> str:
    """Extract text payload from chat completion message content."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                chunks.append(getattr(item, "text", ""))
        return "".join(chunks)
    return ""


def clean_caption_text(response_text: str) -> str:
    """Normalize model output into plain caption text."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.lower().startswith("caption:"):
        text = text.split(":", 1)[1].strip()
    # Fallback for outputs like "thinking ... caption ..."
    marker = "caption"
    lower = text.lower()
    idx = lower.rfind(marker)
    if idx != -1 and idx > 0:
        tail = text[idx + len(marker):].lstrip(" :\n\t")
        if tail:
            text = tail
    return text


def save_debug_response(
    debug_dir: Path | None,
    video_id: str,
    attempt: int,
    reason: str,
    response_text: str,
    finish_reason: Any,
) -> None:
    """Persist raw model responses for debugging parse/format issues."""
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = reason.replace(" ", "_")
    out_path = debug_dir / f"{video_id}_attempt{attempt:02d}_{safe_reason}.txt"
    payload = (
        f"video_id: {video_id}\n"
        f"attempt: {attempt}\n"
        f"reason: {reason}\n"
        f"finish_reason: {finish_reason}\n"
        f"response_text_start\n{response_text}\nresponse_text_end\n"
    )
    out_path.write_text(payload, encoding="utf-8")


def call_cosmos_api(
    client: OpenAI,
    model_name: str,
    frames_base64: list[str],
    video_id: str,
    max_retries: int = 10,
    response_format_mode: str = "json_schema",
    debug_dir: Path | None = None,
) -> str:
    """Call cosmos-reason2-2b endpoint and return caption text."""
    system_prompt = (
        "You are a video annotator for first-person cafe clerk perspectives. "
        "The camera wearer is a clerk at a cafe; other visible people are customers. "
        "Only describe what is directly visible in the frames and do not invent details."
    )

    user_prompt = (
        "Analyze this complete first-person video sequence from a clerk perspective. "
        "Write only one English caption (1-2 sentences) that explains both clerk intent and customer intent based only on visible evidence. "
        "Do not output JSON. Do not output labels. Output only caption text."
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for b64 in frames_base64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    for attempt in range(max_retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.7,
                "max_tokens": 1200,
            }
            if response_format_mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            elif response_format_mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "caption_only",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "caption": {"type": "string"},
                            },
                            "required": ["caption"],
                            "additionalProperties": False,
                        },
                    },
                }

            response = client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            response_text = extract_response_text(choice.message).strip()
            logger.info(
                "[%s] finish_reason=%s completion_tokens=%s",
                video_id,
                choice.finish_reason,
                getattr(response.usage, "completion_tokens", None),
            )

            if not response_text:
                save_debug_response(
                    debug_dir=debug_dir,
                    video_id=video_id,
                    attempt=attempt + 1,
                    reason="empty_response",
                    response_text=response_text,
                    finish_reason=choice.finish_reason,
                )
                raise ValueError(
                    "Empty response content "
                    f"(finish_reason={choice.finish_reason}, refusal={getattr(choice.message, 'refusal', None)})"
                )

            caption = clean_caption_text(response_text)
            if not caption:
                save_debug_response(
                    debug_dir=debug_dir,
                    video_id=video_id,
                    attempt=attempt + 1,
                    reason="empty_caption_after_normalize",
                    response_text=response_text,
                    finish_reason=choice.finish_reason,
                )
                raise ValueError("Caption text is empty after normalization")

            return caption

        except Exception as e:
            logger.warning("API failed for %s (attempt %s): %s", video_id, attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    raise RuntimeError(f"Failed for {video_id} after {max_retries} retries")


def save_result(
    video_info: dict[str, Any],
    caption: str,
    output_dir: Path,
) -> None:
    video_id = video_info["video_id"]
    result = {
        "video_id": video_id,
        "video_path": video_info["video_path"],
        "caption": caption,
        "language": "en",
        "model": "nvidia/Cosmos-Reason2-2B",
    }

    output_file = output_dir / f"{video_id}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def append_to_jsonl(
    video_info: dict[str, Any],
    caption: str,
    jsonl_path: Path,
) -> None:
    result = {
        "video_id": video_info["video_id"],
        "video_path": video_info["video_path"],
        "caption": caption,
        "language": "en",
        "model": "nvidia/Cosmos-Reason2-2B",
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def process_video(
    client: OpenAI,
    model_name: str,
    video_info: dict[str, Any],
    output_dir: Path,
    jsonl_path: Path,
    skip_existing: bool,
    response_format_mode: str,
    debug_dir: Path | None,
) -> bool:
    video_id = video_info["video_id"]
    video_path = Path(video_info["video_path"])

    output_file = output_dir / f"{video_id}.json"
    if skip_existing and output_file.exists():
        logger.info("Skipping %s (already processed)", video_id)
        return True

    try:
        frames_bytes = extract_frames_from_video(
            video_path=video_path,
            max_frames=None,
            sampling_hz=2.0,
            resize_size=(640, 480),
        )
        if not frames_bytes:
            logger.error("No frames extracted from %s", video_path)
            return False

        frames_base64 = encode_frames_to_base64(frames_bytes)
        caption = call_cosmos_api(
            client=client,
            model_name=model_name,
            frames_base64=frames_base64,
            video_id=video_id,
            response_format_mode=response_format_mode,
            debug_dir=debug_dir,
        )

        save_result(video_info, caption, output_dir)
        append_to_jsonl(video_info, caption, jsonl_path)
        logger.info("Saved result for %s", video_id)
        return True

    except Exception as e:
        logger.error("Failed to process %s: %s", video_id, e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate teacher data with cosmos-reason2-2b on OpenAI-compatible endpoint"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/video_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cosmos-reason2-2b"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Reason2-2B",
        help="Model name served by your OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("COSMOS_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL for cosmos server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("COSMOS_API_KEY", "EMPTY"),
        help="API key for cosmos endpoint (or OPENAI_API_KEY-compatible token)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--response-format-mode",
        type=str,
        choices=["none", "json_object", "json_schema"],
        default="json_schema",
        help="Structured output mode for response_format. Use 'none' if your server does not support it.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "teacher_raw.jsonl"
    debug_dir = args.output_dir / "_debug_raw"

    videos = load_manifest(args.manifest)
    if not videos:
        logger.warning("No active videos in manifest")
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    success = 0
    for video_info in videos:
        if process_video(
            client=client,
            model_name=args.model,
            video_info=video_info,
            output_dir=args.output_dir,
            jsonl_path=jsonl_path,
            skip_existing=args.skip_existing,
            response_format_mode=args.response_format_mode,
            debug_dir=debug_dir,
        ):
            success += 1

    logger.info("Successfully processed %s/%s video(s)", success, len(videos))


if __name__ == "__main__":
    main()
