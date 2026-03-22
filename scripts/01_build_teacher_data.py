#!/usr/bin/env python3
"""
Generate thinking and caption for each video using GPT-5.4.

Reads video_manifest.csv, extracts frames, encodes as base64, 
and calls OpenAI API to generate teacher data (thinking + caption).

Output:
  - Individual JSON files in data/ground_truth/raw/{video_id}.json
  - Aggregated JSONL in data/ground_truth/raw/teacher_raw.jsonl
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
    """Load video manifest from CSV."""
    videos = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("active", "").lower() in ("true", "1", "yes"):
                videos.append(row)
    logger.info(f"Loaded {len(videos)} active video(s) from {manifest_path}")
    return videos


def extract_frames_from_video(
    video_path: Path,
    max_frames: int | None = None,
    sampling_hz: float = 2.0,
    resize_size: tuple[int, int] = (640, 480),
) -> list[bytes]:
    """
    Extract frames from video at specified sampling rate with resizing.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None = all video frames)
        sampling_hz: Sampling rate in Hz (2.0 = 2 frames per second)
        resize_size: Resize target (width, height). Default 640x480
    
    Returns:
        List of JPEG frame bytes
    """
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
    
    frames_bytes = []
    try:
        for i in range(target_count):
            target_sec = i / sampling_hz
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ok, frame = cap.read()
            
            if not ok:
                logger.warning(f"Could not read frame at {target_sec}s")
                break
            
            # Resize frame
            frame_resized = cv2.resize(frame, resize_size)
            
            # Encode frame to JPEG bytes
            ok, encoded = cv2.imencode(".jpg", frame_resized)
            if not ok:
                logger.warning(f"Could not encode frame at {target_sec}s")
                continue
            
            frames_bytes.append(encoded.tobytes())
    finally:
        cap.release()
    
    logger.info(f"Extracted {len(frames_bytes)} frame(s) from {video_path}")
    return frames_bytes


def encode_frames_to_base64(frames_bytes: list[bytes]) -> list[str]:
    """Encode frame bytes to base64 strings."""
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


def call_gpt54_api(
    client: OpenAI,
    frames_base64: list[str],
    video_id: str,
    max_retries: int = 3,
) -> dict[str, str]:
    """
    Call GPT-5.4 API with frame images to generate thinking and caption.
    
    Returns dict with keys: thinking, caption
    """
    system_prompt = (
'''You are a video annotator for first-person cafe clerk perspectives.'''
)
    
    user_prompt = (
'''Analyze this complete first-person video sequence from a CLERK's perspective. The camera wearer (you) is a clerk at a cafe. Other people are customers.
Task:
1. In 'thinking': Provide a detailed chronological sequence of visible actions, positions, and objects. Describe your movements as the clerk and the specific behaviors of the customers.
2. In 'caption': For both the CLERK (you) and the CUSTOMERS, explain the "Intent-Action Pairs" based on the visual evidence.
   - For the CLERK: What was your objective, and what specific action did you take to achieve it?
   - For the CUSTOMERS: What did they appear to want/need, and how did they express that through their actions?
Return valid JSON with these two keys: "thinking" and "caption".
Keep output concise: thinking <= 4 sentences, caption <= 2 sentences.
Output raw JSON only (no markdown, no extra text).'''
)
    
    # Build message content with images
    content = [{"type": "text", "text": user_prompt}]
    for b64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=0.7,
                max_completion_tokens=1200,
                response_format={"type": "json_object"},
            )

            choice = response.choices[0]
            response_text = extract_response_text(choice.message).strip()
            logger.info(
                "API finish_reason=%s completion_tokens=%s",
                choice.finish_reason,
                getattr(response.usage, "completion_tokens", None),
            )
            logger.info("API response: %s", response_text[:200])

            if not response_text:
                raise ValueError(
                    "Empty response content "
                    f"(finish_reason={choice.finish_reason}, refusal={getattr(choice.message, 'refusal', None)})"
                )
            
            # Remove markdown code block formatting if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate required keys
            if "thinking" not in result or "caption" not in result:
                raise ValueError("Missing 'thinking' or 'caption' in response")
            
            logger.info(f"Successfully generated thinking and caption for {video_id}")
            return result
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response for {video_id} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.warning(f"API call failed for {video_id} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    raise RuntimeError(f"Failed to generate thinking/caption for {video_id} after {max_retries} retries")


def save_result(
    video_info: dict[str, Any],
    thinking: str,
    caption: str,
    output_dir: Path,
) -> None:
    """Save individual JSON result."""
    video_id = video_info["video_id"]
    result = {
        "video_id": video_id,
        "video_path": video_info["video_path"],
        "thinking": thinking,
        "caption": caption,
        "language": "en",
    }
    
    output_file = output_dir / f"{video_id}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved result to {output_file}")


def append_to_jsonl(
    video_info: dict[str, Any],
    thinking: str,
    caption: str,
    jsonl_path: Path,
) -> None:
    """Append result to JSONL file."""
    result = {
        "video_id": video_info["video_id"],
        "video_path": video_info["video_path"],
        "thinking": thinking,
        "caption": caption,
        "language": "en",
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def process_video(
    client: OpenAI,
    video_info: dict[str, Any],
    raw_output_dir: Path,
    jsonl_path: Path,
    skip_existing: bool = True,
) -> bool:
    """
    Process a single video: extract frames, call API, save results.
    
    Returns True if successful, False otherwise.
    """
    video_id = video_info["video_id"]
    video_path = Path(video_info["video_path"])
    
    # Check if already processed
    individual_json = raw_output_dir / f"{video_id}.json"
    if skip_existing and individual_json.exists():
        logger.info(f"Skipping {video_id} (already processed)")
        return True
    
    try:
        # Extract frames
        frames_bytes = extract_frames_from_video(
            video_path,
            max_frames=None,  # Use entire video
            sampling_hz=2.0,   # 2 frames per second
            resize_size=(640, 480),  # Resize to 640x480
        )
        
        if not frames_bytes:
            logger.error(f"No frames extracted from {video_path}")
            return False
        
        # Encode to base64
        frames_base64 = encode_frames_to_base64(frames_bytes)
        
        # Call API
        result = call_gpt54_api(client, frames_base64, video_id)
        thinking = result.get("thinking", "")
        caption = result.get("caption", "")
        
        # Save results
        save_result(video_info, thinking, caption, raw_output_dir)
        append_to_jsonl(video_info, thinking, caption, jsonl_path)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to process {video_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate teacher data using GPT-5.4")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/video_manifest.csv"),
        help="Path to video manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ground_truth/raw"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip videos already processed",
    )
    args = parser.parse_args()
    
    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "teacher_raw.jsonl"
    
    # Load manifest
    videos = load_manifest(args.manifest)
    
    if not videos:
        logger.warning("No active videos found in manifest")
        return
    
    # Process each video
    success_count = 0
    for video_info in videos:
        if process_video(client, video_info, args.output_dir, jsonl_path, args.skip_existing):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(videos)} video(s)")


if __name__ == "__main__":
    main()
