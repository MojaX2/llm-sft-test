#!/usr/bin/env python3
"""
Pairwise winner-only evaluation between GPT-5.4 captions and Cosmos captions.

Inputs per sample:
- Video frames (sampled and resized)
- Caption from GPT output dir
- Caption from Cosmos output dir

Judge model decides winner only: gpt54 / cosmos / tie

Output:
- JSONL with per-video winner decisions
- Summary JSON with counts
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import random
import re
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
    rows: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("active", "").lower() in ("true", "1", "yes"):
                rows.append(row)
    return rows


def read_caption(caption_dir: Path, video_id: str) -> str:
    path = caption_dir / f"{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Caption file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    caption = str(data.get("caption", "")).strip()
    if not caption:
        raise ValueError(f"Caption is empty: {path}")
    return caption


def extract_frames(
    video_path: Path,
    sampling_hz: float,
    max_frames: int,
    resize: tuple[int, int],
) -> list[str]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS/frame_count for {video_path}")

    duration = frame_count / fps
    target = min(int(duration * sampling_hz), max_frames)
    frames_b64: list[str] = []

    try:
        for i in range(target):
            t = i / sampling_hz
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, resize)
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frames_b64.append(base64.b64encode(encoded.tobytes()).decode("utf-8"))
    finally:
        cap.release()

    return frames_b64


def parse_winner(text: str) -> str:
    t = text.strip().upper()
    # Accept common variants
    if re.search(r"\bTIE\b|\bDRAW\b", t):
        return "tie"
    if re.search(r"\bA\b", t):
        return "A"
    if re.search(r"\bB\b", t):
        return "B"
    raise ValueError(f"Could not parse winner from response: {text!r}")


def judge_one(
    client: OpenAI,
    model: str,
    video_id: str,
    video_frames_b64: list[str],
    cap_gpt: str,
    cap_cosmos: str,
    rng: random.Random,
    max_retries: int,
) -> dict[str, Any]:
    # Randomize A/B to reduce position bias
    if rng.random() < 0.5:
        a_model, a_caption = "gpt54", cap_gpt
        b_model, b_caption = "cosmos", cap_cosmos
    else:
        a_model, a_caption = "cosmos", cap_cosmos
        b_model, b_caption = "gpt54", cap_gpt

    system_prompt = (
        "You are a strict video-caption judge using a deduction-based rubric. "
        "Compare two candidate captions for the same video, count violations, "
        "and pick the one with fewer or less severe violations."
    )
    user_prompt = (
        "Task: Choose the better caption for this first-person cafe clerk video.\n"
        "Judge by deduction (lower total penalty is better):\n"
        "Critical (-3 each):\n"
        "- Hallucination: claims not visually supported by the video\n"
        "- Missing either clerk intent or customer intent entirely\n"
        "Major (-2 each):\n"
        "- Weak or vague intent wording when intent is visible\n"
        "- Noticeable mismatch in action/order of events\n"
        "Minor (-1 each):\n"
        "- Awkward or unnatural English\n"
        "- Too verbose beyond 1-2 sentences\n"
        "Decision rule:\n"
        "- Pick A or B unless both are truly equivalent in both factuality and intent coverage.\n"
        "- Use TIE only for near-identical quality and error profile.\n"
        "Output exactly one token: A, B, or TIE."
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for b64 in video_frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": f"Caption A:\n{a_caption}\n\nCaption B:\n{b_caption}",
        }
    )

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_completion_tokens=30,
            )
            raw = str(resp.choices[0].message.content or "").strip()
            winner_ab = parse_winner(raw)

            if winner_ab == "tie":
                winner_model = "tie"
            elif winner_ab == "A":
                winner_model = a_model
            else:
                winner_model = b_model

            return {
                "video_id": video_id,
                "winner": winner_model,
                "winner_ab": winner_ab,
                "a_model": a_model,
                "b_model": b_model,
                "a_caption": a_caption,
                "b_caption": b_caption,
                "judge_raw": raw,
            }
        except Exception as e:
            last_err = e
            logger.warning("Judge failed for %s (attempt %s/%s): %s", video_id, attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"Failed to judge {video_id}: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Winner-only caption evaluation")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/video_manifest.csv"))
    parser.add_argument("--gpt-dir", type=Path, default=Path("data/ground_truth/raw"))
    parser.add_argument("--cosmos-dir", type=Path, default=Path("data/cosmos-reason2-2b"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/eval/winner_only.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/eval/winner_only_summary.json"))
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--sampling-hz", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--resize-height", type=int, default=480)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)
    rng = random.Random(args.seed)

    rows = load_manifest(args.manifest)
    logger.info("Loaded %s active videos", len(rows))

    results: list[dict[str, Any]] = []
    wins = {"gpt54": 0, "cosmos": 0, "tie": 0}

    for row in rows:
        video_id = row["video_id"]
        video_path = Path(row["video_path"])

        cap_gpt = read_caption(args.gpt_dir, video_id)
        cap_cosmos = read_caption(args.cosmos_dir, video_id)
        frames_b64 = extract_frames(
            video_path=video_path,
            sampling_hz=args.sampling_hz,
            max_frames=args.max_frames,
            resize=(args.resize_width, args.resize_height),
        )

        result = judge_one(
            client=client,
            model=args.model,
            video_id=video_id,
            video_frames_b64=frames_b64,
            cap_gpt=cap_gpt,
            cap_cosmos=cap_cosmos,
            rng=rng,
            max_retries=args.max_retries,
        )
        results.append(result)
        wins[result["winner"]] += 1
        logger.info("%s winner=%s", video_id, result["winner"])

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total": len(results),
        "wins": wins,
        "model": args.model,
        "sampling_hz": args.sampling_hz,
        "max_frames": args.max_frames,
        "resize": [args.resize_width, args.resize_height],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Done. total=%s gpt54=%s cosmos=%s tie=%s", len(results), wins["gpt54"], wins["cosmos"], wins["tie"])
    logger.info("Saved: %s", args.output_jsonl)
    logger.info("Saved: %s", args.summary_json)


if __name__ == "__main__":
    main()
