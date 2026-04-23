"""VLM evaluation of robot policy rollout videos with Gemini Robotics ER.

For every ``*exo*.mp4`` under the given eval-output directory, extract 8
evenly-spaced frames and ask a Gemini Robotics ER model to classify the
success/failure mode of the rollout. Results are written to a JSONL file.

Usage:
    export GEMINI_API_KEY=...
    python scripts/vlm_rollout_eval.py \\
        --eval-dir /weka/prior/aguru/mujoco-thor/eval_output/.../20260320_022322 \\
        --output rollout_eval.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from google import genai
from google.genai import types
from PIL import Image

log = logging.getLogger("vlm_rollout_eval")

NUM_FRAMES = 8

PROMPT = """I want you to evaluate the performance of a vision language action model on the given robotics task (a cluttered scene task in a MuJoCo-based simulation benchmark). The task of the robot is to pick up a particular object within the cluttered scene. I will pass in 9 images relevant to the task and the robot's policy rollout. The first image depics the target object that needs to be picked up. It includes a frame early in the task, with the target object boxed in red. That image also includes a zoomed in crop of the target object. The next 8 images shown are in chronological order and come from the robot executing the task. Carefully analyze these sequential images and classify the success or failure mode that the policy experiences- i.e. whether the robot is successful in picking up the target object specified in the first image, based on how the rollout progresses in the last 8 images. Use the following partition of the outcome space for this task:
The robot doesn't attempt to grasp anything
Attempts to grasp an object in the scene —>
a. Does it go for the wrong object
b. Does it go for the right object
i. Does it successfully grasp the object
ii. Does it fail
1) Does it fail due to clutter
2) Does it just have some more "idiosyncratic" failure, general grasping failure
Do your diagnosis after carefully considering these factors:
Always be aware of which object is the target. For this, refer to the first image passed in, and the object boxed in red and with a zoomed in crop.
The path the policy takes. Does it run into other objects on its path toward the target object
Does the robot stop and attempt to grasp other objects, or does it keep going towards the target object
Always be aware of how long the robot is spending next to other objects, prior to its gripper being around the target object. If it is spending a lot of time next to another object, that should count as being distracted by that object or colliding with it.
Note that for a grasp wrong object classification, it doesn't have to successfully grasp the wrong object. The gripper just needs to move in that direction pretty noticeably/potentially touch that other object. These are all
valid cases for the grasp wrong object class.
If the robot gripper's trajectory appears to be getting progressively closer to the target object across frames, but it runs into a different object along the way, that should count as a failure due to clutter. Depending on the context, either it tries to grasp
the wrong object, or it runs runs into the wrong object on the way to the right object, which should count as a grasp_right_object_fail_clutter.
If the gripper gets close to the target object across frames, but then fails to grasp it without any clear collision with a different object, that should count as a grasp_right_object_fail_idiosyncratic.
Closely examine the gripper in all frames. What is it holding, if it is holding anything? Does the held item match the target item boxed in red on the right side of each image? Is it fully closed around the object?"""


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "outcome": {
            "type": "STRING",
            "enum": [
                "no_grasp_attempt",
                "grasp_wrong_object",
                "grasp_right_object_success",
                "grasp_right_object_fail_clutter",
                "grasp_right_object_fail_idiosyncratic",
            ],
            "description": "Leaf of the outcome partition described in the prompt.",
        },
        "path_analysis": {
            "type": "STRING",
            "description": "What path did the robot take? Did it collide with non-target objects on the way?",
        },
        "distraction_analysis": {
            "type": "STRING",
            "description": "Did the robot stop and attempt non-target objects, or drive toward the target?",
        },
        "gripper_analysis": {
            "type": "STRING",
            "description": "What is the gripper holding across frames? Does it match the red-boxed target? Is it fully closed around the object?",
        },
        "rationale": {
            "type": "STRING",
            "description": "One-paragraph overall justification for the chosen outcome.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in the classification in [0, 1].",
        },
    },
    "required": [
        "outcome",
        "path_analysis",
        "distraction_analysis",
        "gripper_analysis",
        "rationale",
        "confidence",
    ],
}


@dataclass(frozen=True)
class VideoJob:
    house: str
    episode: str
    video_path: Path
    target_image_path: Path


def discover_jobs(eval_dir: Path) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for house_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("house_")):
        for video in sorted(house_dir.glob("*exo*.mp4")):
            # episode_00000000_exo_camera_1_batch_1_of_1.mp4 -> episode_00000000
            parts = video.stem.split("_")
            episode = "_".join(parts[:2]) if len(parts) >= 2 and parts[0] == "episode" else video.stem
            target_image_path = house_dir / f"{episode}_target.png"
            jobs.append(VideoJob(
                house=house_dir.name,
                episode=episode,
                video_path=video,
                target_image_path=target_image_path,
            ))
    return jobs


def extract_frames(video_path: Path, num_frames: int = NUM_FRAMES) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Fall back to sequential read
            frames_bgr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames_bgr.append(f)
            if not frames_bgr:
                raise RuntimeError(f"No frames decoded from {video_path}")
            total = len(frames_bgr)
            indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
            sampled = [frames_bgr[i] for i in indices]
        else:
            indices = [round(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
            sampled = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
                sampled.append(f)
    finally:
        cap.release()

    images: list[Image.Image] = []
    for bgr in sampled:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    return images


def pil_to_part(img: Image.Image, jpeg_quality: int = 90) -> types.Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def analyze_video(
    client: genai.Client,
    model: str,
    video_path: Path,
    target_image_path: Path,
    max_retries: int = 3,
    debug_dir: Path | None = None,
) -> dict:
    if not target_image_path.is_file():
        raise FileNotFoundError(f"Target image not found: {target_image_path}")
    target_image = Image.open(target_image_path)
    frames = extract_frames(video_path, NUM_FRAMES)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        target_image.convert("RGB").save(debug_dir / "00_target.png")
        for i, frame in enumerate(frames, start=1):
            frame.convert("RGB").save(debug_dir / f"{i:02d}_frame.png")
        (debug_dir / "prompt.txt").write_text(PROMPT)

    parts: list = [pil_to_part(target_image)]
    parts.extend(pil_to_part(f) for f in frames)
    parts.append(PROMPT)

    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=config,
            )
            text = (response.text or "").strip()
            return json.loads(text)
        except Exception as e:  # noqa: BLE001
            # Fail fast on client errors (bad model id, bad request, auth, etc.).
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            msg = str(e)
            is_client_error = (
                isinstance(status, int) and 400 <= status < 500
            ) or any(tok in msg for tok in ("404", "400", "401", "403", "PERMISSION_DENIED", "INVALID_ARGUMENT", "NOT_FOUND"))
            if is_client_error:
                raise RuntimeError(f"Gemini client error (not retrying): {e}") from e
            last_err = e
            backoff = 2**attempt
            log.warning("Gemini call failed for %s (attempt %d/%d): %s; retrying in %ds",
                        video_path.name, attempt + 1, max_retries, e, backoff)
            time.sleep(backoff)
    raise RuntimeError(f"Gemini failed after {max_retries} attempts: {last_err}") from last_err


def process_job(
    job: VideoJob,
    client: genai.Client,
    model: str,
    debug_image_root: Path | None = None,
) -> dict:
    t0 = time.time()
    debug_dir = (
        debug_image_root / job.house / job.episode if debug_image_root is not None else None
    )
    try:
        result = analyze_video(
            client, model, job.video_path, job.target_image_path, debug_dir=debug_dir,
        )
        return {
            "house": job.house,
            "episode": job.episode,
            "video_path": str(job.video_path),
            "model": model,
            "elapsed_s": round(time.time() - t0, 2),
            "result": result,
        }
    except Exception as e:  # noqa: BLE001
        log.error("Failed on %s: %s", job.video_path, e)
        return {
            "house": job.house,
            "episode": job.episode,
            "video_path": str(job.video_path),
            "model": model,
            "elapsed_s": round(time.time() - t0, 2),
            "error": str(e),
        }


def load_done_keys(output_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                continue  # retry errored entries
            done.add((rec.get("house", ""), rec.get("episode", "")))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="Root evaluation directory containing house_* subdirectories.")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSONL output path. Appended to; existing successful entries are skipped.")
    parser.add_argument("--model", default="gemini-robotics-er-1.6-preview",
                        help="Gemini model id (default: %(default)s).")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent Gemini requests (default: %(default)s).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max videos to process (after resume filter). Handy for smoke tests.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--list-models", action="store_true",
                        help="List models your API key can call generateContent on, then exit.")
    parser.add_argument("--debug-image-dir", type=Path, default=None,
                        help="If set, dump the 9 images (and prompt) sent per query to "
                             "{dir}/{house}/{episode}/. Useful for inspecting exactly what the VLM sees.")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY in the environment.", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)

    if args.list_models:
        print("Models supporting generateContent for this API key:")
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None) or []
            if not actions or "generateContent" in actions:
                print(f"  {m.name}")
        return 0

    if args.eval_dir is None or args.output is None:
        print("ERROR: --eval-dir and --output are required unless --list-models is used.", file=sys.stderr)
        return 2

    if not args.eval_dir.is_dir():
        print(f"ERROR: --eval-dir does not exist: {args.eval_dir}", file=sys.stderr)
        return 2

    jobs = discover_jobs(args.eval_dir)
    log.info("Discovered %d exo videos across %d houses.",
             len(jobs), len({j.house for j in jobs}))

    done = load_done_keys(args.output)
    if done:
        log.info("Resuming: %d episodes already have successful results in %s.",
                 len(done), args.output)
    pending = [j for j in jobs if (j.house, j.episode) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    log.info("Processing %d pending videos with concurrency=%d.",
             len(pending), args.concurrency)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a") as out_f, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {
            pool.submit(process_job, j, client, args.model, args.debug_image_dir): j
            for j in pending
        }
        for fut in concurrent.futures.as_completed(futures):
            job = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {
                    "house": job.house,
                    "episode": job.episode,
                    "video_path": str(job.video_path),
                    "model": args.model,
                    "error": f"unhandled: {e}",
                }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            written += 1
            status = rec.get("result", {}).get("outcome") or rec.get("error", "?")
            log.info("[%d/%d] %s %s -> %s (%.1fs)",
                     written, len(pending), job.house, job.episode, status,
                     rec.get("elapsed_s", 0.0))

    log.info("Done. Wrote %d new records to %s", written, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
